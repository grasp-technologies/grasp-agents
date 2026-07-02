import asyncio
import contextvars
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .agent.approval_store import ApprovalStore
from .durability.checkpoint_store import CheckpointStore
from .durability.checkpoints import CheckpointKind, SessionCheckpoint
from .durability.context_serialization import (
    ContextKind,
    rehydrate_context,
    serialize_context,
)
from .durability.store_keys import make_store_key
from .file_backend.base import FileBackend
from .memory.provider import MemoryProvider
from .printer import Printer
from .sandbox.environment import ExecutionEnvironment, SnapshotCapable
from .sandbox.exec_backend import ExecBackend
from .skills.registry import SkillRegistry
from .types.io import ProcName
from .types.response import Response
from .usage_tracker import UsageTracker

logger = logging.getLogger(__name__)

DEFAULT_SESSION_KEY = "default"
"""Sentinel ``session_key`` for an unnamed session.

Session-scoped stores route lookups by ``session_key``; trace grouping treats
this value as "no session" (each run is its own trace root). Set a real
``session_key`` to opt a session into single-trace grouping across runs.
"""


class SessionContext[CtxT](BaseModel):
    state: CtxT = None  # type: ignore

    # When True, ``state`` is persisted into the per-session
    # :class:`SessionCheckpoint` at every checkpoint boundary and restored
    # by :meth:`load_checkpoint` on a cold start. Opt in for tests / simple
    # workloads where ``state`` is a plain serializable container (pydantic /
    # dataclass / mapping); otherwise rebuild state yourself (e.g. from your
    # DB) and pass it at construction.
    serialize_state: bool = Field(default=False, exclude=True)

    # Filesystem-snapshot policy for checkpoints — session-scoped, like the
    # filesystem it snapshots. Requires :attr:`environment` to be
    # ``SnapshotCapable`` (e.g. E2B). ``"off"`` (default): never snapshot.
    # ``"final"``: snapshot at run-end boundaries (final answer / max turns /
    # resident turn). ``"turn"``: snapshot at every checkpoint boundary,
    # including after each tool batch — strongest rewind granularity, but
    # each snapshot costs a provider round-trip. Only the opaque ref is
    # stored (in the session checkpoint and per-step watermarks); the bytes
    # live with the snapshot owner.
    fs_snapshot_policy: Literal["off", "final", "turn"] = Field(
        default="off", exclude=True
    )

    responses: defaultdict[ProcName, list[Response]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    max_responses_per_agent: int | None = Field(default=None, exclude=True)
    usage_tracker: UsageTracker = Field(default_factory=UsageTracker, exclude=True)
    printer: Printer | None = Field(default=None, exclude=True)
    checkpoint_store: CheckpointStore | None = Field(default=None, exclude=True)

    # Identifier for the conversational session this SessionContext is
    # currently serving. Used by every session-scoped store attached
    # below (``approval_store``, ``file_backend``, etc.) to route
    # lookups.
    session_key: str = Field(default=DEFAULT_SESSION_KEY, exclude=True)

    # Operator-facing labels for this session (user / tenant / task ids,
    # titles, …), persisted into the session checkpoint for external
    # inspection. Write-only: never restored — the caller sets it fresh
    # each construction, like ``state``.
    session_metadata: dict[str, Any] = Field(default_factory=dict, exclude=True)

    # When True (default), every run sharing this ``session_key`` is parented
    # to a common session root derived from the key, so all runs of the session
    # form ONE OTel trace in any backend. Set False to make each run its own
    # trace root — e.g. when the backend groups runs itself (Phoenix Sessions
    # via a ``session.id`` span attribute, which expects one trace per turn).
    # No effect while ``session_key`` is the default sentinel.
    session_trace_grouping: bool = Field(default=True, exclude=True)

    # Set ``approval_store`` to enable the approval gate built via
    # ``build_store_approval``; it scopes its session allowlist by
    # ``session_key``.
    approval_store: ApprovalStore | None = Field(default=None, exclude=True)

    # Single :class:`FileBackend` for all file-shaped I/O during this
    # session — consumed by the file-edit + file-search tools and by
    # :class:`MemoryProvider`. The backend owns its own
    # ``allowed_roots``; read-before-write bookkeeping lives on the
    # active :class:`AgentLoop` and is passed to each tool call on its
    # :class:`AgentContext` (``file_edit_state``).
    file_backend: FileBackend | None = Field(default=None, exclude=True)

    # Optional :class:`ExecutionEnvironment` owning two co-located surfaces
    # (file + exec) under one shared :class:`SandboxPolicy`. This is the ONLY
    # way to grant an exec surface: :attr:`exec_backend` is a read-only
    # property off this field (below), so an exec backend without a co-located
    # ``file_backend`` — the divergent-views trap the environment exists to
    # prevent — is unrepresentable. When set, the validator below sources
    # :attr:`file_backend` from it.
    environment: ExecutionEnvironment | None = Field(default=None, exclude=True)

    # Skills registry consumed by the top-level ``load_skill`` /
    # ``list_skills`` tools and the ``skills_system_prompt_section``.
    skills: SkillRegistry | None = Field(default=None, exclude=True)

    # Cross-session memory provider consumed by the
    # ``memory_system_prompt_section`` and ``relevant_memories_attachment``.
    # Memory I/O is routed through :attr:`file_backend`; the validator
    # below enforces that linkage.
    memory: MemoryProvider | None = Field(default=None, exclude=True)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    _ambient_tokens: list[contextvars.Token[Any]] = PrivateAttr(
        default_factory=list["contextvars.Token[Any]"]
    )

    _session_restored: bool = PrivateAttr(default=False)
    _session_fs_restored: bool = PrivateAttr(default=False)
    _session_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def __enter__(self) -> "SessionContext[CtxT]":
        """
        Make this the **ambient** ctx for bare-constructed processors.

        Inside the ``with`` block, any ``Processor`` / ``LLMAgent`` /
        container built without an explicit ``ctx`` binds to *this* instance
        (resolved by :func:`current_session_context`), so a group of agents shares
        one session without threading ctx through every constructor::

            with SessionContext(state=..., checkpoint_store=store) as ctx:
                planner = LLMAgent("planner", llm=...)
                critic = LLMAgent("critic", llm=...)   # same ctx

        Binding happens at *construction* time (ctx is read once in
        ``__init__``); an agent built outside any block binds to the process
        default. ``ContextVar`` is task-local, so concurrent sessions in one
        process stay isolated.
        """
        self._ambient_tokens.append(_AMBIENT_SESSION_CONTEXT.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        if self._ambient_tokens:
            _AMBIENT_SESSION_CONTEXT.reset(self._ambient_tokens.pop())

    @property
    def exec_backend(self) -> ExecBackend | None:
        """
        Co-located exec surface, sourced solely from :attr:`environment`.

        Read-only on purpose: exec capability is reachable only through an
        :class:`ExecutionEnvironment`, which pairs it with a co-located
        ``file_backend`` under one ``SandboxPolicy``. An exec surface without
        a co-located filesystem view — the divergent-views trap — is therefore
        unrepresentable. ``None`` when no environment is wired (file-only or
        capability-less sessions). Consumed by the ``Bash`` / code-execution
        tools.
        """
        return self.environment.exec_backend if self.environment is not None else None

    def __deepcopy__(
        self, memo: dict[int, Any] | None = None
    ) -> "SessionContext[CtxT]":
        """
        :class:`SessionContext` is the session-scoped DI container — every
        processor in a run shares one instance. Returning ``self`` ensures
        ``Processor.copy()`` (and any other ``deepcopy``) preserves that
        identity rather than spawning a divergent ctx that silently drops
        checkpoints / usage tracking / file_backend bindings.
        """
        return self

    # --- Session persistence (the session-scoped half of durability) ---
    #
    # Processor checkpoints persist each processor's own working state
    # (transcripts, ledgers, watermarks). The state shared by every processor
    # bound to this ctx — ``state`` and the environment filesystem — is
    # persisted here instead, in one ``SessionCheckpoint`` per ``session_key``
    # (:meth:`save_checkpoint` / :meth:`load_checkpoint`, the same pair every
    # resumable processor exposes), so a multi-processor session restores it
    # exactly once.

    def _session_store_key(self) -> str | None:
        if self.checkpoint_store is None:
            return None
        return make_store_key(self.session_key, CheckpointKind.SESSION)

    @property
    def session_fs_restored(self) -> bool:
        """
        Whether this process rewound the shared filesystem to a snapshot.

        Set by :meth:`load_checkpoint`; kernel re-attach is only meaningful
        inside a restored filesystem, so consumers gate on this.
        """
        return self._session_fs_restored

    async def load_checkpoint(self) -> SessionCheckpoint | None:
        """
        Restore session-scoped state from the checkpoint store — once per ctx.

        Idempotent: called at the start of every processor run, but only the
        first call on this ctx does the work — containers, team members, and
        sub-agents sharing the session no-op, so nothing an earlier participant
        restored (or mutated live) is clobbered by a later one. Rehydrates
        ``state`` when it was persisted (see ``serialize_state``) and rewinds
        the shared environment filesystem to the last snapshotted checkpoint
        boundary. A record carrying a snapshot ref without a
        ``SnapshotCapable`` :attr:`environment` crashes rather than resuming
        with divergent files.
        """
        async with self._session_lock:
            if self._session_restored:
                return None
            key = self._session_store_key()
            if self.checkpoint_store is None or key is None:
                self._session_restored = True
                return None
            record = await self.checkpoint_store.load_json(
                key, SessionCheckpoint, subject=f"session checkpoint at {key}"
            )
            if record is None:
                self._session_restored = True
                return None

            if record.fs_snapshot_ref is not None:
                if not isinstance(self.environment, SnapshotCapable):
                    raise RuntimeError(
                        "Session checkpoint carries fs_snapshot_ref="
                        f"{record.fs_snapshot_ref!r} but ctx.environment "
                        f"({type(self.environment).__name__}) is not "
                        "SnapshotCapable; wire the same kind of environment "
                        "the session was saved with."
                    )
                await self.environment.restore(record.fs_snapshot_ref)
                self._session_fs_restored = True

            self.state = rehydrate_context(
                record.context_kind, record.context_data, self.state
            )
            self._session_restored = True
            logger.info(
                "Restored session %s (state=%s, fs_snapshot_ref=%s)",
                key,
                (record.context_kind or ContextKind.OMITTED).value,
                record.fs_snapshot_ref or "none",
            )
            return record

    async def save_checkpoint(self, *, fs_snapshot_ref: str | None = None) -> None:
        """
        Persist session-scoped state as of the current checkpoint boundary.

        Called by agents alongside their own checkpoints; no-ops unless a
        session-scoped feature is on (``serialize_state`` / ``fs_snapshot_policy``
        / ``session_metadata``). ``fs_snapshot_ref`` is recorded exactly as
        given: ``None`` means no snapshot describes the session as of this
        boundary, so a cold resume keeps the live filesystem instead of
        rewinding to a stale ref.
        """
        key = self._session_store_key()
        if self.checkpoint_store is None or key is None:
            return
        if (
            not self.serialize_state
            and self.fs_snapshot_policy == "off"
            and not self.session_metadata
        ):
            return
        context_kind: ContextKind | None = None
        context_data: Any | None = None
        if self.serialize_state:
            context_kind, context_data = serialize_context(self.state)
        record = SessionCheckpoint(
            session_key=self.session_key,
            context_kind=context_kind,
            context_data=context_data,
            fs_snapshot_ref=fs_snapshot_ref,
            session_metadata=self.session_metadata,
        )
        await self.checkpoint_store.save(key, record.model_dump_json().encode("utf-8"))

    def record_response(self, agent_name: ProcName, response: Response) -> None:
        """
        Append ``response`` to :attr:`responses`, trimming the agent's bucket to
        the most recent :attr:`max_responses_per_agent` entries when that cap is
        set. Keeps a long-lived shared ctx from accumulating responses without
        bound.
        """
        bucket = self.responses[agent_name]
        bucket.append(response)
        cap = self.max_responses_per_agent
        if cap is not None and len(bucket) > cap:
            del bucket[: len(bucket) - cap]

    @model_validator(mode="after")
    def _reconcile_environment(self) -> "SessionContext[CtxT]":
        if self.environment is None:
            return self
        env_file_backend = self.environment.file_backend
        if self.file_backend is None:
            self.file_backend = env_file_backend
        elif self.file_backend is not env_file_backend:
            raise ValueError(
                "SessionContext was given both an environment and a different "
                "file_backend. The environment's file_backend is co-located "
                "with its exec_backend under one SandboxPolicy; a divergent "
                "standalone file_backend would break that guarantee. Pass only "
                "the environment, or set file_backend to "
                "environment.file_backend."
            )
        return self

    @model_validator(mode="after")
    def _validate_memory_under_backend_roots(self) -> "SessionContext[CtxT]":
        if self.memory is None:
            return self
        root = self.memory.root
        if root == Path():
            return self  # "no explicit memdir" sentinel
        if self.file_backend is None:
            raise ValueError(
                "SessionContext.memory requires file_backend. Memory I/O is "
                "routed through the session's file backend; pass "
                "LocalFileBackend(allowed_roots=...) or "
                "MCPFileBackend(client=..., allowed_roots=...) explicitly."
            )
        self.file_backend.add_allowed_root(root)
        self.memory.bind_backend(self.file_backend)
        return self


_AMBIENT_SESSION_CONTEXT: contextvars.ContextVar["SessionContext[Any] | None"] = (
    contextvars.ContextVar("grasp_agents_ambient_session_context", default=None)
)

_default_session_context: "SessionContext[Any] | None" = None


def current_session_context() -> "SessionContext[Any]":
    """
    The ctx a processor binds to when constructed without an explicit one.

    Resolution: the innermost active ``with SessionContext(...)`` block, else the
    lazily-created process-wide default (created on first use).
    """
    ambient = _AMBIENT_SESSION_CONTEXT.get()
    if ambient is not None:
        return ambient
    global _default_session_context
    if _default_session_context is None:
        _default_session_context = SessionContext()
    return _default_session_context


def reset_default_session_context() -> None:
    """
    Drop the process-wide default so the next bare construction makes a fresh
    one. For test isolation (the default otherwise accumulates state across
    unrelated bare agents); production rarely needs it. Does not affect any
    active ``with SessionContext`` block.
    """
    global _default_session_context
    _default_session_context = None


# Deprecated aliases (pre-rename names) — will be removed in a future release.
RunContext = SessionContext
current_run_context = current_session_context
reset_default_run_context = reset_default_session_context
