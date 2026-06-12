import contextvars
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .agent.approval_store import ApprovalStore
from .durability.checkpoint_store import CheckpointStore
from .memory.provider import MemoryProvider
from .printer import Printer
from .sandbox.environment import ExecutionEnvironment
from .sandbox.exec_backend import ExecBackend
from .skills.registry import SkillRegistry
from .tools.file_backend.base import FileBackend
from .types.io import ProcName
from .types.response import Response
from .usage_tracker import UsageTracker


class RunContext[CtxT](BaseModel):
    state: CtxT = None  # type: ignore

    # When True, the agent persists ``state`` into its checkpoints (via
    # ``serialize_context``) and restores it on resume. Default False:
    # business state is rebuilt on resume via ``@agent.add_state_builder``
    # (the application's database is the source of truth), keeping the
    # checkpoint small. Opt in for tests / simple workloads where ``state``
    # is a plain serializable container (pydantic / dataclass / mapping).
    serialize_state: bool = Field(default=False, exclude=True)

    responses: defaultdict[ProcName, list[Response]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    max_responses_per_agent: int | None = Field(default=None, exclude=True)
    usage_tracker: UsageTracker = Field(default_factory=UsageTracker, exclude=True)
    printer: Printer | None = Field(default=None, exclude=True)
    checkpoint_store: CheckpointStore | None = Field(default=None, exclude=True)

    # Identifier for the conversational session this RunContext is
    # currently serving. Used by every session-scoped store attached
    # below (``approval_store``, ``file_backend``, etc.) to route
    # lookups.
    session_key: str = Field(default="default", exclude=True)

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

    # Stack of ContextVar tokens, one per active ``with self:`` frame, so a
    # context can be entered re-entrantly and each exit restores the right
    # outer ambient ctx.
    _ambient_tokens: list[contextvars.Token[Any]] = PrivateAttr(
        default_factory=list["contextvars.Token[Any]"]
    )

    def __enter__(self) -> "RunContext[CtxT]":
        """
        Make this the **ambient** ctx for bare-constructed processors.

        Inside the ``with`` block, any ``Processor`` / ``LLMAgent`` /
        container built without an explicit ``ctx`` binds to *this* instance
        (resolved by :func:`current_run_context`), so a group of agents shares
        one session without threading ctx through every constructor::

            with RunContext(state=..., checkpoint_store=store) as ctx:
                planner = LLMAgent("planner", llm=...)
                critic = LLMAgent("critic", llm=...)   # same ctx

        Binding happens at *construction* time (ctx is read once in
        ``__init__``); an agent built outside any block binds to the process
        default. ``ContextVar`` is task-local, so concurrent sessions in one
        process stay isolated.
        """
        self._ambient_tokens.append(_AMBIENT_RUN_CONTEXT.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        if self._ambient_tokens:
            _AMBIENT_RUN_CONTEXT.reset(self._ambient_tokens.pop())

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

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "RunContext[CtxT]":
        """
        :class:`RunContext` is the session-scoped DI container — every
        processor in a run shares one instance. Returning ``self`` ensures
        ``Processor.copy()`` (and any other ``deepcopy``) preserves that
        identity rather than spawning a divergent ctx that silently drops
        checkpoints / usage tracking / file_backend bindings.
        """
        return self

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
    def _reconcile_environment(self) -> "RunContext[CtxT]":
        """
        Source the file surface from the :class:`ExecutionEnvironment` when one
        is wired, and reject a standalone ``file_backend`` that diverges from
        it.

        Exec capability is exposed only via the :attr:`exec_backend` property
        (off :attr:`environment`), so there is no separate exec field to
        reconcile — co-location with the environment's ``file_backend`` is
        structural. A ``file_backend`` may still be wired alone for file-only
        sessions (memory authoring, MCP artifact backends) that need no exec
        surface. No-op when no environment was provided.
        """
        if self.environment is None:
            return self
        env_file_backend = self.environment.file_backend
        if self.file_backend is None:
            self.file_backend = env_file_backend
        elif self.file_backend is not env_file_backend:
            raise ValueError(
                "RunContext was given both an environment and a different "
                "file_backend. The environment's file_backend is co-located "
                "with its exec_backend under one SandboxPolicy; a divergent "
                "standalone file_backend would break that guarantee. Pass only "
                "the environment, or set file_backend to "
                "environment.file_backend."
            )
        return self

    @model_validator(mode="after")
    def _validate_memory_under_backend_roots(self) -> "RunContext[CtxT]":
        """
        Wire memory to the session's file backend at start:

        - When :attr:`memory` is set with a concrete ``root``, :attr:`file_backend`
          is required (memory I/O routes through it). The memdir is admitted
          into the backend's address space (:meth:`FileBackend.add_allowed_root`)
          so memory authoring via the file tools works without the host
          repeating the memdir in ``allowed_roots``. The backend is then bound
          on the provider so its methods can fetch bytes without re-receiving
          a ctx kwarg.
        - When :attr:`memory.root` is the ``Path()`` sentinel (e.g.
          :class:`InMemoryMemoryProvider`), the backend is not required —
          the provider serves a static snapshot.
        """
        if self.memory is None:
            return self
        root = self.memory.root
        if root == Path():
            return self  # "no explicit memdir" sentinel
        if self.file_backend is None:
            raise ValueError(
                "RunContext.memory requires file_backend. Memory I/O is "
                "routed through the session's file backend; pass "
                "LocalFileBackend(allowed_roots=...) or "
                "MCPFileBackend(client=..., allowed_roots=...) explicitly."
            )
        # Admit the configured memdir so memory authoring through the file
        # tools is allowed without the host repeating it in allowed_roots
        # (idempotent if already covered).
        self.file_backend.add_allowed_root(root)
        # Wire the backend onto the provider so its methods can read
        # bytes without callers threading ctx through again.
        self.memory.bind_backend(self.file_backend)
        return self


# The ambient ctx set by ``with RunContext(...)`` (task-local). ``None`` =
# no active block; bare construction then falls back to the process default.
_AMBIENT_RUN_CONTEXT: contextvars.ContextVar["RunContext[Any] | None"] = (
    contextvars.ContextVar("grasp_agents_ambient_run_context", default=None)
)

# Lazily-created process-wide default. The single session every
# bare-constructed processor shares when no ctx is passed and no ``with
# RunContext`` block is active — so two agents that are never composed still
# share one usage tracker / state / store set, instead of each minting a
# private throwaway ctx that silently drops those bindings.
_default_run_context: "RunContext[Any] | None" = None


def current_run_context() -> "RunContext[Any]":
    """
    The ctx a processor binds to when constructed without an explicit one.

    Resolution: the innermost active ``with RunContext(...)`` block, else the
    lazily-created process-wide default (created on first use). Never mints a
    fresh per-call ctx — that was the old placeholder behavior, which left
    uncomposed agents in separate sessions.
    """
    ambient = _AMBIENT_RUN_CONTEXT.get()
    if ambient is not None:
        return ambient
    global _default_run_context
    if _default_run_context is None:
        _default_run_context = RunContext()
    return _default_run_context


def reset_default_run_context() -> None:
    """
    Drop the process-wide default so the next bare construction makes a fresh
    one. For test isolation (the default otherwise accumulates state across
    unrelated bare agents); production rarely needs it. Does not affect any
    active ``with RunContext`` block.
    """
    global _default_run_context
    _default_run_context = None
