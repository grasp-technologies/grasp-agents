import contextvars
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .agent.approval_store import ApprovalStore
from .durability.checkpoint_store import CheckpointStore
from .file_backend.base import FileBackend
from .memory.provider import MemoryProvider
from .printer import Printer
from .sandbox.environment import ExecutionEnvironment
from .sandbox.exec_backend import ExecBackend
from .skills.registry import SkillRegistry
from .types.io import ProcName
from .types.response import Response
from .usage_tracker import UsageTracker

DEFAULT_SESSION_KEY = "default"
"""Sentinel ``session_key`` for an unnamed session.

Session-scoped stores route lookups by ``session_key``; trace grouping treats
this value as "no session" (each run is its own trace root). Set a real
``session_key`` to opt a session into single-trace grouping across runs.
"""


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
    session_key: str = Field(default=DEFAULT_SESSION_KEY, exclude=True)

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
        self.file_backend.add_allowed_root(root)
        self.memory.bind_backend(self.file_backend)
        return self


_AMBIENT_RUN_CONTEXT: contextvars.ContextVar["RunContext[Any] | None"] = (
    contextvars.ContextVar("grasp_agents_ambient_run_context", default=None)
)

_default_run_context: "RunContext[Any] | None" = None


def current_run_context() -> "RunContext[Any]":
    """
    The ctx a processor binds to when constructed without an explicit one.

    Resolution: the innermost active ``with RunContext(...)`` block, else the
    lazily-created process-wide default (created on first use).
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
