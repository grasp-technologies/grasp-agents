from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

# CtxT must be defined before the skills import below, because
# ``skills.registry`` (via injection → prompt_builder) imports CtxT from this
# module. Keeping it at the top breaks the load-order circular dependency
# without forcing every consumer through ``TYPE_CHECKING``.
CtxT = TypeVar("CtxT")

from .agent.approval_store import ApprovalStore  # noqa: E402
from .durability.checkpoint_store import CheckpointStore  # noqa: E402
from .memory.provider import MemoryProvider  # noqa: E402
from .printer import Printer  # noqa: E402
from .skills.registry import SkillRegistry  # noqa: E402
from .tools.file_edit.backend import FileBackend  # noqa: E402
from .types.io import ProcName  # noqa: E402
from .types.response import Response  # noqa: E402
from .usage_tracker import UsageTracker  # noqa: E402


class RunContext(BaseModel, Generic[CtxT]):
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
    # active :class:`AgentLoop` and is reached through the
    # :mod:`tools.file_edit.agent_state` ContextVar.
    file_backend: FileBackend | None = Field(default=None, exclude=True)

    # Skills registry consumed by the top-level ``load_skill`` /
    # ``list_skills`` tools and the ``skills_system_prompt_section``.
    skills: SkillRegistry | None = Field(default=None, exclude=True)

    # Cross-session memory provider consumed by the
    # ``memory_system_prompt_section`` and ``relevant_memories_attachment``.
    # Memory I/O is routed through :attr:`file_backend`; the validator
    # below enforces that linkage.
    memory: MemoryProvider | None = Field(default=None, exclude=True)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def __deepcopy__(
        self, memo: dict[int, Any] | None = None
    ) -> "RunContext[CtxT]":
        """
        :class:`RunContext` is the session-scoped DI container — every
        processor in a run shares one instance. Returning ``self`` ensures
        ``Processor.copy()`` (and any other ``deepcopy``) preserves that
        identity rather than spawning a divergent ctx that silently drops
        checkpoints / usage tracking / file_backend bindings.
        """
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


def shared_child_ctx(children: Iterable[Any]) -> "RunContext[Any] | None":
    """
    The single :class:`RunContext` explicitly provided to ``children``, or
    ``None`` when none was.

    Container processors (``WorkflowProcessor`` / ``ParallelProcessor`` /
    ``Runner``) call this when no ``ctx`` was passed to them: the shared
    session is then inherited from whichever children were *built* with one,
    rather than borrowed from an arbitrary child. Children carrying only the
    fresh placeholder default (no ``ctx`` at construction) are ignored. The
    container cascades the chosen ctx to every child via ``on_adopted``, so
    all subprocessors end up sharing one instance regardless.

    Raises:
        ValueError: if children were built with *different* ctx instances —
            an ambiguous setup the caller should resolve by passing a single
            ctx to the container.

    """
    explicit = {
        id(c.ctx): c.ctx for c in children if getattr(c, "_ctx_explicit", False)
    }
    if len(explicit) > 1:
        raise ValueError(
            "Subprocessors were built with different RunContext instances. "
            "Pass a single ctx to the container so all subprocessors share "
            "one session."
        )
    return next(iter(explicit.values()), None)
