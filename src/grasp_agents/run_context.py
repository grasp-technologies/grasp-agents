from collections import defaultdict
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# CtxT must be defined before the skills import below, because
# ``skills.registry`` (via injection → prompt_builder) imports CtxT from this
# module. Keeping it at the top breaks the load-order circular dependency
# without forcing every consumer through ``TYPE_CHECKING``.
CtxT = TypeVar("CtxT")

from .agent.approval_store import ApprovalStore  # noqa: E402
from .durability.checkpoint_store import CheckpointStore  # noqa: E402
from .printer import Printer  # noqa: E402
from .skills.registry import SkillRegistry  # noqa: E402
from .tools.file_edit.store import FileEditStore  # noqa: E402
from .types.io import ProcName  # noqa: E402
from .types.response import Response  # noqa: E402
from .usage_tracker import UsageTracker  # noqa: E402


class RunContext(BaseModel, Generic[CtxT]):
    state: CtxT = None  # type: ignore

    responses: defaultdict[ProcName, list[Response]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    usage_tracker: UsageTracker = Field(default_factory=UsageTracker, exclude=True)
    printer: Printer | None = Field(default=None, exclude=True)
    checkpoint_store: CheckpointStore | None = Field(default=None, exclude=True)

    # Identifier for the conversational session this RunContext is
    # currently serving. Used by every session-scoped store attached
    # below (``approval_store``, ``file_edit_store``, etc.) to route
    # lookups. Callers mutate this as sessions begin, resume, or
    # continue — re-keying into the same slot recovers the prior
    # session's state if the store has it.
    session_key: str = Field(default="default", exclude=True)

    # Set ``approval_store`` to enable the approval gate built via
    # ``build_store_approval``; it scopes its session allowlist by
    # ``session_key``.
    approval_store: ApprovalStore | None = Field(default=None, exclude=True)

    # Session-keyed backing store for the file-edit tools' read-before-write
    # and mtime-staleness state. When set, the tools from
    # ``grasp_agents.tools.file_edit`` route their state lookups through
    # this store keyed by ``session_key``.
    file_edit_store: FileEditStore | None = Field(default=None, exclude=True)

    # Skills registry consumed by the top-level ``load_skill`` /
    # ``list_skills`` tools and the ``skills_system_prompt_section``.
    # Attach via ``grasp_agents.skills.attach_skills(agent)`` and pass the
    # populated registry on the ``RunContext``.
    skills: SkillRegistry | None = Field(default=None, exclude=True)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
