from collections import defaultdict
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .agent.approval_store import ApprovalStore
from .durability.checkpoint_store import CheckpointStore
from .printer import Printer
from .tools.file_edit.store import FileEditStore
from .types.io import ProcName
from .types.response import Response
from .usage_tracker import UsageTracker

CtxT = TypeVar("CtxT")


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

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
