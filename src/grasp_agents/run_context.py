from collections import defaultdict
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .agent.approval_store import ApprovalStore
from .durability.checkpoint_store import CheckpointStore
from .printer import Printer
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

    # Set ``approval_store`` to enable the approval gate built via
    # ``build_store_approval``; ``approval_session_key`` scopes its
    # session allowlist per user or per run.
    approval_store: ApprovalStore | None = Field(default=None, exclude=True)
    approval_session_key: str = Field(default="default", exclude=True)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
