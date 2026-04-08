from collections import defaultdict
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

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
    store: CheckpointStore | None = Field(default=None, exclude=True)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
