from collections import defaultdict
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from grasp_agents.typing.completion import Completion

from .printer import Printer
from .typing.io import ProcName
from .usage_tracker import UsageTracker

CtxT = TypeVar("CtxT")


class RunContext(BaseModel, Generic[CtxT]):
    state: CtxT = None  # type: ignore

    completions: dict[ProcName, list[Completion]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    usage_tracker: UsageTracker = Field(default_factory=UsageTracker)
    printer: Printer | None = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
