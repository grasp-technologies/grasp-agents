from collections import defaultdict
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from grasp_agents.typing.completion import Completion

from .printer import ColoringMode, Printer
from .typing.io import LLMPromptArgs, ProcName
from .usage_tracker import UsageTracker

CtxT = TypeVar("CtxT")


class RunContext(BaseModel, Generic[CtxT]):
    state: CtxT | None = None

    sys_args: dict[ProcName, LLMPromptArgs] = Field(default_factory=dict)

    is_streaming: bool = False
    result: Any | None = None

    completions: dict[ProcName, list[Completion]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    usage_tracker: UsageTracker = Field(default_factory=UsageTracker)

    printer: Printer | None = None
    print_messages: bool = False
    color_messages_by: ColoringMode = "role"

    def model_post_init(self, context: Any) -> None:  # noqa: ARG002
        if self.print_messages:
            self.printer = Printer(color_by=self.color_messages_by)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
