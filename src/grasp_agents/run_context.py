from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from grasp_agents.typing.completion import Completion

from .printer import ColoringMode, Printer
from .typing.io import LLMPromptArgs, ProcName
from .usage_tracker import UsageTracker


class RunArgs(BaseModel):
    sys: LLMPromptArgs = Field(default_factory=LLMPromptArgs)
    usr: LLMPromptArgs = Field(default_factory=LLMPromptArgs)

    model_config = ConfigDict(extra="forbid")


CtxT = TypeVar("CtxT")


class RunContext(BaseModel, Generic[CtxT]):
    run_id: str = Field(default_factory=lambda: str(uuid4())[:8], frozen=True)

    state: CtxT | None = None

    run_args: dict[ProcName, RunArgs] = Field(default_factory=dict)
    completions: Mapping[ProcName, list[Completion]] = Field(
        default_factory=lambda: defaultdict(list)
    )

    print_messages: bool = False
    color_messages_by: ColoringMode = "role"

    _usage_tracker: UsageTracker = PrivateAttr()
    _printer: Printer = PrivateAttr()

    def model_post_init(self, context: Any) -> None:  # noqa: ARG002
        self._usage_tracker = UsageTracker(source_id=self.run_id)
        self._printer = Printer(
            source_id=self.run_id,
            print_messages=self.print_messages,
            color_by=self.color_messages_by,
        )

    @property
    def usage_tracker(self) -> UsageTracker:
        return self._usage_tracker

    @property
    def printer(self) -> Printer:
        return self._printer

    model_config = ConfigDict(extra="forbid")
