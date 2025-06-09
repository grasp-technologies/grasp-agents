from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeAlias, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from grasp_agents.typing.completion import Completion

from .printer import ColoringMode, Printer
from .typing.content import ImageData
from .typing.io import (
    InT_contra,
    LLMPrompt,
    LLMPromptArgs,
    MemT_co,
    OutT_co,
    ProcessorName,
)
from .usage_tracker import UsageTracker


class RunArgs(BaseModel):
    sys: LLMPromptArgs = Field(default_factory=LLMPromptArgs)
    usr: LLMPromptArgs | Sequence[LLMPromptArgs] = Field(default_factory=LLMPromptArgs)

    model_config = ConfigDict(extra="forbid")


class InteractionRecord(BaseModel, Generic[InT_contra, OutT_co, MemT_co]):
    source_id: str
    recipients: Sequence[ProcessorName]
    memory: MemT_co
    chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None
    sys_prompt: LLMPrompt | None = None
    in_prompt: LLMPrompt | None = None
    sys_args: LLMPromptArgs | None = None
    usr_args: LLMPromptArgs | Sequence[LLMPromptArgs] | None = None
    in_args: InT_contra | Sequence[InT_contra] | None = None
    outputs: Sequence[OutT_co]

    model_config = ConfigDict(extra="forbid", frozen=True)


InteractionHistory: TypeAlias = list[InteractionRecord[Any, Any, Any]]


CtxT = TypeVar("CtxT")


class RunContext(BaseModel, Generic[CtxT]):
    state: CtxT | None = None
    run_id: str = Field(default_factory=lambda: str(uuid4())[:8], frozen=True)
    run_args: dict[ProcessorName, RunArgs] = Field(default_factory=dict)

    completions: Mapping[ProcessorName, list[Completion]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    interaction_history: InteractionHistory = Field(default_factory=list)  # type: ignore[valid-type]

    print_messages: bool = False
    color_messages_by: ColoringMode = "role"

    _usage_tracker: UsageTracker = PrivateAttr()
    _printer: Printer = PrivateAttr()

    def model_post_init(self, context: Any) -> None:  # noqa: ARG002
        self._usage_tracker = UsageTracker(source_id=self.run_id)
        self._printer = Printer(
            source_id=self.run_id, print_messages=self.print_messages
        )

    @property
    def usage_tracker(self) -> UsageTracker:
        return self._usage_tracker

    @property
    def printer(self) -> Printer:
        return self._printer

    model_config = ConfigDict(extra="forbid")
