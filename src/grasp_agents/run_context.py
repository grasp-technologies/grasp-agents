from collections.abc import Sequence
from typing import Any, Generic, TypeAlias, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .printer import Printer
from .typing.content import ImageData
from .typing.io import (
    AgentID,
    AgentState,
    InT,
    LLMPrompt,
    LLMPromptArgs,
    OutT,
    StateT,
)
from .usage_tracker import UsageTracker

SystemRunArgs: TypeAlias = LLMPromptArgs
UserRunArgs: TypeAlias = LLMPromptArgs | list[LLMPromptArgs]


class RunArgs(BaseModel):
    sys: SystemRunArgs = Field(default_factory=LLMPromptArgs)
    usr: UserRunArgs = Field(default_factory=LLMPromptArgs)

    model_config = ConfigDict(extra="forbid")


class InteractionRecord(BaseModel, Generic[InT, OutT, StateT]):
    source_id: str
    recipient_ids: Sequence[AgentID]
    state: StateT
    inp_items: LLMPrompt | list[str | ImageData] | None = None
    sys_prompt: LLMPrompt | None = None
    inp_prompt: LLMPrompt | None = None
    sys_args: SystemRunArgs | None = None
    usr_args: UserRunArgs | None = None
    rcv_args: Sequence[InT] | None = None
    outputs: Sequence[OutT]

    model_config = ConfigDict(extra="forbid", frozen=True)


InteractionHistory: TypeAlias = list[InteractionRecord[Any, Any, AgentState]]


CtxT = TypeVar("CtxT")


class RunContextWrapper(BaseModel, Generic[CtxT]):
    context: CtxT | None = None
    run_id: str = Field(default_factory=lambda: str(uuid4())[:8], frozen=True)
    run_args: dict[AgentID, RunArgs] = Field(default_factory=dict)
    interaction_history: InteractionHistory = Field(default_factory=list)  # type: ignore[valid-type]

    print_messages: bool = False

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
