from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import Field

from .memory import Memory
from .run_context import RunContext
from .typing.io import LLMPrompt
from .typing.message import Message, Messages, SystemMessage


class MemoryPreparator(Protocol):
    def __call__(
        self,
        memory: "LLMAgentMemory",
        *,
        in_args: Any | None,
        sys_prompt: LLMPrompt | None,
        ctx: RunContext[Any],
        call_id: str,
    ) -> None: ...


class LLMAgentMemory(Memory):
    message_history: Messages = Field(default_factory=Messages)
    sys_prompt: LLMPrompt | None = Field(default=None)

    def __init__(self, sys_prompt: LLMPrompt | None = None) -> None:
        super().__init__()
        self.reset(sys_prompt)

    def reset(
        self, sys_prompt: LLMPrompt | None = None, ctx: RunContext[Any] | None = None
    ):
        if sys_prompt is not None:
            self.sys_prompt = sys_prompt

        self.message_history = (
            [SystemMessage(content=self.sys_prompt)]
            if self.sys_prompt is not None
            else []
        )

    def erase(self) -> None:
        self.message_history = []

    def update(
        self, messages: Sequence[Message], *, ctx: RunContext[Any] | None = None
    ):
        self.message_history.extend(messages)

    @property
    def is_empty(self) -> bool:
        return len(self.message_history) == 0

    def __repr__(self) -> str:
        return (
            f"LLMAgentMemory with message history of length {len(self.message_history)}"
        )
