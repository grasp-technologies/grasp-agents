from collections.abc import Sequence
from typing import Any

from pydantic import Field

from .memory import Memory
from .run_context import RunContext
from .typing.io import LLMPrompt
from .typing.message import AssistantMessage, Message, Messages, SystemMessage


class LLMAgentMemory(Memory):
    messages: Messages = Field(default_factory=Messages)

    def reset(
        self, instructions: LLMPrompt | None = None, ctx: RunContext[Any] | None = None
    ):
        self.messages = (
            [SystemMessage(content=instructions)] if instructions is not None else []
        )

    def erase(self) -> None:
        self.messages = []

    def update(
        self, new_messages: Sequence[Message], *, ctx: RunContext[Any] | None = None
    ):
        self.messages.extend(new_messages)

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0

    @property
    def instructions(self) -> LLMPrompt | None:
        if not self.is_empty and isinstance(self.messages[0], SystemMessage):
            return self.messages[0].content
        return None

    def __repr__(self) -> str:
        return f"LLMAgentMemory with message history of length {len(self.messages)}"
