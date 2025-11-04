from collections.abc import Sequence
from typing import Any

from pydantic import Field

from .memory import Memory
from .run_context import RunContext
from .typing.io import LLMPrompt
from .typing.message import Message, Messages, SystemMessage


class LLMAgentMemory(Memory):
    messages: Messages = Field(default_factory=Messages)
    instructions: LLMPrompt | None = Field(default=None)

    def __init__(self, instructions: LLMPrompt | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reset(instructions)

    def reset(
        self, instructions: LLMPrompt | None = None, ctx: RunContext[Any] | None = None
    ):
        if instructions is not None:
            self.instructions = instructions

        self.messages = (
            [SystemMessage(content=self.instructions)]
            if self.instructions is not None
            else []
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

    def __repr__(self) -> str:
        return f"LLMAgentMemory with message history of length {len(self.messages)}"
