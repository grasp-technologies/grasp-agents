from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import Field

from .memory import Memory
from .message_history import MessageHistory
from .run_context import RunContext
from .typing.io import LLMPrompt
from .typing.message import Message


class SetMemoryHandler(Protocol):
    def __call__(
        self,
        prev_memory: "LLMAgentMemory",
        in_args: Any | None,
        sys_prompt: LLMPrompt | None,
        ctx: RunContext[Any] | None,
    ) -> "LLMAgentMemory": ...


class LLMAgentMemory(Memory):
    message_history: MessageHistory = Field(default_factory=MessageHistory)

    def reset(
        self, sys_prompt: LLMPrompt | None = None, ctx: RunContext[Any] | None = None
    ):
        self.message_history.reset(sys_prompt=sys_prompt)

    def update(
        self,
        *,
        message_batch: Sequence[Message] | None = None,
        message_list: Sequence[Message] | None = None,
        ctx: RunContext[Any] | None = None,
    ):
        if message_batch is not None and message_list is not None:
            raise ValueError(
                "Only one of message_batch or messages should be provided."
            )
        if message_batch is not None:
            self.message_history.add_message_batch(message_batch)
        elif message_list is not None:
            self.message_history.add_message_list(message_list)
        else:
            raise ValueError("Either message_batch or messages must be provided.")

    @property
    def is_empty(self) -> bool:
        return len(self.message_history) == 0

    @property
    def batch_size(self) -> int:
        return self.message_history.batch_size

    def __repr__(self) -> str:
        return f"Message History: {len(self.message_history)}"
