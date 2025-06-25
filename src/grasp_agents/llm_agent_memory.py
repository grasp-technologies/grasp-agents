from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import PrivateAttr

from .memory import Memory
from .run_context import RunContext
from .typing.io import LLMPrompt
from .typing.message import Message, Messages, SystemMessage


class MakeMemoryHandler(Protocol):
    def __call__(
        self,
        prev_memory: "LLMAgentMemory",
        in_args: Any | None,
        sys_prompt: LLMPrompt | None,
        ctx: RunContext[Any] | None,
    ) -> "LLMAgentMemory": ...


class LLMAgentMemory(Memory):
    _message_history: Messages = PrivateAttr(default_factory=list)  # type: ignore
    _sys_prompt: LLMPrompt | None = PrivateAttr(default=None)

    def __init__(self, sys_prompt: LLMPrompt | None = None) -> None:
        super().__init__()
        self._sys_prompt = sys_prompt
        self.reset(sys_prompt)

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._sys_prompt

    @property
    def message_history(self) -> Messages:
        return self._message_history

    def reset(
        self, sys_prompt: LLMPrompt | None = None, ctx: RunContext[Any] | None = None
    ):
        if sys_prompt is not None:
            self._sys_prompt = sys_prompt

        self._message_history = (
            [SystemMessage(content=self._sys_prompt)]
            if self._sys_prompt is not None
            else []
        )

    def erase(self) -> None:
        self._message_history = []

    def update(
        self, messages: Sequence[Message], *, ctx: RunContext[Any] | None = None
    ):
        self._message_history.extend(messages)

    @property
    def is_empty(self) -> bool:
        return len(self._message_history) == 0

    def __repr__(self) -> str:
        return (
            "LLMAgentMemory with message history of "
            f"length {len(self._message_history)}"
        )
