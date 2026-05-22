from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from ..run_context import RunContext
from ..types.io import LLMPrompt
from ..types.items import InputItem, InputMessageItem


class LLMAgentTranscript(BaseModel):
    """
    Per-run message history for :class:`LLMAgent`.

    Owned by the agent (``agent.transcript``), persisted via the agent's
    checkpoint, and rebuilt on resume. Distinct from cross-session memory
    on :class:`RunContext.memory` (the memdir-backed knowledge store).
    """

    messages: list[InputItem] = Field(default_factory=list[InputItem])

    def reset(
        self,
        instructions: LLMPrompt | None = None,
        ctx: RunContext[Any] | None = None,
    ) -> None:
        del ctx
        self.messages = (
            [InputMessageItem.from_text(instructions, role="system")]
            if instructions is not None
            else []
        )

    def erase(self) -> None:
        self.messages = []

    def update(
        self,
        new_messages: Sequence[InputItem],
        *,
        ctx: RunContext[Any] | None = None,
    ) -> None:
        del ctx
        self.messages.extend(new_messages)

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0

    @property
    def instructions(self) -> LLMPrompt | None:
        if (
            not self.is_empty
            and isinstance(self.messages[0], InputMessageItem)
            and self.messages[0].role == "system"
        ):
            return self.messages[0].text or None
        return None

    def __repr__(self) -> str:
        return f"LLMAgentTranscript(len={len(self.messages)})"
