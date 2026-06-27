from collections.abc import Sequence

from pydantic import BaseModel, Field

from grasp_agents.types.errors import TranscriptInvariantError
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
)


class LLMAgentTranscript(BaseModel):
    """
    Per-run message history for :class:`LLMAgent` — the pure conversation log.

    Owned by the agent (``agent.transcript``), persisted via the agent's
    checkpoint, and rebuilt on resume. Distinct from cross-session memory on
    :class:`RunContext.memory` (the memdir-backed knowledge store). The system
    prompt is not stored here — it lives in the ephemeral header
    (``initial_context``) the agent prepends to the model-facing view each turn.
    """

    messages: list[InputItem] = Field(default_factory=list[InputItem])

    def clear(self) -> None:
        self.messages = []

    def truncate(self, message_count: int) -> None:
        if 0 <= message_count < len(self.messages):
            del self.messages[message_count:]

    def update(self, new_messages: Sequence[InputItem]) -> None:
        self.messages.extend(new_messages)

    def validate_tool_call_pairing(self) -> None:
        """
        Raise if a tool call isn't immediately resolved by its result.

        Enforces the provider invariant that every ``FunctionToolCallItem``
        is followed by its ``FunctionToolOutputItem`` before any input
        (user / system / developer) message, and that none dangle
        unresolved at the end. Same-turn assistant items (reasoning, output
        text) between a call and its result are allowed — they're part of
        the same assistant message. Called before each LLM generation.

        Raises:
            TranscriptInvariantError: On a wedged input message or a
                dangling tool call.

        """
        open_calls: list[str] = []
        for item in self.messages:
            if isinstance(item, FunctionToolCallItem):
                open_calls.append(item.call_id)
            elif isinstance(item, FunctionToolOutputItem):
                if item.call_id in open_calls:
                    open_calls.remove(item.call_id)
            elif isinstance(item, InputMessageItem) and open_calls:
                raise TranscriptInvariantError(
                    f"Tool call(s) {open_calls} not resolved before a "
                    f"{item.role!r} message: tool calls must be immediately "
                    "followed by their tool results."
                )
        if open_calls:
            raise TranscriptInvariantError(
                f"Transcript has unresolved tool call(s) with no result: {open_calls}."
            )

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def __repr__(self) -> str:
        return f"LLMAgentTranscript(len={len(self.messages)})"
