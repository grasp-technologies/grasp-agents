from dataclasses import dataclass
from enum import StrEnum, auto

from ..types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    OutputMessageItem,
    ReasoningItem,
)


class InterruptionType(StrEnum):
    """How the previous session ended."""

    NONE = auto()
    AFTER_TOOL_RESULT = auto()  # Tools completed, LLM never responded
    MID_ASSISTANT_TURN = auto()  # Incomplete assistant turn was stripped
    PENDING_USER_MESSAGE = auto()  # User message waiting for LLM response


@dataclass
class ResumeState:
    """Result of preparing messages for resume."""

    messages: list[InputItem]
    interruption: InterruptionType
    removed_count: int = 0


def _strip_incomplete_turn(messages: list[InputItem]) -> tuple[list[InputItem], int]:
    """Strip trailing incomplete assistant turn (unresolved tool calls + their context).

    Walks backward from the end. If any trailing FunctionToolCallItem has no
    matching FunctionToolOutputItem, the entire assistant turn (reasoning +
    text + tool calls + partial outputs) is removed.
    """
    if not messages:
        return [], 0

    # Collect all resolved call_ids
    resolved_ids = {
        msg.call_id
        for msg in messages
        if isinstance(msg, FunctionToolOutputItem)
    }

    # Walk backward through trailing tool-related items
    i = len(messages) - 1
    has_unresolved = False

    while i >= 0:
        msg = messages[i]
        if isinstance(msg, FunctionToolOutputItem):
            i -= 1
            continue
        if isinstance(msg, FunctionToolCallItem):
            if msg.call_id not in resolved_ids:
                has_unresolved = True
            i -= 1
            continue
        break  # Hit non-tool message

    if not has_unresolved:
        return list(messages), 0

    # Also strip the assistant's text and reasoning from the same turn
    while i >= 0 and isinstance(messages[i], (OutputMessageItem, ReasoningItem)):
        i -= 1

    cleaned = list(messages[: i + 1])
    return cleaned, len(messages) - len(cleaned)


def _detect_interruption(messages: list[InputItem]) -> InterruptionType:
    """Classify what state the conversation is in after cleanup."""
    if not messages:
        return InterruptionType.NONE

    last = messages[-1]

    if isinstance(last, FunctionToolOutputItem):
        return InterruptionType.AFTER_TOOL_RESULT

    if isinstance(last, InputItem) and hasattr(last, "role") and last.role == "user":  # type: ignore[union-attr]
        return InterruptionType.PENDING_USER_MESSAGE

    return InterruptionType.NONE


def prepare_messages_for_resume(
    messages: list[InputItem],
) -> ResumeState:
    """CC-style message cleanup for session resume.

    1. Strip trailing incomplete assistant turn (dangling tool calls)
    2. Detect interruption type for diagnostics

    The cleaned messages can be fed directly into an agent loop.
    Stripping ensures no unmatched tool_use/tool_result pairs, which
    would cause LLM API errors.
    """
    if not messages:
        return ResumeState(messages=[], interruption=InterruptionType.NONE)

    cleaned, removed = _strip_incomplete_turn(messages)

    interruption: InterruptionType
    if removed > 0:
        interruption = InterruptionType.MID_ASSISTANT_TURN
    else:
        interruption = _detect_interruption(cleaned)

    return ResumeState(
        messages=cleaned,
        interruption=interruption,
        removed_count=removed,
    )
