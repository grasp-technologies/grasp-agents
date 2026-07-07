from dataclasses import dataclass
from enum import StrEnum, auto

from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
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


def _detect_interruption(messages: list[InputItem]) -> InterruptionType:
    """Classify what state the conversation is in after cleanup."""
    if not messages:
        return InterruptionType.NONE

    last = messages[-1]

    if isinstance(last, FunctionToolOutputItem):
        return InterruptionType.AFTER_TOOL_RESULT

    if isinstance(last, InputMessageItem) and last.role == "user":
        return InterruptionType.PENDING_USER_MESSAGE

    return InterruptionType.NONE


def prepare_messages_for_resume(
    messages: list[InputItem], *, drop_trailing_response: bool = False
) -> ResumeState:
    """
    Settle a transcript to its last closed boundary, dropping only the tool
    round in flight.

    Serves both session resume (the persisted log) and live-interrupt settling
    (Esc / consumer abort): when the trailing round's calls lack results, the
    whole round — its response text / reasoning, calls, and any partial
    results — is discarded, preventing unmatched tool_use/tool_result pairs
    that would cause LLM API errors. Completed rounds always survive: their
    results are already durable (checkpoints only ever write at closed
    boundaries), and re-running them would re-issue their tools' side effects.

    ``drop_trailing_response`` additionally drops a trailing *call-less*
    response block. A failed run's final answer (e.g. one that did not parse)
    is not a closed turn — pruning it lets a retry regenerate it.
    """
    if not messages:
        return ResumeState(messages=[], interruption=InterruptionType.NONE)

    resolved_ids = {
        msg.call_id for msg in messages if isinstance(msg, FunctionToolOutputItem)
    }

    # The last round: trailing results, preceded by the response block that
    # issued them (text / reasoning / calls emitted by one LLM response).
    i = len(messages) - 1
    while i >= 0 and isinstance(messages[i], FunctionToolOutputItem):
        i -= 1
    start = i
    while start >= 0 and isinstance(
        messages[start], (FunctionToolCallItem, OutputMessageItem, ReasoningItem)
    ):
        start -= 1

    calls = [
        msg
        for msg in messages[start + 1 : i + 1]
        if isinstance(msg, FunctionToolCallItem)
    ]
    dangling = any(call.call_id not in resolved_ids for call in calls)
    open_response = drop_trailing_response and not calls and start < i
    if dangling or open_response:
        cleaned = list(messages[: start + 1])
    else:
        cleaned = list(messages)

    removed = len(messages) - len(cleaned)
    interruption = (
        InterruptionType.MID_ASSISTANT_TURN
        if removed
        else _detect_interruption(cleaned)
    )
    return ResumeState(
        messages=cleaned, interruption=interruption, removed_count=removed
    )
