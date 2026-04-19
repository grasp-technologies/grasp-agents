"""
Classification of the next :class:`AgentLoop` step after an LLM turn.

After each response the loop picks one of four outcomes: stop with a
final answer, force-generate a final answer (budget exhausted), run
tool calls, or continue to another turn. :func:`decide_next_step` is
the pure function that makes the choice; the four ``NextStep*``
dataclasses carry any payload the handler needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from ..types.events import StopReason

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..types.items import FunctionToolCallItem


@dataclass(frozen=True, slots=True)
class NextStepStop:
    """Terminal: final answer extracted. Loop ends with ``FINAL_ANSWER``."""

    final_answer: str
    stop_reason: StopReason = StopReason.FINAL_ANSWER


@dataclass(frozen=True, slots=True)
class NextStepForceFinalAnswer:
    """
    Terminal: turn budget exhausted. Caller force-generates a final
    answer and ends the loop with ``stop_reason=MAX_TURNS``.
    """


@dataclass(frozen=True, slots=True)
class NextStepRunTools:
    """Non-terminal: run these tool calls, then continue."""

    tool_calls: Sequence[FunctionToolCallItem]


@dataclass(frozen=True, slots=True)
class NextStepContinue:
    """
    Non-terminal: no tools and no final answer — loop again.
    Covers reason-turns under ``force_react_mode`` and final-answer
    suppression while background tasks are pending.
    """


NextStep: TypeAlias = (
    NextStepStop | NextStepForceFinalAnswer | NextStepRunTools | NextStepContinue
)


def decide_next_step(
    *,
    final_answer: str | None,
    tool_calls: Sequence[FunctionToolCallItem],
    turn: int,
    max_turns: int,
    bg_tasks_pending: bool,
) -> NextStep:
    """
    Classify the next :class:`AgentLoop` step. Pure function — no I/O,
    no side effects.

    Precedence: stop on final answer unless background tasks still
    block it; force-generate when the turn budget is exhausted; run
    tools if the response has any; otherwise continue.
    """
    wait_for_bg = bg_tasks_pending and turn < max_turns

    if final_answer is not None and not wait_for_bg:
        return NextStepStop(final_answer=final_answer)

    if turn >= max_turns:
        return NextStepForceFinalAnswer()

    if tool_calls:
        return NextStepRunTools(tool_calls=tool_calls)

    return NextStepContinue()
