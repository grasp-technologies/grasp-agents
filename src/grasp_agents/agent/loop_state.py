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
from typing import TYPE_CHECKING

from grasp_agents.types.events import StopReason

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grasp_agents.types.items import FunctionToolCallItem


@dataclass(frozen=True, slots=True)
class NextStepStop:
    """Terminal: final answer extracted. Loop ends with ``FINAL_ANSWER``."""

    final_answer: str
    stop_reason: StopReason = StopReason.FINAL_ANSWER


@dataclass(frozen=True, slots=True)
class NextStepForceFinalAnswer:
    """
    Terminal: budget exhausted (turn count or wall-clock deadline). Caller
    force-generates a final answer and ends the loop with ``stop_reason`` —
    ``MAX_TURNS`` by default, ``TIMEOUT`` when a run deadline was hit.
    """

    stop_reason: StopReason = StopReason.MAX_TURNS


@dataclass(frozen=True, slots=True)
class NextStepRunTools:
    """Non-terminal: run these tool calls, then continue."""

    tool_calls: Sequence[FunctionToolCallItem]


@dataclass(frozen=True, slots=True)
class NextStepResidentReply:
    """
    Non-terminal: a resident actor (open inbox) produced its reply to the current
    inbox message — the resident analog of :class:`NextStepStop`, which would end
    a lone agent's run. The handler persists the turn (a resident reply otherwise
    never checkpoints, so the answer would be re-generated on resume); the loop
    then parks for the next message.
    """

    final_answer: str


@dataclass(frozen=True, slots=True)
class NextStepForceResidentReply:
    """
    Non-terminal: a resident actor burned its per-message turn budget without
    producing an answer it can return — the resident analog of
    :class:`NextStepForceFinalAnswer`. The handler force-generates a final answer
    for the current message (closing any dangling tool calls), persists it, then
    parks for the next message. Caps one runaway message's work without ever
    ending the resident's run.
    """


@dataclass(frozen=True, slots=True)
class NextStepContinue:
    """
    Non-terminal: no tools and no final answer — loop again.
    Covers reason-turns under ``force_react_mode`` and final-answer
    suppression while background tasks are pending.
    """


type NextStep = (
    NextStepStop
    | NextStepForceFinalAnswer
    | NextStepRunTools
    | NextStepResidentReply
    | NextStepForceResidentReply
    | NextStepContinue
)


def decide_next_step(
    *,
    final_answer: str | None,
    tool_calls: Sequence[FunctionToolCallItem],
    turn: int,
    max_turns: int,
    bg_tasks_pending: bool,
    deadline_exceeded: bool = False,
    inbox_open: bool = False,
    turns_on_message: int = 0,
) -> NextStep:
    """
    Classify the next :class:`AgentLoop` step. Pure function — no I/O,
    no side effects.

    Precedence: stop on final answer (unless background tasks still block it,
    and the run deadline has not passed); force-generate when the wall-clock
    deadline is exceeded (``TIMEOUT``) or the turn budget is exhausted
    (``MAX_TURNS``); run tools if the response has any; otherwise continue.

    ``inbox_open`` marks a resident actor. Its budget is **per inbox message**,
    not per run: ``turns_on_message`` (turns generated since the current message
    was delivered) is capped at ``max_turns`` rather than the lifetime ``turn``,
    which grows unbounded across messages. A resident never stops or force-ends
    its run; instead, finishing a message recycles the loop to wait for the next
    one. A satisfied answer becomes a ``NextStepResidentReply``; blowing the
    per-message budget without one becomes a ``NextStepForceResidentReply`` (force
    an answer, then move on) — so a model stuck in a tool loop on one message
    cannot spin forever.
    """
    # A resident's budget is per message; a lone agent's is the whole run.
    budget_turn = turns_on_message if inbox_open else turn
    over_budget = budget_turn >= max_turns

    # A blown deadline or an exhausted budget overrides waiting on background work.
    wait_for_bg = bg_tasks_pending and not over_budget and not deadline_exceeded

    if final_answer is not None and not wait_for_bg and not inbox_open:
        return NextStepStop(final_answer=final_answer)

    if deadline_exceeded and not inbox_open:
        return NextStepForceFinalAnswer(stop_reason=StopReason.TIMEOUT)

    if over_budget and not inbox_open:
        return NextStepForceFinalAnswer()

    # The resident's reply for this message, if it has one it can return now.
    resident_answer = final_answer if (inbox_open and not wait_for_bg) else None

    if over_budget and inbox_open:
        # Resident past its per-message budget: wrap this message up now —
        # never run more tools or continue. Reply with the answer if there is
        # one, else force one (closing dangling tool calls), then park.
        if resident_answer is not None:
            return NextStepResidentReply(final_answer=resident_answer)
        return NextStepForceResidentReply()

    if tool_calls:
        return NextStepRunTools(tool_calls=tool_calls)

    if resident_answer is not None:
        # The resident analog of Stop: the message is fully handled, but the open
        # inbox recycles the loop (the handler persists + releases it) rather than
        # ending the run. Checked after tool_calls so a turn that both answers and
        # calls tools still runs the tools first (matching a resident's prior
        # behavior); reasoning turns (no final answer) fall through to Continue.
        return NextStepResidentReply(final_answer=resident_answer)

    return NextStepContinue()
