"""
Tests for the Loop-state ADT and the pure ``decide_next_step`` function.

``decide_next_step`` is the JUDGE-phase state machine for :class:`AgentLoop`.
Keeping the decision logic as a pure function lets us unit-test every
transition exhaustively without stitching up an LLM, memory, or tool
registry.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

import grasp_agents
from grasp_agents.agent.loop_state import (
    NextStep,
    NextStepContinue,
    NextStepForceFinalAnswer,
    NextStepForceResidentAnswer,
    NextStepResidentAnswer,
    NextStepRunTools,
    NextStepStop,
    decide_next_step,
)
from grasp_agents.types.events import StopReason, TurnEndInfo
from grasp_agents.types.items import FunctionToolCallItem

if TYPE_CHECKING:
    from collections.abc import Sequence


def _call(name: str = "get_weather", arguments: str = "{}") -> FunctionToolCallItem:
    return FunctionToolCallItem(
        id=f"fc_{name}",
        call_id=f"call_{name}",
        name=name,
        arguments=arguments,
    )


# ---------- Terminal: NextStepStop ----------


class TestNextStepStop:
    def test_final_answer_stops_loop(self) -> None:
        step = decide_next_step(
            final_answer="answer",
            tool_calls=[],
            turn=0,
            max_turns=10,
            blocking_bg_tasks=False,
        )
        assert isinstance(step, NextStepStop)
        assert step.final_answer == "answer"
        assert step.stop_reason is StopReason.FINAL_ANSWER

    def test_final_answer_with_tool_calls_still_stops(self) -> None:
        # Once we have a final answer and no bg tasks are holding us, we
        # stop even if the model also returned tool calls.
        step = decide_next_step(
            final_answer="answer",
            tool_calls=[_call()],
            turn=1,
            max_turns=10,
            blocking_bg_tasks=False,
        )
        assert isinstance(step, NextStepStop)
        assert step.final_answer == "answer"

    def test_final_answer_on_last_turn_stops_even_with_bg(self) -> None:
        # Budget exhausted → we can't wait for bg any longer, so stop.
        step = decide_next_step(
            final_answer="answer",
            tool_calls=[],
            turn=10,
            max_turns=10,
            blocking_bg_tasks=True,
        )
        assert isinstance(step, NextStepStop)

    def test_frozen(self) -> None:
        step = NextStepStop(final_answer="x")
        with pytest.raises(FrozenInstanceError):
            step.final_answer = "y"  # type: ignore[misc]


# ---------- Terminal: NextStepForceFinalAnswer ----------


class TestNextStepForceFinalAnswer:
    def test_budget_exhausted_with_no_final_answer(self) -> None:
        step = decide_next_step(
            final_answer=None,
            tool_calls=[_call()],
            turn=10,
            max_turns=10,
            blocking_bg_tasks=False,
        )
        assert isinstance(step, NextStepForceFinalAnswer)

    def test_budget_exhausted_without_tool_calls(self) -> None:
        step = decide_next_step(
            final_answer=None,
            tool_calls=[],
            turn=10,
            max_turns=10,
            blocking_bg_tasks=False,
        )
        assert isinstance(step, NextStepForceFinalAnswer)

    def test_budget_exhausted_takes_priority_over_tools(self) -> None:
        # No final answer, turn budget exhausted, tool calls present — we
        # still force-generate rather than run more tools.
        step = decide_next_step(
            final_answer=None,
            tool_calls=[_call("a"), _call("b")],
            turn=5,
            max_turns=5,
            blocking_bg_tasks=True,
        )
        assert isinstance(step, NextStepForceFinalAnswer)


# ---------- Deadline (wall-clock) → TIMEOUT ----------


class TestDeadline:
    def test_deadline_exceeded_forces_final_with_timeout(self) -> None:
        step = decide_next_step(
            final_answer=None,
            tool_calls=[_call()],
            turn=2,
            max_turns=10,
            blocking_bg_tasks=False,
            deadline_exceeded=True,
        )
        assert isinstance(step, NextStepForceFinalAnswer)
        assert step.stop_reason is StopReason.TIMEOUT

    def test_max_turns_force_keeps_max_turns_reason(self) -> None:
        step = decide_next_step(
            final_answer=None,
            tool_calls=[_call()],
            turn=10,
            max_turns=10,
            blocking_bg_tasks=False,
        )
        assert isinstance(step, NextStepForceFinalAnswer)
        assert step.stop_reason is StopReason.MAX_TURNS

    def test_real_final_answer_wins_over_expired_deadline(self) -> None:
        # We got an answer in time-ish; an expired deadline must not discard it.
        step = decide_next_step(
            final_answer="done",
            tool_calls=[],
            turn=2,
            max_turns=10,
            blocking_bg_tasks=False,
            deadline_exceeded=True,
        )
        assert isinstance(step, NextStepStop)

    def test_deadline_overrides_background_wait(self) -> None:
        # Final answer present, bg tasks pending (would normally wait) — but a
        # blown deadline stops now instead of waiting on background work.
        step = decide_next_step(
            final_answer="done",
            tool_calls=[],
            turn=2,
            max_turns=10,
            blocking_bg_tasks=True,
            deadline_exceeded=True,
        )
        assert isinstance(step, NextStepStop)


# ---------- Non-terminal: NextStepRunTools ----------


class TestNextStepRunTools:
    def test_tool_calls_no_final_answer(self) -> None:
        calls = [_call("a")]
        step = decide_next_step(
            final_answer=None,
            tool_calls=calls,
            turn=1,
            max_turns=10,
            blocking_bg_tasks=False,
        )
        assert isinstance(step, NextStepRunTools)
        assert list(step.tool_calls) == calls

    def test_tool_calls_preserved_verbatim(self) -> None:
        calls = [_call("a", '{"x":1}'), _call("b")]
        step = decide_next_step(
            final_answer=None,
            tool_calls=calls,
            turn=3,
            max_turns=10,
            blocking_bg_tasks=False,
        )
        assert isinstance(step, NextStepRunTools)
        assert len(step.tool_calls) == 2
        assert step.tool_calls[0].name == "a"
        assert json.loads(step.tool_calls[0].arguments) == {"x": 1}
        assert step.tool_calls[1].name == "b"


# ---------- Non-terminal: NextStepContinue ----------


class TestNextStepContinue:
    def test_no_final_no_tools_no_budget_exhaustion(self) -> None:
        step = decide_next_step(
            final_answer=None,
            tool_calls=[],
            turn=3,
            max_turns=10,
            blocking_bg_tasks=False,
        )
        assert isinstance(step, NextStepContinue)

    def test_final_answer_suppressed_by_bg_tasks(self) -> None:
        # Final answer extracted but bg tasks are pending and turns remain —
        # continue so the LLM can observe the bg results on the next turn.
        step = decide_next_step(
            final_answer="answer",
            tool_calls=[],
            turn=1,
            max_turns=10,
            blocking_bg_tasks=True,
        )
        assert isinstance(step, NextStepContinue)

    def test_final_answer_suppressed_bg_with_tool_calls(self) -> None:
        # Final answer AND tool calls AND bg pending — since the final
        # answer is suppressed, we then look at tool_calls: they're present,
        # so we run them. (RunTools wins over bare Continue.)
        step = decide_next_step(
            final_answer="answer",
            tool_calls=[_call()],
            turn=1,
            max_turns=10,
            blocking_bg_tasks=True,
        )
        assert isinstance(step, NextStepRunTools)


# ---------- Resident actor: NextStepResidentAnswer ----------


class TestNextStepResidentAnswer:
    def test_final_answer_becomes_resident_answer(self) -> None:
        # A resident (open inbox) that produced a reply: the analog of Stop, but
        # the loop recycles instead of ending.
        step = decide_next_step(
            final_answer="reply",
            tool_calls=[],
            turn=3,
            max_turns=10,
            blocking_bg_tasks=False,
            inbox_open=True,
        )
        assert isinstance(step, NextStepResidentAnswer)
        assert step.final_answer == "reply"

    def test_tool_calls_still_win_over_resident_answer(self) -> None:
        # Final answer AND tool calls under an open inbox: run the tools first
        # (matching a resident's prior behavior), don't release the message yet.
        step = decide_next_step(
            final_answer="reply",
            tool_calls=[_call()],
            turn=3,
            max_turns=10,
            blocking_bg_tasks=False,
            inbox_open=True,
        )
        assert isinstance(step, NextStepRunTools)

    def test_reasoning_turn_continues_not_reply(self) -> None:
        # No final answer yet (a reasoning turn): keep going, don't release the
        # message — only a produced answer is a ResidentAnswer.
        step = decide_next_step(
            final_answer=None,
            tool_calls=[],
            turn=3,
            max_turns=10,
            blocking_bg_tasks=False,
            inbox_open=True,
        )
        assert isinstance(step, NextStepContinue)

    def test_resident_answers_despite_blocking_bg_tasks(self) -> None:
        # Answer-blocking bg work gates only a bounded run's final answer
        # (ending the run would strand the pending result). A resident's loop
        # outlives its answer and receives the completion on wake, so it
        # answers now instead of holding the message open.
        step = decide_next_step(
            final_answer="reply",
            tool_calls=[],
            turn=1,
            max_turns=10,
            blocking_bg_tasks=True,
            inbox_open=True,
        )
        assert isinstance(step, NextStepResidentAnswer)
        assert step.final_answer == "reply"

    def test_resident_never_force_finalizes_past_budget(self) -> None:
        # A resident past its budget / deadline must never Stop or
        # ForceFinalAnswer (which would END the run) — with an answer in hand it
        # replies (recycling the loop), even past the per-message budget.
        step = decide_next_step(
            final_answer="reply",
            tool_calls=[],
            turn=99,
            max_turns=10,
            blocking_bg_tasks=False,
            deadline_exceeded=True,
            inbox_open=True,
            turns_on_message=99,
        )
        assert isinstance(step, NextStepResidentAnswer)

    def test_lifetime_turn_does_not_trip_per_message_budget(self) -> None:
        # A long-lived resident (huge lifetime ``turn``) is bounded per message,
        # not per run: with only a few turns on the current message it keeps
        # working (here: runs its tools) rather than being force-finalized.
        step = decide_next_step(
            final_answer=None,
            tool_calls=[_call()],
            turn=999,
            max_turns=10,
            blocking_bg_tasks=False,
            inbox_open=True,
            turns_on_message=2,
        )
        assert isinstance(step, NextStepRunTools)


# ---------- Resident actor: NextStepForceResidentAnswer ----------


class TestNextStepForceResidentAnswer:
    def test_per_message_budget_without_answer_forces_one(self) -> None:
        # Stuck in a tool loop on one message past the per-message budget, with no
        # answer: force a reply (and move on) rather than spinning forever.
        step = decide_next_step(
            final_answer=None,
            tool_calls=[_call()],
            turn=50,
            max_turns=10,
            blocking_bg_tasks=False,
            inbox_open=True,
            turns_on_message=10,
        )
        assert isinstance(step, NextStepForceResidentAnswer)

    def test_force_reply_preempts_tools(self) -> None:
        # Over the per-message budget, pending tool calls must NOT run — the cap
        # wraps the message up (mirrors a lone agent's force-final at max_turns).
        step = decide_next_step(
            final_answer=None,
            tool_calls=[_call("a"), _call("b")],
            turn=20,
            max_turns=5,
            blocking_bg_tasks=True,
            inbox_open=True,
            turns_on_message=7,
        )
        assert isinstance(step, NextStepForceResidentAnswer)

    def test_answer_at_budget_answers_not_forces(self) -> None:
        # Over the per-message budget but the model produced an answer: deliver it
        # as a normal reply (no need to force-generate one).
        step = decide_next_step(
            final_answer="done",
            tool_calls=[],
            turn=20,
            max_turns=10,
            blocking_bg_tasks=False,
            inbox_open=True,
            turns_on_message=10,
        )
        assert isinstance(step, NextStepResidentAnswer)
        assert step.final_answer == "done"

    def test_under_budget_no_answer_continues_not_forces(self) -> None:
        # Within the per-message budget, a reasoning turn just continues — the
        # force path is only for budget exhaustion.
        step = decide_next_step(
            final_answer=None,
            tool_calls=[],
            turn=20,
            max_turns=10,
            blocking_bg_tasks=False,
            inbox_open=True,
            turns_on_message=3,
        )
        assert isinstance(step, NextStepContinue)


# ---------- Precedence table (parametrized) ----------


@pytest.mark.parametrize(
    ("final_answer", "tool_calls_len", "turn", "max_turns", "bg", "expected"),
    [
        # (final, n_tools, turn, max_turns, bg_pending) -> expected class
        ("ans", 0, 0, 10, False, NextStepStop),
        ("ans", 1, 0, 10, False, NextStepStop),
        ("ans", 0, 10, 10, True, NextStepStop),  # budget exhausted, bg can't save us
        ("ans", 0, 1, 10, True, NextStepContinue),  # suppressed by bg
        ("ans", 1, 1, 10, True, NextStepRunTools),  # suppressed → tools win
        (None, 0, 10, 10, False, NextStepForceFinalAnswer),
        (None, 1, 10, 10, False, NextStepForceFinalAnswer),  # budget > tools
        (None, 1, 5, 10, False, NextStepRunTools),
        (None, 0, 5, 10, False, NextStepContinue),
        (None, 0, 5, 10, True, NextStepContinue),  # bg pending, no final → continue
    ],
)
def test_precedence_table(
    final_answer: str | None,
    tool_calls_len: int,
    turn: int,
    max_turns: int,
    bg: bool,
    expected: type[NextStep],
) -> None:
    calls: Sequence[FunctionToolCallItem] = [
        _call(f"t{i}") for i in range(tool_calls_len)
    ]
    step = decide_next_step(
        final_answer=final_answer,
        tool_calls=calls,
        turn=turn,
        max_turns=max_turns,
        blocking_bg_tasks=bg,
    )
    assert isinstance(step, expected)


# ---------- ADT shape contract ----------


class TestADTShape:
    def test_default_stop_reason_is_final_answer(self) -> None:
        step = NextStepStop(final_answer="x")
        assert step.stop_reason is StopReason.FINAL_ANSWER

    def test_stop_reason_override(self) -> None:
        # Not used by decide_next_step today, but the ADT accepts an
        # explicit stop_reason so future transitions (e.g. TIMEOUT) can
        # surface it without introducing a new variant.
        step = NextStepStop(final_answer="x", stop_reason=StopReason.TIMEOUT)
        assert step.stop_reason is StopReason.TIMEOUT


# ---------- Public export ----------


def test_loop_state_is_public_api() -> None:
    assert grasp_agents.agent.NextStep is NextStep
    assert grasp_agents.agent.NextStepStop is NextStepStop
    assert grasp_agents.agent.NextStepForceFinalAnswer is NextStepForceFinalAnswer
    assert grasp_agents.agent.NextStepRunTools is NextStepRunTools
    assert grasp_agents.agent.NextStepContinue is NextStepContinue
    assert grasp_agents.agent.decide_next_step is decide_next_step
    assert grasp_agents.StopReason is StopReason


class TestTurnEndInfoStopReason:
    """
    stop_reason stays a StopReason member through TurnEndInfo, so
    isinstance / is comparisons work — not only ==.
    """

    def test_construction_keeps_enum_member(self) -> None:
        info = TurnEndInfo(turn=1, had_tool_calls=False, stop_reason=StopReason.TIMEOUT)
        assert isinstance(info.stop_reason, StopReason)
        assert info.stop_reason is StopReason.TIMEOUT

    def test_roundtrip_keeps_enum_member(self) -> None:
        info = TurnEndInfo(turn=1, had_tool_calls=False, stop_reason=StopReason.TIMEOUT)
        back = TurnEndInfo.model_validate_json(info.model_dump_json())
        assert isinstance(back.stop_reason, StopReason)
        assert back.stop_reason is StopReason.TIMEOUT

    def test_json_still_serializes_to_string(self) -> None:
        info = TurnEndInfo(turn=1, had_tool_calls=False, stop_reason=StopReason.TIMEOUT)
        assert '"timeout"' in info.model_dump_json()
