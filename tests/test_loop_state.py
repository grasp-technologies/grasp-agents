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
    NextStepRunTools,
    NextStepStop,
    decide_next_step,
)
from grasp_agents.types.events import StopReason
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
            bg_tasks_pending=False,
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
            bg_tasks_pending=False,
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
            bg_tasks_pending=True,
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
            bg_tasks_pending=False,
        )
        assert isinstance(step, NextStepForceFinalAnswer)

    def test_budget_exhausted_without_tool_calls(self) -> None:
        step = decide_next_step(
            final_answer=None,
            tool_calls=[],
            turn=10,
            max_turns=10,
            bg_tasks_pending=False,
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
            bg_tasks_pending=True,
        )
        assert isinstance(step, NextStepForceFinalAnswer)


# ---------- Non-terminal: NextStepRunTools ----------


class TestNextStepRunTools:
    def test_tool_calls_no_final_answer(self) -> None:
        calls = [_call("a")]
        step = decide_next_step(
            final_answer=None,
            tool_calls=calls,
            turn=1,
            max_turns=10,
            bg_tasks_pending=False,
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
            bg_tasks_pending=False,
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
            bg_tasks_pending=False,
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
            bg_tasks_pending=True,
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
            bg_tasks_pending=True,
        )
        assert isinstance(step, NextStepRunTools)


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
        bg_tasks_pending=bg,
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
    assert grasp_agents.NextStep is NextStep
    assert grasp_agents.NextStepStop is NextStepStop
    assert grasp_agents.NextStepForceFinalAnswer is NextStepForceFinalAnswer
    assert grasp_agents.NextStepRunTools is NextStepRunTools
    assert grasp_agents.NextStepContinue is NextStepContinue
    assert grasp_agents.decide_next_step is decide_next_step
    assert grasp_agents.StopReason is StopReason
