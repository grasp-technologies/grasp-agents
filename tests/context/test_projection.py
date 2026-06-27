"""
View-time provider-invariant repair (``context.projection``).

A view projector may drop / reorder messages and orphan a tool_call /
tool_result pair; ``repair_tool_call_pairing`` makes the projected view
valid again before it reaches the provider. The repaired output must satisfy
the same invariant ``LLMAgentTranscript.validate_tool_call_pairing`` enforces
on the log, so that is the oracle here.
"""

import logging
from collections.abc import Sequence

import pytest

from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.context.projection import apply_folds, repair_tool_call_pairing
from grasp_agents.context.system_reminder import wrap_in_system_reminder
from grasp_agents.types.folds import FoldSpec
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
)


def _user(text: str) -> InputMessageItem:
    return InputMessageItem.from_text(text, role="user")


def _call(call_id: str) -> FunctionToolCallItem:
    return FunctionToolCallItem(call_id=call_id, name="tool", arguments="{}")


def _result(call_id: str, output: str = "ok") -> FunctionToolOutputItem:
    return FunctionToolOutputItem.from_tool_result(call_id=call_id, output=output)


def _assert_valid(messages: Sequence[InputItem]) -> None:
    transcript = LLMAgentTranscript()
    transcript.messages = list(messages)
    transcript.validate_tool_call_pairing()  # raises if the view is invalid


def _output_call_ids(messages: Sequence[InputItem]) -> list[str]:
    return [m.call_id for m in messages if isinstance(m, FunctionToolOutputItem)]


def test_valid_view_is_unchanged() -> None:
    messages = [_user("q"), _call("c1"), _result("c1"), _user("next")]
    repaired = repair_tool_call_pairing(messages)
    assert repaired == messages
    _assert_valid(repaired)


def test_prefix_cut_on_turn_boundary_is_noop() -> None:
    # A projection that drops a whole leading turn (call+result together) leaves
    # no orphan — the repair is a no-op.
    full = [_user("q0"), _call("c1"), _result("c1"), _user("q1")]
    cut = full[3:]  # just [_user("q1")]
    assert repair_tool_call_pairing(cut) == cut


def test_free_floating_result_is_dropped() -> None:
    # The projection kept a result whose call it dropped.
    messages = [_user("q"), _result("c1"), _user("next")]
    repaired = repair_tool_call_pairing(messages)
    assert _output_call_ids(repaired) == []
    # Kept items are the same objects (only the free-floating result dropped).
    assert repaired == [messages[0], messages[2]]
    _assert_valid(repaired)


def test_duplicate_result_is_dropped() -> None:
    messages = [_call("c1"), _result("c1"), _result("c1")]
    repaired = repair_tool_call_pairing(messages)
    assert _output_call_ids(repaired) == ["c1"]
    _assert_valid(repaired)


def test_missing_result_is_stubbed_before_next_input() -> None:
    # The projection dropped the result but kept the call and a later input.
    messages = [_call("c1"), _user("next")]
    repaired = repair_tool_call_pairing(messages)
    assert [type(m).__name__ for m in repaired] == [
        "FunctionToolCallItem",
        "FunctionToolOutputItem",
        "InputMessageItem",
    ]
    assert _output_call_ids(repaired) == ["c1"]
    _assert_valid(repaired)


def test_dangling_call_at_tail_is_stubbed() -> None:
    messages = [_user("q"), _call("c1")]
    repaired = repair_tool_call_pairing(messages)
    assert _output_call_ids(repaired) == ["c1"]
    _assert_valid(repaired)


def test_multiple_open_calls_all_resolved() -> None:
    # Parallel tool calls in one turn, both results dropped by the projection.
    messages = [_call("c1"), _call("c2"), _user("next")]
    repaired = repair_tool_call_pairing(messages)
    assert sorted(_output_call_ids(repaired)) == ["c1", "c2"]
    _assert_valid(repaired)


def test_empty_view() -> None:
    assert repair_tool_call_pairing([]) == []


# --- apply_folds (summarized-span replacement) ---


def _summary(messages: Sequence[InputItem]) -> list[str]:
    return [m.text for m in messages if isinstance(m, InputMessageItem)]


def _sum(text: str) -> str:
    return wrap_in_system_reminder(
        text,
        subject="your previous conversation, summarized to compact the input context",
    )


def test_foldspec_rejects_empty_span() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FoldSpec(start=2, end=2, summary="x")


def test_no_folds_is_identity() -> None:
    messages = [_user("a"), _call("c1"), _result("c1")]
    assert apply_folds(messages, []) == messages


def test_single_fold_replaces_span() -> None:
    messages = [_user("a"), _user("b"), _user("c"), _user("d")]
    out = apply_folds(messages, [FoldSpec(start=1, end=3, summary="b and c")])
    assert len(out) == 3
    assert out[0] is messages[0]
    assert out[2] is messages[3]
    assert isinstance(out[1], InputMessageItem)
    assert "b and c" in out[1].text
    assert "<system-reminder" in out[1].text  # wrapped, not a magic phrase


def test_multiple_disjoint_folds() -> None:
    messages = [_user(c) for c in "abcdef"]
    out = apply_folds(
        messages,
        [
            FoldSpec(start=0, end=2, summary="S1"),
            FoldSpec(start=3, end=5, summary="S2"),
        ],
    )
    # [S1] c [S2] f
    assert _summary(out) == [_sum("S1"), "c", _sum("S2"), "f"]


def test_out_of_bounds_fold_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    messages = [_user("a"), _user("b")]
    with caplog.at_level(logging.WARNING, logger="grasp_agents.context.projection"):
        out = apply_folds(messages, [FoldSpec(start=1, end=99, summary="S")])
    assert out == messages  # end past the log → skipped
    assert any("out-of-bounds" in r.getMessage() for r in caplog.records)


def test_overlapping_fold_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    messages = [_user(c) for c in "abcd"]
    with caplog.at_level(logging.WARNING, logger="grasp_agents.context.projection"):
        out = apply_folds(
            messages,
            [
                FoldSpec(start=0, end=2, summary="S1"),
                FoldSpec(start=1, end=3, summary="S2"),
            ],
        )
    # S2 overlaps S1's consumed range → skipped; S1 applied.
    assert _summary(out) == [_sum("S1"), "c", "d"]
    assert any("overlapping" in r.getMessage() for r in caplog.records)


def test_fold_preserves_tool_pairing() -> None:
    # Folding a whole call+result turn leaves no orphan.
    messages = [_user("q"), _call("c1"), _result("c1"), _user("next")]
    out = apply_folds(messages, [FoldSpec(start=1, end=3, summary="used a tool")])
    _assert_valid(out)
    assert repair_tool_call_pairing(out) == out  # nothing to repair
