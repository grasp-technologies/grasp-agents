"""``ContextWindowManager`` transcript surgery and the view state it repairs."""

import logging
from unittest.mock import patch

import pytest

from grasp_agents.agent.context_window import ContextWindowManager
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.folds import CompactionPlan, FoldSpec
from grasp_agents.types.items import InputMessageItem, OutputMessageItem
from grasp_agents.types.response import ResponseUsage

_COUNT = "grasp_agents.agent.context_window.estimate_input_tokens"


def _user(text: str) -> InputMessageItem:
    return InputMessageItem.from_text(text, role="user")


def _cw(n: int) -> ContextWindowManager:
    cw = ContextWindowManager(
        model_name="mock",
    )
    cw.add_messages([_user(str(i)) for i in range(n)])
    return cw


def _answer(text: str) -> OutputMessageItem:
    return OutputMessageItem(content=[OutputMessageText(text=text)], status="completed")


def _usage(input_tokens: int) -> ResponseUsage:
    return ResponseUsage(input_tokens=input_tokens, output_tokens=10)


def _fold(start: int, end: int) -> FoldSpec:
    return FoldSpec(start=start, end=end, summary=f"summary {start}-{end}")


async def _anchored(n: int, input_tokens: int = 1000) -> ContextWindowManager:
    """A manager whose usage anchor covers its first ``n`` messages."""
    cw = _cw(n)
    await cw.project_view(exec_id="t")
    cw.note_response_usage(_usage(input_tokens))
    return cw


class TestTruncate:
    def test_negative_count_is_rejected(self) -> None:
        cw = _cw(3)
        with pytest.raises(ValueError, match="message_count"):
            cw.truncate_transcript(-1)
        assert len(cw.transcript) == 3

    def test_count_past_the_end_keeps_log_and_folds(self) -> None:
        cw = _cw(4)
        cw.folds = [_fold(0, 2)]
        cw.truncate_transcript(10)
        assert len(cw.transcript) == 4
        assert cw.folds == [_fold(0, 2)]

    def test_cut_drops_folds_reaching_past_it(self) -> None:
        cw = _cw(6)
        cw.folds = [_fold(0, 2), _fold(2, 5)]
        cw.truncate_transcript(4)
        assert len(cw.transcript) == 4
        assert cw.folds == [_fold(0, 2)]


class TestUsageAnchor:
    @pytest.mark.asyncio
    async def test_cut_below_counted_prefix_drops_the_usage(self) -> None:
        cw = await _anchored(5)
        cw.truncate_transcript(3)

        assert cw.last_usage is None
        with patch(_COUNT, return_value=7) as count:
            assert cw.effective_input_tokens() == 7
        count.assert_called_once()

    @pytest.mark.asyncio
    async def test_cut_at_counted_prefix_keeps_the_anchor(self) -> None:
        cw = await _anchored(5)
        cw.add_messages([_user("x"), _user("y")])
        cw.truncate_transcript(5)

        with patch(_COUNT, return_value=7) as count:
            assert cw.effective_input_tokens() == 1000
        count.assert_not_called()

    @pytest.mark.asyncio
    async def test_cut_clamps_the_basis_of_the_view_in_flight(self) -> None:
        cw = _cw(5)
        await cw.project_view(exec_id="t")  # view basis: 5 messages
        cw.truncate_transcript(2)
        cw.note_response_usage(_usage(500))

        # The response's usage now anchors the 2 messages that remain, not 5.
        cw.add_messages([_user("new")])
        with patch(_COUNT, return_value=3) as count:
            assert cw.effective_input_tokens() == 503
        count.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_fold_stops_anchoring_but_keeps_the_usage(self) -> None:
        cw = await _anchored(5)
        cw.record_fold(
            CompactionPlan(end=3), "summary", tokens_before=1000, elapsed_s=0.0
        )

        assert cw.last_usage == _usage(1000)
        with patch(_COUNT, return_value=7) as count:
            assert cw.effective_input_tokens() == 7
        count.assert_called_once()

    def test_resume_anchors_on_the_request_before_the_latest_response(self) -> None:
        cw = _cw(0)
        log = [_user("a"), _answer("b"), _user("c"), _answer("d"), _user("e")]
        cw.replace_transcript(log, usage=_usage(1000))

        # The usage measured "a".."c"; "d" and "e" are estimated on top.
        with patch(_COUNT, return_value=3) as count:
            assert cw.effective_input_tokens() == 1003
        assert count.call_args.args[0] == log[3:]

    def test_usage_without_a_response_in_the_log_is_dropped(self) -> None:
        cw = _cw(0)
        cw.replace_transcript([_user("a"), _user("b")], usage=_usage(1000))
        assert cw.last_usage is None


class TestReplace:
    def test_folds_past_the_new_log_are_dropped_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cw = _cw(0)
        with caplog.at_level(
            logging.WARNING, logger="grasp_agents.agent.context_window"
        ):
            cw.replace_transcript(
                [_user(str(i)) for i in range(4)], folds=[_fold(0, 2), _fold(2, 6)]
            )

        assert cw.folds == [_fold(0, 2)]
        assert any("dropped 1 summary fold" in r.getMessage() for r in caplog.records)

    def test_folds_within_the_new_log_are_kept_silently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cw = _cw(0)
        with caplog.at_level(
            logging.WARNING, logger="grasp_agents.agent.context_window"
        ):
            cw.replace_transcript(
                [_user(str(i)) for i in range(4)], folds=[_fold(0, 2), _fold(2, 4)]
            )

        assert cw.folds == [_fold(0, 2), _fold(2, 4)]
        assert not caplog.records

    @pytest.mark.asyncio
    async def test_a_dropped_fold_never_summarizes_later_messages(self) -> None:
        cw = _cw(0)
        cw.replace_transcript([_user(str(i)) for i in range(4)], folds=[_fold(2, 6)])
        cw.add_messages([_user("new4"), _user("new5"), _user("new6")])

        view = await cw.project_view(exec_id="t")
        texts = [m.text for m in view if isinstance(m, InputMessageItem)]
        assert texts == ["0", "1", "2", "3", "new4", "new5", "new6"]
