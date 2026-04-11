"""Tests for EventConsole event display."""

import re
from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from grasp_agents.console import (
    EventConsole,
    _extract_input_text,
    _truncate,
    _truncate_lines,
    _unescape_json_string,
    stream_events,
)
from grasp_agents.types.content import InputImage, InputText, OutputMessageText
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskInfo,
    BackgroundTaskLaunchedEvent,
    Event,
    GenerationEndEvent,
    LLMStreamEvent,
    LLMStreamingErrorData,
    LLMStreamingErrorEvent,
    OutputMessageItemEvent,
    ProcPacketOutEvent,
    ReasoningItemEvent,
    SystemMessageEvent,
    ToolCallItemEvent,
    ToolErrorEvent,
    ToolErrorInfo,
    ToolResultEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_events import (
    FunctionCallArgumentsDelta,
    OutputItemAdded,
    OutputItemDone,
    OutputMessageTextPartTextDelta,
    ReasoningSummaryPartTextDelta,
    ResponseCompleted,
)
from grasp_agents.types.response import Response, ResponseUsage


def _make_console() -> tuple[EventConsole, StringIO]:
    """Create an EventConsole with captured output."""
    buf = StringIO()
    console = Console(file=buf, no_color=True, highlight=False, width=80)
    ec = EventConsole(console=console)
    return ec, buf


def _make_console_with(**kwargs: Any) -> tuple[EventConsole, StringIO]:
    buf = StringIO()
    console = Console(file=buf, no_color=True, highlight=False, width=80)
    ec = EventConsole(console=console, **kwargs)
    return ec, buf


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


async def _collect(ec: EventConsole, events: list[Event[Any]]) -> list[Event[Any]]:
    """Run events through console and collect yielded events."""
    async def gen():
        for e in events:
            yield e

    collected: list[Event[Any]] = []
    async for event in ec.stream(gen()):
        collected.append(event)
    return collected


# ── Helpers ──


class TestHelpers:
    def test_extract_input_text(self):
        msg = InputMessageItem.from_text("Hello world", role="user")
        assert _extract_input_text(msg) == "Hello world"

    def test_extract_input_text_with_image(self):
        msg = InputMessageItem(
            content_parts=[
                InputText(text="Look at this"),
                InputImage.from_url("https://example.com/pic.jpg"),
            ],
            role="user",
        )
        text = _extract_input_text(msg)
        assert "Look at this" in text
        assert "https://example.com/pic.jpg" in text

    def test_extract_input_text_base64_image(self):
        msg = InputMessageItem(
            content_parts=[InputImage.from_base64("abc123")],
            role="user",
        )
        assert "[image]" in _extract_input_text(msg)

    def test_truncate_short(self):
        assert _truncate("short", 100) == "short"

    def test_truncate_long(self):
        result = _truncate("a" * 200, 50)
        assert len(result) == 51  # 50 + "…"
        assert result.endswith("…")

    def test_unescape_json_string(self):
        assert _unescape_json_string('"hello\\nworld"') == "hello\nworld"
        assert _unescape_json_string('{"a": 1}') == '{\n  "a": 1\n}'
        assert _unescape_json_string("not json") == "not json"

    def test_unescape_non_string_json(self):
        result = _unescape_json_string('{"a": 1}')
        assert '"a": 1' in result

    def test_truncate_lines_short(self):
        assert _truncate_lines("a\nb\nc", 5) == "a\nb\nc"

    def test_truncate_lines_exact(self):
        assert _truncate_lines("a\nb\nc", 3) == "a\nb\nc"

    def test_truncate_lines_over(self):
        result = _truncate_lines("a\nb\nc\nd\ne", 3)
        assert result == "a\nb\nc\n… 2 more lines"


# ── Turn lifecycle ──


class TestTurnEvents:
    @pytest.mark.asyncio
    async def test_turn_start(self):
        ec, buf = _make_console()
        events = [TurnStartEvent(data=TurnInfo(turn=0), source="teacher")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "teacher" in output
        assert "turn 1" in output

    @pytest.mark.asyncio
    async def test_turn_end_final_answer(self):
        """Final answer stop reason produces no extra output."""
        ec, buf = _make_console()
        events = [
            TurnEndEvent(
                data=TurnEndInfo(turn=0, had_tool_calls=False, stop_reason="final_answer"),  # type: ignore[arg-type]
                source="agent",
            )
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "stopped" not in output

    @pytest.mark.asyncio
    async def test_turn_end_max_turns(self):
        ec, buf = _make_console()
        events = [
            TurnEndEvent(
                data=TurnEndInfo(turn=2, had_tool_calls=True, stop_reason="max_turns"),  # type: ignore[arg-type]
                source="agent",
            )
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "stopped" in output
        assert "max_turns" in output


# ── Streaming mode (LLMStreamEvent) ──


class TestStreamingText:
    @pytest.mark.asyncio
    async def test_text_deltas_stream(self):
        ec, buf = _make_console()
        events = [
            LLMStreamEvent(
                data=OutputMessageTextPartTextDelta(
                    content_index=0, delta="Hello ", output_index=0,
                    sequence_number=1, item_id="i1", logprobs=[],
                ),
                source="agent", exec_id="c1",
            ),
            LLMStreamEvent(
                data=OutputMessageTextPartTextDelta(
                    content_index=0, delta="world!", output_index=0,
                    sequence_number=2, item_id="i1", logprobs=[],
                ),
                source="agent", exec_id="c1",
            ),
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "Hello " in output
        assert "world!" in output

    @pytest.mark.asyncio
    async def test_text_block_ends_with_newline(self):
        ec, buf = _make_console()
        msg_item = OutputMessageItem(
            content_parts=[OutputMessageText(text="Hello")],
            status="completed",
        )
        events = [
            LLMStreamEvent(
                data=OutputMessageTextPartTextDelta(
                    content_index=0, delta="Hello", output_index=0,
                    sequence_number=1, item_id="i1", logprobs=[],
                ),
                source="agent", exec_id="c1",
            ),
            LLMStreamEvent(
                data=OutputItemDone(item=msg_item, output_index=0, sequence_number=2),
                source="agent", exec_id="c1",
            ),
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "Hello\n" in output


class TestStreamingToolCalls:
    @pytest.mark.asyncio
    async def test_tool_call_header_from_promoted_event(self):
        """Tool name comes from ToolCallItemEvent, not OutputItemAdded."""
        ec, buf = _make_console()
        item = FunctionToolCallItem(call_id="tc_1", name="search", arguments='{"q": "test"}')
        events = [
            LLMStreamEvent(
                data=OutputItemAdded(item=item, output_index=0, sequence_number=1),
                source="agent", exec_id="c1",
            ),
            ToolCallItemEvent(data=item, source="agent", exec_id="c1"),
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "search" in output
        assert "╭" in output  # Panel with tool name as title

    @pytest.mark.asyncio
    async def test_tool_call_args_shown(self):
        ec, buf = _make_console()
        item = FunctionToolCallItem(call_id="tc_1", name="search", arguments='{"query": "test"}')
        events = [
            LLMStreamEvent(
                data=OutputItemAdded(item=item, output_index=0, sequence_number=1),
                source="agent", exec_id="c1",
            ),
            ToolCallItemEvent(data=item, source="agent", exec_id="c1"),
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "query" in output
        assert "test" in output
        assert "╭" in output  # Panel border

    @pytest.mark.asyncio
    async def test_tool_call_args_hidden(self):
        ec, buf = _make_console_with(show_tool_args=False)
        item = FunctionToolCallItem(call_id="tc_1", name="search", arguments='{"query": "test"}')
        events = [
            LLMStreamEvent(
                data=OutputItemAdded(item=item, output_index=0, sequence_number=1),
                source="agent", exec_id="c1",
            ),
            ToolCallItemEvent(data=item, source="agent", exec_id="c1"),
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "search" in output  # header still shown
        assert "query" not in output  # args hidden


class TestStreamingThinking:
    @pytest.mark.asyncio
    async def test_thinking_hidden_by_default(self):
        ec, buf = _make_console()
        events = [
            LLMStreamEvent(
                data=OutputItemAdded(
                    item=ReasoningItem(), output_index=0, sequence_number=1
                ),
                source="agent", exec_id="c1",
            ),
            LLMStreamEvent(
                data=ReasoningSummaryPartTextDelta(
                    content_index=0, delta="deep thought", summary_index=0,
                    output_index=0, sequence_number=2, item_id="i1",
                ),
                source="agent", exec_id="c1",
            ),
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "deep thought" not in output
        assert "thinking" not in output

    @pytest.mark.asyncio
    async def test_thinking_shown_when_enabled(self):
        ec, buf = _make_console_with(show_thinking=True)
        events = [
            LLMStreamEvent(
                data=OutputItemAdded(
                    item=ReasoningItem(), output_index=0, sequence_number=1
                ),
                source="agent", exec_id="c1",
            ),
            LLMStreamEvent(
                data=ReasoningSummaryPartTextDelta(
                    content_index=0, delta="deep thought", summary_index=0,
                    output_index=0, sequence_number=2, item_id="i1",
                ),
                source="agent", exec_id="c1",
            ),
            LLMStreamEvent(
                data=OutputItemDone(
                    item=ReasoningItem(), output_index=0, sequence_number=3,
                ),
                source="agent", exec_id="c1",
            ),
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "thinking" in output
        assert "deep thought" in output


# ── Non-streaming mode (promoted events) ──


class TestNonStreamingMode:
    @pytest.mark.asyncio
    async def test_output_message_full_text(self):
        """When no deltas seen, OutputMessageItemEvent prints full text."""
        ec, buf = _make_console()
        msg = OutputMessageItem(
            content_parts=[OutputMessageText(text="Complete answer")],
            status="completed",
        )
        events = [OutputMessageItemEvent(data=msg, source="agent", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "Complete answer" in output

    @pytest.mark.asyncio
    async def test_output_message_skipped_after_streaming(self):
        """When deltas were seen, promoted OutputMessageItemEvent is skipped."""
        ec, buf = _make_console()
        msg = OutputMessageItem(
            content_parts=[OutputMessageText(text="Hello world")],
            status="completed",
        )
        events = [
            LLMStreamEvent(
                data=OutputMessageTextPartTextDelta(
                    content_index=0, delta="Hello world", output_index=0,
                    sequence_number=1, item_id="i1", logprobs=[],
                ),
                source="agent", exec_id="c1",
            ),
            OutputMessageItemEvent(data=msg, source="agent", exec_id="c1"),
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        # "Hello world" should appear once (from delta), not twice
        assert output.count("Hello world") == 1

    @pytest.mark.asyncio
    async def test_tool_call_item_full(self):
        """Non-streaming tool call shows name and args."""
        ec, buf = _make_console()
        item = FunctionToolCallItem(
            call_id="tc_1", name="web_search", arguments='{"q": "test"}'
        )
        events = [ToolCallItemEvent(data=item, source="agent", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "web_search" in output
        assert "q" in output
        assert "test" in output
        assert "╭" in output  # Panel with tool name as title

    @pytest.mark.asyncio
    async def test_tool_call_item_skipped_after_streaming(self):
        ec, buf = _make_console()
        item = FunctionToolCallItem(
            call_id="tc_1", name="search", arguments='{"q": "test"}'
        )
        events = [
            LLMStreamEvent(
                data=OutputItemAdded(item=item, output_index=0, sequence_number=1),
                source="agent", exec_id="c1",
            ),
            ToolCallItemEvent(data=item, source="agent", exec_id="c1"),
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        # Tool name appears once in panel title, not duplicated
        assert output.count("search") == 1

    @pytest.mark.asyncio
    async def test_reasoning_item_non_streaming(self):
        ec, buf = _make_console_with(show_thinking=True)
        item = ReasoningItem()
        events = [ReasoningItemEvent(data=item, source="agent", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "thinking" in output


# ── Tool result and error events ──


class TestToolEvents:
    @pytest.mark.asyncio
    async def test_tool_result(self):
        ec, buf = _make_console()
        msg = FunctionToolOutputItem.from_tool_result(
            call_id="tc_1", output="found 3 results"
        )
        events = [ToolResultEvent(data=msg, source="agent", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "found 3 results" in output

    @pytest.mark.asyncio
    async def test_tool_result_json_unescaped(self):
        """JSON-encoded string results are unescaped for readability."""
        ec, buf = _make_console()
        msg = FunctionToolOutputItem.from_tool_result(
            call_id="tc_1", output={"status": "ok"}
        )
        events = [ToolResultEvent(data=msg, source="agent", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "status" in output

    @pytest.mark.asyncio
    async def test_tool_error(self):
        ec, buf = _make_console()
        events = [
            ToolErrorEvent(
                data=ToolErrorInfo(tool_name="search", error="API down"),
                source="agent", exec_id="c1",
            )
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "search" in output
        assert "API down" in output
        assert "✗" in output

    @pytest.mark.asyncio
    async def test_tool_error_timeout(self):
        ec, buf = _make_console()
        events = [
            ToolErrorEvent(
                data=ToolErrorInfo(tool_name="slow", error="took too long", timed_out=True),
                source="agent", exec_id="c1",
            )
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "timed out" in output


# ── Message events ──


class TestMessageEvents:
    @pytest.mark.asyncio
    async def test_user_message_hidden_by_default(self):
        ec, buf = _make_console()
        msg = InputMessageItem.from_text("Hello agent", role="user")
        events = [UserMessageEvent(data=msg, source="agent", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "User:" not in output

    @pytest.mark.asyncio
    async def test_user_message_shown_when_enabled(self):
        ec, buf = _make_console_with(show_input_messages=True)
        msg = InputMessageItem.from_text("Hello agent", role="user")
        events = [UserMessageEvent(data=msg, source="agent", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "User" in output
        assert "Hello agent" in output

    @pytest.mark.asyncio
    async def test_system_message_shown_when_enabled(self):
        ec, buf = _make_console_with(show_input_messages=True)
        msg = InputMessageItem.from_text("You are helpful.", role="system")
        events = [SystemMessageEvent(data=msg, source="agent", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "System" in output
        assert "You are helpful." in output


# ── Usage display ──


class TestUsageDisplay:
    @pytest.mark.asyncio
    async def test_generation_end_with_usage(self):
        ec, buf = _make_console()
        resp = Response(
            model="claude-sonnet-4-5",
            output_items=[],
            usage_with_cost=ResponseUsage(
                input_tokens=100, output_tokens=50, total_tokens=150, cost=0.0015
            ),
        )
        events = [GenerationEndEvent(data=resp, source="agent", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "claude-sonnet-4-5" in output
        assert "100" in output
        assert "50" in output
        assert "$0.0015" in output

    @pytest.mark.asyncio
    async def test_usage_hidden(self):
        ec, buf = _make_console_with(show_usage=False)
        resp = Response(
            model="test-model",
            output_items=[],
            usage_with_cost=ResponseUsage(
                input_tokens=100, output_tokens=50, total_tokens=150
            ),
        )
        events = [GenerationEndEvent(data=resp, source="agent", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "100" not in output

    @pytest.mark.asyncio
    async def test_cumulative_cost(self):
        """After 2+ generations, cumulative total is shown."""
        ec, buf = _make_console()
        resp1 = Response(
            model="test-model",
            output_items=[],
            usage_with_cost=ResponseUsage(
                input_tokens=100, output_tokens=50, total_tokens=150, cost=0.005
            ),
        )
        resp2 = Response(
            model="test-model",
            output_items=[],
            usage_with_cost=ResponseUsage(
                input_tokens=200, output_tokens=80, total_tokens=280, cost=0.008
            ),
        )
        events = [
            GenerationEndEvent(data=resp1, source="a", exec_id="c1"),
            GenerationEndEvent(data=resp2, source="a", exec_id="c2"),
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "Σ" in _strip_ansi(output)
        assert "$0.0130" in output

    @pytest.mark.asyncio
    async def test_cached_tokens_shown(self):
        """Cached input tokens are displayed when present."""
        from grasp_agents.types.response import InputTokensDetails

        ec, buf = _make_console()
        resp = Response(
            model="test-model",
            output_items=[],
            usage_with_cost=ResponseUsage(
                input_tokens=500,
                input_tokens_details=InputTokensDetails(cached_tokens=300),
                output_tokens=50,
                total_tokens=550,
                cost=0.001,
            ),
        )
        events = [GenerationEndEvent(data=resp, source="a", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "500" in output
        assert "300 cached" in output

    @pytest.mark.asyncio
    async def test_thinking_tokens_shown(self):
        """Reasoning/thinking tokens are displayed when present."""
        from grasp_agents.types.response import OutputTokensDetails

        ec, buf = _make_console()
        resp = Response(
            model="test-model",
            output_items=[],
            usage_with_cost=ResponseUsage(
                input_tokens=100,
                output_tokens=200,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=150),
                total_tokens=300,
                cost=0.002,
            ),
        )
        events = [GenerationEndEvent(data=resp, source="a", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "200" in output
        assert "150 thinking" in output


# ── Tool panel tests ──


class TestToolPanels:
    @pytest.mark.asyncio
    async def test_tool_args_nested_json(self):
        """Nested dicts render in panel."""
        ec, buf = _make_console()
        item = FunctionToolCallItem(
            call_id="tc_1", name="api",
            arguments='{"config": {"temp": 0.7, "max": 500}, "query": "hello"}'
        )
        events = [ToolCallItemEvent(data=item, source="a", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "config" in output
        assert "temp" in output
        assert "query" in output
        assert "hello" in output
        assert "╭" in output

    @pytest.mark.asyncio
    async def test_tool_args_non_dict(self):
        """Non-dict args handled gracefully."""
        ec, buf = _make_console()
        item = FunctionToolCallItem(
            call_id="tc_1", name="simple", arguments='"just a string"'
        )
        events = [ToolCallItemEvent(data=item, source="a", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "just a string" in output
        assert "╭" in output

    @pytest.mark.asyncio
    async def test_tool_result_panel_has_border(self):
        """Tool result rendered in a panel with borders."""
        ec, buf = _make_console()
        msg = FunctionToolOutputItem.from_tool_result(
            call_id="tc_1", output="result text"
        )
        events = [ToolResultEvent(data=msg, source="a", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "──" in output  # HORIZONTALS box style
        assert "result text" in output

    @pytest.mark.asyncio
    async def test_tool_error_panel_has_border(self):
        """Tool error rendered in a panel with borders."""
        ec, buf = _make_console()
        events = [
            ToolErrorEvent(
                data=ToolErrorInfo(tool_name="broken", error="something failed"),
                source="a", exec_id="c1",
            )
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "╭" in output
        assert "╰" in output
        assert "✗" in output
        assert "something failed" in output

    @pytest.mark.asyncio
    async def test_content_sanitization(self):
        """Rich markup in tool output doesn't break rendering."""
        ec, buf = _make_console()
        msg = FunctionToolOutputItem.from_tool_result(
            call_id="tc_1", output="[bold red]injection attempt[/]"
        )
        events = [ToolResultEvent(data=msg, source="a", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        # The raw markup should appear escaped, not interpreted
        assert "injection attempt" in output


# ── Error events ──


class TestErrorEvents:
    @pytest.mark.asyncio
    async def test_llm_streaming_error(self):
        ec, buf = _make_console()
        events = [
            LLMStreamingErrorEvent(
                data=LLMStreamingErrorData(
                    error=RuntimeError("connection lost"),
                    model_name="test",
                ),
                source="agent", exec_id="c1",
            )
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "Error:" in output
        assert "connection lost" in output


# ── Background tasks ──


class TestBackgroundTasks:
    @pytest.mark.asyncio
    async def test_bg_launched_silent(self):
        """Launch event is silent — placeholder result panel covers it."""
        ec, buf = _make_console()
        events = [
            BackgroundTaskLaunchedEvent(
                data=BackgroundTaskInfo(
                    task_id="t1", tool_name="web_search", tool_call_id="tc_1"
                ),
                source="agent", exec_id="c1",
            )
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert output.strip() == ""

    @pytest.mark.asyncio
    async def test_bg_completed(self):
        ec, buf = _make_console()
        events = [
            BackgroundTaskCompletedEvent(
                data=BackgroundTaskInfo(
                    task_id="t1", tool_name="web_search", tool_call_id="tc_1"
                ),
                source="agent", exec_id="c1",
            )
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "web_search" in output
        assert "completed" in output
        assert "t1" in output


# ── Packet events ──


class TestPacketEvents:
    @pytest.mark.asyncio
    async def test_packets_hidden_by_default(self):
        from grasp_agents.packet import Packet

        ec, buf = _make_console()
        packet = Packet(payloads=["result"], sender="agent")
        events = [
            ProcPacketOutEvent(data=packet, source="agent", exec_id="c1")
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "result" not in output

    @pytest.mark.asyncio
    async def test_packets_shown_when_enabled(self):
        from grasp_agents.packet import Packet

        ec, buf = _make_console_with(show_packets=True)
        packet = Packet(payloads=["hello"], sender="my_agent")
        events = [
            ProcPacketOutEvent(data=packet, source="my_agent", exec_id="c1")
        ]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "my_agent" in output
        assert "hello" in output


# ── Event passthrough ──


class TestPassthrough:
    @pytest.mark.asyncio
    async def test_all_events_yielded(self):
        ec, _ = _make_console()
        msg = InputMessageItem.from_text("Hi", role="user")
        events: list[Event[Any]] = [
            TurnStartEvent(data=TurnInfo(turn=0), source="a"),
            UserMessageEvent(data=msg, source="a", exec_id="c1"),
            LLMStreamEvent(
                data=OutputMessageTextPartTextDelta(
                    content_index=0, delta="Hi", output_index=0,
                    sequence_number=1, item_id="i1", logprobs=[],
                ),
                source="a", exec_id="c1",
            ),
        ]
        collected = await _collect(ec, events)
        assert len(collected) == 3
        assert isinstance(collected[0], TurnStartEvent)
        assert isinstance(collected[1], UserMessageEvent)
        assert isinstance(collected[2], LLMStreamEvent)


# ── stream_events convenience function ──


class TestStreamEventsFunction:
    @pytest.mark.asyncio
    async def test_stream_events_wrapper(self):
        """stream_events() is a convenience wrapper that works like EventConsole.stream()."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, highlight=False, width=80)

        async def gen():
            yield TurnStartEvent(data=TurnInfo(turn=0), source="agent")

        collected = []
        async for event in stream_events(gen(), console=console):
            collected.append(event)

        assert len(collected) == 1
        output = buf.getvalue()
        assert "agent" in output
        assert "turn 1" in output


# ── Full turn sequence ──


class TestFullSequence:
    @pytest.mark.asyncio
    async def test_streaming_turn_with_tool_call(self):
        """Simulate a full streaming turn: text → tool call → tool result."""
        ec, buf = _make_console()

        text_item = OutputMessageItem(
            content_parts=[OutputMessageText(text="Let me search.")],
            status="completed",
        )
        tool_item = FunctionToolCallItem(
            call_id="tc_1", name="search", arguments='{"q": "stats"}'
        )
        tool_result = FunctionToolOutputItem.from_tool_result(
            call_id="tc_1", output="3 results found"
        )
        resp = Response(
            model="test-model",
            output_items=[text_item, tool_item],
            usage_with_cost=ResponseUsage(
                input_tokens=200, output_tokens=80, total_tokens=280
            ),
        )

        events: list[Event[Any]] = [
            TurnStartEvent(data=TurnInfo(turn=0), source="teacher"),
            # Text streaming
            LLMStreamEvent(
                data=OutputItemAdded(item=text_item, output_index=0, sequence_number=1),
                source="teacher", exec_id="c1",
            ),
            LLMStreamEvent(
                data=OutputMessageTextPartTextDelta(
                    content_index=0, delta="Let me search.",
                    output_index=0, sequence_number=2, item_id="i1", logprobs=[],
                ),
                source="teacher", exec_id="c1",
            ),
            # Promoted message (should be skipped — already streamed)
            OutputMessageItemEvent(data=text_item, source="teacher", exec_id="c1"),
            LLMStreamEvent(
                data=OutputItemDone(item=text_item, output_index=0, sequence_number=3),
                source="teacher", exec_id="c1",
            ),
            # Tool call
            LLMStreamEvent(
                data=OutputItemAdded(item=tool_item, output_index=1, sequence_number=4),
                source="teacher", exec_id="c1",
            ),
            LLMStreamEvent(
                data=FunctionCallArgumentsDelta(
                    call_id="tc_1", delta='{"q": "stats"}',
                    output_index=1, sequence_number=5, item_id="i2",
                ),
                source="teacher", exec_id="c1",
            ),
            ToolCallItemEvent(data=tool_item, source="teacher", exec_id="c1"),
            LLMStreamEvent(
                data=OutputItemDone(item=tool_item, output_index=1, sequence_number=6),
                source="teacher", exec_id="c1",
            ),
            # Generation complete
            LLMStreamEvent(
                data=ResponseCompleted(response=resp, sequence_number=7),
                source="teacher", exec_id="c1",
            ),
            GenerationEndEvent(data=resp, source="teacher", exec_id="c1"),
            # Tool result
            ToolResultEvent(data=tool_result, source="teacher", exec_id="c1"),
            # Turn end
            TurnEndEvent(
                data=TurnEndInfo(turn=0, had_tool_calls=True),
                source="teacher",
            ),
        ]

        collected = await _collect(ec, events)
        output = buf.getvalue()

        # All events yielded
        assert len(collected) == len(events)

        # Turn header
        assert "teacher" in output
        assert "turn 1" in output

        # Text streamed (not duplicated)
        assert output.count("Let me search.") == 1

        # Tool call displayed in panel
        assert "search" in output
        assert "╭" in output

        # Tool result displayed (indented)
        assert "3 results found" in output

        # Usage displayed
        assert "200" in output
        assert "80" in output

    @pytest.mark.asyncio
    async def test_tool_content_indented(self):
        """Tool args, usage, and result are indented under tool call."""
        ec, buf = _make_console()
        tool_item = FunctionToolCallItem(
            call_id="tc_1", name="fetch", arguments='{"url": "example.com"}'
        )
        tool_result = FunctionToolOutputItem.from_tool_result(
            call_id="tc_1", output="page content"
        )
        resp = Response(
            model="test-model",
            output_items=[tool_item],
            usage_with_cost=ResponseUsage(
                input_tokens=50, output_tokens=20, total_tokens=70, cost=0.001
            ),
        )
        events: list[Event[Any]] = [
            ToolCallItemEvent(data=tool_item, source="a", exec_id="c1"),
            GenerationEndEvent(data=resp, source="a", exec_id="c1"),
            ToolResultEvent(data=tool_result, source="a", exec_id="c1"),
        ]
        await _collect(ec, events)
        output = buf.getvalue()

        # Tool args in panel
        assert "url" in output
        assert "example.com" in output
        assert "╭" in output  # Panel borders
        # Usage shown
        assert "test-model" in output
        # Tool result present
        assert "page content" in output

    @pytest.mark.asyncio
    async def test_tool_result_filters_blank_lines(self):
        """Blank lines in tool result content are filtered out."""
        ec, buf = _make_console()
        msg = FunctionToolOutputItem.from_tool_result(
            call_id="tc_1", output="line 1\n\n\nline 2"
        )
        events = [ToolResultEvent(data=msg, source="a", exec_id="c1")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "line 1" in output
        assert "line 2" in output
        # No triple blank lines in output
        assert "\n\n\n" not in output

    @pytest.mark.asyncio
    async def test_turn_header_format(self):
        """Turn header shows agent name and turn number."""
        ec, buf = _make_console()
        events = [TurnStartEvent(data=TurnInfo(turn=0), source="agent")]
        await _collect(ec, events)
        output = buf.getvalue()
        assert "agent" in output
        assert "turn 1" in output
