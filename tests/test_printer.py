"""Tests for Printer with Responses API items and LLMStreamEvent."""

import asyncio
from unittest.mock import patch

import pytest

from grasp_agents.printer import (
    Printer,
    _input_message_text,
    get_color,
    print_event_stream,
    truncate_content_str,
)
from grasp_agents.types.content import (
    InputImage,
    InputText,
    OutputMessageText,
)
from grasp_agents.types.events import (
    LLMStreamEvent,
    SystemMessageEvent,
    ToolMessageEvent,
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
    ResponseCreated,
)
from grasp_agents.types.response import Response

# ---------- Utility functions ----------


class TestGetColor:
    def test_role_based_colors(self):
        """Role-based coloring returns expected colors."""
        assert get_color(role="system", color_by="role") == "magenta"
        assert get_color(role="user", color_by="role") == "green"
        assert get_color(role="assistant", color_by="role") == "light_blue"
        assert get_color(role="tool", color_by="role") == "blue"

    def test_unknown_role_defaults(self):
        """Unknown role defaults to light_blue."""
        assert get_color(role="unknown", color_by="role") == "light_blue"

    def test_agent_based_coloring(self):
        """Agent-based coloring is deterministic per agent name."""
        c1 = get_color(agent_name="agent_a", color_by="agent")
        c2 = get_color(agent_name="agent_a", color_by="agent")
        c3 = get_color(agent_name="agent_b", color_by="agent")

        assert c1 == c2  # same agent → same color
        # different agents may or may not have different colors (hash collision)


class TestInputMessageText:
    def test_text_only(self):
        """Extracts text from InputMessageItem."""
        msg = InputMessageItem.from_text("Hello world", role="user")
        assert _input_message_text(msg) == "Hello world"

    def test_with_image_url(self):
        """Image URLs are included in text output."""
        msg = InputMessageItem(
            content_parts=[
                InputText(text="Look at this"),
                InputImage.from_url("https://example.com/pic.jpg"),
            ],
            role="user",
        )
        text = _input_message_text(msg)
        assert "Look at this" in text
        assert "https://example.com/pic.jpg" in text

    def test_with_base64_image(self):
        """Base64 images show placeholder."""
        msg = InputMessageItem(
            content_parts=[
                InputText(text="Describe"),
                InputImage.from_base64("abc123"),
            ],
            role="user",
        )
        text = _input_message_text(msg)
        assert "<ENCODED_IMAGE>" in text


class TestTruncate:
    def test_no_truncation(self):
        assert truncate_content_str("short", trunc_len=100) == "short"

    def test_truncation(self):
        result = truncate_content_str("a" * 200, trunc_len=50)
        assert len(result) == 55  # 50 + len("[...]")
        assert result.endswith("[...]")


# ---------- Printer.print_message ----------


class TestPrinterMessage:
    def test_print_system_message(self, capsys):
        """InputMessageItem with role=system prints <system> tags."""
        printer = Printer(output_to="stdout")
        msg = InputMessageItem.from_text("You are helpful.", role="system")

        printer.print_message(msg, agent_name="test", call_id="c1")

        output = capsys.readouterr().out
        assert "<system>" in output
        assert "You are helpful." in output
        assert "</system>" in output

    def test_print_user_message(self, capsys):
        """InputMessageItem with role=user prints <input> tags."""
        printer = Printer(output_to="stdout")
        msg = InputMessageItem.from_text("Hello", role="user")

        printer.print_message(msg, agent_name="test", call_id="c1")

        output = capsys.readouterr().out
        assert "<input>" in output
        assert "Hello" in output

    def test_print_tool_output(self, capsys):
        """FunctionToolOutputItem prints <tool result> tags."""
        printer = Printer(output_to="stdout")
        msg = FunctionToolOutputItem.from_tool_result(
            call_id="call_1", output={"status": "ok"}
        )

        printer.print_message(msg, agent_name="test", call_id="c1")

        output = capsys.readouterr().out
        assert "<tool result>" in output
        assert "call_1" in output
        assert "status" in output

    def test_print_unknown_message_type(self, capsys):
        """Unknown message types are printed as strings."""
        printer = Printer(output_to="stdout")
        printer.print_message("raw string", agent_name="test", call_id="c1")

        output = capsys.readouterr().out
        assert "raw string" in output


# ---------- print_event_stream with LLMStreamEvent ----------


class TestPrintEventStream:
    def _make_response(self) -> Response:
        return Response(
            model="test-model",
            output_items=[
                OutputMessageItem(
                    content_parts=[OutputMessageText(text="Hello")],
                    status="completed",
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_response_created_event(self, capsys):
        """ResponseCreated event prints agent name and call_id header."""
        response = self._make_response()

        async def gen():
            yield LLMStreamEvent(
                data=ResponseCreated(
                    response=response,
                    sequence_number=1,  # type: ignore[arg-type]
                ),
                src_name="my_agent",
                call_id="call_1",
            )

        collected = []
        async for event in print_event_stream(gen()):
            collected.append(event)

        output = capsys.readouterr().out
        assert "my_agent" in output
        assert "call_1" in output
        assert len(collected) == 1

    @pytest.mark.asyncio
    async def test_text_delta_event(self, capsys):
        """OutputMessageTextPartTextDelta events stream text content."""

        async def gen():
            yield LLMStreamEvent(
                data=OutputMessageTextPartTextDelta(
                    content_index=0,
                    delta="Hello ",
                    output_index=0,
                    sequence_number=1,
                    item_id="item_1",
                    logprobs=[],
                ),
                src_name="agent",
                call_id="c1",
            )
            yield LLMStreamEvent(
                data=OutputMessageTextPartTextDelta(
                    content_index=0,
                    delta="world!",
                    output_index=0,
                    sequence_number=2,
                    item_id="item_1",
                    logprobs=[],
                ),
                src_name="agent",
                call_id="c1",
            )

        collected = []
        async for event in print_event_stream(gen()):
            collected.append(event)

        output = capsys.readouterr().out
        assert "Hello " in output
        assert "world!" in output
        assert len(collected) == 2

    @pytest.mark.asyncio
    async def test_output_item_added_response(self, capsys):
        """OutputItemAdded with OutputMessageItem prints <response> tag."""
        item = OutputMessageItem(
            content_parts=[OutputMessageText(text="test")],
            status="in_progress",
        )

        async def gen():
            yield LLMStreamEvent(
                data=OutputItemAdded(item=item, output_index=0, sequence_number=1),
                src_name="agent",
                call_id="c1",
            )

        async for _ in print_event_stream(gen()):
            pass

        output = capsys.readouterr().out
        assert "<response>" in output

    @pytest.mark.asyncio
    async def test_output_item_added_tool_call(self, capsys):
        """OutputItemAdded with FunctionToolCallItem prints <tool call> tag."""
        item = FunctionToolCallItem(call_id="tc_1", name="search", arguments="{}")

        async def gen():
            yield LLMStreamEvent(
                data=OutputItemAdded(item=item, output_index=0, sequence_number=1),
                src_name="agent",
                call_id="c1",
            )

        async for _ in print_event_stream(gen()):
            pass

        output = capsys.readouterr().out
        assert "<tool call>" in output
        assert "search" in output

    @pytest.mark.asyncio
    async def test_output_item_added_reasoning(self, capsys):
        """OutputItemAdded with ReasoningItem prints <thinking> tag."""
        item = ReasoningItem()

        async def gen():
            yield LLMStreamEvent(
                data=OutputItemAdded(item=item, output_index=0, sequence_number=1),
                src_name="agent",
                call_id="c1",
            )

        async for _ in print_event_stream(gen()):
            pass

        output = capsys.readouterr().out
        assert "<thinking>" in output

    @pytest.mark.asyncio
    async def test_output_item_done_closing_tags(self, capsys):
        """OutputItemDone events print closing tags."""
        msg_item = OutputMessageItem(
            content_parts=[OutputMessageText(text="done")],
            status="completed",
        )
        tc_item = FunctionToolCallItem(call_id="tc_1", name="search", arguments="{}")

        async def gen():
            yield LLMStreamEvent(
                data=OutputItemDone(item=msg_item, output_index=0, sequence_number=1),
                src_name="agent",
                call_id="c1",
            )
            yield LLMStreamEvent(
                data=OutputItemDone(item=tc_item, output_index=1, sequence_number=2),
                src_name="agent",
                call_id="c1",
            )

        async for _ in print_event_stream(gen()):
            pass

        output = capsys.readouterr().out
        assert "</response>" in output
        assert "</tool call>" in output

    @pytest.mark.asyncio
    async def test_function_call_arguments_delta(self, capsys):
        """FunctionCallArgumentsDelta streams tool call arguments."""

        async def gen():
            yield LLMStreamEvent(
                data=FunctionCallArgumentsDelta(
                    call_id="tc_1",
                    delta='{"query":',
                    output_index=0,
                    sequence_number=1,
                    item_id="item_1",
                ),
                src_name="agent",
                call_id="c1",
            )

        async for _ in print_event_stream(gen()):
            pass

        output = capsys.readouterr().out
        assert '{"query":' in output

    @pytest.mark.asyncio
    async def test_system_message_event(self, capsys):
        """SystemMessageEvent prints <system> tags."""
        msg = InputMessageItem.from_text("Be helpful.", role="system")

        async def gen():
            yield SystemMessageEvent(data=msg, src_name="agent", call_id="c1")

        async for _ in print_event_stream(gen()):
            pass

        output = capsys.readouterr().out
        assert "<system>" in output
        assert "Be helpful." in output

    @pytest.mark.asyncio
    async def test_user_message_event(self, capsys):
        """UserMessageEvent prints <input> tags."""
        msg = InputMessageItem.from_text("Hello", role="user")

        async def gen():
            yield UserMessageEvent(data=msg, src_name="agent", call_id="c1")

        async for _ in print_event_stream(gen()):
            pass

        output = capsys.readouterr().out
        assert "<input>" in output
        assert "Hello" in output

    @pytest.mark.asyncio
    async def test_tool_message_event(self, capsys):
        """ToolMessageEvent prints <tool result> tags."""
        msg = FunctionToolOutputItem.from_tool_result(call_id="tc_1", output="result")

        async def gen():
            yield ToolMessageEvent(data=msg, src_name="agent", call_id="c1")

        async for _ in print_event_stream(gen()):
            pass

        output = capsys.readouterr().out
        assert "<tool result>" in output

    @pytest.mark.asyncio
    async def test_events_are_yielded_through(self):
        """print_event_stream yields all events through unchanged."""
        msg = InputMessageItem.from_text("Hello", role="user")

        async def gen():
            yield UserMessageEvent(data=msg, src_name="a", call_id="c1")
            yield LLMStreamEvent(
                data=OutputMessageTextPartTextDelta(
                    content_index=0,
                    delta="Hi",
                    output_index=0,
                    sequence_number=1,
                    item_id="item_1",
                    logprobs=[],
                ),
                src_name="a",
                call_id="c1",
            )

        collected = []
        async for event in print_event_stream(gen()):
            collected.append(event)

        assert len(collected) == 2
        assert isinstance(collected[0], UserMessageEvent)
        assert isinstance(collected[1], LLMStreamEvent)
