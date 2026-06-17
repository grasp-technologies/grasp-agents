"""Tests for Printer with Responses API items and LLMStreamEvent."""

import pytest
from pydantic import BaseModel

from grasp_agents.printer import (
    Printer,
    _input_message_text,
    get_style,
    print_events,
    render_payload,
    sanitize_terminal_text,
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
    ToolOutputItemEvent,
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
    ResponseCreated,
)
from grasp_agents.types.response import Response
from grasp_agents.ui._event_render import truncate, truncate_lines

# ---------- Utility functions ----------


class TestGetStyle:
    def test_role_based_styles(self):
        """Role-based coloring returns expected styles."""
        assert get_style(role="system", color_by="role") == "magenta"
        assert get_style(role="user", color_by="role") == "green"
        assert get_style(role="assistant", color_by="role") == "bright_blue"
        assert get_style(role="tool", color_by="role") == "blue"

    def test_unknown_role_defaults(self):
        """Unknown role defaults to bright_blue."""
        assert get_style(role="unknown", color_by="role") == "bright_blue"

    def test_agent_based_coloring(self):
        """Agent-based coloring is deterministic per agent name."""
        c1 = get_style(agent_name="agent_a", color_by="agent")
        c2 = get_style(agent_name="agent_a", color_by="agent")

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


class TestRenderPayload:
    """Packet payloads render without redundant quoting / escaping."""

    def test_str_shown_verbatim(self):
        # A string payload is shown as-is — no wrapping quotes, and a non-ASCII
        # char (here the multiplication sign) is not escaped to a \\uXXXX
        # sequence (the bug: json.dumps gave '"**347 \\u00d7 ..."').
        s = "**347 × 829 = 287,663**"  # noqa: RUF001
        assert render_payload(s) == s

    def test_model_as_indented_json(self):
        class M(BaseModel):
            a: int
            b: str

        out = render_payload(M(a=1, b="hi"))
        assert '"a": 1' in out
        assert '"b": "hi"' in out

    def test_dict_keeps_non_ascii(self):
        out = render_payload({"label": "café"})
        assert "café" in out  # kept as-is, not escaped to a \\uXXXX sequence
        assert "\\u" not in out


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

        printer.print_message(msg, agent_name="test", exec_id="c1")

        output = capsys.readouterr().out
        assert "<system>" in output
        assert "You are helpful." in output
        assert "</system>" in output

    def test_print_user_message(self, capsys):
        """InputMessageItem with role=user prints <input> tags."""
        printer = Printer(output_to="stdout")
        msg = InputMessageItem.from_text("Hello", role="user")

        printer.print_message(msg, agent_name="test", exec_id="c1")

        output = capsys.readouterr().out
        assert "<input>" in output
        assert "Hello" in output

    def test_print_tool_output(self, capsys):
        """FunctionToolOutputItem prints <tool result> tags."""
        printer = Printer(output_to="stdout")
        msg = FunctionToolOutputItem.from_tool_result(
            call_id="call_1", output={"status": "ok"}
        )

        printer.print_message(msg, agent_name="test", exec_id="c1")

        output = capsys.readouterr().out
        assert "<tool result>" in output
        assert "call_1" in output
        assert "status" in output

    def test_print_unknown_message_type(self, capsys):
        """Unknown message types are printed as strings."""
        printer = Printer(output_to="stdout")
        printer.print_message("raw string", agent_name="test", exec_id="c1")

        output = capsys.readouterr().out
        assert "raw string" in output

    def test_print_assistant_response(self, capsys):
        """OutputMessageItem (generated answer) prints <response> tags."""
        printer = Printer(output_to="stdout")
        msg = OutputMessageItem(
            content_parts=[OutputMessageText(text="The answer is 42.")],
            status="completed",
        )

        printer.print_message(msg, agent_name="test", exec_id="c1")

        output = capsys.readouterr().out
        assert "<response>" in output
        assert "The answer is 42." in output
        assert "</response>" in output

    def test_print_reasoning(self, capsys):
        """ReasoningItem (thinking) prints <thinking> tags with the summary."""
        from grasp_agents.types.content import ReasoningSummary

        printer = Printer(output_to="stdout")
        item = ReasoningItem(
            summary_parts=[ReasoningSummary(text="Let me work through this.")],
            status="completed",
        )

        printer.print_message(item, agent_name="test", exec_id="c1")

        output = capsys.readouterr().out
        assert "<thinking>" in output
        assert "Let me work through this." in output
        assert "</thinking>" in output

    def test_print_tool_call(self, capsys):
        """FunctionToolCallItem (a tool call) prints <tool call> tags + args."""
        printer = Printer(output_to="stdout")
        item = FunctionToolCallItem(
            call_id="call_1", name="search", arguments='{"query": "weather"}'
        )

        printer.print_message(item, agent_name="test", exec_id="c1")

        output = capsys.readouterr().out
        assert "<tool call>" in output
        assert "search" in output
        assert "query" in output
        assert "weather" in output


# ---------- AgentLoop routes generated output to ctx.printer (issue 1) ----------


class TestPrinterShowsGeneratedOutput:
    @pytest.mark.asyncio
    async def test_non_streaming_run_prints_generated_output(self, capsys):
        """
        A non-streaming run with ctx.printer set shows the model's generated
        output (response / thinking / tool calls), not just inputs + tool
        results — the raw printer is a full debug view of the conversation.
        """
        from grasp_agents import LLMAgent, RunContext
        from tests._helpers import MockLLM, _text_response

        ctx = RunContext[None](printer=Printer(output_to="stdout"))
        agent = LLMAgent[None, str, None](
            name="dbg",
            ctx=ctx,
            llm=MockLLM(responses_queue=[_text_response("Generated reply.")]),
        )
        await agent.run("hello")

        output = capsys.readouterr().out
        assert "<input>" in output  # input still shown
        assert "<response>" in output  # ...and now the generated answer too
        assert "Generated reply." in output


# ---------- print_events with LLMStreamEvent ----------


class TestPrintEvents:
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
        """ResponseCreated event prints agent name and exec_id header."""
        response = self._make_response()

        async def gen():
            yield LLMStreamEvent(
                data=ResponseCreated(
                    response=response,
                    sequence_number=1,  # type: ignore[arg-type]
                ),
                source="my_agent",
                exec_id="call_1",
            )

        collected = []
        async for event in print_events(gen()):
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
                source="agent",
                exec_id="c1",
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
                source="agent",
                exec_id="c1",
            )

        collected = []
        async for event in print_events(gen()):
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
                source="agent",
                exec_id="c1",
            )

        async for _ in print_events(gen()):
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
                source="agent",
                exec_id="c1",
            )

        async for _ in print_events(gen()):
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
                source="agent",
                exec_id="c1",
            )

        async for _ in print_events(gen()):
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
                source="agent",
                exec_id="c1",
            )
            yield LLMStreamEvent(
                data=OutputItemDone(item=tc_item, output_index=1, sequence_number=2),
                source="agent",
                exec_id="c1",
            )

        async for _ in print_events(gen()):
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
                    delta='{"query":',
                    output_index=0,
                    sequence_number=1,
                    item_id="item_1",
                ),
                source="agent",
                exec_id="c1",
            )

        async for _ in print_events(gen()):
            pass

        output = capsys.readouterr().out
        assert '{"query":' in output

    @pytest.mark.asyncio
    async def test_system_message_event(self, capsys):
        """SystemMessageEvent prints <system> tags."""
        msg = InputMessageItem.from_text("Be helpful.", role="system")

        async def gen():
            yield SystemMessageEvent(data=msg, source="agent", exec_id="c1")

        async for _ in print_events(gen()):
            pass

        output = capsys.readouterr().out
        assert "<system>" in output
        assert "Be helpful." in output

    @pytest.mark.asyncio
    async def test_user_message_event(self, capsys):
        """UserMessageEvent prints <input> tags."""
        msg = InputMessageItem.from_text("Hello", role="user")

        async def gen():
            yield UserMessageEvent(data=msg, source="agent", exec_id="c1")

        async for _ in print_events(gen()):
            pass

        output = capsys.readouterr().out
        assert "<input>" in output
        assert "Hello" in output

    @pytest.mark.asyncio
    async def test_tool_message_event(self, capsys):
        """ToolOutputItemEvent prints <tool result> tags."""
        msg = FunctionToolOutputItem.from_tool_result(call_id="tc_1", output="result")

        async def gen():
            yield ToolOutputItemEvent(data=msg, source="agent", exec_id="c1")

        async for _ in print_events(gen()):
            pass

        output = capsys.readouterr().out
        assert "<tool result>" in output

    @pytest.mark.asyncio
    async def test_events_are_yielded_through(self):
        """print_events yields all events through unchanged."""
        msg = InputMessageItem.from_text("Hello", role="user")

        async def gen():
            yield UserMessageEvent(data=msg, source="a", exec_id="c1")
            yield LLMStreamEvent(
                data=OutputMessageTextPartTextDelta(
                    content_index=0,
                    delta="Hi",
                    output_index=0,
                    sequence_number=1,
                    item_id="item_1",
                    logprobs=[],
                ),
                source="a",
                exec_id="c1",
            )

        collected = []
        async for event in print_events(gen()):
            collected.append(event)

        assert len(collected) == 2
        assert isinstance(collected[0], UserMessageEvent)
        assert isinstance(collected[1], LLMStreamEvent)

    @pytest.mark.asyncio
    async def test_tool_result_output_truncated(self, capsys):
        """A large tool result is truncated for display, not dumped whole."""
        item = FunctionToolOutputItem(call_id="c1", output="x" * 50_000)

        async def gen():
            yield ToolOutputItemEvent(source="tool", exec_id="e", data=item)

        async for _ in print_events(gen(), trunc_len=2000):
            pass

        output = capsys.readouterr().out
        assert "[...]" in output
        assert len(output) < 5_000


# ---------- terminal escape-sequence sanitization ----------


class TestTerminalSanitization:
    def test_csi_clear_screen_neutralized(self) -> None:
        out = sanitize_terminal_text("before\x1b[2Jafter")
        assert "\x1b" not in out
        assert "before" in out
        assert "after" in out

    def test_osc_title_spoof_neutralized(self) -> None:
        out = sanitize_terminal_text("\x1b]0;you-have-been-pwned\x07rest")
        assert "\x1b" not in out
        assert "\x07" not in out
        assert "rest" in out

    def test_carriage_return_overwrite_neutralized(self) -> None:
        # "\r" rewinds the line — classic approval-prompt spoof.
        out = sanitize_terminal_text("rm -rf /\rls -la    ")
        assert "\r" not in out
        assert "rm -rf /" in out

    def test_newlines_and_tabs_kept(self) -> None:
        assert sanitize_terminal_text("a\n\tb\r\nc") == "a\n\tb\nc"

    def test_render_truncate_helpers_sanitize(self) -> None:
        assert "\x1b" not in truncate("x\x1b[2Jy", 100)
        assert "\x1b" not in truncate_lines("x\x1b[2Jy\nz", 10)
