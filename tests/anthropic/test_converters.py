"""
Tests for Anthropic provider converters.

Tests the full conversion pipeline: Anthropic SDK types → grasp-agents types.
Uses real Anthropic SDK model objects (not mocks) to ensure type compatibility.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from anthropic.types import (
    ContentBlock,
    DocumentBlock,
    InputJSONDelta,
    Message,
    MessageDeltaUsage,
    PlainTextSource,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RedactedThinkingBlock,
    ServerToolUseBlock,
    SignatureDelta,
    StopReason,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    Usage,
    WebFetchBlock,
    WebFetchToolResultBlock,
    WebFetchToolResultErrorBlock,
    WebSearchResultBlock,
    WebSearchToolResultBlock,
)
from anthropic.types.raw_message_delta_event import Delta as MessageDelta
from openai.types.responses.response_function_web_search import (
    ActionOpenPage,
    ActionSearch,
    ActionSearchSource,
)
from pydantic import BaseModel

from grasp_agents.llm_providers.anthropic.llm_event_converters import (
    AnthropicStreamConverter,
)
from grasp_agents.llm_providers.anthropic.provider_output_to_response import (
    _anthropic_message_to_items_and_web_search_info as items_from_anthropic_message,
)
from grasp_agents.llm_providers.anthropic.provider_output_to_response import (
    provider_output_to_response as from_anthropic_message,
)
from grasp_agents.llm_providers.anthropic.response_to_provider_inputs import (
    items_to_provider_inputs as items_to_anthropic_messages,
)
from grasp_agents.llm_providers.anthropic.tool_converters import (
    to_api_tool,
    to_api_tool_choice,
)
from grasp_agents.types.content import OutputMessageText as OutputMessageText
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)
from grasp_agents.types.llm_events import (
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    OutputItemAdded,
    OutputItemDone,
    OutputMessageTextPartTextDone,
    ReasoningSummaryPartTextDelta,
    ResponseCompleted,
)
from grasp_agents.types.llm_events import (
    OutputMessageTextPartTextDelta as LlmTextDelta,
)
from grasp_agents.types.tool import BaseTool, NamedToolChoice

# ==== Helpers ====


def _make_message(
    content: list[ContentBlock],
    *,
    model: str = "claude-sonnet-4-20250514",
    stop_reason: StopReason = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> Message:
    return Message(
        id="msg_test123",
        type="message",
        role="assistant",
        model=model,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


async def _collect_events(
    events: list[Any],
) -> list[Any]:
    """Run AnthropicStreamConverter on a list of raw events."""
    converter = AnthropicStreamConverter()

    async def event_stream():  # type: ignore[reportReturnType]  # noqa: RUF029
        for event in events:
            yield event

    return [e async for e in converter.convert(event_stream())]


# ==== assistant_message_to_items ====


class TestAssistantMessageToItems:
    def test_text_only(self):
        msg = _make_message([TextBlock(type="text", text="Hello world")])
        items, _ = items_from_anthropic_message(msg)

        assert len(items) == 1
        assert isinstance(items[0], OutputMessageItem)
        assert items[0].text == "Hello world"
        assert items[0].status == "completed"

    def test_consecutive_text_blocks_merge(self):
        """Multiple consecutive text blocks should merge into one OutputMessageItem."""
        msg = _make_message(
            [
                TextBlock(type="text", text="First"),
                TextBlock(type="text", text="Second"),
            ]
        )
        items, _ = items_from_anthropic_message(msg)

        assert len(items) == 1
        assert isinstance(items[0], OutputMessageItem)
        assert items[0].text == "FirstSecond"

    def test_thinking_then_text(self):
        """Thinking block → ReasoningItem, text → OutputMessageItem."""
        msg = _make_message(
            [
                ThinkingBlock(
                    type="thinking",
                    thinking="Let me think...",
                    signature="sig123",
                ),
                TextBlock(type="text", text="The answer is 42"),
            ]
        )
        items, _ = items_from_anthropic_message(msg)

        assert len(items) == 2
        assert isinstance(items[0], ReasoningItem)
        assert items[0].summary_text == "Let me think..."
        assert items[0].encrypted_content == "sig123"
        assert not items[0].redacted

        assert isinstance(items[1], OutputMessageItem)
        assert items[1].text == "The answer is 42"

    def test_redacted_thinking(self):
        msg = _make_message(
            [
                RedactedThinkingBlock(
                    type="redacted_thinking", data="encrypted_data_123"
                ),
                TextBlock(type="text", text="Answer"),
            ]
        )
        items, _ = items_from_anthropic_message(msg)

        assert len(items) == 2
        assert isinstance(items[0], ReasoningItem)
        assert items[0].redacted is True
        assert items[0].encrypted_content == "encrypted_data_123"
        assert items[0].summary_parts == []

    def test_tool_use(self):
        msg = _make_message(
            [
                ToolUseBlock(
                    type="tool_use",
                    id="toolu_123",
                    name="get_weather",
                    input={"city": "Paris"},
                ),
            ],
            stop_reason="tool_use",
        )
        items, _ = items_from_anthropic_message(msg)

        assert len(items) == 1
        assert isinstance(items[0], FunctionToolCallItem)
        assert items[0].call_id == "toolu_123"
        assert items[0].name == "get_weather"
        assert json.loads(items[0].arguments) == {"city": "Paris"}

    def test_text_then_tool_use(self):
        """Text + tool call: separate OutputMessageItem and FunctionToolCallItem."""
        msg = _make_message(
            [
                TextBlock(type="text", text="I'll check the weather"),
                ToolUseBlock(
                    type="tool_use",
                    id="toolu_456",
                    name="get_weather",
                    input={"city": "London"},
                ),
            ],
            stop_reason="tool_use",
        )
        items, _ = items_from_anthropic_message(msg)

        assert len(items) == 2
        assert isinstance(items[0], OutputMessageItem)
        assert items[0].text == "I'll check the weather"
        assert isinstance(items[1], FunctionToolCallItem)
        assert items[1].name == "get_weather"

    def test_thinking_text_tool_full_pipeline(self):
        """Full response with thinking + text + tool use."""
        msg = _make_message(
            [
                ThinkingBlock(
                    type="thinking",
                    thinking="Planning...",
                    signature="sig_abc",
                ),
                TextBlock(type="text", text="I'll look that up"),
                ToolUseBlock(
                    type="tool_use",
                    id="toolu_789",
                    name="search",
                    input={"query": "test"},
                ),
            ],
            stop_reason="tool_use",
        )
        items, _ = items_from_anthropic_message(msg)

        assert len(items) == 3
        assert isinstance(items[0], ReasoningItem)
        assert isinstance(items[1], OutputMessageItem)
        assert isinstance(items[2], FunctionToolCallItem)


# ==== response_converters ====


class TestResponseConverters:
    def test_basic_response(self):
        msg = _make_message([TextBlock(type="text", text="Hello")])
        response = from_anthropic_message(msg)

        assert response.id == "msg_test123"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.status == "completed"
        assert response.output_text == "Hello"
        assert response.usage_with_cost is not None
        assert response.usage_with_cost.input_tokens == 100
        assert response.usage_with_cost.output_tokens == 50
        assert response.usage_with_cost.total_tokens == 150

    def test_max_tokens_incomplete(self):
        msg = _make_message(
            [TextBlock(type="text", text="Partial")],
            stop_reason="max_tokens",
        )
        response = from_anthropic_message(msg)

        assert response.status == "incomplete"
        assert response.incomplete_details is not None
        assert response.incomplete_details.reason == "max_output_tokens"


# ==== response_to_message ====


class TestResponseToMessage:
    def test_system_message_extracted(self):
        items = [
            InputMessageItem.from_text("You are helpful", role="system"),
            InputMessageItem.from_text("Hi"),
        ]
        system, messages = items_to_anthropic_messages(items)

        assert system == "You are helpful"
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_multiple_system_messages_concatenated(self):
        items = [
            InputMessageItem.from_text("Rule 1", role="system"),
            InputMessageItem.from_text("Rule 2", role="developer"),
            InputMessageItem.from_text("Hello"),
        ]
        system, messages = items_to_anthropic_messages(items)

        assert system == "Rule 1\n\nRule 2"
        assert len(messages) == 1

    def test_user_message_simple_text(self):
        items = [InputMessageItem.from_text("What is 2+2?")]
        system, messages = items_to_anthropic_messages(items)

        assert system is None
        assert len(messages) == 1
        # Simple text is passed as string, not list
        assert messages[0]["content"] == "What is 2+2?"

    def test_assistant_output_group(self):
        """Consecutive output items group into one assistant message."""
        items = [
            InputMessageItem.from_text("Hi"),
            ReasoningItem(
                status="completed",
                summary_parts=[],
                encrypted_content="sig123",
            ),
            OutputMessageItem(
                status="completed",
                content_parts=[OutputMessageText(text="Hello!")],
            ),
        ]
        _system, messages = items_to_anthropic_messages(items)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        # Assistant content should have thinking + text blocks
        content = messages[1]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        # ThinkingBlockParam (redacted=False, but no text → empty thinking)
        assert content[0]["type"] == "thinking"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "Hello!"

    def test_tool_call_roundtrip(self):
        """Tool call + tool output should produce correct Anthropic message format."""
        items = [
            InputMessageItem.from_text("What's the weather?"),
            FunctionToolCallItem(
                call_id="toolu_abc",
                name="get_weather",
                arguments='{"city": "Paris"}',
                status="completed",
            ),
            FunctionToolOutputItem(
                call_id="toolu_abc",
                output_parts="Sunny, 22°C",
            ),
            OutputMessageItem(
                status="completed",
                content_parts=[OutputMessageText(text="It's sunny in Paris!")],
            ),
        ]
        _system, messages = items_to_anthropic_messages(items)

        assert len(messages) == 4
        # User message
        assert messages[0]["role"] == "user"
        # Assistant with tool_use
        assert messages[1]["role"] == "assistant"
        assistant_content = messages[1]["content"]
        assert isinstance(assistant_content, list)
        assert assistant_content[0]["type"] == "tool_use"
        assert assistant_content[0]["name"] == "get_weather"
        assert assistant_content[0]["input"] == {"city": "Paris"}
        # Tool result (user role in Anthropic)
        assert messages[2]["role"] == "user"
        tool_result_content = messages[2]["content"]
        assert isinstance(tool_result_content, list)
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "toolu_abc"
        # Final assistant response
        assert messages[3]["role"] == "assistant"

    def test_redacted_thinking_roundtrip(self):
        """Redacted thinking should produce RedactedThinkingBlockParam."""
        items = [
            InputMessageItem.from_text("Think hard"),
            ReasoningItem(
                status="completed",
                summary_parts=[],
                redacted=True,
                encrypted_content="encrypted_data",
            ),
            OutputMessageItem(
                status="completed",
                content_parts=[OutputMessageText(text="Done")],
            ),
        ]
        _, messages = items_to_anthropic_messages(items)

        assistant_content = messages[1]["content"]
        assert isinstance(assistant_content, list)
        assert assistant_content[0]["type"] == "redacted_thinking"
        assert assistant_content[0]["data"] == "encrypted_data"


# ==== tool_converters ====


class _WeatherInput(BaseModel):
    city: str
    units: str = "celsius"


class _WeatherTool(BaseTool[_WeatherInput, str, None]):
    def __init__(self) -> None:
        super().__init__(
            name="get_weather",
            description="Get current weather for a city",
        )

    async def _run(
        self,
        inp: _WeatherInput,
        *,
        ctx: Any = None,  # noqa: ARG002
        call_id: str | None = None,  # noqa: ARG002
        progress_callback: Any = None,  # noqa: ARG002
    ) -> str:
        return f"Weather in {inp.city}"


class TestToolConverters:
    def test_to_api_tool(self):
        tool = _WeatherTool()
        api_tool = to_api_tool(tool)

        assert api_tool["name"] == "get_weather"
        assert api_tool["description"] == "Get current weather for a city"
        schema = api_tool["input_schema"]
        assert "city" in schema["properties"]
        assert "units" in schema["properties"]

    def test_to_api_tool_choice_auto(self):
        result = to_api_tool_choice("auto")
        assert result["type"] == "auto"

    def test_to_api_tool_choice_required(self):
        result = to_api_tool_choice("required")
        assert result["type"] == "any"

    def test_to_api_tool_choice_none(self):
        result = to_api_tool_choice("none")
        assert result["type"] == "none"

    def test_to_api_tool_choice_named(self):
        result = to_api_tool_choice(NamedToolChoice(name="get_weather"))
        assert result["type"] == "tool"
        assert result["name"] == "get_weather"  # type: ignore[typeddict-item]


# ==== Stream converter ====


def _msg_start_event(
    *,
    msg_id: str = "msg_stream_123",
    model: str = "claude-sonnet-4-20250514",
    input_tokens: int = 50,
    output_tokens: int = 0,
) -> RawMessageStartEvent:
    return RawMessageStartEvent(
        type="message_start",
        message=Message(
            id=msg_id,
            type="message",
            role="assistant",
            model=model,
            content=[],
            stop_reason=None,
            stop_sequence=None,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        ),
    )


def _block_start(idx: int, block: ContentBlock) -> RawContentBlockStartEvent:
    return RawContentBlockStartEvent(
        type="content_block_start",
        index=idx,
        content_block=block,
    )


def _block_delta(idx: int, delta: Any) -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=idx,
        delta=delta,
    )


def _block_stop(idx: int) -> RawContentBlockStopEvent:
    return RawContentBlockStopEvent(
        type="content_block_stop",
        index=idx,
    )


def _msg_delta(
    stop_reason: StopReason = "end_turn",
    output_tokens: int = 20,
) -> RawMessageDeltaEvent:
    return RawMessageDeltaEvent(
        type="message_delta",
        delta=MessageDelta(stop_reason=stop_reason),
        usage=MessageDeltaUsage(output_tokens=output_tokens),
    )


def _msg_stop() -> RawMessageStopEvent:
    return RawMessageStopEvent(type="message_stop")


class TestAnthropicStreamConverter:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.loop = asyncio.get_event_loop()

    def _run(self, events: list[Any]) -> list[Any]:
        return self.loop.run_until_complete(_collect_events(events))

    def test_text_stream(self):
        """Basic text streaming: message_start → text block → deltas → stop."""
        events = [
            _msg_start_event(),
            _block_start(0, TextBlock(type="text", text="")),
            _block_delta(0, TextDelta(type="text_delta", text="Hello ")),
            _block_delta(0, TextDelta(type="text_delta", text="world")),
            _block_stop(0),
            _msg_delta(),
            _msg_stop(),
        ]
        llm_events = self._run(events)
        types = [type(e).__name__ for e in llm_events]

        assert "ResponseCreated" in types
        assert "ResponseInProgress" in types
        assert "ResponseCompleted" in types

        # Check text deltas
        text_deltas = [e for e in llm_events if isinstance(e, LlmTextDelta)]
        assert len(text_deltas) == 2
        assert text_deltas[0].delta == "Hello "
        assert text_deltas[1].delta == "world"

        # Check final text
        text_dones = [
            e for e in llm_events if isinstance(e, OutputMessageTextPartTextDone)
        ]
        assert len(text_dones) == 1
        assert text_dones[0].text == "Hello world"

        # Check completed response
        completed = [e for e in llm_events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        assert completed[0].response.output_text == "Hello world"
        assert completed[0].response.status == "completed"

    def test_thinking_then_text(self):
        """Extended thinking followed by text response."""
        events = [
            _msg_start_event(),
            # Thinking block
            _block_start(0, ThinkingBlock(type="thinking", thinking="", signature="")),
            _block_delta(0, ThinkingDelta(type="thinking_delta", thinking="Let me ")),
            _block_delta(0, ThinkingDelta(type="thinking_delta", thinking="think...")),
            _block_delta(
                0,
                SignatureDelta(type="signature_delta", signature="sig_xyz"),
            ),
            _block_stop(0),
            # Text block
            _block_start(1, TextBlock(type="text", text="")),
            _block_delta(1, TextDelta(type="text_delta", text="Answer: 42")),
            _block_stop(1),
            _msg_delta(),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        # Reasoning events
        reasoning_deltas = [
            e for e in llm_events if isinstance(e, ReasoningSummaryPartTextDelta)
        ]
        assert len(reasoning_deltas) == 2
        assert reasoning_deltas[0].delta == "Let me "
        assert reasoning_deltas[1].delta == "think..."

        # Reasoning item done
        reasoning_items = [
            e.item
            for e in llm_events
            if isinstance(e, OutputItemDone) and isinstance(e.item, ReasoningItem)
        ]
        assert len(reasoning_items) == 1
        assert reasoning_items[0].summary_text == "Let me think..."
        assert reasoning_items[0].encrypted_content == "sig_xyz"

        # Text output
        completed = [e for e in llm_events if isinstance(e, ResponseCompleted)]
        assert completed[0].response.output_text == "Answer: 42"

    def test_redacted_thinking(self):
        """Redacted thinking block should produce redacted ReasoningItem."""
        events = [
            _msg_start_event(),
            _block_start(
                0,
                RedactedThinkingBlock(type="redacted_thinking", data="encrypted_abc"),
            ),
            _block_stop(0),
            _block_start(1, TextBlock(type="text", text="")),
            _block_delta(1, TextDelta(type="text_delta", text="Result")),
            _block_stop(1),
            _msg_delta(),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        reasoning_items = [
            e.item
            for e in llm_events
            if isinstance(e, OutputItemDone) and isinstance(e.item, ReasoningItem)
        ]
        assert len(reasoning_items) == 1
        assert reasoning_items[0].redacted is True
        assert reasoning_items[0].encrypted_content == "encrypted_abc"

    def test_interleaved_thinking_and_redacted(self):
        """Thinking → redacted → thinking produces 3 separate ReasoningItems."""
        events = [
            _msg_start_event(),
            # First thinking block
            _block_start(0, ThinkingBlock(type="thinking", thinking="", signature="")),
            _block_delta(0, ThinkingDelta(type="thinking_delta", thinking="first")),
            _block_delta(
                0,
                SignatureDelta(type="signature_delta", signature="sig1"),
            ),
            _block_stop(0),
            # Redacted thinking block
            _block_start(
                1,
                RedactedThinkingBlock(type="redacted_thinking", data="secret_data"),
            ),
            _block_stop(1),
            # Second thinking block
            _block_start(2, ThinkingBlock(type="thinking", thinking="", signature="")),
            _block_delta(2, ThinkingDelta(type="thinking_delta", thinking="second")),
            _block_delta(
                2,
                SignatureDelta(type="signature_delta", signature="sig2"),
            ),
            _block_stop(2),
            # Text block
            _block_start(3, TextBlock(type="text", text="")),
            _block_delta(3, TextDelta(type="text_delta", text="answer")),
            _block_stop(3),
            _msg_delta(),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        reasoning_items = [
            e.item
            for e in llm_events
            if isinstance(e, OutputItemDone) and isinstance(e.item, ReasoningItem)
        ]
        assert len(reasoning_items) == 3

        # First: normal thinking
        assert reasoning_items[0].redacted is False
        assert reasoning_items[0].summary_text == "first"
        assert reasoning_items[0].encrypted_content == "sig1"

        # Second: redacted block
        assert reasoning_items[1].redacted is True
        assert reasoning_items[1].encrypted_content == "secret_data"
        assert reasoning_items[1].summary_parts == []

        # Third: normal thinking resumed
        assert reasoning_items[2].redacted is False
        assert reasoning_items[2].summary_text == "second"
        assert reasoning_items[2].encrypted_content == "sig2"

        # Text output still present
        completed = [e for e in llm_events if isinstance(e, ResponseCompleted)]
        assert completed[0].response.output_text == "answer"

    def test_consecutive_redacted_blocks(self):
        """Multiple redacted blocks in a row → separate ReasoningItems each."""
        events = [
            _msg_start_event(),
            _block_start(
                0,
                RedactedThinkingBlock(type="redacted_thinking", data="enc_1"),
            ),
            _block_stop(0),
            _block_start(
                1,
                RedactedThinkingBlock(type="redacted_thinking", data="enc_2"),
            ),
            _block_stop(1),
            _block_start(2, TextBlock(type="text", text="")),
            _block_delta(2, TextDelta(type="text_delta", text="ok")),
            _block_stop(2),
            _msg_delta(),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        reasoning_items = [
            e.item
            for e in llm_events
            if isinstance(e, OutputItemDone) and isinstance(e.item, ReasoningItem)
        ]
        assert len(reasoning_items) == 2
        assert reasoning_items[0].redacted is True
        assert reasoning_items[0].encrypted_content == "enc_1"
        assert reasoning_items[1].redacted is True
        assert reasoning_items[1].encrypted_content == "enc_2"

    def test_tool_use_stream(self):
        """Tool use streaming with incremental JSON arguments."""
        events = [
            _msg_start_event(),
            _block_start(
                0,
                ToolUseBlock(
                    type="tool_use",
                    id="toolu_stream_1",
                    name="get_weather",
                    input={},
                ),
            ),
            _block_delta(
                0,
                InputJSONDelta(type="input_json_delta", partial_json='{"ci'),
            ),
            _block_delta(
                0,
                InputJSONDelta(type="input_json_delta", partial_json='ty": '),
            ),
            _block_delta(
                0,
                InputJSONDelta(type="input_json_delta", partial_json='"Paris"}'),
            ),
            _block_stop(0),
            _msg_delta(stop_reason="tool_use"),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        # Function call arguments delta
        arg_deltas = [
            e for e in llm_events if isinstance(e, FunctionCallArgumentsDelta)
        ]
        assert len(arg_deltas) == 3

        # Function call arguments done
        arg_dones = [e for e in llm_events if isinstance(e, FunctionCallArgumentsDone)]
        assert len(arg_dones) == 1
        assert arg_dones[0].name == "get_weather"
        assert json.loads(arg_dones[0].arguments) == {"city": "Paris"}

        # Final response
        completed = [e for e in llm_events if isinstance(e, ResponseCompleted)]
        assert len(completed[0].response.tool_call_items) == 1
        tc = completed[0].response.tool_call_items[0]
        assert tc.call_id == "toolu_stream_1"
        assert tc.name == "get_weather"

    def test_text_then_tool_use(self):
        """Text block followed by tool use — message closes before tool call opens."""
        events = [
            _msg_start_event(),
            _block_start(0, TextBlock(type="text", text="")),
            _block_delta(0, TextDelta(type="text_delta", text="I'll check")),
            _block_stop(0),
            _block_start(
                1,
                ToolUseBlock(type="tool_use", id="toolu_2", name="search", input={}),
            ),
            _block_delta(
                1,
                InputJSONDelta(type="input_json_delta", partial_json='{"q":"test"}'),
            ),
            _block_stop(1),
            _msg_delta(stop_reason="tool_use"),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        completed = [e for e in llm_events if isinstance(e, ResponseCompleted)]
        response = completed[0].response
        assert response.output_text == "I'll check"
        assert len(response.tool_call_items) == 1

        # Verify output ordering: message then tool call
        output = response.output_items
        assert isinstance(output[0], OutputMessageItem)
        assert isinstance(output[1], FunctionToolCallItem)

    def test_usage_updated_from_message_delta(self):
        """Usage should reflect final output_tokens from message_delta."""
        events = [
            _msg_start_event(input_tokens=100, output_tokens=0),
            _block_start(0, TextBlock(type="text", text="")),
            _block_delta(0, TextDelta(type="text_delta", text="Hi")),
            _block_stop(0),
            _msg_delta(output_tokens=42),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        completed = [e for e in llm_events if isinstance(e, ResponseCompleted)]
        usage = completed[0].response.usage_with_cost
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 42
        assert usage.total_tokens == 142

    def test_sequence_numbers_monotonic(self):
        """All sequence numbers should be strictly increasing."""
        events = [
            _msg_start_event(),
            _block_start(0, TextBlock(type="text", text="")),
            _block_delta(0, TextDelta(type="text_delta", text="Hi")),
            _block_stop(0),
            _msg_delta(),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        seq_nums = [e.sequence_number for e in llm_events]
        for i in range(1, len(seq_nums)):
            assert seq_nums[i] > seq_nums[i - 1], (
                f"Sequence numbers not monotonic at index {i}: "
                f"{seq_nums[i - 1]} >= {seq_nums[i]}"
            )

    def test_output_indices_correct(self):
        """Each output item should have a unique, increasing output_index."""
        events = [
            _msg_start_event(),
            # Thinking (output_index 0)
            _block_start(0, ThinkingBlock(type="thinking", thinking="", signature="")),
            _block_delta(0, ThinkingDelta(type="thinking_delta", thinking="hmm")),
            _block_stop(0),
            # Text (output_index 1)
            _block_start(1, TextBlock(type="text", text="")),
            _block_delta(1, TextDelta(type="text_delta", text="ok")),
            _block_stop(1),
            # Tool (output_index 2)
            _block_start(
                2,
                ToolUseBlock(type="tool_use", id="t1", name="fn", input={}),
            ),
            _block_delta(2, InputJSONDelta(type="input_json_delta", partial_json="{}")),
            _block_stop(2),
            _msg_delta(stop_reason="tool_use"),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        added_events = [e for e in llm_events if isinstance(e, OutputItemAdded)]
        output_indices = [e.output_index for e in added_events]
        assert output_indices == [0, 1, 2]


# ==== Web search: extraction + round-trip + streaming ====


def _web_search_blocks() -> tuple[
    ServerToolUseBlock, WebSearchToolResultBlock, TextBlock
]:
    """Build a realistic server_tool_use + web_search_tool_result + text sequence."""
    server = ServerToolUseBlock(
        type="server_tool_use",
        id="srvtoolu_abc",
        name="web_search",
        input={"query": "python 3.13 release date"},
    )
    result = WebSearchToolResultBlock(
        type="web_search_tool_result",
        tool_use_id="srvtoolu_abc",
        content=[
            WebSearchResultBlock(
                type="web_search_result",
                url="https://python.org/downloads/",
                title="Python Releases",
                page_age="2d",
                encrypted_content="enc_aaa",
            ),
            WebSearchResultBlock(
                type="web_search_result",
                url="https://blog.python.org/new",
                title="Python Blog",
                page_age=None,
                encrypted_content="enc_bbb",
            ),
        ],
    )
    text = TextBlock(type="text", text="Python 3.13 was released in October 2024.")
    return server, result, text


class TestWebSearchExtraction:
    """server_tool_use + web_search_tool_result → WebSearchCallItem."""

    def test_extracts_web_search_call_item(self):
        server, result, text = _web_search_blocks()
        msg = _make_message([server, result, text])
        items, web_search = items_from_anthropic_message(msg)

        # Should produce WebSearchCallItem + OutputMessageItem
        assert len(items) == 2
        ws_item = items[0]
        assert isinstance(ws_item, WebSearchCallItem)
        assert ws_item.id == "srvtoolu_abc"
        assert ws_item.status == "completed"

        # Action carries query + sources
        assert isinstance(ws_item.action, ActionSearch)
        assert ws_item.action.query == "python 3.13 release date"
        assert ws_item.action.sources is not None
        assert len(ws_item.action.sources) == 2
        assert ws_item.action.sources[0].url == "https://python.org/downloads/"
        assert ws_item.action.sources[1].url == "https://blog.python.org/new"

        # provider_specific_fields preserves per-url encrypted data
        assert ws_item.provider_specific_fields is not None
        encrypted = ws_item.provider_specific_fields["anthropic:encrypted_content"]
        assert encrypted["https://python.org/downloads/"] == "enc_aaa"
        assert encrypted["https://blog.python.org/new"] == "enc_bbb"

        # Text block still creates OutputMessageItem
        assert isinstance(items[1], OutputMessageItem)
        assert "3.13" in items[1].text

        # WebSearchInfo backward compat
        assert web_search is not None
        assert len(web_search.sources) == 2
        assert web_search.sources[0].title == "Python Releases"
        assert web_search.sources[1].page_age is None

    def test_web_search_interleaved_with_tool_calls(self):
        """Web search + user tool call in same message → both extracted correctly."""
        server, result, _ = _web_search_blocks()
        tool = ToolUseBlock(
            type="tool_use",
            id="toolu_xyz",
            name="get_date",
            input={"format": "iso"},
        )
        text = TextBlock(type="text", text="Here's what I found.")
        msg = _make_message([server, result, text, tool], stop_reason="tool_use")
        items, web_search = items_from_anthropic_message(msg)

        assert len(items) == 3
        assert isinstance(items[0], WebSearchCallItem)
        assert isinstance(items[1], OutputMessageItem)
        assert isinstance(items[2], FunctionToolCallItem)
        assert items[2].name == "get_date"
        assert web_search is not None


class TestWebSearchRoundtrip:
    """WebSearchCallItem → server_tool_use + web_search_tool_result."""

    def test_roundtrip_preserves_structure(self):
        """Extract → reconstruct → verify block types and encrypted_content."""
        server, result, text = _web_search_blocks()
        msg = _make_message([server, result, text])
        items, _ = items_from_anthropic_message(msg)

        # Now convert back to Anthropic messages
        _, messages = items_to_anthropic_messages(items)
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)

        # Should have: server_tool_use, web_search_tool_result, text
        assert len(content) == 3
        assert content[0]["type"] == "server_tool_use"
        assert content[0]["id"] == "srvtoolu_abc"
        assert content[0]["input"]["query"] == "python 3.13 release date"

        assert content[1]["type"] == "web_search_tool_result"
        assert content[1]["tool_use_id"] == "srvtoolu_abc"
        results = content[1]["content"]
        assert len(results) == 2
        assert results[0]["url"] == "https://python.org/downloads/"
        assert results[0]["encrypted_content"] == "enc_aaa"
        assert results[1]["url"] == "https://blog.python.org/new"
        assert results[1]["encrypted_content"] == "enc_bbb"

        assert content[2]["type"] == "text"

    def test_roundtrip_in_multi_turn(self):
        """WebSearchCallItem in a multi-turn conversation round-trips correctly."""
        ws_item = WebSearchCallItem(
            id="srvtoolu_multi",
            status="completed",
            action=ActionSearch(
                type="search",
                query="latest news",
                sources=[
                    ActionSearchSource(type="url", url="https://example.com"),
                ],
            ),
            provider_specific_fields={
                "anthropic:encrypted_content": {"https://example.com": "enc_999"}
            },
        )
        items = [
            InputMessageItem.from_text("Search for latest news"),
            ws_item,
            OutputMessageItem(
                status="completed",
                content_parts=[OutputMessageText(text="Here's what I found.")],
            ),
            InputMessageItem.from_text("Tell me more"),
        ]
        _, messages = items_to_anthropic_messages(items)

        # user, assistant (ws + text), user
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        assistant_content = messages[1]["content"]
        assert isinstance(assistant_content, list)
        assert assistant_content[0]["type"] == "server_tool_use"
        assert assistant_content[1]["type"] == "web_search_tool_result"
        assert assistant_content[2]["type"] == "text"


class TestWebSearchStream:
    """Streaming web search → WebSearchCallItem events."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.loop = asyncio.get_event_loop()

    def _run(self, events: list[Any]) -> list[Any]:
        return self.loop.run_until_complete(_collect_events(events))

    def test_stream_web_search_produces_item_events(self):
        """server_tool_use + web_search_tool_result blocks emit OutputItemAdded/Done."""
        server_block = ServerToolUseBlock(
            type="server_tool_use",
            id="srvtoolu_stream1",
            name="web_search",
            input={},
        )
        result_block = WebSearchToolResultBlock(
            type="web_search_tool_result",
            tool_use_id="srvtoolu_stream1",
            content=[
                WebSearchResultBlock(
                    type="web_search_result",
                    url="https://example.com/a",
                    title="Example A",
                    page_age=None,
                    encrypted_content="enc_stream_a",
                ),
            ],
        )
        events = [
            _msg_start_event(),
            # server_tool_use block
            _block_start(0, server_block),
            _block_delta(
                0,
                InputJSONDelta(
                    type="input_json_delta",
                    partial_json='{"query": "test query"}',
                ),
            ),
            _block_stop(0),
            # web_search_tool_result block
            _block_start(1, result_block),
            _block_stop(1),
            # text block
            _block_start(2, TextBlock(type="text", text="")),
            _block_delta(2, TextDelta(type="text_delta", text="Found it")),
            _block_stop(2),
            _msg_delta(),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        # WebSearchCallItem should appear as OutputItemAdded + OutputItemDone
        ws_added = [
            e
            for e in llm_events
            if isinstance(e, OutputItemAdded) and isinstance(e.item, WebSearchCallItem)
        ]
        ws_done = [
            e
            for e in llm_events
            if isinstance(e, OutputItemDone) and isinstance(e.item, WebSearchCallItem)
        ]
        assert len(ws_added) == 1
        assert len(ws_done) == 1

        ws_item = ws_done[0].item
        assert isinstance(ws_item, WebSearchCallItem)
        assert ws_item.id == "srvtoolu_stream1"
        assert isinstance(ws_item.action, ActionSearch)
        assert ws_item.action.query == "test query"
        assert ws_item.action.sources is not None
        assert len(ws_item.action.sources) == 1
        assert ws_item.provider_specific_fields == {
            "anthropic:encrypted_content": {"https://example.com/a": "enc_stream_a"}
        }

        # WebSearchInfo on ResponseCompleted (backward compat)
        completed = [e for e in llm_events if isinstance(e, ResponseCompleted)]
        assert completed[0].response.web_search is not None
        assert len(completed[0].response.web_search.sources) == 1

        # Output ordering: web_search (0), message (1)
        added_indices = [
            e.output_index for e in llm_events if isinstance(e, OutputItemAdded)
        ]
        assert added_indices == [0, 1]

    def test_stream_web_search_then_tool_call(self):
        """Web search followed by a user tool call in the same stream."""
        events = [
            _msg_start_event(),
            # server_tool_use
            _block_start(
                0,
                ServerToolUseBlock(
                    type="server_tool_use",
                    id="srvtoolu_ws",
                    name="web_search",
                    input={},
                ),
            ),
            _block_delta(
                0,
                InputJSONDelta(
                    type="input_json_delta",
                    partial_json='{"query": "q"}',
                ),
            ),
            _block_stop(0),
            # web_search_tool_result
            _block_start(
                1,
                WebSearchToolResultBlock(
                    type="web_search_tool_result",
                    tool_use_id="srvtoolu_ws",
                    content=[
                        WebSearchResultBlock(
                            type="web_search_result",
                            url="https://x.com",
                            title="X",
                            page_age=None,
                            encrypted_content="enc_x",
                        ),
                    ],
                ),
            ),
            _block_stop(1),
            # text
            _block_start(2, TextBlock(type="text", text="")),
            _block_delta(2, TextDelta(type="text_delta", text="let me check")),
            _block_stop(2),
            # user tool call
            _block_start(
                3,
                ToolUseBlock(type="tool_use", id="toolu_fn", name="calc", input={}),
            ),
            _block_delta(
                3,
                InputJSONDelta(type="input_json_delta", partial_json='{"x": 1}'),
            ),
            _block_stop(3),
            _msg_delta(stop_reason="tool_use"),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        completed = [e for e in llm_events if isinstance(e, ResponseCompleted)]
        response = completed[0].response

        # 3 output items: WebSearchCallItem, OutputMessageItem, FunctionToolCallItem
        assert len(response.output_items) == 3
        assert isinstance(response.output_items[0], WebSearchCallItem)
        assert isinstance(response.output_items[1], OutputMessageItem)
        assert isinstance(response.output_items[2], FunctionToolCallItem)
        assert response.output_items[2].name == "calc"


# ==== Web Fetch helpers ====


def _web_fetch_blocks() -> tuple[
    ServerToolUseBlock, WebFetchToolResultBlock, TextBlock
]:
    server = ServerToolUseBlock(
        type="server_tool_use",
        id="srvtoolu_wf1",
        name="web_fetch",
        input={"url": "https://example.com/page"},
    )
    result = WebFetchToolResultBlock(
        type="web_fetch_tool_result",
        tool_use_id="srvtoolu_wf1",
        content=WebFetchBlock(
            type="web_fetch_result",
            url="https://example.com/page",
            retrieved_at="2026-03-22T12:00:00Z",
            content=DocumentBlock(
                type="document",
                title="Example Page",
                source=PlainTextSource(
                    type="text",
                    media_type="text/plain",
                    data="This is the page content.",
                ),
            ),
        ),
    )
    text = TextBlock(type="text", text="The page contains example content.")
    return server, result, text


def _web_fetch_error_blocks() -> tuple[
    ServerToolUseBlock, WebFetchToolResultBlock, TextBlock
]:
    server = ServerToolUseBlock(
        type="server_tool_use",
        id="srvtoolu_wf_err",
        name="web_fetch",
        input={"url": "https://unreachable.invalid"},
    )
    result = WebFetchToolResultBlock(
        type="web_fetch_tool_result",
        tool_use_id="srvtoolu_wf_err",
        content=WebFetchToolResultErrorBlock(
            type="web_fetch_tool_result_error",
            error_code="url_not_accessible",
        ),
    )
    text = TextBlock(type="text", text="I couldn't access that URL.")
    return server, result, text


# ==== Web Fetch extraction ====


class TestWebFetchExtraction:
    """server_tool_use + web_fetch_tool_result → WebSearchCallItem(ActionOpenPage)."""

    def test_success(self):
        server, result, text = _web_fetch_blocks()
        msg = _make_message([server, result, text])
        items, _ = items_from_anthropic_message(msg)

        assert len(items) == 2
        wf = items[0]
        assert isinstance(wf, WebSearchCallItem)
        assert wf.id == "srvtoolu_wf1"
        assert wf.status == "completed"
        assert isinstance(wf.action, ActionOpenPage)
        assert wf.action.url == "https://example.com/page"

        psf = wf.provider_specific_fields
        assert psf is not None
        assert psf["anthropic:retrieved_at"] == "2026-03-22T12:00:00Z"
        assert psf["anthropic:title"] == "Example Page"
        assert psf["anthropic:media_type"] == "text/plain"
        assert psf["anthropic:data"] == "This is the page content."

        assert isinstance(items[1], OutputMessageItem)

    def test_error(self):
        server, result, text = _web_fetch_error_blocks()
        msg = _make_message([server, result, text])
        items, _ = items_from_anthropic_message(msg)

        assert len(items) == 2
        wf = items[0]
        assert isinstance(wf, WebSearchCallItem)
        assert wf.status == "failed"
        assert isinstance(wf.action, ActionOpenPage)
        assert wf.action.url == "https://unreachable.invalid"
        assert wf.provider_specific_fields is not None
        assert (
            wf.provider_specific_fields["anthropic:error_code"] == "url_not_accessible"
        )

    def test_interleaved_with_tool_calls(self):
        server, result, _ = _web_fetch_blocks()
        tool = ToolUseBlock(
            type="tool_use",
            id="toolu_xyz",
            name="summarize",
            input={"text": "hello"},
        )
        text = TextBlock(type="text", text="Here's the summary.")
        msg = _make_message([server, result, text, tool], stop_reason="tool_use")
        items, _ = items_from_anthropic_message(msg)

        assert len(items) == 3
        assert isinstance(items[0], WebSearchCallItem)
        assert isinstance(items[0].action, ActionOpenPage)
        assert isinstance(items[1], OutputMessageItem)
        assert isinstance(items[2], FunctionToolCallItem)


# ==== Web Fetch roundtrip ====


class TestWebFetchRoundtrip:
    """WebSearchCallItem(ActionOpenPage) → server_tool_use + web_fetch_tool_result."""

    def test_success_roundtrip(self):
        server, result, text = _web_fetch_blocks()
        msg = _make_message([server, result, text])
        items, _ = items_from_anthropic_message(msg)

        _, messages = items_to_anthropic_messages(items)
        assert len(messages) == 1
        content: list[Any] = messages[0]["content"]  # type: ignore[assignment]
        assert isinstance(content, list)

        assert len(content) == 3
        assert content[0]["type"] == "server_tool_use"
        assert content[0]["name"] == "web_fetch"
        assert content[0]["input"]["url"] == "https://example.com/page"

        assert content[1]["type"] == "web_fetch_tool_result"
        assert content[1]["tool_use_id"] == "srvtoolu_wf1"
        wf_content = content[1]["content"]
        assert wf_content["type"] == "web_fetch_result"
        assert wf_content["url"] == "https://example.com/page"
        assert wf_content["content"]["source"]["data"] == "This is the page content."

    def test_error_roundtrip(self):
        server, result, text = _web_fetch_error_blocks()
        msg = _make_message([server, result, text])
        items, _ = items_from_anthropic_message(msg)

        _, messages = items_to_anthropic_messages(items)
        content: list[Any] = messages[0]["content"]  # type: ignore[assignment]
        assert isinstance(content, list)

        assert content[1]["type"] == "web_fetch_tool_result"
        wf_content = content[1]["content"]
        assert wf_content["type"] == "web_fetch_tool_result_error"
        assert wf_content["error_code"] == "url_not_accessible"

    def test_multi_turn(self):
        wf_item = WebSearchCallItem(
            id="srvtoolu_wf_mt",
            status="completed",
            action=ActionOpenPage(type="open_page", url="https://example.com"),
            provider_specific_fields={
                "anthropic:title": "Example",
                "anthropic:media_type": "text/plain",
                "anthropic:data": "content",
            },
        )
        items = [
            InputMessageItem.from_text("Fetch this page"),
            wf_item,
            OutputMessageItem(
                status="completed",
                content_parts=[OutputMessageText(text="The page says...")],
            ),
            InputMessageItem.from_text("Tell me more"),
        ]
        _, messages = items_to_anthropic_messages(items)

        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        content: list[Any] = messages[1]["content"]  # type: ignore[assignment]
        assert isinstance(content, list)
        assert content[0]["type"] == "server_tool_use"
        assert content[0]["name"] == "web_fetch"
        assert content[1]["type"] == "web_fetch_tool_result"
        assert content[2]["type"] == "text"


# ==== Web Fetch streaming ====


class TestWebFetchStream:
    """Streaming web_fetch → WebSearchCallItem events."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.loop = asyncio.get_event_loop()

    def _run(self, events: list[Any]) -> list[Any]:
        return self.loop.run_until_complete(_collect_events(events))

    def test_success(self):
        server_block = ServerToolUseBlock(
            type="server_tool_use",
            id="srvtoolu_wf_s1",
            name="web_fetch",
            input={},
        )
        result_block = WebFetchToolResultBlock(
            type="web_fetch_tool_result",
            tool_use_id="srvtoolu_wf_s1",
            content=WebFetchBlock(
                type="web_fetch_result",
                url="https://example.com/streamed",
                retrieved_at="2026-03-22T12:00:00Z",
                content=DocumentBlock(
                    type="document",
                    title="Streamed Page",
                    source=PlainTextSource(
                        type="text",
                        media_type="text/plain",
                        data="streamed content",
                    ),
                ),
            ),
        )
        events = [
            _msg_start_event(),
            _block_start(0, server_block),
            _block_delta(
                0,
                InputJSONDelta(
                    type="input_json_delta",
                    partial_json='{"url": "https://example.com/streamed"}',
                ),
            ),
            _block_stop(0),
            _block_start(1, result_block),
            _block_stop(1),
            _block_start(2, TextBlock(type="text", text="")),
            _block_delta(2, TextDelta(type="text_delta", text="Got it")),
            _block_stop(2),
            _msg_delta(),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        ws_done = [
            e
            for e in llm_events
            if isinstance(e, OutputItemDone) and isinstance(e.item, WebSearchCallItem)
        ]
        assert len(ws_done) == 1
        wf = ws_done[0].item
        assert isinstance(wf, WebSearchCallItem)
        assert isinstance(wf.action, ActionOpenPage)
        assert wf.action.url == "https://example.com/streamed"
        assert wf.status == "completed"
        psf = wf.provider_specific_fields
        assert psf is not None
        assert psf["anthropic:data"] == "streamed content"

    def test_error(self):
        server_block = ServerToolUseBlock(
            type="server_tool_use",
            id="srvtoolu_wf_err_s",
            name="web_fetch",
            input={},
        )
        result_block = WebFetchToolResultBlock(
            type="web_fetch_tool_result",
            tool_use_id="srvtoolu_wf_err_s",
            content=WebFetchToolResultErrorBlock(
                type="web_fetch_tool_result_error",
                error_code="url_not_accessible",
            ),
        )
        events = [
            _msg_start_event(),
            _block_start(0, server_block),
            _block_delta(
                0,
                InputJSONDelta(
                    type="input_json_delta",
                    partial_json='{"url": "https://bad.invalid"}',
                ),
            ),
            _block_stop(0),
            _block_start(1, result_block),
            _block_stop(1),
            _block_start(2, TextBlock(type="text", text="")),
            _block_delta(2, TextDelta(type="text_delta", text="Failed")),
            _block_stop(2),
            _msg_delta(),
            _msg_stop(),
        ]
        llm_events = self._run(events)

        ws_done = [
            e
            for e in llm_events
            if isinstance(e, OutputItemDone) and isinstance(e.item, WebSearchCallItem)
        ]
        assert len(ws_done) == 1
        wf = ws_done[0].item
        assert isinstance(wf, WebSearchCallItem)
        assert wf.status == "failed"
        assert isinstance(wf.action, ActionOpenPage)
        assert wf.action.url == "https://bad.invalid"
        assert wf.provider_specific_fields is not None
        assert (
            wf.provider_specific_fields["anthropic:error_code"] == "url_not_accessible"
        )
