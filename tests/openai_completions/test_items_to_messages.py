"""
Tests for items_to_completions_messages conversion.

Verifies that Responses API items stored in memory are correctly converted
to Chat Completions message format for the Completions API providers.
"""

import json

import pytest

from grasp_agents.llm_providers.openai_completions.response_to_provider_inputs import (
    items_to_provider_inputs as items_to_completions_messages,
)
from grasp_agents.llm_providers.openai_completions.response_to_provider_inputs import (
    response_to_provider_input as response_to_completions_message,
)
from grasp_agents.types.content import (
    InputImage,
    InputText,
    OutputMessageText,
    ReasoningText,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.response import Response

# ---------- Input messages ----------


class TestInputMessageConversion:
    def test_system_message_text(self):
        """System message with single text part → plain string content."""
        items = [InputMessageItem.from_text("You are helpful.", role="system")]
        msgs = items_to_completions_messages(items)

        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."

    def test_user_message_text(self):
        """User message with single text part → plain string content."""
        items = [InputMessageItem.from_text("Hello", role="user")]
        msgs = items_to_completions_messages(items)

        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"

    def test_user_message_with_image(self):
        """User message with text + image → list of content parts."""
        img = InputImage.from_url("https://example.com/img.png", detail="high")
        item = InputMessageItem(
            content_parts=[InputText(text="Describe this"), img],
            role="user",
        )
        msgs = items_to_completions_messages([item])

        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Describe this"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/img.png"
        assert content[1]["image_url"]["detail"] == "high"

    def test_developer_message(self):
        """Developer role messages are emitted as user-style messages."""
        items = [InputMessageItem.from_text("Dev instructions", role="developer")]
        msgs = items_to_completions_messages(items)

        assert len(msgs) == 1
        # developer role maps to user message param with role=developer
        assert msgs[0]["role"] == "developer"
        assert msgs[0]["content"] == "Dev instructions"


# ---------- Assistant output ----------


class TestOutputMessageConversion:
    def test_output_message_text_only(self):
        """OutputMessageItem without tool calls → assistant with content."""
        item = OutputMessageItem(
            content_parts=[OutputMessageText(text="The answer is 42.")],
            status="completed",
        )
        msgs = items_to_completions_messages([item])

        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "The answer is 42."
        assert "tool_calls" not in msgs[0]

    def test_output_message_with_tool_calls(self):
        """OutputMessageItem + FunctionToolCallItems → grouped assistant msg."""
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="Let me search.")],
            status="completed",
        )
        tc1 = FunctionToolCallItem(
            call_id="call_1",
            name="search",
            arguments='{"query": "python"}',
        )
        tc2 = FunctionToolCallItem(
            call_id="call_2",
            name="lookup",
            arguments='{"id": 123}',
        )

        msgs = items_to_completions_messages([output, tc1, tc2])

        assert len(msgs) == 1
        msg = msgs[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me search."
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["id"] == "call_1"
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"query": "python"}'
        assert msg["tool_calls"][1]["id"] == "call_2"

    def test_output_empty_content_with_tool_calls(self):
        """When content is empty but tool calls exist, content should not be None."""
        output = OutputMessageItem(
            content_parts=[],
            status="completed",
        )
        tc = FunctionToolCallItem(call_id="call_1", name="search", arguments="{}")
        msgs = items_to_completions_messages([output, tc])

        # Should have non-None content to avoid API errors
        assert msgs[0]["content"] is not None


# ---------- Standalone tool calls ----------


class TestStandaloneToolCalls:
    def test_tool_calls_without_output_message(self):
        """FunctionToolCallItem without preceding OutputMessageItem."""
        tc = FunctionToolCallItem(
            call_id="call_x", name="get_weather", arguments='{"city": "NYC"}'
        )
        msgs = items_to_completions_messages([tc])

        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert "tool_calls" in msgs[0]
        assert len(msgs[0]["tool_calls"]) == 1
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_consecutive_standalone_tool_calls(self):
        """Multiple consecutive FunctionToolCallItems without OutputMessageItem."""
        tc1 = FunctionToolCallItem(call_id="c1", name="tool_a", arguments="{}")
        tc2 = FunctionToolCallItem(call_id="c2", name="tool_b", arguments="{}")
        msgs = items_to_completions_messages([tc1, tc2])

        assert len(msgs) == 1
        assert len(msgs[0]["tool_calls"]) == 2


# ---------- Tool outputs ----------


class TestToolOutputConversion:
    def test_tool_output_string(self):
        """FunctionToolOutputItem with string output → tool message."""
        item = FunctionToolOutputItem(call_id="call_1", output_parts='{"result": "ok"}')
        msgs = items_to_completions_messages([item])

        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_1"
        assert msgs[0]["content"] == '{"result": "ok"}'

    def test_tool_output_from_tool_result(self):
        """FunctionToolOutputItem.from_tool_result helper works correctly."""
        item = FunctionToolOutputItem.from_tool_result(
            call_id="call_2", output={"status": "success", "data": [1, 2, 3]}
        )
        msgs = items_to_completions_messages([item])

        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_2"
        parsed = json.loads(msgs[0]["content"])
        assert parsed["status"] == "success"
        assert parsed["data"] == [1, 2, 3]


# ---------- Reasoning items ----------


class TestReasoningConversion:
    def test_reasoning_grouped_with_output(self):
        """ReasoningItem + OutputMessageItem → single assistant message with thinking_blocks."""
        reasoning = ReasoningItem(content_parts=[ReasoningText(text="Let me think...")])
        user = InputMessageItem.from_text("What is 2+2?", role="user")
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="4")],
            status="completed",
        )

        msgs = items_to_completions_messages([user, reasoning, output])

        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "4"
        # Reasoning is included as thinking_blocks
        assert "thinking_blocks" in msgs[1]
        assert len(msgs[1]["thinking_blocks"]) == 1
        assert msgs[1]["thinking_blocks"][0]["type"] == "thinking"
        assert msgs[1]["thinking_blocks"][0]["thinking"] == "Let me think..."

    def test_reasoning_with_signature(self):
        """ReasoningItem with encrypted_content → thinking block has signature."""
        reasoning = ReasoningItem(
            content_parts=[ReasoningText(text="Deep thought")],
            encrypted_content="sig_abc123",
        )
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="Result")],
            status="completed",
        )

        msgs = items_to_completions_messages([reasoning, output])

        assert len(msgs) == 1
        block = msgs[0]["thinking_blocks"][0]
        assert block["type"] == "thinking"
        assert block["thinking"] == "Deep thought"
        assert block["signature"] == "sig_abc123"

    def test_redacted_reasoning(self):
        """Redacted ReasoningItem → redacted_thinking block with data."""
        reasoning = ReasoningItem(
            encrypted_content="encrypted_data_xyz",
            redacted=True,
        )
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="Answer")],
            status="completed",
        )

        msgs = items_to_completions_messages([reasoning, output])

        assert len(msgs) == 1
        block = msgs[0]["thinking_blocks"][0]
        assert block["type"] == "redacted_thinking"
        assert block["data"] == "encrypted_data_xyz"

    def test_multiple_reasoning_items_grouped(self):
        """Multiple consecutive ReasoningItems are all included in thinking_blocks."""
        r1 = ReasoningItem(
            content_parts=[ReasoningText(text="Step 1")],
            encrypted_content="sig_1",
        )
        r2 = ReasoningItem(
            encrypted_content="encrypted_step2",
            redacted=True,
        )
        r3 = ReasoningItem(
            content_parts=[ReasoningText(text="Step 3")],
        )
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="Final answer")],
            status="completed",
        )

        msgs = items_to_completions_messages([r1, r2, r3, output])

        assert len(msgs) == 1
        assert msgs[0]["content"] == "Final answer"
        blocks = msgs[0]["thinking_blocks"]
        assert len(blocks) == 3
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["thinking"] == "Step 1"
        assert blocks[0]["signature"] == "sig_1"
        assert blocks[1]["type"] == "redacted_thinking"
        assert blocks[1]["data"] == "encrypted_step2"
        assert blocks[2]["type"] == "thinking"
        assert blocks[2]["thinking"] == "Step 3"

    def test_reasoning_with_tool_calls_no_output(self):
        """ReasoningItem followed by tool calls (no OutputMessageItem) → assistant msg."""
        reasoning = ReasoningItem(
            content_parts=[ReasoningText(text="I need to search")],
        )
        tc = FunctionToolCallItem(
            call_id="call_1", name="search", arguments='{"q": "test"}'
        )

        msgs = items_to_completions_messages([reasoning, tc])

        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert "content" not in msgs[0]
        assert len(msgs[0]["tool_calls"]) == 1
        assert msgs[0]["thinking_blocks"][0]["thinking"] == "I need to search"

    def test_reasoning_standalone(self):
        """ReasoningItem not followed by output or tool calls → standalone assistant msg."""
        reasoning = ReasoningItem(content_parts=[ReasoningText(text="Just thinking")])
        user = InputMessageItem.from_text("Next question", role="user")

        msgs = items_to_completions_messages([reasoning, user])

        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert "content" not in msgs[0]
        assert "thinking_blocks" in msgs[0]
        assert msgs[1]["role"] == "user"

    def test_reasoning_output_tool_calls_all_grouped(self):
        """ReasoningItem + OutputMessageItem + FunctionToolCallItems → single message."""
        reasoning = ReasoningItem(
            content_parts=[ReasoningText(text="Let me check two things")],
            encrypted_content="sig_xyz",
        )
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="Checking...")],
            status="completed",
        )
        tc1 = FunctionToolCallItem(call_id="c1", name="tool_a", arguments="{}")
        tc2 = FunctionToolCallItem(call_id="c2", name="tool_b", arguments="{}")

        msgs = items_to_completions_messages([reasoning, output, tc1, tc2])

        assert len(msgs) == 1
        msg = msgs[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Checking..."
        assert len(msg["tool_calls"]) == 2
        assert len(msg["thinking_blocks"]) == 1
        assert msg["thinking_blocks"][0]["signature"] == "sig_xyz"


# ---------- Full conversation flows ----------


class TestFullConversation:
    def test_multi_turn_with_tool_use(self):
        """Full multi-turn conversation: system → user → assistant+tools → tool_output → assistant."""
        items = [
            InputMessageItem.from_text("You are a calculator.", role="system"),
            InputMessageItem.from_text("What is 15 * 7?", role="user"),
            OutputMessageItem(
                content_parts=[OutputMessageText(text="I'll calculate that.")],
                status="completed",
            ),
            FunctionToolCallItem(
                call_id="calc_1",
                name="multiply",
                arguments='{"a": 15, "b": 7}',
            ),
            FunctionToolOutputItem.from_tool_result(
                call_id="calc_1", output={"result": 105}
            ),
            OutputMessageItem(
                content_parts=[OutputMessageText(text="15 * 7 = 105")],
                status="completed",
            ),
        ]

        msgs = items_to_completions_messages(items)

        assert len(msgs) == 5
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a calculator."

        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "What is 15 * 7?"

        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "I'll calculate that."
        assert len(msgs[2]["tool_calls"]) == 1
        assert msgs[2]["tool_calls"][0]["function"]["name"] == "multiply"

        assert msgs[3]["role"] == "tool"
        assert msgs[3]["tool_call_id"] == "calc_1"

        assert msgs[4]["role"] == "assistant"
        assert msgs[4]["content"] == "15 * 7 = 105"
        assert "tool_calls" not in msgs[4]

    def test_parallel_tool_calls_and_outputs(self):
        """Multiple tool calls followed by their respective outputs."""
        items = [
            InputMessageItem.from_text("Compare weather", role="user"),
            OutputMessageItem(content_parts=[], status="completed"),
            FunctionToolCallItem(
                call_id="w1", name="weather", arguments='{"city": "NYC"}'
            ),
            FunctionToolCallItem(
                call_id="w2", name="weather", arguments='{"city": "LA"}'
            ),
            FunctionToolOutputItem.from_tool_result(call_id="w1", output="Sunny, 72F"),
            FunctionToolOutputItem.from_tool_result(call_id="w2", output="Cloudy, 65F"),
            OutputMessageItem(
                content_parts=[OutputMessageText(text="NYC is warmer.")],
                status="completed",
            ),
        ]

        msgs = items_to_completions_messages(items)

        assert len(msgs) == 5
        # user
        assert msgs[0]["role"] == "user"
        # assistant with 2 tool calls
        assert msgs[1]["role"] == "assistant"
        assert len(msgs[1]["tool_calls"]) == 2
        # 2 tool outputs
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "w1"
        assert msgs[3]["role"] == "tool"
        assert msgs[3]["tool_call_id"] == "w2"
        # final assistant
        assert msgs[4]["role"] == "assistant"
        assert msgs[4]["content"] == "NYC is warmer."

    def test_empty_input(self):
        """Empty items list → empty messages list."""
        assert items_to_completions_messages([]) == []

    @pytest.mark.xfail(reason="Unknown item filtering not yet implemented")
    def test_unknown_item_type_skipped(self):
        """Unknown item types are silently skipped."""
        items = [
            InputMessageItem.from_text("Hello", role="user"),
            {"type": "unknown", "data": "something"},  # not a recognized item
            OutputMessageItem(
                content_parts=[OutputMessageText(text="Hi!")],
                status="completed",
            ),
        ]
        msgs = items_to_completions_messages(items)

        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"


# ---------- OpenRouter reasoning_details format ----------


class TestReasoningDetailsFormat:
    def test_reasoning_summary_format(self):
        """Normal reasoning → reasoning.summary entry."""
        reasoning = ReasoningItem(content_parts=[ReasoningText(text="Let me think...")])
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="4")],
            status="completed",
        )

        msgs = items_to_completions_messages(
            [reasoning, output], reasoning_block_format="openrouter"
        )

        assert len(msgs) == 1
        assert "reasoning_details" in msgs[0]
        assert "thinking_blocks" not in msgs[0]
        details = msgs[0]["reasoning_details"]
        assert len(details) == 1
        assert details[0]["type"] == "reasoning.summary"
        assert details[0]["summary"] == "Let me think..."

    def test_reasoning_with_signature_format(self):
        """Reasoning with signature → reasoning.text entry."""
        reasoning = ReasoningItem(
            content_parts=[ReasoningText(text="Deep thought")],
            encrypted_content="sig_abc123",
        )
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="Result")],
            status="completed",
        )

        msgs = items_to_completions_messages(
            [reasoning, output], reasoning_block_format="openrouter"
        )

        detail = msgs[0]["reasoning_details"][0]
        assert detail["type"] == "reasoning.text"
        assert detail["text"] == "Deep thought"
        assert detail["signature"] == "sig_abc123"

    def test_redacted_reasoning_format(self):
        """Redacted reasoning → reasoning.encrypted entry."""
        reasoning = ReasoningItem(
            encrypted_content="encrypted_data_xyz",
            redacted=True,
        )
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="Answer")],
            status="completed",
        )

        msgs = items_to_completions_messages(
            [reasoning, output], reasoning_block_format="openrouter"
        )

        detail = msgs[0]["reasoning_details"][0]
        assert detail["type"] == "reasoning.encrypted"
        assert detail["data"] == "encrypted_data_xyz"

    def test_multiple_reasoning_items(self):
        """Multiple reasoning items → multiple reasoning_details entries."""
        r1 = ReasoningItem(
            content_parts=[ReasoningText(text="Step 1")],
            encrypted_content="sig_1",
        )
        r2 = ReasoningItem(
            encrypted_content="encrypted_step2",
            redacted=True,
        )
        r3 = ReasoningItem(
            content_parts=[ReasoningText(text="Step 3")],
        )
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="Done")],
            status="completed",
        )

        msgs = items_to_completions_messages(
            [r1, r2, r3, output], reasoning_block_format="openrouter"
        )

        details = msgs[0]["reasoning_details"]
        assert len(details) == 3
        assert details[0]["type"] == "reasoning.text"
        assert details[0]["text"] == "Step 1"
        assert details[0]["signature"] == "sig_1"
        assert details[1]["type"] == "reasoning.encrypted"
        assert details[1]["data"] == "encrypted_step2"
        assert details[2]["type"] == "reasoning.summary"
        assert details[2]["summary"] == "Step 3"

    def test_reasoning_format_none_omits_reasoning(self):
        """reasoning_block_format=None omits reasoning entirely."""
        reasoning = ReasoningItem(
            content_parts=[ReasoningText(text="Thinking...")],
        )
        output = OutputMessageItem(
            content_parts=[OutputMessageText(text="Answer")],
            status="completed",
        )

        msgs = items_to_completions_messages([reasoning, output], reasoning_block_format=None)

        assert len(msgs) == 1
        assert msgs[0]["content"] == "Answer"
        assert "thinking_blocks" not in msgs[0]
        assert "reasoning_details" not in msgs[0]


# ---------- response_to_completions_message ----------


class TestResponseToCompletionsMessage:
    def test_basic_text_response(self):
        """Response with text output → assistant message with content."""
        response = Response(
            model="test-model",
            output_items=[
                OutputMessageItem(
                    content_parts=[OutputMessageText(text="Hello!")],
                    status="completed",
                )
            ],
        )

        msg = response_to_completions_message(response)

        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert "tool_calls" not in msg
        assert "thinking_blocks" not in msg

    def test_response_with_tool_calls(self):
        """Response with tool calls → assistant message with tool_calls."""
        response = Response(
            model="test-model",
            output_items=[
                OutputMessageItem(content_parts=[], status="completed"),
                FunctionToolCallItem(
                    call_id="call_1",
                    name="search",
                    arguments='{"q": "test"}',
                ),
            ],
        )

        msg = response_to_completions_message(response)

        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "call_1"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"]["name"] == "search"

    def test_response_with_reasoning_thinking_blocks(self):
        """Response with reasoning → thinking_blocks by default."""
        response = Response(
            model="test-model",
            output_items=[
                ReasoningItem(
                    content_parts=[ReasoningText(text="Let me think")],
                    encrypted_content="sig_123",
                ),
                OutputMessageItem(
                    content_parts=[OutputMessageText(text="Answer")],
                    status="completed",
                ),
            ],
        )

        msg = response_to_completions_message(response)

        assert msg["content"] == "Answer"
        assert len(msg["thinking_blocks"]) == 1
        assert msg["thinking_blocks"][0]["type"] == "thinking"
        assert msg["thinking_blocks"][0]["signature"] == "sig_123"

    def test_response_with_reasoning_openrouter_format(self):
        """Response with reasoning + reasoning_block_format="openrouter"."""
        response = Response(
            model="test-model",
            output_items=[
                ReasoningItem(
                    content_parts=[ReasoningText(text="Thinking")],
                ),
                OutputMessageItem(
                    content_parts=[OutputMessageText(text="Result")],
                    status="completed",
                ),
            ],
        )

        msg = response_to_completions_message(
            response, reasoning_block_format="openrouter"
        )

        assert "thinking_blocks" not in msg
        assert len(msg["reasoning_details"]) == 1
        assert msg["reasoning_details"][0]["type"] == "reasoning.summary"
        assert msg["reasoning_details"][0]["summary"] == "Thinking"
