"""Tests for LLMAgentMemory with Responses API items."""

import pytest

from grasp_agents.llm_agent_memory import LLMAgentMemory
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
)


class TestMemoryReset:
    def test_reset_with_instructions(self):
        """Reset creates a system InputMessageItem."""
        mem = LLMAgentMemory()
        mem.reset(instructions="You are helpful.")

        assert len(mem.messages) == 1
        msg = mem.messages[0]
        assert isinstance(msg, InputMessageItem)
        assert msg.role == "system"
        assert msg.texts == ["You are helpful."]

    def test_reset_without_instructions(self):
        """Reset with no instructions creates empty memory."""
        mem = LLMAgentMemory()
        mem.reset(instructions="temp")
        assert len(mem.messages) == 1

        mem.reset()
        assert len(mem.messages) == 0

    def test_reset_clears_old_messages(self):
        """Reset replaces any existing messages."""
        mem = LLMAgentMemory()
        mem.update([InputMessageItem.from_text("old", role="user")])
        assert len(mem.messages) == 1

        mem.reset(instructions="new system")
        assert len(mem.messages) == 1
        assert mem.messages[0].role == "system"


class TestMemoryInstructions:
    def test_instructions_property_returns_text(self):
        """Instructions property returns the first text from system message."""
        mem = LLMAgentMemory()
        mem.reset(instructions="Be concise.")
        assert mem.instructions == "Be concise."

    def test_instructions_property_no_system(self):
        """Instructions returns None when first message is not system."""
        mem = LLMAgentMemory()
        mem.update([InputMessageItem.from_text("user msg", role="user")])
        assert mem.instructions is None

    def test_instructions_property_empty(self):
        """Instructions returns None for empty memory."""
        mem = LLMAgentMemory()
        assert mem.instructions is None


class TestMemoryUpdate:
    def test_update_appends_items(self):
        """update() extends the message list with new items."""
        mem = LLMAgentMemory()
        mem.reset(instructions="sys")

        user_msg = InputMessageItem.from_text("Hello", role="user")
        output_msg = OutputMessageItem(
            content_parts=[OutputMessageText(text="Hi!")],
            status="completed",
        )
        mem.update([user_msg, output_msg])

        assert len(mem.messages) == 3
        assert mem.messages[0].role == "system"
        assert isinstance(mem.messages[1], InputMessageItem)
        assert isinstance(mem.messages[2], OutputMessageItem)

    def test_update_with_tool_call_items(self):
        """update() can store tool call and tool output items."""
        mem = LLMAgentMemory()
        tc = FunctionToolCallItem(call_id="tc1", name="search", arguments="{}")
        to = FunctionToolOutputItem.from_tool_result(call_id="tc1", output="result")
        mem.update([tc, to])

        assert len(mem.messages) == 2
        assert isinstance(mem.messages[0], FunctionToolCallItem)
        assert isinstance(mem.messages[1], FunctionToolOutputItem)

    def test_update_preserves_order(self):
        """Items are stored in the order they are provided."""
        mem = LLMAgentMemory()
        items = [
            InputMessageItem.from_text("Hello", role="user"),
            OutputMessageItem(
                content_parts=[OutputMessageText(text="Hi")],
                status="completed",
            ),
            FunctionToolCallItem(call_id="c1", name="t", arguments="{}"),
            FunctionToolOutputItem.from_tool_result(call_id="c1", output="r"),
        ]
        mem.update(items)

        assert len(mem.messages) == 4
        assert isinstance(mem.messages[0], InputMessageItem)
        assert isinstance(mem.messages[1], OutputMessageItem)
        assert isinstance(mem.messages[2], FunctionToolCallItem)
        assert isinstance(mem.messages[3], FunctionToolOutputItem)


class TestMemoryState:
    def test_is_empty(self):
        """is_empty reflects actual state."""
        mem = LLMAgentMemory()
        assert mem.is_empty

        mem.reset(instructions="sys")
        assert not mem.is_empty

    def test_erase(self):
        """erase() clears all messages."""
        mem = LLMAgentMemory()
        mem.reset(instructions="sys")
        mem.update([InputMessageItem.from_text("x", role="user")])

        mem.erase()
        assert mem.is_empty
        assert len(mem.messages) == 0

    def test_repr(self):
        """Repr shows message count."""
        mem = LLMAgentMemory()
        mem.reset(instructions="sys")
        mem.update([InputMessageItem.from_text("x", role="user")])
        assert "2" in repr(mem)


class TestMemoryFullConversation:
    def test_simulates_agentic_loop(self):
        """Simulates a full agentic loop: system → user → assistant(+tools) → tool_output → assistant."""
        mem = LLMAgentMemory()
        mem.reset(instructions="You are a calculator.")

        # User turn
        mem.update([InputMessageItem.from_text("What is 2+2?", role="user")])

        # First LLM response (with tool call)
        first_response_output = [
            OutputMessageItem(
                content_parts=[OutputMessageText(text="Let me calculate.")],
                status="completed",
            ),
            FunctionToolCallItem(
                call_id="add_1", name="add", arguments='{"a":2,"b":2}'
            ),
        ]
        mem.update(first_response_output)

        # Tool execution result
        tool_output = FunctionToolOutputItem.from_tool_result(
            call_id="add_1", output={"result": 4}
        )
        mem.update([tool_output])

        # Second LLM response (final answer)
        final_output = [
            OutputMessageItem(
                content_parts=[OutputMessageText(text="2 + 2 = 4")],
                status="completed",
            ),
        ]
        mem.update(final_output)

        assert len(mem.messages) == 6
        assert mem.instructions == "You are a calculator."

        # Verify types in order
        types = [type(m).__name__ for m in mem.messages]
        assert types == [
            "InputMessageItem",  # system
            "InputMessageItem",  # user
            "OutputMessageItem",  # assistant text
            "FunctionToolCallItem",  # tool call
            "FunctionToolOutputItem",  # tool result
            "OutputMessageItem",  # final answer
        ]
