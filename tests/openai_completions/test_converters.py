"""Unit tests for OpenAI Completions API converters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai.types import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import (
    Annotation as ChatCompletionAnnotation,
)
from openai.types.chat.chat_completion_message import (
    AnnotationURLCitation,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

from grasp_agents.llm_providers.openai_completions.provider_output_to_response import (
    convert_annotations,
    convert_usage,
    _chat_completion_to_items,
    provider_output_to_response,
)
from grasp_agents.llm_providers.openai_completions.tool_converters import (
    to_api_tool,
    to_api_tool_choice,
)
from grasp_agents.types.content import OutputMessageRefusal, OutputMessageText, UrlCitation
from grasp_agents.types.items import (
    FunctionToolCallItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.tool import NamedToolChoice

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _make_add_tool() -> Any:
    """Reuse the same AddTool definition from conftest."""
    from tests.conftest import AddTool

    return AddTool()


def _make_completion(
    content: str | None = None,
    refusal: str | None = None,
    tool_calls: list[ChatCompletionMessageFunctionToolCall] | None = None,
    finish_reason: str = "stop",
    usage: CompletionUsage | None = None,
) -> ChatCompletion:
    msg = ChatCompletionMessage(
        role="assistant",
        content=content,
        refusal=refusal,
        tool_calls=tool_calls,
    )
    return ChatCompletion(
        id="chatcmpl-test",
        created=1234567890,
        model="gpt-4.1-nano",
        object="chat.completion",
        choices=[Choice(index=0, finish_reason=finish_reason, message=msg)],
        usage=usage
        or CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


# ================================================================== #
#  items_extraction.generated_message_to_items                         #
# ================================================================== #


class TestGeneratedMessageToItems:
    def test_text_only(self) -> None:
        msg = ChatCompletionMessage(role="assistant", content="Hello world")
        items = _chat_completion_to_items(msg, output_message_status="completed")

        assert len(items) == 1
        assert isinstance(items[0], OutputMessageItem)
        assert items[0].text == "Hello world"

    def test_refusal(self) -> None:
        msg = ChatCompletionMessage(
            role="assistant", content=None, refusal="I can't do that"
        )
        items = _chat_completion_to_items(msg, output_message_status="completed")

        assert len(items) == 1
        assert isinstance(items[0], OutputMessageItem)
        out_msg = items[0]
        assert len(out_msg.content_parts) == 1
        assert isinstance(out_msg.content_parts[0], OutputMessageRefusal)
        assert out_msg.content_parts[0].refusal == "I can't do that"

    def test_text_and_refusal(self) -> None:
        msg = ChatCompletionMessage(
            role="assistant", content="Partial answer", refusal="But also refusing"
        )
        items = _chat_completion_to_items(msg, output_message_status="completed")

        assert len(items) == 1
        assert isinstance(items[0], OutputMessageItem)
        assert len(items[0].content_parts) == 2
        assert isinstance(items[0].content_parts[0], OutputMessageText)
        assert isinstance(items[0].content_parts[1], OutputMessageRefusal)

    def test_tool_calls(self) -> None:
        tc = ChatCompletionMessageFunctionToolCall(
            id="call_abc",
            type="function",
            function=Function(name="add", arguments='{"a": 1, "b": 2}'),
        )
        msg = ChatCompletionMessage(role="assistant", content=None, tool_calls=[tc])
        items = _chat_completion_to_items(msg, output_message_status="completed")

        assert len(items) == 1
        assert isinstance(items[0], FunctionToolCallItem)
        assert items[0].name == "add"
        assert items[0].call_id == "call_abc"
        assert items[0].arguments == '{"a": 1, "b": 2}'

    def test_text_and_tool_calls(self) -> None:
        tc = ChatCompletionMessageFunctionToolCall(
            id="call_xyz",
            type="function",
            function=Function(name="add", arguments='{"a": 3, "b": 4}'),
        )
        msg = ChatCompletionMessage(
            role="assistant", content="Let me add those", tool_calls=[tc]
        )
        items = _chat_completion_to_items(msg, output_message_status="completed")

        assert len(items) == 2
        assert isinstance(items[0], OutputMessageItem)
        assert isinstance(items[1], FunctionToolCallItem)

    def test_empty_message(self) -> None:
        msg = ChatCompletionMessage(role="assistant", content=None)
        items = _chat_completion_to_items(msg, output_message_status="completed")

        assert items == []

    def test_reasoning_content(self) -> None:
        """Message with reasoning_content attr (e.g. DeepSeek via OpenAI compat)."""
        msg = ChatCompletionMessage(role="assistant", content="Answer: 42")
        # Simulate a non-standard field via SimpleNamespace overlay
        msg_ns = SimpleNamespace(**msg.__dict__, reasoning_content="Let me think...")
        # Patch getattr to work
        msg.__dict__["reasoning_content"] = "Let me think..."

        items = _chat_completion_to_items(msg, output_message_status="completed")

        reasoning = [i for i in items if isinstance(i, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].summary_text == "Let me think..."

    def test_reasoning_details(self) -> None:
        """Message with reasoning_details (OpenRouter format)."""
        msg = ChatCompletionMessage(role="assistant", content="Answer")
        msg.__dict__["reasoning_details"] = [
            {"type": "reasoning.text", "text": "Step 1: think", "signature": "sig123"}
        ]

        items = _chat_completion_to_items(msg, output_message_status="completed")

        reasoning = [i for i in items if isinstance(i, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].encrypted_content == "sig123"
        assert reasoning[0].summary_text == "Step 1: think"


# ================================================================== #
#  items_extraction.convert_annotations                                #
# ================================================================== #


class TestConvertAnnotations:
    def test_pydantic_annotations(self) -> None:
        ann = ChatCompletionAnnotation(
            type="url_citation",
            url_citation=AnnotationURLCitation(
                end_index=50,
                start_index=10,
                title="Test Source",
                url="https://example.com",
            ),
        )
        citations = convert_annotations([ann])

        assert len(citations) == 1
        assert isinstance(citations[0], UrlCitation)
        assert citations[0].url == "https://example.com"
        assert citations[0].title == "Test Source"
        assert citations[0].start_index == 10
        assert citations[0].end_index == 50

    def test_dict_annotations(self) -> None:
        ann_dict: dict[str, Any] = {
            "url_citation": {
                "url": "https://example.com/dict",
                "title": "Dict Source",
                "start_index": 0,
                "end_index": 20,
            }
        }
        citations = convert_annotations([ann_dict])

        assert len(citations) == 1
        assert citations[0].url == "https://example.com/dict"
        assert citations[0].title == "Dict Source"

    def test_incomplete_dict_annotation(self) -> None:
        """Missing url should cause the annotation to be skipped."""
        ann_dict: dict[str, Any] = {
            "url_citation": {
                "title": "No URL",
                "start_index": 0,
                "end_index": 10,
            }
        }
        citations = convert_annotations([ann_dict])

        assert citations == []


# ================================================================== #
#  tool_converters                                                     #
# ================================================================== #


class TestToolConverters:
    def test_to_api_tool_non_strict(self) -> None:
        tool = _make_add_tool()
        result = to_api_tool(tool, strict=None)

        assert result["type"] == "function"
        fn = result["function"]
        assert fn["name"] == "add"
        assert fn["description"] == "Add two integers and return their sum."
        assert "properties" in fn["parameters"]
        assert "strict" not in fn

    def test_to_api_tool_strict(self) -> None:
        tool = _make_add_tool()
        result = to_api_tool(tool, strict=True)

        assert result["type"] == "function"
        fn = result["function"]
        assert fn["name"] == "add"
        assert fn.get("strict") is True

    def test_to_api_tool_choice_auto(self) -> None:
        assert to_api_tool_choice("auto") == "auto"

    def test_to_api_tool_choice_required(self) -> None:
        assert to_api_tool_choice("required") == "required"

    def test_to_api_tool_choice_named(self) -> None:
        result = to_api_tool_choice(NamedToolChoice(name="add"))

        assert result["type"] == "function"
        assert result["function"]["name"] == "add"


# ================================================================== #
#  provider_output_to_response                                         #
# ================================================================== #


class TestProviderOutputToResponse:
    def test_basic_response(self) -> None:
        completion = _make_completion(content="Hello")
        response = provider_output_to_response(completion)

        assert response.status == "completed"
        assert response.model == "gpt-4.1-nano"
        assert response.output_text == "Hello"
        assert len(response.output_items) == 1
        assert isinstance(response.output_items[0], OutputMessageItem)
        assert response.usage_with_cost is not None
        assert response.usage_with_cost.input_tokens == 10
        assert response.usage_with_cost.output_tokens == 5

    def test_length_finish_reason(self) -> None:
        completion = _make_completion(content="Partial...", finish_reason="length")
        response = provider_output_to_response(completion)

        assert response.status == "incomplete"
        assert response.incomplete_details is not None
        assert response.incomplete_details.reason == "max_output_tokens"

    def test_content_filter(self) -> None:
        completion = _make_completion(
            content="Filtered", finish_reason="content_filter"
        )
        response = provider_output_to_response(completion)

        assert response.status == "incomplete"
        assert response.incomplete_details is not None
        assert response.incomplete_details.reason == "content_filter"

    def test_usage_with_cached_and_reasoning(self) -> None:
        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details={"cached_tokens": 30},  # type: ignore[arg-type]
            completion_tokens_details={"reasoning_tokens": 20},  # type: ignore[arg-type]
        )
        completion = _make_completion(content="Answer", usage=usage)
        response = provider_output_to_response(completion)

        assert response.usage_with_cost is not None
        assert response.usage_with_cost.input_tokens == 100
        assert response.usage_with_cost.output_tokens == 50
        assert response.usage_with_cost.input_tokens_details.cached_tokens == 30
        assert response.usage_with_cost.output_tokens_details.reasoning_tokens == 20

    def test_convert_usage_no_details(self) -> None:
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        result = convert_usage(usage)

        assert result.input_tokens_details.cached_tokens == 0
        assert result.output_tokens_details.reasoning_tokens == 0
