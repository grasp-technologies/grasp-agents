"""
Provider request/response behavior: structured-output schema gating, parallel
tool outputs, custom-provider threading, SDK retry defaults, context-window
error mapping, safety finish-reasons, and citation / thinking-block handling.
"""

from __future__ import annotations

from typing import Any

import litellm
import pytest
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    SignatureDelta,
    TextBlock,
    ThinkingBlock,
    ThinkingDelta,
)
from google.genai.types import Candidate, Content, FinishReason, Part
from google.genai.types import GenerateContentResponse as GeminiResponse
from pydantic import BaseModel

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm_providers.anthropic.llm_event_converters import (
    AnthropicStreamConverter,
)
from grasp_agents.llm_providers.anthropic.provider_output_to_response import (
    _anthropic_message_to_items,  # pyright: ignore[reportPrivateUsage]
)
from grasp_agents.llm_providers.anthropic.response_to_provider_inputs import (
    _output_message_to_blocks,  # pyright: ignore[reportPrivateUsage]
)
from grasp_agents.llm_providers.gemini.provider_output_to_response import (
    provider_output_to_response as gemini_output_to_response,
)
from grasp_agents.llm_providers.litellm.error_mapping import (
    map_api_error as litellm_map_api_error,
)
from grasp_agents.llm_providers.openai_responses.responses_llm import (
    OpenAIResponsesLLM,
    _items_after_last_response,  # pyright: ignore[reportPrivateUsage]
)
from grasp_agents.llm_providers.openai_responses.tool_converters import (
    to_api_tool as responses_to_api_tool,
)
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import OutputMessageText, UrlCitation
from grasp_agents.types.items import OutputMessageItem, ReasoningItem
from grasp_agents.types.llm_errors import LlmBadRequestError, LlmContextWindowError
from grasp_agents.types.llm_events import OutputItemDone
from tests.anthropic.test_converters import (
    _make_message,  # pyright: ignore[reportPrivateUsage]
)

# ---------- Item 9: responses to_api_tool strict gate ----------


class _OptionalInput(BaseModel):
    required_field: str
    optional_field: int | None = None


class _OptTool(BaseTool[_OptionalInput, Any, Any]):
    def __init__(self) -> None:
        super().__init__(name="opt", description="d")

    async def _run(self, inp: _OptionalInput, **kwargs: Any) -> str:
        return "ok"


class TestResponsesStrictGate:
    def test_non_strict_keeps_plain_schema(self) -> None:
        param = responses_to_api_tool(_OptTool(), strict=False)
        assert param["strict"] is False
        params_schema = param["parameters"]
        assert params_schema is not None
        assert params_schema.get("required") == ["required_field"]
        assert "additionalProperties" not in params_schema

    def test_strict_applies_strict_schema(self) -> None:
        param = responses_to_api_tool(_OptTool(), strict=True)
        assert param["strict"] is True
        params_schema = param["parameters"]
        assert params_schema is not None
        assert set(params_schema["required"]) == {"required_field", "optional_field"}
        assert params_schema["additionalProperties"] is False


# ---------- Item 10: previous_response_id input slicing ----------


class TestPreviousResponseInputSlice:
    def test_parallel_tool_outputs_all_kept(self) -> None:
        api_input: list[Any] = [
            {"role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "a", "name": "t", "arguments": "{}"},
            {"type": "function_call", "call_id": "b", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "1"},
            {"type": "function_call_output", "call_id": "b", "output": "2"},
        ]
        assert _items_after_last_response(api_input) == api_input[3:]

    def test_new_user_message(self) -> None:
        api_input: list[Any] = [
            {"role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "content": "hello"},
            {"role": "user", "content": "next"},
        ]
        assert _items_after_last_response(api_input) == api_input[2:]

    def test_no_model_output_returns_all(self) -> None:
        api_input: list[Any] = [{"role": "user", "content": "hi"}]
        assert _items_after_last_response(api_input) == api_input


# ---------- Item 11: responses provider honors api_provider ----------


class TestResponsesApiProvider:
    def test_custom_provider_threaded_into_client(self) -> None:
        llm = OpenAIResponsesLLM(
            model_name="gpt-4o-mini",
            api_provider=APIProvider(
                name="custom",
                base_url="https://example.com/v1",
                api_key="sk-test-123",
            ),
        )
        assert str(llm.client.base_url).startswith("https://example.com/v1")
        assert llm.client.api_key == "sk-test-123"

    def test_sdk_retries_default_zero(self) -> None:
        llm = OpenAIResponsesLLM(
            model_name="gpt-4o-mini",
            api_provider=APIProvider(name="openai", base_url=None, api_key="sk-x"),
        )
        assert llm.client.max_retries == 0


# ---------- Item 12: litellm context-window mapping order ----------


class TestLitellmContextWindowMapping:
    def test_context_window_error_maps_to_context_window(self) -> None:
        err = litellm.ContextWindowExceededError(
            message="context length exceeded",
            model="gpt-4o-mini",
            llm_provider="openai",
        )
        mapped = litellm_map_api_error(err)
        assert isinstance(mapped, LlmContextWindowError)

    def test_plain_bad_request_still_maps(self) -> None:
        err = litellm.BadRequestError(
            message="bad request",
            model="gpt-4o-mini",
            llm_provider="openai",
        )
        mapped = litellm_map_api_error(err)
        assert isinstance(mapped, LlmBadRequestError)
        assert not isinstance(mapped, LlmContextWindowError)


# ---------- Item 13: gemini abnormal finish reasons ----------


def _gemini_response(finish_reason: FinishReason) -> GeminiResponse:
    return GeminiResponse(
        response_id="r1",
        candidates=[
            Candidate(
                content=Content(role="model", parts=[Part(text="")]),
                finish_reason=finish_reason,
            )
        ],
    )


class TestGeminiFinishReasons:
    @pytest.mark.parametrize(
        "reason",
        [
            FinishReason.PROHIBITED_CONTENT,
            FinishReason.SPII,
            FinishReason.IMAGE_SAFETY,
        ],
    )
    def test_safety_reasons_map_to_content_filter(
        self, reason: FinishReason
    ) -> None:
        out = gemini_output_to_response(_gemini_response(reason))
        assert out.status == "incomplete"
        assert out.incomplete_details is not None
        assert out.incomplete_details.reason == "content_filter"

    @pytest.mark.parametrize(
        "reason",
        [
            FinishReason.MALFORMED_FUNCTION_CALL,
            FinishReason.UNEXPECTED_TOOL_CALL,
            FinishReason.LANGUAGE,
            FinishReason.OTHER,
        ],
    )
    def test_abnormal_reasons_not_completed(self, reason: FinishReason) -> None:
        out = gemini_output_to_response(_gemini_response(reason))
        assert out.status == "incomplete"

    def test_stop_still_completed(self) -> None:
        out = gemini_output_to_response(_gemini_response(FinishReason.STOP))
        assert out.status == "completed"


# ---------- Item 14a: citations stay with their own text block ----------


class TestAnthropicCitationsPerBlock:
    def test_citations_not_duplicated_across_blocks(self) -> None:
        cit1 = UrlCitation(
            type="url_citation", url="https://a.example", title="A", start_index=0,
            end_index=1,
        )
        cit2 = UrlCitation(
            type="url_citation", url="https://b.example", title="B", start_index=0,
            end_index=1,
        )
        item = OutputMessageItem(
            status="completed",
            content_parts=[
                OutputMessageText(text="first", citations=[cit1]),
                OutputMessageText(text="second", citations=[cit2]),
            ],
        )
        blocks = _output_message_to_blocks(item)
        assert len(blocks) == 2
        urls_per_block = [
            [c["url"] for c in (b.get("citations") or [])] for b in blocks
        ]
        assert urls_per_block == [["https://a.example"], ["https://b.example"]]


# ---------- Item 14b: thinking blocks keep their own signatures ----------


class TestAnthropicThinkingSignatures:
    def test_consecutive_thinking_blocks_stay_separate(self) -> None:
        msg = _make_message(
            [
                ThinkingBlock(type="thinking", thinking="first", signature="sig1"),
                ThinkingBlock(type="thinking", thinking="second", signature="sig2"),
                TextBlock(type="text", text="answer"),
            ]
        )
        items = _anthropic_message_to_items(msg)
        reasoning = [i for i in items if isinstance(i, ReasoningItem)]
        assert [r.encrypted_content for r in reasoning] == ["sig1", "sig2"]
        assert [r.summary_text for r in reasoning] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_streamed_thinking_blocks_without_stop_stay_separate(self) -> None:
        """
        Even on a malformed stream (no content_block_stop between thinking
        blocks), each block keeps its own signature.
        """
        events = [
            RawMessageStartEvent(
                type="message_start", message=_make_message([])
            ),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=ThinkingBlock(
                    type="thinking", thinking="", signature=""
                ),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=ThinkingDelta(type="thinking_delta", thinking="first"),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=SignatureDelta(type="signature_delta", signature="sig1"),
            ),
            # NOTE: no content_block_stop here — the merge-defect trigger.
            RawContentBlockStartEvent(
                type="content_block_start",
                index=1,
                content_block=ThinkingBlock(
                    type="thinking", thinking="", signature=""
                ),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=1,
                delta=ThinkingDelta(type="thinking_delta", thinking="second"),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=1,
                delta=SignatureDelta(type="signature_delta", signature="sig2"),
            ),
            RawContentBlockStopEvent(type="content_block_stop", index=1),
            RawMessageStopEvent(type="message_stop"),
        ]

        converter = AnthropicStreamConverter()

        async def stream() -> Any:
            for e in events:
                yield e

        collected = [e async for e in converter.convert(stream())]
        done_reasoning = [
            e.item
            for e in collected
            if isinstance(e, OutputItemDone) and isinstance(e.item, ReasoningItem)
        ]
        assert [r.encrypted_content for r in done_reasoning] == ["sig1", "sig2"]
