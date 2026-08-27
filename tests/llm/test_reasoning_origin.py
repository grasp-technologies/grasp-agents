"""
Reasoning origin: tag / stamp / filter reasoning items by provider format
identity (docs/plans/reasoning-origin.md).

* stamping — an untagged reasoning item gets this provider's origin on the
  non-stream and stream paths alike; other item types and a pre-existing
  origin are left untouched
* filtering — a foreign-origin reasoning item never reaches the wire; own
  and untagged items pass through verbatim, and the caller's input list is
  never mutated
* fallback rescue (the S1 fix) — a reasoning item tagged for the failed
  primary's origin does not poison the fallback member's request
* LiteLLM resolves its origin through the backend it proxies to; a provider
  with no reasoning-origin identity is a no-op filter
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ThinkingBlock
from anthropic.types import Usage as AnthropicUsage
from openai.types.responses import (
    Response as OpenAIResponse,
)
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import (
    Summary as OpenAIReasoningSummary,
)

from grasp_agents.llm.cloud_llm import ApiCallParams, APIProvider, CloudLLM
from grasp_agents.llm.fallback_llm import FallbackLLM
from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM
from grasp_agents.llm_providers.litellm.lite_llm import LiteLLM
from grasp_agents.llm_providers.openai_completions.completions_llm import (
    OpenAILLM,
)
from grasp_agents.llm_providers.openai_responses.responses_llm import OpenAIResponsesLLM
from grasp_agents.types.content import OutputMessageText, ReasoningSummary
from grasp_agents.types.items import (
    InputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_errors import LlmInternalServerError
from grasp_agents.types.llm_events import LlmEvent, ResponseCompleted
from grasp_agents.types.response import Response
from tests.llm.test_resilience import ErrorLLM, LazyStreamCloudLLM, _resp

_ANTHROPIC_PROVIDER = APIProvider(name="anthropic", base_url=None, api_key="test-key")
_OPENAI_PROVIDER = APIProvider(name="openai", base_url=None, api_key="test-key")

_USER_MSG = [InputMessageItem.from_text("go", role="user")]


# ---------- Helpers: real SDK payloads ----------


def _anthropic_message(content: list[Any]) -> AnthropicMessage:
    return AnthropicMessage(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=content,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=10, output_tokens=5),
    )


def _openai_response(output: list[Any]) -> OpenAIResponse:
    return OpenAIResponse(
        id="resp_test",
        created_at=0,
        model="gpt-5.1",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _openai_message_item(text: str = "ok") -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_test",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _openai_reasoning_item(
    encrypted_content: str, text: str = "thinking"
) -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id="rs_test",
        type="reasoning",
        summary=[OpenAIReasoningSummary(type="summary_text", text=text)],
        encrypted_content=encrypted_content,
    )


def _anthropic_thinking_blocks(messages: list[Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            blocks.extend(b for b in content if b.get("type") == "thinking")
    return blocks


# ---------- Helpers: stub LLMs ----------


@dataclass(frozen=True)
class _StubAnthropicLLM(AnthropicLLM):
    """Real Anthropic converters/request-building; only the wire call is stubbed."""

    served: Any = None
    captured_api_input: list[Any] = field(default_factory=list)

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: Any | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        object.__setattr__(self, "captured_api_input", api_input)
        return self.served


@dataclass(frozen=True)
class _ServedAnthropicLLM(AnthropicLLM):
    """Serves an already-built internal Response, bypassing conversion entirely."""

    served: Response = field(default_factory=lambda: Response(model="mock", output=[]))

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: Any | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        return self.served

    def _convert_api_response(self, raw: Any) -> Response:
        return raw


@dataclass(frozen=True)
class _StubOpenAIResponsesLLM(OpenAIResponsesLLM):
    """Real Responses converters/request-building; only the wire call is stubbed."""

    served: Any = None
    captured_api_input: list[Any] = field(default_factory=list)

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: Any | None = None,
        previous_response_id: str | None = None,
        conversation: Any | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        object.__setattr__(self, "captured_api_input", api_input)
        return self.served


@dataclass(frozen=True)
class _StreamStampingLLM(CloudLLM):
    """
    Minimal CloudLLM whose stream directly yields a ``ResponseCompleted`` —
    exercises ``_finalize_response`` on the streaming path (``ResponseCompleted``
    handling in ``_generate_response_stream_once``) without a provider-specific
    event grammar.
    """

    _native_provider_name: ClassVar[str] = "openai"

    served: Response = field(default_factory=lambda: Response(model="mock", output=[]))

    def _make_api_input(
        self,
        input: Sequence[InputItem],
        tools: Any = None,
        tool_choice: Any = None,
        output_schema: Any = None,
        **extra_llm_settings: Any,
    ) -> ApiCallParams:
        return {"api_input": list(input)}

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: Any | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        raise NotImplementedError

    def _convert_api_response(self, raw: Any) -> Response:
        raise NotImplementedError

    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: Any | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[Any]:
        async def iterator() -> AsyncIterator[Any]:
            yield "done"

        return iterator()

    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[LlmEvent]:
        async for _ in api_stream:
            yield ResponseCompleted(response=self.served, sequence_number=1)


# ---------- 1. Stamping ----------


class TestStamping:
    @pytest.mark.asyncio
    async def test_anthropic_thinking_block_stamped_anthropic_origin(self) -> None:
        message = _anthropic_message(
            [
                ThinkingBlock(type="thinking", thinking="hmm", signature="fake-sig"),
                TextBlock(type="text", text="answer"),
            ]
        )
        llm = _StubAnthropicLLM(
            model_name="claude-sonnet-4-5",
            api_provider=_ANTHROPIC_PROVIDER,
            served=message,
        )

        response = await llm.generate_response(_USER_MSG)

        reasoning = next(i for i in response.output if isinstance(i, ReasoningItem))
        assert reasoning.origin == "anthropic"
        message_item = next(
            i for i in response.output if isinstance(i, OutputMessageItem)
        )
        assert message_item.text == "answer"

    @pytest.mark.asyncio
    async def test_openai_responses_reasoning_item_stamped_openai_origin(
        self,
    ) -> None:
        raw = _openai_response(
            [_openai_reasoning_item("enc-abc"), _openai_message_item("hi")]
        )
        llm = _StubOpenAIResponsesLLM(
            model_name="gpt-5.1", api_provider=_OPENAI_PROVIDER, served=raw
        )

        response = await llm.generate_response(_USER_MSG)

        reasoning = next(i for i in response.output if isinstance(i, ReasoningItem))
        assert reasoning.origin == "openai"
        message_item = next(
            i for i in response.output if isinstance(i, OutputMessageItem)
        )
        assert message_item.text == "hi"

    @pytest.mark.asyncio
    async def test_streamed_reasoning_item_stamped_on_completion(self) -> None:
        served = Response(
            model="mock",
            output=[ReasoningItem(summary=[ReasoningSummary(text="thinking")])],
        )
        llm = _StreamStampingLLM(model_name="mock", served=served)

        events = [e async for e in llm.generate_response_stream(_USER_MSG)]

        completed = next(e for e in events if isinstance(e, ResponseCompleted))
        reasoning = next(
            i for i in completed.response.output if isinstance(i, ReasoningItem)
        )
        assert reasoning.origin == "openai"

    @pytest.mark.asyncio
    async def test_pretagged_origin_never_overwritten(self) -> None:
        served = Response(
            model="claude-sonnet-4-5",
            output=[
                ReasoningItem(
                    origin="gemini", summary=[ReasoningSummary(text="pre-tagged")]
                ),
                OutputMessageItem(
                    status="completed", content=[OutputMessageText(text="hi")]
                ),
            ],
        )
        llm = _ServedAnthropicLLM(
            model_name="claude-sonnet-4-5",
            api_provider=_ANTHROPIC_PROVIDER,
            served=served,
        )

        response = await llm.generate_response(_USER_MSG)

        reasoning = next(i for i in response.output if isinstance(i, ReasoningItem))
        assert reasoning.origin == "gemini"


# ---------- 2. Filtering ----------


class TestFiltering:
    @pytest.mark.asyncio
    async def test_anthropic_drops_foreign_keeps_own_and_untagged(self) -> None:
        llm = _StubAnthropicLLM(
            model_name="claude-sonnet-4-5",
            api_provider=_ANTHROPIC_PROVIDER,
            served=_anthropic_message([TextBlock(type="text", text="ok")]),
        )
        history: list[InputItem] = [
            InputMessageItem.from_text("hi"),
            ReasoningItem(origin="openai", encrypted_content="foreign"),
            ReasoningItem(
                origin="anthropic",
                encrypted_content="own-sig",
                summary=[ReasoningSummary(text="own thoughts")],
            ),
            ReasoningItem(
                encrypted_content="untagged-sig",
                summary=[ReasoningSummary(text="untagged thoughts")],
            ),
            OutputMessageItem(
                status="completed", content=[OutputMessageText(text="done")]
            ),
            InputMessageItem.from_text("more"),
        ]
        before_len = len(history)

        await llm.generate_response(history)

        signatures = [
            b["signature"] for b in _anthropic_thinking_blocks(llm.captured_api_input)
        ]
        assert signatures == ["own-sig", "untagged-sig"]
        assert len(history) == before_len

    @pytest.mark.asyncio
    async def test_openai_responses_drops_foreign_keeps_own_and_untagged(self) -> None:
        llm = _StubOpenAIResponsesLLM(
            model_name="gpt-5.1",
            api_provider=_OPENAI_PROVIDER,
            served=_openai_response([_openai_message_item("ok")]),
        )
        history: list[InputItem] = [
            InputMessageItem.from_text("hi"),
            ReasoningItem(origin="anthropic", encrypted_content="foreign"),
            ReasoningItem(
                origin="openai",
                encrypted_content="own-sig",
                summary=[ReasoningSummary(text="own")],
            ),
            ReasoningItem(
                encrypted_content="untagged-sig",
                summary=[ReasoningSummary(text="untagged")],
            ),
            OutputMessageItem(
                status="completed", content=[OutputMessageText(text="done")]
            ),
            InputMessageItem.from_text("more"),
        ]
        before_len = len(history)

        await llm.generate_response(history)

        captured = llm.captured_api_input
        reasoning_dicts = [
            d for d in captured if isinstance(d, dict) and d.get("type") == "reasoning"
        ]
        assert [d["encrypted_content"] for d in reasoning_dicts] == [
            "own-sig",
            "untagged-sig",
        ]
        assert not any("origin" in d for d in captured if isinstance(d, dict))
        assert len(history) == before_len


# ---------- 3. Fallback rescue (the S1 fix) ----------


class TestFallbackRescue:
    @pytest.mark.asyncio
    async def test_fallback_filters_its_own_foreign_reasoning(self) -> None:
        primary = ErrorLLM(
            model_name="primary",
            retry_policy=None,
            error_to_raise=LlmInternalServerError(
                "down", response=_resp(503), body=None
            ),
        )
        fallback = _StubAnthropicLLM(
            model_name="claude-sonnet-4-5",
            api_provider=_ANTHROPIC_PROVIDER,
            served=_anthropic_message([TextBlock(type="text", text="rescued")]),
        )
        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))
        history: list[InputItem] = [
            InputMessageItem.from_text("hi"),
            ReasoningItem(origin="openai", encrypted_content="foreign"),
        ]

        result = await llm.generate_response(history)

        assert result.output_text == "rescued"
        assert _anthropic_thinking_blocks(fallback.captured_api_input) == []


# ---------- 4. Origin resolution ----------


class TestOriginResolution:
    def test_openai_dialects_share_one_origin(self) -> None:
        responses = OpenAIResponsesLLM(
            model_name="gpt-5.6-luna", api_provider=_OPENAI_PROVIDER
        )
        completions = OpenAILLM(
            model_name="gpt-5.6-luna", api_provider=_OPENAI_PROVIDER
        )

        assert responses._resolve_reasoning_origin() == "openai"
        assert completions._resolve_reasoning_origin() == "openai"

    def test_custom_endpoint_does_not_claim_the_vendor_origin(self) -> None:
        llm = OpenAILLM(
            model_name="anthropic/claude-opus-5",
            api_provider=APIProvider(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="or-key",
            ),
        )

        assert llm._resolve_reasoning_origin() == "openrouter"

    def test_litellm_origin_is_backend_prefixed(self) -> None:
        llm = LiteLLM(
            model_name="openai/some-proxy-model",
            api_provider=APIProvider(
                name="custom-proxy",
                base_url="http://localhost:8000/v1",
                api_key="proxy-key",
            ),
        )
        assert llm._resolve_reasoning_origin() == "litellm:custom-proxy"

    def test_provider_without_reasoning_origin_filters_nothing(self) -> None:
        llm = LazyStreamCloudLLM(model_name="mock")
        tagged = [
            ReasoningItem(origin="anthropic", summary=[ReasoningSummary(text="x")])
        ]

        assert llm._resolve_reasoning_origin() is None
        assert llm._filter_foreign_reasoning(tagged) is tagged
