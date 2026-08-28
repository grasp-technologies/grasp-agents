"""
Item origin: stamp and drop origin-bound items by the vendor of the client that
produced them.

* stamping — an untagged reasoning item gets this client's origin on the
  non-stream and stream paths alike, as does a message or tool call carrying a
  thought signature; items without a signature and a pre-existing origin are
  left untouched
* dropping — a foreign-origin reasoning item never reaches the wire; a
  foreign-origin signed message or tool call is forwarded without its
  signature; own and untagged items pass through verbatim, and neither the
  caller's input list nor its items are ever mutated
* fallback rescue — a reasoning item tagged for the failed primary's origin
  does not poison the fallback member's request
* resolution — the origin is the client class's vendor, whatever model name,
  endpoint or platform is in play; a class with no vendor of its own has none,
  which stamps and drops nothing
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ThinkingBlock
from anthropic.types import Usage as AnthropicUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
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

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm.fallback_llm import FallbackLLM
from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM
from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM
from grasp_agents.llm_providers.litellm.lite_llm import LiteLLM
from grasp_agents.llm_providers.openai_completions.completions_llm import (
    OpenAILLM,
)
from grasp_agents.llm_providers.openai_responses.responses_llm import OpenAIResponsesLLM
from grasp_agents.types.content import OutputMessageText, ReasoningSummary
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_errors import LlmInternalServerError
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemDone,
    ResponseCompleted,
    ResponseIncomplete,
)
from grasp_agents.types.response import Response
from tests.llm.test_resilience import (
    _USER_MSG,
    ErrorLLM,
    LazyStreamCloudLLM,
    _resp,
)

_ANTHROPIC_PROVIDER = APIProvider(name="anthropic", base_url=None, api_key="test-key")
_OPENAI_PROVIDER = APIProvider(name="openai", base_url=None, api_key="test-key")
_GEMINI_PROVIDER = APIProvider(name="gemini", base_url=None, api_key="test-key")
_OPENROUTER_PROVIDER = APIProvider(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key="test-key",
)


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


def _chat_completion(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        created=0,
        model="gpt-5.1",
        object="chat.completion",
        choices=[
            ChatCompletionChoice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
    )


def _anthropic_thinking_blocks(messages: list[Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            blocks.extend(b for b in content if b.get("type") == "thinking")
    return blocks


def _first_reasoning(items: Iterable[Any]) -> ReasoningItem:
    return next(i for i in items if isinstance(i, ReasoningItem))


def _signed_message(origin: str | None) -> OutputMessageItem:
    return OutputMessageItem(
        status="completed",
        origin=origin,
        content=[OutputMessageText(text="calling add")],
        provider_specific_fields={"thought_signature": "msg-sig", "other": 1},
    )


def _signed_tool_call(origin: str | None) -> FunctionToolCallItem:
    return FunctionToolCallItem(
        call_id="call_1",
        name="add",
        arguments='{"a": 1, "b": 2}',
        origin=origin,
        provider_specific_fields={"thought_signature": "fc-sig"},
    )


def _signed_turn(origin: str | None) -> list[InputItem]:
    """
    An assistant turn whose message and tool call carry a thought signature.

    Gemini signs regular text and function-call parts rather than thinking
    parts, so this is where a cross-provider replay leaks.
    """
    return [
        InputMessageItem.from_text("hi"),
        _signed_message(origin),
        _signed_tool_call(origin),
        FunctionToolOutputItem(call_id="call_1", output="3"),
    ]


def _signatures(items: Iterable[Any]) -> list[Any]:
    return [
        (item.provider_specific_fields or {}).get("thought_signature")
        for item in items
        if isinstance(item, (OutputMessageItem, FunctionToolCallItem))
    ]


def _history_with_reasoning(*, foreign: str, own: str) -> list[InputItem]:
    """A turn carrying foreign, own-origin and untagged reasoning."""
    return [
        InputMessageItem.from_text("hi"),
        ReasoningItem(origin=foreign, encrypted_content="foreign-sig"),
        ReasoningItem(
            origin=own,
            encrypted_content="own-sig",
            summary=[ReasoningSummary(text="own")],
        ),
        ReasoningItem(
            encrypted_content="untagged-sig",
            summary=[ReasoningSummary(text="untagged")],
        ),
        OutputMessageItem(status="completed", content=[OutputMessageText(text="done")]),
        InputMessageItem.from_text("more"),
    ]


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
class _StubOpenAICompletionsLLM(OpenAILLM):
    """Real Completions converters/request-building; only the wire call is stubbed."""

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
class _StreamingLLM(LazyStreamCloudLLM):
    """
    Streams a reasoning item the way real converters do: the object delivered
    by ``OutputItemDone`` is a different one from its counterpart in the
    terminal response, so stamping the response alone would leave the item the
    caller actually consumes untagged.
    """

    _native_provider_name: ClassVar[str] = "openai"

    terminal_incomplete: bool = False

    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[LlmEvent]:
        async for _ in api_stream:
            yield OutputItemDone(
                item=ReasoningItem(summary=[ReasoningSummary(text="thinking")]),
                output_index=0,
                sequence_number=1,
            )
            response = Response(
                model=self.model_name,
                output=[ReasoningItem(summary=[ReasoningSummary(text="thinking")])],
            )
            if self.terminal_incomplete:
                yield ResponseIncomplete(response=response, sequence_number=2)
            else:
                yield ResponseCompleted(response=response, sequence_number=2)


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

        assert _first_reasoning(response.output).origin == "anthropic"
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

        assert _first_reasoning(response.output).origin == "openai"
        message_item = next(
            i for i in response.output if isinstance(i, OutputMessageItem)
        )
        assert message_item.text == "hi"

    @pytest.mark.asyncio
    async def test_streamed_item_stamped_as_it_is_emitted(self) -> None:
        llm = _StreamingLLM(model_name="gpt-5.1", fail_attempts=0)

        events = [e async for e in llm.generate_response_stream(_USER_MSG)]

        streamed = next(e for e in events if isinstance(e, OutputItemDone))
        assert _first_reasoning([streamed.item]).origin == "openai"
        completed = next(e for e in events if isinstance(e, ResponseCompleted))
        assert _first_reasoning(completed.response.output).origin == "openai"

    @pytest.mark.asyncio
    async def test_incomplete_stream_is_finalized_like_a_completed_one(self) -> None:
        llm = _StreamingLLM(
            model_name="gpt-5.1", fail_attempts=0, terminal_incomplete=True
        )

        events = [e async for e in llm.generate_response_stream(_USER_MSG)]

        incomplete = next(e for e in events if isinstance(e, ResponseIncomplete))
        assert _first_reasoning(incomplete.response.output).origin == "openai"
        streamed = next(e for e in events if isinstance(e, OutputItemDone))
        assert _first_reasoning([streamed.item]).origin == "openai"

    def test_pretagged_origin_never_overwritten(self) -> None:
        llm = AnthropicLLM(
            model_name="claude-sonnet-4-5", api_provider=_ANTHROPIC_PROVIDER
        )
        response = Response(
            model="claude-sonnet-4-5",
            output=[
                ReasoningItem(
                    origin="gemini", summary=[ReasoningSummary(text="pre-tagged")]
                ),
                ReasoningItem(summary=[ReasoningSummary(text="untagged")]),
            ],
        )

        llm._stamp_origin(response)

        assert [i.origin for i in response.output] == ["gemini", "anthropic"]


# ---------- 2. Dropping foreign reasoning ----------


class TestDroppingForeignReasoning:
    @pytest.mark.asyncio
    async def test_anthropic_drops_foreign_keeps_own_and_untagged(self) -> None:
        llm = _StubAnthropicLLM(
            model_name="claude-sonnet-4-5",
            api_provider=_ANTHROPIC_PROVIDER,
            served=_anthropic_message([TextBlock(type="text", text="ok")]),
        )
        history = _history_with_reasoning(foreign="openai", own="anthropic")
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
        history = _history_with_reasoning(foreign="anthropic", own="openai")
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


# ---------- 3. Thought signatures ----------


class TestThoughtSignatures:
    def test_signed_message_and_tool_call_are_stamped(self) -> None:
        llm = GeminiLLM(model_name="gemini-3-pro", api_provider=_GEMINI_PROVIDER)
        message = _signed_message(origin=None)
        tool_call = _signed_tool_call(origin=None)
        unsigned = OutputMessageItem(
            status="completed", content=[OutputMessageText(text="plain")]
        )

        llm._stamp_origin(
            Response(model="gemini-3-pro", output=[message, tool_call, unsigned])
        )

        assert message.origin == "gemini"
        assert tool_call.origin == "gemini"
        # An unsigned item has nothing a backend must verify, so it stays free
        # to replay anywhere.
        assert unsigned.origin is None

    def test_signed_item_origin_never_overwritten(self) -> None:
        llm = OpenAILLM(model_name="gpt-5.1", api_provider=_OPENAI_PROVIDER)
        message = _signed_message(origin="gemini")

        llm._stamp_origin(Response(model="gpt-5.1", output=[message]))

        assert message.origin == "gemini"

    def test_foreign_signatures_stripped_without_touching_history(self) -> None:
        llm = OpenAILLM(model_name="gpt-5.1", api_provider=_OPENAI_PROVIDER)
        history = _signed_turn(origin="gemini")

        filtered = llm._drop_foreign_reasoning(history)

        # Dropping either item would break tool-call pairing, so both survive.
        assert len(filtered) == len(history)
        assert [type(i) for i in filtered] == [type(i) for i in history]
        assert _signatures(filtered) == [None, None]
        # Non-signature provider data rides along untouched.
        message = filtered[1]
        assert isinstance(message, OutputMessageItem)
        assert message.provider_specific_fields == {"other": 1}
        # The transcript keeps its signatures for a later Gemini turn.
        assert _signatures(history) == ["msg-sig", "fc-sig"]
        assert filtered[1] is not history[1]

    def test_own_origin_keeps_signatures(self) -> None:
        llm = GeminiLLM(model_name="gemini-3-pro", api_provider=_GEMINI_PROVIDER)
        history = _signed_turn(origin="gemini")

        filtered = llm._drop_foreign_reasoning(history)

        assert _signatures(filtered) == ["msg-sig", "fc-sig"]
        assert all(a is b for a, b in zip(filtered, history, strict=True))

    def test_untagged_signatures_are_kept(self) -> None:
        llm = OpenAILLM(model_name="gpt-5.1", api_provider=_OPENAI_PROVIDER)
        history = _signed_turn(origin=None)

        filtered = llm._drop_foreign_reasoning(history)

        assert _signatures(filtered) == ["msg-sig", "fc-sig"]
        assert all(a is b for a, b in zip(filtered, history, strict=True))

    @pytest.mark.asyncio
    async def test_foreign_signature_never_reaches_the_completions_wire(self) -> None:
        llm = _StubOpenAICompletionsLLM(
            model_name="gpt-5.1",
            api_provider=_OPENAI_PROVIDER,
            served=_chat_completion("ok"),
        )
        history = _signed_turn(origin="gemini")

        await llm.generate_response(history)

        assistant = [m for m in llm.captured_api_input if m.get("role") == "assistant"]
        assert assistant
        for message in assistant:
            assert "thought_signature" not in str(message)
        assert _signatures(history) == ["msg-sig", "fc-sig"]


# ---------- 4. Fallback rescue ----------


class TestFallbackRescue:
    @pytest.mark.asyncio
    async def test_fallback_drops_its_own_foreign_reasoning(self) -> None:
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


# ---------- 5. Origin resolution ----------


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

    def test_origin_is_the_client_vendor_regardless_of_endpoint(self) -> None:
        llm = OpenAILLM(
            model_name="gpt-5.6-luna",
            api_provider=APIProvider(
                name="proxy",
                base_url="https://proxy.example.com/v1",
                api_key="proxy-key",
            ),
        )

        assert llm._resolve_reasoning_origin() == "openai"

    def test_gateway_served_model_takes_the_client_vendor(self) -> None:
        llm = OpenAILLM(
            model_name="google/gemma-4-31b-it", api_provider=_OPENROUTER_PROVIDER
        )

        assert llm._resolve_reasoning_origin() == "openai"

    def test_litellm_leaves_every_model_untagged(self) -> None:
        for model_name in (
            "anthropic/claude-sonnet-4-5",
            "gemini/gemini-2.5-pro",
            "gpt-5.1",
        ):
            assert LiteLLM(model_name=model_name)._resolve_reasoning_origin() is None

        proxied = LiteLLM(
            model_name="openai/some-proxy-model",
            api_provider=APIProvider(
                name="custom-proxy",
                base_url="http://localhost:8000/v1",
                api_key="proxy-key",
            ),
        )

        assert proxied._resolve_reasoning_origin() is None

    def test_llm_without_reasoning_origin_drops_nothing(self) -> None:
        llm = LazyStreamCloudLLM(model_name="mock")
        tagged = [
            ReasoningItem(origin="anthropic", summary=[ReasoningSummary(text="x")])
        ]

        assert llm._resolve_reasoning_origin() is None
        assert llm._drop_foreign_reasoning(tagged) is tagged
