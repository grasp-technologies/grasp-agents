"""LiteLLM stream converter: thought-signature distribution over a stream."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)

from grasp_agents.llm_providers.litellm.llm_event_converters import (
    LiteLLMStreamConverter,
)
from grasp_agents.types.items import FunctionToolCallItem, ReasoningItem
from grasp_agents.types.llm_events import (
    OutputItemDone,
    ResponseCompleted,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from grasp_agents.types.llm_events import LlmEvent

_CHUNK_ID = "chatcmpl-litellm-test"
_MODEL = "gemini-2.5-flash"
_CREATED = 1700000000


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[ChatCompletionDeltaToolCall] | None = None,
    finish_reason: str | None = None,
    provider_specific_fields: dict[str, Any] | None = None,
) -> ModelResponseStream:
    delta = Delta(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )
    if provider_specific_fields is not None:
        delta.provider_specific_fields = provider_specific_fields

    chunk = ModelResponseStream(
        id=_CHUNK_ID,
        model=_MODEL,
        choices=[StreamingChoices(delta=delta, finish_reason=finish_reason)],
    )
    object.__setattr__(chunk, "created", _CREATED)
    return chunk


def _tool_call_delta(
    *, call_id: str, name: str, arguments: str
) -> ChatCompletionDeltaToolCall:
    return ChatCompletionDeltaToolCall(
        id=call_id,
        index=0,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


async def _collect(chunks: list[ModelResponseStream]) -> list[LlmEvent]:
    async def _stream() -> AsyncIterator[ModelResponseStream]:
        for chunk in chunks:
            yield chunk

    converter = LiteLLMStreamConverter()
    return [event async for event in converter.convert(_stream())]  # type: ignore[arg-type]


def _done_items(events: list[LlmEvent], typ: type[Any]) -> list[Any]:
    return [
        e.item
        for e in events
        if isinstance(e, OutputItemDone) and isinstance(e.item, typ)
    ]


def _final_output(events: list[LlmEvent]) -> list[Any]:
    completed = [e for e in events if isinstance(e, ResponseCompleted)]
    assert len(completed) == 1
    return list(completed[0].response.output)


class TestThoughtSignaturePatching:
    """
    ``thought_signatures`` arrive in provider_specific_fields after the items
    they belong to have already been streamed out.
    """

    @pytest.mark.asyncio
    async def test_signature_reaches_already_streamed_reasoning_item(self) -> None:
        events = await _collect(
            [
                _chunk(reasoning_content="thinking"),
                _chunk(
                    content="answer",
                    provider_specific_fields={"thought_signatures": ["SIG_ABC"]},
                ),
                _chunk(finish_reason="stop"),
            ]
        )

        streamed = _done_items(events, ReasoningItem)
        assert len(streamed) == 1
        assert streamed[0].encrypted_content == "SIG_ABC"

        final = [o for o in _final_output(events) if isinstance(o, ReasoningItem)]
        assert len(final) == 1
        assert final[0] is streamed[0]

    @pytest.mark.asyncio
    async def test_signature_applied_to_reasoning_still_open_at_stream_end(
        self,
    ) -> None:
        events = await _collect(
            [
                _chunk(reasoning_content="thinking"),
                _chunk(
                    finish_reason="stop",
                    provider_specific_fields={"thought_signatures": ["SIG_Z"]},
                ),
            ]
        )

        streamed = _done_items(events, ReasoningItem)
        assert len(streamed) == 1
        assert streamed[0].encrypted_content == "SIG_Z"

        final = [o for o in _final_output(events) if isinstance(o, ReasoningItem)]
        assert len(final) == 1
        assert final[0] is streamed[0]

    @pytest.mark.asyncio
    async def test_signatures_split_between_reasoning_and_tool_call(self) -> None:
        events = await _collect(
            [
                _chunk(reasoning_content="thinking"),
                _chunk(
                    tool_calls=[
                        _tool_call_delta(
                            call_id="call_1", name="add", arguments='{"a": 1, "b": 2}'
                        )
                    ],
                    provider_specific_fields={"thought_signatures": ["SIG_R", "SIG_T"]},
                ),
                _chunk(finish_reason="tool_calls"),
            ]
        )

        reasoning = _done_items(events, ReasoningItem)
        assert len(reasoning) == 1
        assert reasoning[0].encrypted_content == "SIG_R"

        tool_calls = _done_items(events, FunctionToolCallItem)
        assert len(tool_calls) == 1
        assert tool_calls[0].provider_specific_fields == {"thought_signature": "SIG_T"}

        final = _final_output(events)
        assert next(o for o in final if isinstance(o, ReasoningItem)) is reasoning[0]
        assert (
            next(o for o in final if isinstance(o, FunctionToolCallItem))
            is tool_calls[0]
        )

    @pytest.mark.asyncio
    async def test_fewer_signatures_than_items_leaves_the_rest_unsigned(self) -> None:
        events = await _collect(
            [
                _chunk(reasoning_content="first"),
                _chunk(content="mid"),
                _chunk(reasoning_content="second"),
                _chunk(
                    content="answer",
                    provider_specific_fields={"thought_signatures": ["ONLY_SIG"]},
                ),
                _chunk(finish_reason="stop"),
            ]
        )

        reasoning = _done_items(events, ReasoningItem)
        assert len(reasoning) == 2
        assert reasoning[0].encrypted_content == "ONLY_SIG"
        assert reasoning[1].encrypted_content is None
