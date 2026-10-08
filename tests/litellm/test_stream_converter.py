"""LiteLLM stream converter: Gemini thought signatures over a stream."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
)
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
from grasp_agents.llm_providers.openai_completions.response_to_provider_inputs import (
    items_to_provider_inputs,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_events import OutputItemDone, ResponseCompleted

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from grasp_agents.types.llm_events import LlmEvent

_CHUNK_ID = "chatcmpl-litellm-test"
_MODEL = "gemini-3.1-pro-preview"
_CREATED = 1700000000

_SIG = "EogGCoUGAR-thought-signature"
_SIG_2 = "EqwFCqkFAR-second-signature"


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
    *,
    index: int = 0,
    call_id: str,
    name: str,
    arguments: str,
    provider_specific_fields: dict[str, Any] | None = None,
) -> ChatCompletionDeltaToolCall:
    tool_call = ChatCompletionDeltaToolCall(
        id=call_id,
        index=index,
        type="function",
        function=Function(name=name, arguments=arguments),
    )
    if provider_specific_fields is not None:
        tool_call.provider_specific_fields = provider_specific_fields
    return tool_call


def _signed_call_chunk(
    *,
    index: int = 0,
    call_id: str,
    name: str,
    arguments: str,
    sig: str,
) -> ModelResponseStream:
    """A tool call as LiteLLM streams it for Gemini: signed on the call and delta."""
    return _chunk(
        tool_calls=[
            _tool_call_delta(
                index=index,
                call_id=f"{call_id}__thought__{sig}",
                name=name,
                arguments=arguments,
                provider_specific_fields={"thought_signature": sig},
            )
        ],
        provider_specific_fields={"thought_signatures": [sig]},
    )


async def _collect(
    chunks: list[ModelResponseStream],
) -> tuple[list[LlmEvent], list[OutputItem]]:
    """Run the converter; also snapshot every item at the moment it is emitted."""

    async def _stream() -> AsyncIterator[ModelResponseStream]:
        for chunk in chunks:
            yield chunk

    converter = LiteLLMStreamConverter()
    events: list[LlmEvent] = []
    emitted: list[OutputItem] = []
    async for event in converter.convert(_stream()):
        events.append(event)
        if isinstance(event, OutputItemDone):
            emitted.append(event.item.model_copy(deep=True))
    return events, emitted


def _final_output(events: list[LlmEvent]) -> list[OutputItem]:
    completed = [e for e in events if isinstance(e, ResponseCompleted)]
    assert len(completed) == 1
    return list(completed[0].response.output)


def _reasoning(items: Sequence[OutputItem]) -> list[ReasoningItem]:
    return [i for i in items if isinstance(i, ReasoningItem)]


def _messages(items: Sequence[OutputItem]) -> list[OutputMessageItem]:
    return [i for i in items if isinstance(i, OutputMessageItem)]


def _tool_calls(items: Sequence[OutputItem]) -> list[FunctionToolCallItem]:
    return [i for i in items if isinstance(i, FunctionToolCallItem)]


def _text_answer_chunks() -> list[ModelResponseStream]:
    return [
        _chunk(reasoning_content="**Considering the basics**"),
        _chunk(content="The sky is"),
        _chunk(content=" blue."),
        _chunk(provider_specific_fields={"thought_signatures": [_SIG]}),
        _chunk(finish_reason="stop"),
    ]


def _tool_call_chunks() -> list[ModelResponseStream]:
    return [
        _chunk(reasoning_content="**Initiating the calculation**"),
        _signed_call_chunk(
            call_id="call_1", name="add", arguments='{"a": 17, "b": 25}', sig=_SIG
        ),
        _chunk(finish_reason="tool_calls"),
    ]


class TestThoughtSignatures:
    """
    Chunk shapes are the ones LiteLLM emits for Gemini: a tool call's
    signature arrives in the call's own chunk (on the delta and on the call),
    a text answer's signature in a trailing chunk of its own after the text,
    and only the first of several parallel calls is signed.
    """

    @pytest.mark.asyncio
    async def test_text_answer_signature_lands_on_message(self) -> None:
        _, emitted = await _collect(_text_answer_chunks())

        (message,) = _messages(emitted)
        assert message.provider_specific_fields == {"thought_signatures": [_SIG]}
        (reasoning,) = _reasoning(emitted)
        assert reasoning.encrypted_content is None

    @pytest.mark.asyncio
    async def test_tool_call_signature_lands_on_tool_call(self) -> None:
        _, emitted = await _collect(_tool_call_chunks())

        (call,) = _tool_calls(emitted)
        assert call.provider_specific_fields == {"thought_signature": _SIG}
        assert call.call_id == "call_1"
        (reasoning,) = _reasoning(emitted)
        assert reasoning.encrypted_content is None

    @pytest.mark.asyncio
    async def test_signature_embedded_in_call_id_is_split_out(self) -> None:
        # LiteLLM also encodes the signature into the id; other providers
        # reject such ids, so it must travel in provider_specific_fields.
        _, emitted = await _collect(
            [
                _chunk(
                    tool_calls=[
                        _tool_call_delta(
                            call_id=f"call_1__thought__{_SIG}",
                            name="add",
                            arguments='{"a": 1, "b": 2}',
                        )
                    ]
                ),
                _chunk(finish_reason="tool_calls"),
            ]
        )

        (call,) = _tool_calls(emitted)
        assert call.call_id == "call_1"
        assert call.provider_specific_fields == {"thought_signature": _SIG}

    @pytest.mark.asyncio
    async def test_other_fields_on_the_call_are_kept(self) -> None:
        _, emitted = await _collect(
            [
                _chunk(
                    tool_calls=[
                        _tool_call_delta(
                            call_id=f"call_1__thought__{_SIG}",
                            name="add",
                            arguments='{"a": 1, "b": 2}',
                            provider_specific_fields={"other": "x"},
                        )
                    ]
                ),
                _chunk(finish_reason="tool_calls"),
            ]
        )

        (call,) = _tool_calls(emitted)
        assert call.provider_specific_fields == {
            "other": "x",
            "thought_signature": _SIG,
        }

    @pytest.mark.asyncio
    async def test_only_first_parallel_call_is_signed(self) -> None:
        _, emitted = await _collect(
            [
                _chunk(reasoning_content="**Planning both calls**"),
                _signed_call_chunk(
                    call_id="call_1", name="add", arguments='{"a": 2, "b": 3}', sig=_SIG
                ),
                _chunk(
                    tool_calls=[
                        _tool_call_delta(
                            index=1,
                            call_id="call_2",
                            name="multiply",
                            arguments='{"a": 4, "b": 5}',
                        )
                    ]
                ),
                _chunk(finish_reason="tool_calls"),
            ]
        )

        calls = _tool_calls(emitted)
        assert [c.name for c in calls] == ["add", "multiply"]
        assert [c.provider_specific_fields for c in calls] == [
            {"thought_signature": _SIG},
            None,
        ]

    @pytest.mark.asyncio
    async def test_text_then_tool_call_signs_the_call(self) -> None:
        _, emitted = await _collect(
            [
                _chunk(reasoning_content="**Initiating**"),
                _chunk(content="I will now compute 17 + 25."),
                _signed_call_chunk(
                    call_id="call_1",
                    name="add",
                    arguments='{"a": 17, "b": 25}',
                    sig=_SIG,
                ),
                _chunk(finish_reason="tool_calls"),
            ]
        )

        (message,) = _messages(emitted)
        assert message.provider_specific_fields is None
        (call,) = _tool_calls(emitted)
        assert call.provider_specific_fields == {"thought_signature": _SIG}

    @pytest.mark.asyncio
    async def test_thoughts_only_response_signs_reasoning(self) -> None:
        _, emitted = await _collect(
            [
                _chunk(reasoning_content="**Only thinking**"),
                _chunk(provider_specific_fields={"thought_signatures": [_SIG, _SIG_2]}),
                _chunk(finish_reason="stop"),
            ]
        )

        (reasoning,) = _reasoning(emitted)
        assert reasoning.encrypted_content == _SIG_2

    @pytest.mark.asyncio
    async def test_signatures_kept_on_response(self) -> None:
        events, _ = await _collect(_text_answer_chunks())

        (completed,) = [e for e in events if isinstance(e, ResponseCompleted)]
        assert completed.response.provider_specific_fields == {
            "thought_signatures": [_SIG]
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks", [_text_answer_chunks(), _tool_call_chunks()], ids=["text", "tool"]
    )
    async def test_items_are_final_when_emitted(
        self, chunks: list[ModelResponseStream]
    ) -> None:
        events, emitted = await _collect(chunks)

        final = _final_output(events)
        assert [i.model_dump() for i in emitted] == [i.model_dump() for i in final]


class TestThoughtSignatureReplay:
    """
    Streamed items replay through LiteLLM's Gemini request builder with exactly
    one signature, on the part Gemini signed. A duplicate is not just noise:
    Gemini bills every replayed signature as the previous turn's reasoning.
    """

    @staticmethod
    def _gemini_parts(items: Sequence[Any]) -> list[dict[str, Any]]:
        messages = items_to_provider_inputs(items, reasoning_block_format="anthropic")
        contents = _gemini_convert_messages_with_history(messages=messages)  # type: ignore[arg-type]
        (model_turn,) = [c for c in contents if c["role"] == "model"]
        return [dict(p) for p in model_turn["parts"]]  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_tool_call_signature_replays_on_the_function_call(self) -> None:
        events, _ = await _collect(_tool_call_chunks())
        output = _final_output(events)
        (call,) = _tool_calls(output)

        parts = self._gemini_parts(
            [
                InputMessageItem.from_text("What is 17 + 25?"),
                *output,
                FunctionToolOutputItem.from_tool_result(
                    call_id=call.call_id, output=42
                ),
            ]
        )

        signed = [p for p in parts if "thoughtSignature" in p]
        assert len(signed) == 1
        assert "function_call" in signed[0]
        assert signed[0]["thoughtSignature"] == _SIG
        assert not any("text" in p and "thoughtSignature" in p for p in parts)

    @pytest.mark.asyncio
    async def test_text_answer_signature_replays_on_the_answer(self) -> None:
        events, _ = await _collect(_text_answer_chunks())
        output = _final_output(events)

        parts = self._gemini_parts(
            [
                InputMessageItem.from_text("Why is the sky blue?"),
                *output,
                InputMessageItem.from_text("Now in three words."),
            ]
        )

        signed = [p for p in parts if "thoughtSignature" in p]
        assert len(signed) == 1
        assert signed[0]["text"] == "The sky is blue."
        assert not signed[0].get("thought")
