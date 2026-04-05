"""
Tests for stream → LlmEvent converters.

Tests CompletionsStreamConverter (OpenAI base) and LiteLLMStreamConverter.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai.types import CompletionUsage
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChoiceLogprobs,
)
from openai.types.chat.chat_completion_token_logprob import (
    ChatCompletionTokenLogprob,
    TopLogprob,
)

from grasp_agents.llm_providers.litellm.llm_event_converters import (
    LiteLLMStreamConverter,
)
from grasp_agents.llm_providers.openai_completions.llm_event_converters import (
    CompletionsStreamConverter,
)
from grasp_agents.llm_providers.openai_completions.logprob_converters import (
    convert_logprobs as convert_completion_logprobs,
)
from grasp_agents.types.content import (
    OutputMessageRefusal,
    OutputMessageText,
    ReasoningSummary,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_events import (
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    LlmEvent,
    OutputContentPartAdded,
    OutputContentPartDone,
    OutputItemAdded,
    OutputItemDone,
    OutputMessageRefusalPartDelta,
    OutputMessageRefusalPartDone,
    OutputMessageTextPartTextDelta,
    OutputMessageTextPartTextDone,
    ReasoningSummaryPartAdded,
    ReasoningSummaryPartDone,
    ReasoningSummaryPartTextDelta,
    ReasoningSummaryPartTextDone,
    ResponseCompleted,
    ResponseCreated,
    ResponseInProgress,
)

# ------------------------------------------------------------------ #
#  Helpers                                                              #
# ------------------------------------------------------------------ #

_CHUNK_ID = "chatcmpl-test"
_MODEL = "gpt-4o"
_CREATED = 1700000000


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[ChoiceDeltaToolCall] | None = None,
    refusal: str | None = None,
    finish_reason: str | None = None,
    usage: CompletionUsage | None = None,
    service_tier: str | None = None,
    logprobs: ChoiceLogprobs | None = None,
    # For OpenRouter reasoning
    reasoning_content: str | None = None,
    reasoning_details: list[Any] | None = None,
) -> ChatCompletionChunk:
    delta = ChoiceDelta(
        content=content,
        tool_calls=tool_calls,
        refusal=refusal,
        role="assistant" if content is not None else None,
    )
    # Inject extra attributes via object __dict__ for OpenRouter fields
    if reasoning_content is not None:
        object.__setattr__(delta, "reasoning_content", reasoning_content)
    if reasoning_details is not None:
        object.__setattr__(delta, "reasoning_details", reasoning_details)

    choice = Choice(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
        logprobs=logprobs,
    )
    return ChatCompletionChunk(
        id=_CHUNK_ID,
        choices=[choice],
        created=_CREATED,
        model=_MODEL,
        object="chat.completion.chunk",
        usage=usage,
        service_tier=service_tier,
    )


async def _collect(
    converter: CompletionsStreamConverter,
    chunks: list[ChatCompletionChunk],
) -> list[LlmEvent]:
    """Feed chunks through converter.convert() and collect all events."""

    async def _stream():
        for c in chunks:
            yield c

    events: list[LlmEvent] = []
    async for event in converter.convert(_stream()):  # type: ignore[arg-type]
        events.append(event)
    return events


def _events_of_type(events: list[LlmEvent], typ: type) -> list[Any]:
    return [e for e in events if isinstance(e, typ)]


def _final_response(events: list[LlmEvent]) -> ResponseCompleted:
    completed = _events_of_type(events, ResponseCompleted)
    assert len(completed) == 1
    return completed[0]


# ------------------------------------------------------------------ #
#  LiteLLM chunk helpers (SimpleNamespace-based)                        #
# ------------------------------------------------------------------ #


def _litellm_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict[str, Any]] | None = None,
    annotations: list[dict[str, Any]] | None = None,
    tool_calls: list[Any] | None = None,
    refusal: str | None = None,
    finish_reason: str | None = None,
    provider_specific_fields: dict[str, Any] | None = None,
    hidden_params: dict[str, Any] | None = None,
    response_ms: float | None = None,
    usage: Any | None = None,
    logprobs: Any | None = None,
) -> Any:
    """Build a LiteLLM-style chunk using real LiteLLM types."""
    from litellm.types.utils import (
        Delta as LiteLLMDelta,
    )
    from litellm.types.utils import (
        ModelResponseStream,
        StreamingChoices,
    )

    delta = LiteLLMDelta(
        content=content,
        reasoning_content=reasoning_content,
        thinking_blocks=thinking_blocks,
        annotations=annotations,
        tool_calls=tool_calls,
    )
    # Set extra attrs that Delta.__init__ doesn't handle
    if refusal is not None:
        delta.refusal = refusal
    if provider_specific_fields is not None:
        delta.provider_specific_fields = provider_specific_fields

    choice = StreamingChoices(
        delta=delta,
        finish_reason=finish_reason,
        logprobs=logprobs,
    )
    chunk = ModelResponseStream(
        id=_CHUNK_ID,
        model=_MODEL,
        choices=[choice],
    )
    # Set created time (ModelResponseStream auto-sets it)
    object.__setattr__(chunk, "created", _CREATED)
    if usage is not None:
        chunk.usage = usage  # type: ignore[assignment]
    if hidden_params is not None:
        chunk._hidden_params = hidden_params  # type: ignore[attr-defined]
    if response_ms is not None:
        chunk._response_ms = response_ms  # type: ignore[attr-defined]
    return chunk


async def _collect_litellm(
    converter: LiteLLMStreamConverter,
    chunks: list[Any],
) -> list[LlmEvent]:
    async def _stream():
        for c in chunks:
            yield c

    events: list[LlmEvent] = []
    async for event in converter.convert(_stream()):  # type: ignore[arg-type]
        events.append(event)
    return events


# ================================================================== #
#  CompletionsStreamConverter tests                                     #
# ================================================================== #


class TestTextStream:
    def test_single_chunk(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="Hello"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        deltas = _events_of_type(events, OutputMessageTextPartTextDelta)
        assert len(deltas) == 1
        assert deltas[0].delta == "Hello"

        dones = _events_of_type(events, OutputMessageTextPartTextDone)
        assert len(dones) == 1
        assert dones[0].text == "Hello"

    def test_multi_chunk(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="Hel"),
                    _chunk(content="lo"),
                    _chunk(content=" world"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        deltas = _events_of_type(events, OutputMessageTextPartTextDelta)
        assert len(deltas) == 3
        assert "".join(d.delta for d in deltas) == "Hello world"

        done = _events_of_type(events, OutputMessageTextPartTextDone)[0]
        assert done.text == "Hello world"

    def test_final_response_output(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="Hi"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        assert resp.status == "completed"
        assert len(resp.output_items) == 1
        msg = resp.output_items[0]
        assert isinstance(msg, OutputMessageItem)
        assert msg.text == "Hi"

    def test_content_part_events(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="x"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        added = _events_of_type(events, OutputContentPartAdded)
        assert len(added) == 1
        assert isinstance(added[0].part, OutputMessageText)

        done = _events_of_type(events, OutputContentPartDone)
        assert len(done) == 1
        assert isinstance(done[0].part, OutputMessageText)
        assert done[0].part.text == "x"


class TestToolCallStream:
    def test_single_tool(self):
        tc = ChoiceDeltaToolCall(
            index=0,
            id="call_1",
            function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"c'),
            type="function",
        )
        tc2 = ChoiceDeltaToolCall(
            index=0,
            function=ChoiceDeltaToolCallFunction(arguments='ity": "NYC"}'),
        )
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(tool_calls=[tc]),
                    _chunk(tool_calls=[tc2]),
                    _chunk(finish_reason="tool_calls"),
                ],
            )
        )

        arg_deltas = _events_of_type(events, FunctionCallArgumentsDelta)
        assert len(arg_deltas) == 2
        full_args = "".join(d.delta for d in arg_deltas)
        assert full_args == '{"city": "NYC"}'

        arg_dones = _events_of_type(events, FunctionCallArgumentsDone)
        assert len(arg_dones) == 1
        assert arg_dones[0].name == "get_weather"
        assert arg_dones[0].arguments == '{"city": "NYC"}'

    def test_text_then_tool(self):
        tc = ChoiceDeltaToolCall(
            index=0,
            id="call_1",
            function=ChoiceDeltaToolCallFunction(name="search", arguments="{}"),
            type="function",
        )
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="Let me search"),
                    _chunk(tool_calls=[tc]),
                    _chunk(finish_reason="tool_calls"),
                ],
            )
        )
        # Message should be closed before tool call
        msg_dones = _events_of_type(events, OutputItemDone)
        # One for message, one for tool call
        assert len(msg_dones) == 2
        assert isinstance(msg_dones[0].item, OutputMessageItem)
        assert isinstance(msg_dones[1].item, FunctionToolCallItem)

    def test_multiple_tools(self):
        tc1 = ChoiceDeltaToolCall(
            index=0,
            id="call_1",
            function=ChoiceDeltaToolCallFunction(name="fn_a", arguments="{}"),
            type="function",
        )
        tc2 = ChoiceDeltaToolCall(
            index=1,
            id="call_2",
            function=ChoiceDeltaToolCallFunction(name="fn_b", arguments="{}"),
            type="function",
        )
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(tool_calls=[tc1]),
                    _chunk(tool_calls=[tc2]),
                    _chunk(finish_reason="tool_calls"),
                ],
            )
        )
        resp = _final_response(events).response
        tools = [o for o in resp.output_items if isinstance(o, FunctionToolCallItem)]
        assert len(tools) == 2
        assert {t.name for t in tools} == {"fn_a", "fn_b"}


class TestRefusalStream:
    def test_refusal_events(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(refusal="I can't"),
                    _chunk(refusal=" do that"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        deltas = _events_of_type(events, OutputMessageRefusalPartDelta)
        assert len(deltas) == 2
        assert "".join(d.delta for d in deltas) == "I can't do that"

        dones = _events_of_type(events, OutputMessageRefusalPartDone)
        assert len(dones) == 1
        assert dones[0].refusal == "I can't do that"


class TestReasoningContentStream:
    """Tests for reasoning_content (plain string fallback, e.g. OpenRouter)."""

    def test_reasoning_events(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(reasoning_content="Think"),
                    _chunk(reasoning_content="ing..."),
                    _chunk(content="Answer"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        r_deltas = _events_of_type(events, ReasoningSummaryPartTextDelta)
        assert len(r_deltas) == 2
        assert "".join(d.delta for d in r_deltas) == "Thinking..."

        r_dones = _events_of_type(events, ReasoningSummaryPartTextDone)
        assert len(r_dones) == 1
        assert r_dones[0].text == "Thinking..."

    def test_reasoning_closes_before_text(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(reasoning_content="hmm"),
                    _chunk(content="ok"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        # Reasoning OutputItemDone should come before text OutputItemAdded
        item_events = [
            e for e in events if isinstance(e, (OutputItemAdded, OutputItemDone))
        ]
        types = [(type(e).__name__, type(e.item).__name__) for e in item_events]
        # Reasoning added, reasoning done, message added, message done
        assert types[0] == ("OutputItemAdded", "ReasoningItem")
        assert types[1] == ("OutputItemDone", "ReasoningItem")
        assert types[2] == ("OutputItemAdded", "OutputMessageItem")

    def test_reasoning_in_final_response(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(reasoning_content="thought"),
                    _chunk(content="answer"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        assert len(resp.output_items) == 2
        assert isinstance(resp.output_items[0], ReasoningItem)
        assert resp.output_items[0].summary_text == "thought"
        assert isinstance(resp.output_items[1], OutputMessageItem)


class TestReasoningDetailsStream:
    """Tests for OpenRouter structured reasoning_details."""

    def test_reasoning_text_with_signature(self):
        details = [
            {"type": "reasoning.text", "text": "Deep thought", "index": 0},
            {
                "type": "reasoning.text",
                "text": " continues",
                "signature": "sig123",
                "index": 0,
            },
        ]
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(reasoning_details=details[:1]),
                    _chunk(reasoning_details=details[1:]),
                    _chunk(content="Answer"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        r_deltas = _events_of_type(events, ReasoningSummaryPartTextDelta)
        assert "".join(d.delta for d in r_deltas) == "Deep thought continues"

        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].summary_text == "Deep thought continues"
        assert reasoning[0].encrypted_content == "sig123"

    def test_reasoning_encrypted(self):
        details = [
            {"type": "reasoning.encrypted", "data": "encrypted_data_here", "index": 0},
        ]
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(reasoning_details=details),
                    _chunk(content="ok"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        # Encrypted items must be emitted inline (before text events)
        item_events = [
            e for e in events if isinstance(e, (OutputItemAdded, OutputItemDone))
        ]
        # First pair: encrypted reasoning (Added+Done), then message
        assert isinstance(item_events[0].item, ReasoningItem)  # Added (skeleton)
        assert isinstance(item_events[1].item, ReasoningItem)  # Done (completed)
        assert item_events[1].item.redacted is True
        # No ReasoningSummary events for encrypted blocks
        assert len(_events_of_type(events, ReasoningSummaryPartTextDelta)) == 0

        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].redacted is True
        assert reasoning[0].encrypted_content == "encrypted_data_here"

    def test_reasoning_summary(self):
        details = [
            {"type": "reasoning.summary", "summary": "Brief thought", "index": 0},
        ]
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(reasoning_details=details),
                    _chunk(content="done"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        r_deltas = _events_of_type(events, ReasoningSummaryPartTextDelta)
        assert r_deltas[0].delta == "Brief thought"

        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].summary_text == "Brief thought"

    def test_details_preferred_over_content(self):
        """When reasoning_details is present, reasoning_content should be skipped."""
        detail = {"type": "reasoning.text", "text": "structured", "index": 0}
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(reasoning_details=[detail]),
                    # A later chunk with just reasoning_content should be ignored
                    _chunk(reasoning_content="plain fallback"),
                    _chunk(content="out"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        # Should use the structured detail, not double-count
        assert reasoning[0].summary_text == "structured"

    def test_encrypted_details_preferred_over_content(self):
        """Even encrypted-only details should suppress reasoning_content."""
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(
                        reasoning_details=[
                            {"type": "reasoning.encrypted", "data": "enc", "index": 0},
                        ]
                    ),
                    _chunk(reasoning_content="should be ignored"),
                    _chunk(content="out"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].redacted is True

    def test_interleaved_text_and_encrypted(self):
        """Text → encrypted → text across chunks → 3 separate ReasoningItems."""
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(
                        reasoning_details=[
                            {
                                "type": "reasoning.text",
                                "text": "first thought",
                                "index": 0,
                                "signature": "sig1",
                            },
                        ]
                    ),
                    _chunk(
                        reasoning_details=[
                            {
                                "type": "reasoning.encrypted",
                                "data": "secret_data",
                                "index": 0,
                            },
                        ]
                    ),
                    _chunk(
                        reasoning_details=[
                            {
                                "type": "reasoning.text",
                                "text": "second thought",
                                "index": 0,
                                "signature": "sig2",
                            },
                        ]
                    ),
                    _chunk(content="answer"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 3

        # First: normal reasoning with text
        assert reasoning[0].redacted is False
        assert reasoning[0].summary_text == "first thought"
        assert reasoning[0].encrypted_content == "sig1"

        # Second: redacted block
        assert reasoning[1].redacted is True
        assert reasoning[1].encrypted_content == "secret_data"
        assert reasoning[1].summary_parts == []

        # Third: normal reasoning with text
        assert reasoning[2].redacted is False
        assert reasoning[2].summary_text == "second thought"
        assert reasoning[2].encrypted_content == "sig2"

        # Text output should still be present after reasoning
        msgs = [o for o in resp.output_items if isinstance(o, OutputMessageItem)]
        assert len(msgs) == 1
        assert msgs[0].text == "answer"

    def test_interleaved_summary_and_encrypted(self):
        """Summary → encrypted → summary produces 3 items (summary variant)."""
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(
                        reasoning_details=[
                            {
                                "type": "reasoning.summary",
                                "summary": "thinking about it",
                                "index": 0,
                            },
                        ]
                    ),
                    _chunk(
                        reasoning_details=[
                            {
                                "type": "reasoning.encrypted",
                                "data": "redacted_blob",
                                "index": 0,
                            },
                        ]
                    ),
                    _chunk(
                        reasoning_details=[
                            {
                                "type": "reasoning.summary",
                                "summary": "decided on approach",
                                "index": 0,
                            },
                        ]
                    ),
                    _chunk(content="done"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 3

        assert reasoning[0].redacted is False
        assert reasoning[0].summary_text == "thinking about it"

        assert reasoning[1].redacted is True
        assert reasoning[1].encrypted_content == "redacted_blob"

        assert reasoning[2].redacted is False
        assert reasoning[2].summary_text == "decided on approach"

    def test_consecutive_encrypted_details(self):
        """Multiple encrypted blocks in a row → separate ReasoningItems each."""
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(
                        reasoning_details=[
                            {
                                "type": "reasoning.encrypted",
                                "data": "enc_1",
                                "index": 0,
                            },
                        ]
                    ),
                    _chunk(
                        reasoning_details=[
                            {
                                "type": "reasoning.encrypted",
                                "data": "enc_2",
                                "index": 0,
                            },
                        ]
                    ),
                    _chunk(content="ok"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 2
        assert reasoning[0].redacted is True
        assert reasoning[0].encrypted_content == "enc_1"
        assert reasoning[1].redacted is True
        assert reasoning[1].encrypted_content == "enc_2"


class TestLogprobs:
    def _make_logprobs(self, token: str, lp: float) -> ChoiceLogprobs:
        return ChoiceLogprobs(
            content=[
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=list(token.encode("utf-8")),
                    logprob=lp,
                    top_logprobs=[
                        TopLogprob(
                            token=token,
                            bytes=list(token.encode("utf-8")),
                            logprob=lp,
                        )
                    ],
                )
            ]
        )

    def test_logprobs_in_text_delta(self):
        lp = self._make_logprobs("Hi", -0.5)
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="Hi", logprobs=lp),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        deltas = _events_of_type(events, OutputMessageTextPartTextDelta)
        assert len(deltas) == 1
        assert len(deltas[0].logprobs) == 1

    def test_logprobs_accumulated_in_text_done(self):
        lp1 = self._make_logprobs("A", -0.3)
        lp2 = self._make_logprobs("B", -0.7)
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="A", logprobs=lp1),
                    _chunk(content="B", logprobs=lp2),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        dones = _events_of_type(events, OutputMessageTextPartTextDone)
        assert len(dones) == 1
        assert len(dones[0].logprobs) == 2

    def test_logprobs_in_final_content(self):
        lp = self._make_logprobs("X", -1.0)
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="X", logprobs=lp),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        msg = resp.output_items[0]
        assert isinstance(msg, OutputMessageItem)
        text_part = msg.content_parts[0]
        assert isinstance(text_part, OutputMessageText)
        assert text_part.logprobs is not None
        assert len(text_part.logprobs) == 1

    def test_no_logprobs_default(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="Hi"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        deltas = _events_of_type(events, OutputMessageTextPartTextDelta)
        assert deltas[0].logprobs == []

        resp = _final_response(events).response
        msg = resp.output_items[0]
        assert isinstance(msg, OutputMessageItem)
        text_part = msg.content_parts[0]
        assert isinstance(text_part, OutputMessageText)
        assert text_part.logprobs is None


class TestLifecycle:
    def test_sequence_numbers_monotonic(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="a"),
                    _chunk(content="b"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        seq_nums = [e.sequence_number for e in events]
        assert seq_nums == sorted(seq_nums)
        assert len(set(seq_nums)) == len(seq_nums)  # all unique

    def test_response_created_and_in_progress(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="x"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        assert isinstance(events[0], ResponseCreated)
        assert isinstance(events[1], ResponseInProgress)
        assert events[0].response.status == "in_progress"

    def test_service_tier(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="x", service_tier="scale"),
                    _chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        assert resp.service_tier == "scale"

    def test_usage(self):
        usage = CompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="x"),
                    _chunk(finish_reason="stop", usage=usage),
                ],
            )
        )
        resp = _final_response(events).response
        assert resp.usage_with_cost is not None
        assert resp.usage_with_cost.input_tokens == 10
        assert resp.usage_with_cost.output_tokens == 20


class TestFinishReasons:
    def test_length_incomplete(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect(
                CompletionsStreamConverter(),
                [
                    _chunk(content="partial"),
                    _chunk(finish_reason="length"),
                ],
            )
        )
        resp = _final_response(events).response
        assert resp.status == "incomplete"
        assert resp.incomplete_details is not None
        assert resp.incomplete_details.reason == "max_output_tokens"


# ================================================================== #
#  Helper function tests                                                #
# ================================================================== #


class TestConvertChoiceLogprobs:
    def test_empty_content_returns_empty(self):
        lp = SimpleNamespace(content=[])
        assert convert_completion_logprobs(lp) == []

    def test_no_content_returns_empty(self):
        lp = SimpleNamespace(content=None)
        assert convert_completion_logprobs(lp) == []

    def test_converts_logprobs(self):
        lp = SimpleNamespace(
            content=[
                SimpleNamespace(
                    token="hi",
                    bytes=[104, 105],
                    logprob=-0.5,
                    top_logprobs=[
                        SimpleNamespace(token="hi", bytes=[104, 105], logprob=-0.5),
                    ],
                )
            ]
        )
        result = convert_completion_logprobs(lp)
        assert len(result) == 1
        assert result[0].token == "hi"
        assert result[0].logprob == -0.5
        assert len(result[0].top_logprobs) == 1


# ================================================================== #
#  LiteLLMStreamConverter tests                                         #
# ================================================================== #


class TestLiteLLMThinkingBlocks:
    def test_single_block(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "deep thought",
                                "signature": "sig1",
                            },
                        ],
                    ),
                    _litellm_chunk(content="ok"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].summary_text == "deep thought"
        assert reasoning[0].encrypted_content == "sig1"

    def test_multi_chunk_consolidated(self):
        """Multiple chunks for one thinking segment get consolidated."""
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        thinking_blocks=[
                            {"type": "thinking", "thinking": "part1 ", "signature": ""},
                        ],
                    ),
                    _litellm_chunk(
                        thinking_blocks=[
                            {"type": "thinking", "thinking": "part2 ", "signature": ""},
                        ],
                    ),
                    _litellm_chunk(
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "part3",
                                "signature": "sig_final",
                            },
                        ],
                    ),
                    _litellm_chunk(content="ok"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].summary_text == "part1 part2 part3"
        assert reasoning[0].encrypted_content == "sig_final"

    def test_redacted_block(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        thinking_blocks=[
                            {"type": "redacted_thinking", "data": "encrypted_data"},
                        ],
                    ),
                    _litellm_chunk(content="ok"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].redacted is True
        assert reasoning[0].encrypted_content == "encrypted_data"
        assert reasoning[0].summary_parts == []

    def test_interleaved_thinking_and_redacted(self):
        """Thinking → redacted → thinking produces 3 items with correct flags."""
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "first",
                                "signature": "s1",
                            },
                        ],
                    ),
                    _litellm_chunk(
                        thinking_blocks=[
                            {"type": "redacted_thinking", "data": "secret"},
                        ],
                    ),
                    _litellm_chunk(
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "second",
                                "signature": "s2",
                            },
                        ],
                    ),
                    _litellm_chunk(content="ok"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 3

        # First: normal thinking
        assert reasoning[0].redacted is False
        assert reasoning[0].summary_text == "first"
        assert reasoning[0].encrypted_content == "s1"

        # Second: redacted block — no summary, has encrypted data
        assert reasoning[1].redacted is True
        assert reasoning[1].encrypted_content == "secret"
        assert reasoning[1].summary_parts == []

        # Third: normal thinking resumed after redacted
        assert reasoning[2].redacted is False
        assert reasoning[2].summary_text == "second"
        assert reasoning[2].encrypted_content == "s2"

    def test_multi_chunk_thinking_blocks_consolidated(self):
        """
        Multiple chunks each with a thinking block → single ReasoningItem.

        Real LiteLLM always sends reasoning_content alongside thinking_blocks.
        """
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        reasoning_content="Let me ",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "Let me ",
                                "signature": "",
                            },
                        ],
                    ),
                    _litellm_chunk(
                        reasoning_content="think about ",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "think about ",
                                "signature": "",
                            },
                        ],
                    ),
                    _litellm_chunk(
                        reasoning_content="this.",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "this.",
                                "signature": "final_sig",
                            },
                        ],
                    ),
                    _litellm_chunk(content="The answer is 42"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].summary_text == "Let me think about this."
        assert reasoning[0].encrypted_content == "final_sig"

        # Also verify the text output is present
        msgs = [o for o in resp.output_items if isinstance(o, OutputMessageItem)]
        assert len(msgs) == 1
        assert msgs[0].text == "The answer is 42"

    def test_multi_chunk_with_redacted_block(self):
        """Thinking → redacted → thinking across chunks → 3 ReasoningItems."""
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        reasoning_content="first ",
                        thinking_blocks=[
                            {"type": "thinking", "thinking": "first ", "signature": ""},
                        ],
                    ),
                    _litellm_chunk(
                        reasoning_content="thought",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "thought",
                                "signature": "s1",
                            },
                        ],
                    ),
                    _litellm_chunk(
                        thinking_blocks=[
                            {"type": "redacted_thinking", "data": "secret_data"},
                        ],
                    ),
                    _litellm_chunk(
                        reasoning_content="after redacted",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "after redacted",
                                "signature": "s2",
                            },
                        ],
                    ),
                    _litellm_chunk(content="done"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 3

        # First: accumulated multi-chunk thinking
        assert reasoning[0].redacted is False
        assert reasoning[0].summary_text == "first thought"
        assert reasoning[0].encrypted_content == "s1"

        # Second: redacted block — no summary text
        assert reasoning[1].redacted is True
        assert reasoning[1].encrypted_content == "secret_data"
        assert reasoning[1].summary_parts == []

        # Third: resumed thinking after redacted
        assert reasoning[2].redacted is False
        assert reasoning[2].summary_text == "after redacted"
        assert reasoning[2].encrypted_content == "s2"

    def test_consecutive_redacted_blocks(self):
        """Multiple redacted blocks in a row → separate ReasoningItems each."""
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        thinking_blocks=[
                            {"type": "redacted_thinking", "data": "enc_1"},
                        ],
                    ),
                    _litellm_chunk(
                        thinking_blocks=[
                            {"type": "redacted_thinking", "data": "enc_2"},
                        ],
                    ),
                    _litellm_chunk(content="ok"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 2
        assert reasoning[0].redacted is True
        assert reasoning[0].encrypted_content == "enc_1"
        assert reasoning[1].redacted is True
        assert reasoning[1].encrypted_content == "enc_2"

    def test_thinking_blocks_preferred_over_reasoning_text(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        reasoning_content="plain text",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "structured",
                                "signature": "s",
                            },
                        ],
                    ),
                    _litellm_chunk(content="answer"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].summary_text == "structured"
        assert reasoning[0].encrypted_content == "s"

    def test_redacted_only_blocks_no_reasoning_content(self):
        """Redacted-only blocks emit inline during streaming, not at close."""
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        thinking_blocks=[
                            {"type": "redacted_thinking", "data": "secret1"},
                        ],
                    ),
                    _litellm_chunk(
                        thinking_blocks=[
                            {"type": "redacted_thinking", "data": "secret2"},
                        ],
                    ),
                    _litellm_chunk(content="answer"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        # Redacted items emitted inline before text events
        item_events = [
            e for e in events if isinstance(e, (OutputItemAdded, OutputItemDone))
        ]
        reasoning_events = [e for e in item_events if isinstance(e.item, ReasoningItem)]
        # 2 items × (Added + Done) = 4 events, all before message
        assert len(reasoning_events) == 4
        done_items = [
            e.item
            for e in reasoning_events
            if isinstance(e, OutputItemDone) and isinstance(e.item, ReasoningItem)
        ]
        assert all(item.redacted for item in done_items)

        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 2
        assert reasoning[0].encrypted_content == "secret1"
        assert reasoning[1].encrypted_content == "secret2"

        # Text output still present
        msgs = [o for o in resp.output_items if isinstance(o, OutputMessageItem)]
        assert len(msgs) == 1
        assert msgs[0].text == "answer"

    def test_cache_control_preserved(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "cached",
                                "signature": None,
                                "cache_control": {"type": "ephemeral"},
                            },
                        ],
                    ),
                    _litellm_chunk(content="ok"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].cache_control == {"type": "ephemeral"}


class TestLiteLLMAnnotations:
    def test_url_citations_in_output(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        content="See source",
                        annotations=[
                            {
                                "url_citation": {
                                    "url": "https://example.com",
                                    "title": "Example",
                                    "start_index": 0,
                                    "end_index": 10,
                                }
                            }
                        ],
                    ),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        msg = resp.output_items[0]
        assert isinstance(msg, OutputMessageItem)
        text_part = msg.content_parts[0]
        assert isinstance(text_part, OutputMessageText)
        assert len(text_part.annotations) == 1


class TestLiteLLMMetadata:
    def test_hidden_params_and_cost(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        content="hi",
                        hidden_params={"response_cost": 0.001, "model_id": "test"},
                    ),
                    _litellm_chunk(
                        finish_reason="stop",
                        usage=SimpleNamespace(
                            prompt_tokens=5,
                            completion_tokens=10,
                            total_tokens=15,
                            prompt_tokens_details=None,
                            completion_tokens_details=None,
                        ),
                    ),
                ],
            )
        )
        resp = _final_response(events).response
        assert resp.hidden_params is not None
        assert resp.hidden_params["response_cost"] == 0.001
        assert resp.usage_with_cost is not None
        assert resp.usage_with_cost.cost == 0.001

    def test_response_ms(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(content="fast", response_ms=150.5),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        assert resp.response_ms == 150.5

    def test_provider_specific_fields(self):
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        content="x",
                        provider_specific_fields={"custom_field": "value"},
                    ),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        assert resp.provider_specific_fields is not None
        assert resp.provider_specific_fields["custom_field"] == "value"

    def test_thought_signature_fallback(self):
        """When no thinking_blocks, signature from provider_specific_fields is used."""
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(
                        reasoning_content="thinking",
                        provider_specific_fields={
                            "thought_signatures": ["fallback_sig"],
                        },
                    ),
                    _litellm_chunk(content="result"),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        resp = _final_response(events).response
        reasoning = [o for o in resp.output_items if isinstance(o, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].encrypted_content == "fallback_sig"


class TestLiteLLMLogprobs:
    def test_logprobs_propagated(self):
        lp = SimpleNamespace(
            content=[
                SimpleNamespace(
                    token="hi",
                    bytes=[104, 105],
                    logprob=-0.5,
                    top_logprobs=[
                        SimpleNamespace(token="hi", bytes=[104, 105], logprob=-0.5),
                    ],
                )
            ]
        )
        events = asyncio.get_event_loop().run_until_complete(
            _collect_litellm(
                LiteLLMStreamConverter(),
                [
                    _litellm_chunk(content="hi", logprobs=lp),
                    _litellm_chunk(finish_reason="stop"),
                ],
            )
        )
        deltas = _events_of_type(events, OutputMessageTextPartTextDelta)
        assert len(deltas) == 1
        assert len(deltas[0].logprobs) == 1

        resp = _final_response(events).response
        msg = resp.output_items[0]
        assert isinstance(msg, OutputMessageItem)
        text_part = msg.content_parts[0]
        assert isinstance(text_part, OutputMessageText)
        assert text_part.logprobs is not None
