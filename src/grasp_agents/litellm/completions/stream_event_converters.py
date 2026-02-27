"""
Stateful converter: LiteLLM ModelResponseStream → StreamEvent stream.

Extends the base CompletionsStreamConverter with reasoning content support
(delta.reasoning_content) specific to LiteLLM's handling of thinking/reasoning
models (Anthropic Claude, DeepSeek, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from ...openai.completions.stream_event_converters import CompletionsStreamConverter
from ...typing.content import ReasoningSummaryContent
from ...typing.items import FunctionToolCallItem, OutputMessageItem, ReasoningItem
from ...typing.response import Response, ResponseUsage
from ...typing.stream_events import (
    OutputItemAdded,
    OutputItemDone,
    ReasoningSummaryDelta,
    ReasoningSummaryDone,
    ReasoningSummaryPartAdded,
    ReasoningSummaryPartDone,
    ResponseCompleted,
    ResponseCreated,
    ResponseInProgress,
    StreamEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from openai.types.responses import ResponseStatus

    from . import LiteLLMCompletionChunk, LiteLLMUsage


class LiteLLMStreamConverter(CompletionsStreamConverter):
    """
    Converts a LiteLLM ModelResponseStream async stream into a StreamEvent stream.

    Extends CompletionsStreamConverter with reasoning content handling for models
    that support thinking/reasoning (e.g., Anthropic Claude, DeepSeek).
    """

    def __init__(self) -> None:
        super().__init__()
        self._reasoning_text = ""
        self._reasoning_item_id: str | None = None
        self._reasoning_output_index: int = 0
        self._reasoning_open = False

    async def convert(  # type: ignore[override]
        self, chunk_stream: AsyncIterator[LiteLLMCompletionChunk]
    ) -> AsyncIterator[StreamEvent]:
        """Consume LiteLLM chunk stream and yield StreamEvent instances."""
        async for chunk in chunk_stream:
            for event in self._process_litellm_chunk(chunk):
                yield event

        yield self._build_response_completed()

    # ------------------------------------------------------------------ #
    #  Per-chunk dispatch (LiteLLM-specific)                               #
    # ------------------------------------------------------------------ #

    def _process_litellm_chunk(
        self, chunk: LiteLLMCompletionChunk
    ) -> Iterator[StreamEvent]:
        # Capture metadata (safe access — not all fields on ModelResponseStream)
        service_tier: str | None = getattr(chunk, "service_tier", None)
        if service_tier:
            self._service_tier = service_tier

        raw_usage: Any = getattr(chunk, "usage", None)
        if raw_usage is not None:
            self._usage = _convert_usage(raw_usage)

        if not chunk.choices:
            return

        choice = chunk.choices[0]
        delta = choice.delta

        # First chunk with choices — emit lifecycle events
        if not self._started:
            self._started = True
            self._response_id = chunk.id
            self._model = chunk.model or ""
            self._created_at = float(chunk.created)

            skeleton = self._skeleton_response()
            yield ResponseCreated(
                response=skeleton, sequence_number=self._next_seq()
            )
            yield ResponseInProgress(
                response=skeleton, sequence_number=self._next_seq()
            )

        # LiteLLM's Delta uses dynamic attributes (extends OpenAIObject),
        # so all field access must go through getattr.

        # --- LiteLLM-specific: reasoning content ---

        reasoning_content: str | None = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            yield from self._on_reasoning(reasoning_content)

        # --- Standard deltas ---

        text_content: str | None = getattr(delta, "content", None)
        if text_content:
            # Close reasoning before text starts (reasoning precedes output)
            if self._reasoning_open:
                yield from self._close_reasoning()
            yield from self._on_text(text_content)

        tool_calls: list[Any] | None = getattr(delta, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                yield from self._on_tool_call(tc)

        refusal: str | None = getattr(delta, "refusal", None)
        if refusal:
            yield from self._on_refusal(refusal)

        # --- finish ---

        if choice.finish_reason:
            self._finish_reason = choice.finish_reason
            yield from self._close_all()

    # ------------------------------------------------------------------ #
    #  Reasoning handlers                                                  #
    # ------------------------------------------------------------------ #

    def _on_reasoning(self, text: str) -> Iterator[StreamEvent]:
        if not self._reasoning_open:
            self._reasoning_item_id = str(uuid4())
            self._reasoning_output_index = self._next_output_index
            self._next_output_index += 1
            self._reasoning_open = True

            yield OutputItemAdded(
                item=ReasoningItem(
                    id=self._reasoning_item_id,
                    status="in_progress",
                    summary_ext=[],
                ),
                output_index=self._reasoning_output_index,
                sequence_number=self._next_seq(),
            )
            yield ReasoningSummaryPartAdded(
                item_id=self._reasoning_item_id,
                output_index=self._reasoning_output_index,
                part=ReasoningSummaryContent(text=""),  # type: ignore[arg-type]
                summary_index=0,
                sequence_number=self._next_seq(),
            )

        self._reasoning_text += text
        yield ReasoningSummaryDelta(
            delta=text,
            item_id=self._reasoning_item_id or "",
            output_index=self._reasoning_output_index,
            summary_index=0,
            sequence_number=self._next_seq(),
        )

    def _close_reasoning(self) -> Iterator[StreamEvent]:
        self._reasoning_open = False
        yield ReasoningSummaryDone(
            item_id=self._reasoning_item_id or "",
            output_index=self._reasoning_output_index,
            summary_index=0,
            sequence_number=self._next_seq(),
            text=self._reasoning_text,
        )
        yield ReasoningSummaryPartDone(
            item_id=self._reasoning_item_id or "",
            output_index=self._reasoning_output_index,
            part=ReasoningSummaryContent(text=self._reasoning_text),  # type: ignore[arg-type]
            summary_index=0,
            sequence_number=self._next_seq(),
        )
        yield OutputItemDone(
            item=ReasoningItem(
                id=self._reasoning_item_id or str(uuid4()),
                status="completed",
                summary_ext=[
                    ReasoningSummaryContent(text=self._reasoning_text)
                ],
            ),
            output_index=self._reasoning_output_index,
            sequence_number=self._next_seq(),
        )

    # ------------------------------------------------------------------ #
    #  Overrides                                                           #
    # ------------------------------------------------------------------ #

    def _close_all(self) -> Iterator[StreamEvent]:
        """Close every open item — called when finish_reason is received."""
        if self._reasoning_open:
            yield from self._close_reasoning()
        yield from super()._close_all()

    def _build_response_completed(self) -> ResponseCompleted:
        output_items: list[
            OutputMessageItem | FunctionToolCallItem | ReasoningItem
        ] = []

        # Reasoning items first (matches non-streaming item order)
        if self._reasoning_text:
            output_items.append(
                ReasoningItem(
                    id=self._reasoning_item_id or str(uuid4()),
                    status="completed",
                    summary_ext=[
                        ReasoningSummaryContent(text=self._reasoning_text)
                    ],
                )
            )

        if self._text or self._refusal:
            output_items.append(self._build_message_item("completed"))

        for state in self._tool_calls.values():
            output_items.append(
                FunctionToolCallItem(
                    id=state.item_id,
                    call_id=state.call_id,
                    name=state.name,
                    arguments=state.arguments,
                    status="completed",
                )
            )

        status: ResponseStatus = "completed"
        incomplete_details: IncompleteDetails | None = None
        if self._finish_reason == "length":
            status = "incomplete"
            incomplete_details = IncompleteDetails(reason="max_output_tokens")
        elif self._finish_reason == "content_filter":
            status = "incomplete"
            incomplete_details = IncompleteDetails(reason="content_filter")

        response = Response(
            id=self._response_id,
            created_at=self._created_at,
            model=self._model,
            status=status,
            incomplete_details=incomplete_details,
            output_ext=output_items,
            usage_ext=self._usage,
            service_tier=self._service_tier,  # type: ignore[arg-type]
        )

        return ResponseCompleted(
            response=response, sequence_number=self._next_seq()
        )


# ------------------------------------------------------------------ #
#  Usage conversion                                                    #
# ------------------------------------------------------------------ #


def _convert_usage(raw_usage: LiteLLMUsage) -> ResponseUsage:
    cached_tokens = 0
    reasoning_tokens = 0

    if raw_usage.prompt_tokens_details is not None:
        cached_tokens = raw_usage.prompt_tokens_details.cached_tokens or 0

    if raw_usage.completion_tokens_details is not None:
        reasoning_tokens = (
            raw_usage.completion_tokens_details.reasoning_tokens or 0
        )

    return ResponseUsage(
        input_tokens=raw_usage.prompt_tokens,
        output_tokens=raw_usage.completion_tokens,
        total_tokens=raw_usage.total_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=reasoning_tokens
        ),
    )
