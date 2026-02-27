"""
Stateful converter: ChatCompletionChunk stream → StreamEvent stream.

Synthesizes OpenResponses lifecycle events (OutputItemAdded, ContentPartAdded,
TextDelta, OutputItemDone, ResponseCompleted, etc.) from flat Chat Completions
chunk deltas. Instantiate once per stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from ...typing.content import OutputRefusalContent, OutputTextContent
from ...typing.items import FunctionToolCallItem, OutputMessageItem, ReasoningItem
from ...typing.response import Response, ResponseUsage
from ...typing.stream_events import (
    ContentPartAdded,
    ContentPartDone,
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    OutputItemAdded,
    OutputItemDone,
    RefusalDelta,
    RefusalDone,
    ResponseCompleted,
    ResponseCreated,
    ResponseInProgress,
    StreamEvent,
    TextDelta,
    TextDone,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from openai.types import CompletionUsage as OpenAIUsage
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
    from openai.types.responses import ResponseStatus

    from ...typing.items import ItemStatus
    from . import OpenAICompletionChunk


@dataclass
class _ToolCallState:
    """Accumulated state for one tool call being streamed."""

    output_index: int
    item_id: str
    call_id: str
    name: str
    arguments: str = ""


class CompletionsStreamConverter:
    """
    Converts a ChatCompletionChunk async stream into a StreamEvent stream.

    Tracks accumulated text, tool-call arguments, and refusal content to
    synthesize the full OpenResponses event lifecycle from flat chunk deltas.
    """

    def __init__(self) -> None:
        self._seq = 0
        self._started = False

        # Response-level
        self._response_id = ""
        self._model = ""
        self._created_at: float = 0.0
        self._service_tier: str | None = None
        self._finish_reason: str | None = None

        # Message tracking
        self._message_id: str | None = None
        self._message_output_index: int = 0
        self._message_open = False

        # Text content part
        self._text = ""
        self._text_content_index = 0
        self._text_open = False

        # Refusal content part
        self._refusal = ""
        self._refusal_content_index = 0
        self._refusal_open = False

        # Tool calls (keyed by chunk tool_call index)
        self._tool_calls: dict[int, _ToolCallState] = {}
        self._next_output_index = 0

        # Usage (populated from final chunk)
        self._usage: ResponseUsage | None = None

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    async def convert(
        self, chunk_stream: AsyncIterator[OpenAICompletionChunk]
    ) -> AsyncIterator[StreamEvent]:
        """Consume *chunk_stream* and yield ``StreamEvent`` instances."""
        async for chunk in chunk_stream:
            for event in self._process_chunk(chunk):
                yield event

        # Stream exhausted — emit final ResponseCompleted
        yield self._build_response_completed()

    # ------------------------------------------------------------------ #
    #  Per-chunk dispatch                                                  #
    # ------------------------------------------------------------------ #

    def _process_chunk(
        self, chunk: OpenAICompletionChunk
    ) -> Iterator[StreamEvent]:
        # Capture metadata that may arrive on any chunk
        if chunk.service_tier:
            self._service_tier = chunk.service_tier
        if chunk.usage:
            self._usage = _convert_usage(chunk.usage)

        if not chunk.choices:
            return

        choice = chunk.choices[0]
        delta = choice.delta

        # First chunk with choices — emit lifecycle events
        if not self._started:
            self._started = True
            self._response_id = chunk.id
            self._model = chunk.model
            self._created_at = float(chunk.created)

            skeleton = self._skeleton_response()
            yield ResponseCreated(
                response=skeleton, sequence_number=self._next_seq()
            )
            yield ResponseInProgress(
                response=skeleton, sequence_number=self._next_seq()
            )

        # --- delta handlers ---

        if delta.content:
            yield from self._on_text(delta.content)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                yield from self._on_tool_call(tc)

        if delta.refusal:
            yield from self._on_refusal(delta.refusal)

        # --- finish ---

        if choice.finish_reason:
            self._finish_reason = choice.finish_reason
            yield from self._close_all()

    # ------------------------------------------------------------------ #
    #  Delta handlers                                                      #
    # ------------------------------------------------------------------ #

    def _on_text(self, text: str) -> Iterator[StreamEvent]:
        if not self._message_open:
            yield from self._open_message()

        if not self._text_open:
            self._text_content_index = 1 if self._refusal_open else 0
            self._text_open = True
            yield ContentPartAdded(
                content_index=self._text_content_index,
                item_id=self._message_id or "",
                output_index=self._message_output_index,
                part=OutputTextContent(text="", annotations=[]),
                sequence_number=self._next_seq(),
            )

        self._text += text
        yield TextDelta(
            content_index=self._text_content_index,
            delta=text,
            item_id=self._message_id or "",
            output_index=self._message_output_index,
            sequence_number=self._next_seq(),
            logprobs=[],
        )

    def _on_tool_call(
        self, tc_delta: ChoiceDeltaToolCall
    ) -> Iterator[StreamEvent]:
        idx = tc_delta.index

        if idx not in self._tool_calls:
            # New tool call — close the open message first
            if self._message_open:
                yield from self._close_message()

            item_id = str(uuid4())
            call_id = tc_delta.id or str(uuid4())
            name = (
                tc_delta.function.name
                if tc_delta.function and tc_delta.function.name
                else ""
            )

            output_index = self._next_output_index
            self._next_output_index += 1

            state = _ToolCallState(
                output_index=output_index,
                item_id=item_id,
                call_id=call_id,
                name=name,
            )
            self._tool_calls[idx] = state

            yield OutputItemAdded(
                item=FunctionToolCallItem(
                    id=item_id,
                    call_id=call_id,
                    name=name,
                    arguments="",
                    status="in_progress",
                ),
                output_index=output_index,
                sequence_number=self._next_seq(),
            )
        else:
            state = self._tool_calls[idx]
            # Name may be split across chunks
            if tc_delta.function and tc_delta.function.name:
                state.name += tc_delta.function.name

        # Argument tokens
        if tc_delta.function and tc_delta.function.arguments:
            state = self._tool_calls[idx]
            args_delta = tc_delta.function.arguments
            state.arguments += args_delta
            yield FunctionCallArgumentsDelta(
                delta=args_delta,
                item_id=state.item_id,
                output_index=state.output_index,
                sequence_number=self._next_seq(),
            )

    def _on_refusal(self, refusal: str) -> Iterator[StreamEvent]:
        if not self._message_open:
            yield from self._open_message()

        if not self._refusal_open:
            self._refusal_content_index = 1 if self._text_open else 0
            self._refusal_open = True
            yield ContentPartAdded(
                content_index=self._refusal_content_index,
                item_id=self._message_id or "",
                output_index=self._message_output_index,
                part=OutputRefusalContent(refusal=""),
                sequence_number=self._next_seq(),
            )

        self._refusal += refusal
        yield RefusalDelta(
            content_index=self._refusal_content_index,
            delta=refusal,
            item_id=self._message_id or "",
            output_index=self._message_output_index,
            sequence_number=self._next_seq(),
        )

    # ------------------------------------------------------------------ #
    #  Open / close helpers                                                #
    # ------------------------------------------------------------------ #

    def _open_message(self) -> Iterator[StreamEvent]:
        self._message_id = str(uuid4())
        self._message_output_index = self._next_output_index
        self._next_output_index += 1
        self._message_open = True

        yield OutputItemAdded(
            item=OutputMessageItem(
                id=self._message_id,
                status="in_progress",
                content_ext=[],
            ),
            output_index=self._message_output_index,
            sequence_number=self._next_seq(),
        )

    def _close_message(self) -> Iterator[StreamEvent]:
        if self._text_open:
            yield from self._close_text()
        if self._refusal_open:
            yield from self._close_refusal()

        yield OutputItemDone(
            item=self._build_message_item("completed"),
            output_index=self._message_output_index,
            sequence_number=self._next_seq(),
        )
        self._message_open = False

    def _close_text(self) -> Iterator[StreamEvent]:
        self._text_open = False
        yield TextDone(
            content_index=self._text_content_index,
            item_id=self._message_id or "",
            output_index=self._message_output_index,
            sequence_number=self._next_seq(),
            text=self._text,
            logprobs=[],
        )
        yield ContentPartDone(
            content_index=self._text_content_index,
            item_id=self._message_id or "",
            output_index=self._message_output_index,
            part=OutputTextContent(text=self._text, annotations=[]),
            sequence_number=self._next_seq(),
        )

    def _close_refusal(self) -> Iterator[StreamEvent]:
        self._refusal_open = False
        yield RefusalDone(
            content_index=self._refusal_content_index,
            item_id=self._message_id or "",
            output_index=self._message_output_index,
            sequence_number=self._next_seq(),
            refusal=self._refusal,
        )
        yield ContentPartDone(
            content_index=self._refusal_content_index,
            item_id=self._message_id or "",
            output_index=self._message_output_index,
            part=OutputRefusalContent(refusal=self._refusal),
            sequence_number=self._next_seq(),
        )

    def _close_all(self) -> Iterator[StreamEvent]:
        """Close every open item — called when finish_reason is received."""
        if self._message_open:
            yield from self._close_message()

        for state in self._tool_calls.values():
            yield FunctionCallArgumentsDone(
                arguments=state.arguments,
                item_id=state.item_id,
                name=state.name,
                output_index=state.output_index,
                sequence_number=self._next_seq(),
            )
            yield OutputItemDone(
                item=FunctionToolCallItem(
                    id=state.item_id,
                    call_id=state.call_id,
                    name=state.name,
                    arguments=state.arguments,
                    status="completed",
                ),
                output_index=state.output_index,
                sequence_number=self._next_seq(),
            )

    # ------------------------------------------------------------------ #
    #  Builders                                                            #
    # ------------------------------------------------------------------ #

    def _build_message_item(
        self, status: ItemStatus
    ) -> OutputMessageItem:
        content: list[OutputTextContent | OutputRefusalContent] = []
        if self._text:
            content.append(OutputTextContent(text=self._text, annotations=[]))
        if self._refusal:
            content.append(OutputRefusalContent(refusal=self._refusal))
        return OutputMessageItem(
            id=self._message_id or str(uuid4()),
            status=status,
            content_ext=content,
        )

    def _skeleton_response(self) -> Response:
        return Response(
            id=self._response_id,
            created_at=self._created_at,
            model=self._model,
            status="in_progress",
            output_ext=[],
        )

    def _build_response_completed(self) -> ResponseCompleted:
        output_items: list[
            OutputMessageItem | FunctionToolCallItem | ReasoningItem
        ] = []

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


def _convert_usage(raw_usage: OpenAIUsage) -> ResponseUsage:
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
