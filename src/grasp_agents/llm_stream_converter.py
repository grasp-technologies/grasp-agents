"""
Provider-agnostic base for streaming LLM event converters.

Contains all shared state management and event emission logic.
Subclasses implement ``convert()`` (the async entry point that consumes
a provider-specific chunk/event stream) and hook methods for
provider-specific behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_output_text import Annotation
from openai.types.responses.response_output_text import (
    Logprob as OutputLogprob,
)

from grasp_agents.logprob_converters import (
    to_delta_logprobs,
    to_done_logprobs,
)
from grasp_agents.types.content import (
    Citation,
    OutputMessageContentPart,
    OutputRefusal,
    OutputTextContentPart,
    ReasoningSummaryPart,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
    prefixed_id,
)
from grasp_agents.types.llm_events import (
    ContentPartAdded,
    ContentPartDone,
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ReasoningSummaryDelta,
    ReasoningSummaryPartAdded,
    ReasoningSummaryPartDone,
    ReasoningSummaryTextDone,
    RefusalDelta,
    RefusalDone,
    ResponseCompleted,
    ResponseCreated,
    ResponseInProgress,
    TextDelta,
    TextDone,
)
from grasp_agents.types.response import Response, ResponseUsage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai.types.responses import ResponseStatus


@dataclass
class ToolCallState:
    """Accumulated state for one tool call being streamed."""

    item_index: int
    item_id: str
    call_id: str
    name: str
    arguments: str = ""
    provider_specific_fields: dict[str, Any] | None = None


class BaseLlmStreamConverter:
    """
    Provider-agnostic base for streaming LLM response → LlmEvent converters.

    Tracks accumulated text, tool-call arguments, refusal content, logprobs,
    and reasoning to synthesize the full OpenResponses event lifecycle.
    Subclasses provide ``convert()`` and per-chunk/event dispatch.
    """

    def __init__(self) -> None:
        self._seq = 0

        # == Response-level ==

        # Completed items (populated by close methods, used in ResponseCompleted)
        self._items: list[OutputItem] = []
        self._item_count = 0

        self._started = False
        self._response_id = ""
        self._model = ""
        self._created_at: float = 0.0
        self._service_tier: str | None = None
        self._finish_reason: str | None = None

        # Usage (populated from final chunk)
        self._usage: ResponseUsage | None = None

        # == Output Message ==

        # Content parts (built once in close methods, reused in message item)
        self._message_content_parts: list[OutputMessageContentPart] = []
        self._message_content_part_count = 0
        self._message_content_part_index: int = 0

        self._message_open = False
        self._message_id: str | None = None
        self._message_item_index: int = 0

        # Text content part
        self._text_open: bool = False
        self._text: str | None = None

        # Refusal content part
        self._refusal_open = False
        self._refusal: str | None = None

        # Logprobs (accumulated across chunks)
        self._logprobs: list[OutputLogprob] = []

        # Annotations (accumulated across chunks, e.g. URL citations)
        self._annotations: list[Annotation] = []

        # Provider-specific opaque data for message round-trip fidelity
        self._message_provider_specific_fields: dict[str, Any] | None = None

        # == Reasoning item ==

        self._reasoning_open = False
        self._reasoning_id: str | None = None
        self._reasoning_item_index: int = 0

        self._reasoning_summary_parts: list[ReasoningSummaryPart] = []
        self._reasoning_summary_part_count: int = 0
        self._reasoning_summary_part_index: int = 0

        self._reasoning_summary_part_open = False
        self._reasoning_summary_part_text: str | None = None

        self._reasoning_encrypted_content: str | None = None
        self._reasoning_cache_control: dict[str, Any] | None = None
        self._reasoning_redacted: bool = False

        # Tool calls (keyed by chunk tool_call index)
        self._tool_calls: dict[int, ToolCallState] = {}

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ==== Response lifecycle ====

    def _start_response(
        self,
        *,
        id: str,  # noqa: A002
        model: str,
        created_at: float,
    ) -> Iterator[LlmEvent]:
        self._started = True
        self._response_id = id
        self._model = model
        self._created_at = created_at

        skeleton = Response(
            id=self._response_id,
            created_at=self._created_at,
            model=self._model,
            status="in_progress",
            output_items=[],
        )
        yield ResponseCreated(response=skeleton, sequence_number=self._next_seq())
        yield ResponseInProgress(response=skeleton, sequence_number=self._next_seq())

    def _close_response(self) -> Iterator[LlmEvent]:
        """Close every open item and emit ResponseCompleted."""
        if self._reasoning_open:
            yield from self._close_reasoning()

        if self._message_open:
            yield from self._close_message()

        if self._tool_calls:
            yield from self._close_tool_calls()

        yield self._build_response_completed()

    def _build_response_completed(self) -> ResponseCompleted:
        status, incomplete_details = self._map_finish_reason()

        response = Response(
            id=self._response_id,
            created_at=self._created_at,
            model=self._model,
            status=status,
            incomplete_details=incomplete_details,
            output_items=self._items,
            usage_with_cost=self._usage,
            service_tier=self._service_tier,  # type: ignore[arg-type]
        )

        return ResponseCompleted(response=response, sequence_number=self._next_seq())

    def _map_finish_reason(
        self,
    ) -> tuple[ResponseStatus, IncompleteDetails | None]:
        """Map provider finish_reason to ResponseStatus. Override per provider."""
        if self._finish_reason == "length":
            return "incomplete", IncompleteDetails(reason="max_output_tokens")
        if self._finish_reason == "content_filter":
            return "incomplete", IncompleteDetails(reason="content_filter")
        return "completed", None

    # ==== Reasoning ====

    def _open_reasoning(self) -> Iterator[LlmEvent]:
        self._reasoning_open = True
        self._reasoning_id = prefixed_id("rs")
        self._reasoning_encrypted_content = None
        self._reasoning_cache_control = None
        self._reasoning_redacted = False

        self._reasoning_summary_part_open = False
        self._reasoning_summary_parts = []
        self._reasoning_summary_part_count = 0

        self._reasoning_item_index = self._item_count
        self._item_count += 1

        yield OutputItemAdded(
            item=ReasoningItem(
                id=self._reasoning_id, status="in_progress", summary_parts=[]
            ),
            output_index=self._reasoning_item_index,
            sequence_number=self._next_seq(),
        )

    def _open_reasoning_summary_part(self) -> Iterator[LlmEvent]:
        # if not self._reasoning_open:
        #     yield from self._open_reasoning()

        self._reasoning_summary_part_open = True
        self._reasoning_summary_part_text = ""
        self._reasoning_summary_part_index = self._reasoning_summary_part_count
        self._reasoning_summary_part_count += 1

        assert self._reasoning_id is not None  # for mypy

        yield ReasoningSummaryPartAdded(
            item_id=self._reasoning_id,
            output_index=self._reasoning_item_index,
            part=ReasoningSummaryPart(text=""),  # type: ignore[arg-type]
            summary_index=self._reasoning_summary_part_index,
            sequence_number=self._next_seq(),
        )

    def _on_reasoning_content(self, text: str) -> Iterator[LlmEvent]:
        # if not self._reasoning_summary_part_open:
        #     yield from self._open_reasoning_summary_part()

        assert self._reasoning_summary_part_text is not None  # for mypy
        assert self._reasoning_id is not None  # for mypy

        self._reasoning_summary_part_text += text

        yield ReasoningSummaryDelta(
            delta=text,
            item_id=self._reasoning_id,
            output_index=self._reasoning_item_index,
            summary_index=self._reasoning_summary_part_index,
            sequence_number=self._next_seq(),
        )

    def _close_reasoning_summary_part(self) -> Iterator[LlmEvent]:
        self._reasoning_summary_part_open = False

        assert self._reasoning_id is not None  # for mypy

        if self._reasoning_summary_part_text:
            yield ReasoningSummaryTextDone(
                item_id=self._reasoning_id,
                output_index=self._reasoning_item_index,
                summary_index=self._reasoning_summary_part_index,
                sequence_number=self._next_seq(),
                text=self._reasoning_summary_part_text,
            )

            summary_part = ReasoningSummaryPart(
                text=self._reasoning_summary_part_text,
            )
            self._reasoning_summary_parts.append(summary_part)

            yield ReasoningSummaryPartDone(
                item_id=self._reasoning_id,
                output_index=self._reasoning_item_index,
                part=summary_part,
                summary_index=self._reasoning_summary_part_index,
                sequence_number=self._next_seq(),
            )

    def _close_reasoning(self) -> Iterator[LlmEvent]:
        if self._reasoning_summary_part_open:
            yield from self._close_reasoning_summary_part()

        self._reasoning_open = False

        assert self._reasoning_id is not None  # for mypy

        reasoning_item = ReasoningItem(
            id=self._reasoning_id,
            status="completed",
            summary_parts=list(self._reasoning_summary_parts),
            encrypted_content=self._reasoning_encrypted_content,
            cache_control=self._reasoning_cache_control,
            redacted=self._reasoning_redacted,
        )
        self._items.append(reasoning_item)

        yield OutputItemDone(
            item=reasoning_item,
            output_index=self._reasoning_item_index,
            sequence_number=self._next_seq(),
        )

    # ==== Output message ====

    def _open_message(self) -> Iterator[LlmEvent]:
        # if self._reasoning_open:
        #     yield from self._close_reasoning()

        self._message_open = True
        self._message_id = prefixed_id("msg")

        self._message_item_index = self._item_count
        self._item_count += 1

        self._text_open = False
        self._message_content_parts = []
        self._message_content_part_count = 0
        self._message_provider_specific_fields = None

        yield OutputItemAdded(
            item=OutputMessageItem(
                id=self._message_id, status="in_progress", content_parts=[]
            ),
            output_index=self._message_item_index,
            sequence_number=self._next_seq(),
        )

    def _open_text(self) -> Iterator[LlmEvent]:
        # if not self._message_open:
        #     yield from self._open_message()

        self._text_open = True
        self._text = ""
        self._logprobs = []

        self._message_content_part_index = self._message_content_part_count
        self._message_content_part_count += 1

        yield ContentPartAdded(
            content_index=self._message_content_part_index,
            output_index=self._message_item_index,
            sequence_number=self._next_seq(),
            item_id=self._message_id or "",
            part=OutputTextContentPart(text="", citations=[]),
        )

    def _on_text(
        self, text: str, logprobs: list[OutputLogprob] | None = None
    ) -> Iterator[LlmEvent]:
        # if not self._text_open:
        #     yield from self._open_text()

        assert self._text is not None  # for mypy
        assert self._message_id is not None  # for mypy

        self._text += text
        if logprobs:
            self._logprobs.extend(logprobs)

        yield TextDelta(
            content_index=self._message_content_part_index,
            output_index=self._message_item_index,
            sequence_number=self._next_seq(),
            delta=text,
            item_id=self._message_id,
            logprobs=to_delta_logprobs(logprobs) if logprobs else [],
        )

    def _close_text(self) -> Iterator[LlmEvent]:
        self._text_open = False

        assert self._text is not None  # for mypy
        assert self._message_id is not None  # for mypy

        yield TextDone(
            content_index=self._message_content_part_index,
            output_index=self._message_item_index,
            sequence_number=self._next_seq(),
            item_id=self._message_id,
            text=self._text,
            logprobs=to_done_logprobs(self._logprobs),
        )

        part = OutputTextContentPart(
            text=self._text,
            citations=self._build_text_citations(),
            logprobs=self._logprobs or None,
        )
        self._message_content_parts.append(part)

        yield ContentPartDone(
            content_index=self._message_content_part_index,
            output_index=self._message_item_index,
            sequence_number=self._next_seq(),
            item_id=self._message_id,
            part=part,
        )

    def _build_text_citations(self) -> list[Citation]:
        """Build text citations. Override to convert provider-specific citations."""
        return []

    def _open_refusal(self) -> Iterator[LlmEvent]:
        self._refusal_open = True
        self._refusal = ""

        self._message_content_part_index = self._message_content_part_count
        self._message_content_part_count += 1

        yield ContentPartAdded(
            content_index=self._message_content_part_index,
            output_index=self._message_item_index,
            sequence_number=self._next_seq(),
            item_id=self._message_id or "",
            part=OutputRefusal(refusal=""),
        )

    def _on_refusal(self, refusal: str) -> Iterator[LlmEvent]:
        assert self._refusal is not None  # for mypy
        assert self._message_id is not None  # for mypy

        self._refusal += refusal

        yield RefusalDelta(
            content_index=self._message_content_part_index,
            output_index=self._message_item_index,
            sequence_number=self._next_seq(),
            delta=refusal,
            item_id=self._message_id,
        )

    def _close_refusal(self) -> Iterator[LlmEvent]:
        self._refusal_open = False

        assert self._refusal is not None  # for mypy
        assert self._message_id is not None  # for mypy

        yield RefusalDone(
            content_index=self._message_content_part_index,
            output_index=self._message_item_index,
            sequence_number=self._next_seq(),
            item_id=self._message_id,
            refusal=self._refusal,
        )

        part = OutputRefusal(refusal=self._refusal)
        self._message_content_parts.append(part)

        yield ContentPartDone(
            content_index=self._message_content_part_index,
            output_index=self._message_item_index,
            sequence_number=self._next_seq(),
            item_id=self._message_id,
            part=part,
        )

    def _close_message(self) -> Iterator[LlmEvent]:
        if self._text_open:
            yield from self._close_text()
        if self._refusal_open:
            yield from self._close_refusal()

        self._message_open = False

        assert self._message_id is not None  # for mypy

        item = OutputMessageItem(
            id=self._message_id,
            status="completed",
            content_parts=self._message_content_parts,
            provider_specific_fields=self._message_provider_specific_fields,
        )
        self._items.append(item)

        yield OutputItemDone(
            item=item,
            output_index=self._message_item_index,
            sequence_number=self._next_seq(),
        )

    # ==== Tool calls ====

    def _open_tool_call(
        self, *, call_id: str, name: str, idx: int
    ) -> Iterator[LlmEvent]:
        item_id = prefixed_id("fc")

        item_index = self._item_count
        self._item_count += 1

        state = ToolCallState(
            item_index=item_index,
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
            output_index=item_index,
            sequence_number=self._next_seq(),
        )

    def _on_tool_call_args(self, idx: int, args_delta: str) -> Iterator[LlmEvent]:
        state = self._tool_calls[idx]
        state.arguments += args_delta

        yield FunctionCallArgumentsDelta(
            delta=args_delta,
            item_id=state.item_id,
            output_index=state.item_index,
            sequence_number=self._next_seq(),
        )

    def _close_tool_calls(self) -> Iterator[LlmEvent]:
        for state in self._tool_calls.values():
            yield FunctionCallArgumentsDone(
                arguments=state.arguments,
                item_id=state.item_id,
                name=state.name,
                output_index=state.item_index,
                sequence_number=self._next_seq(),
            )

            item = FunctionToolCallItem(
                id=state.item_id,
                call_id=state.call_id,
                name=state.name,
                arguments=state.arguments,
                status="completed",
                provider_specific_fields=state.provider_specific_fields,
            )
            self._items.append(item)

            yield OutputItemDone(
                item=item,
                output_index=state.item_index,
                sequence_number=self._next_seq(),
            )
