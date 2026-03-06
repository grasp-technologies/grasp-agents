"""
Stateful converter: ChatCompletionChunk stream → LlmEvent stream.

Synthesizes OpenResponses lifecycle events (OutputItemAdded, ContentPartAdded,
TextDelta, OutputItemDone, ResponseCompleted, etc.) from flat Chat Completions
chunk deltas. Instantiate once per stream.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from grasp_agents.llm_providers.openai_completions.items_extraction import (
    convert_annotations,
)
from grasp_agents.llm_providers.openai_completions.logprob_converters import (
    convert_logprobs,
)
from grasp_agents.llm_providers.openai_completions.provider_output_to_response import (
    convert_usage,
)
from grasp_agents.llm_stream_converter import BaseLlmStreamConverter
from grasp_agents.types.reasoning import OpenRouterReasoningDetails

from .utils import validate_chunk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from openai.types.chat import ChatCompletionChunk

    from grasp_agents.types.content import Citation
    from grasp_agents.types.llm_events import LlmEvent


logger = getLogger(__name__)


_REASONING_DETAILS_ADAPTER: TypeAdapter[OpenRouterReasoningDetails] = TypeAdapter(
    OpenRouterReasoningDetails
)


def _validate_reasoning_details(
    details: list[dict[str, Any]],
) -> list[OpenRouterReasoningDetails]:
    validated_details: list[OpenRouterReasoningDetails] = []
    for raw in details:
        try:
            detail = _REASONING_DETAILS_ADAPTER.validate_python(raw)
        except ValidationError:
            continue
        validated_details.append(detail)
    return validated_details


class CompletionsStreamConverter(BaseLlmStreamConverter):
    """
    Converts a ChatCompletionChunk async stream into a LlmEvent stream.

    Tracks accumulated text, tool-call arguments, refusal content, logprobs,
    and reasoning (including OpenRouter reasoning_details) to synthesize the
    full OpenResponses event lifecycle from flat chunk deltas.
    """

    def __init__(self) -> None:
        super().__init__()
        self._has_reasoning_details = False

    async def convert(
        self, chunk_stream: AsyncIterator[ChatCompletionChunk]
    ) -> AsyncIterator[LlmEvent]:
        """Consume *chunk_stream* and yield ``LlmEvent`` instances."""
        async for chunk in chunk_stream:
            for event in self._process_chunk(chunk):
                yield event

        for event in self._close_response():
            yield event

    def _process_chunk(self, chunk: ChatCompletionChunk) -> Iterator[LlmEvent]:
        if not validate_chunk(chunk):
            # Usage-only chunk (no choices) — capture usage and skip
            if chunk.usage:
                self._usage = convert_usage(chunk.usage)
            return

        choice = chunk.choices[0]
        delta = choice.delta

        if chunk.service_tier:
            self._service_tier = chunk.service_tier

        if chunk.usage:
            self._usage = convert_usage(chunk.usage)

        raw_annotations: list[Any] | None = getattr(delta, "annotations", None)
        if raw_annotations:
            self._annotations.extend(raw_annotations)

        # Start the response on the first chunk

        if not self._started:
            yield from self._start_response(
                id=chunk.id,
                model=chunk.model or "",
                created_at=float(chunk.created),
            )

        # Reasoning

        raw_reasoning_details: list[dict[str, Any]] = getattr(
            delta, "reasoning_details", []
        )
        reasoning_details = _validate_reasoning_details(raw_reasoning_details)

        reasoning_content: str | None = getattr(
            delta, "reasoning_content", None
        ) or getattr(delta, "reasoning", None)

        if reasoning_details:
            # Try reasoning_details (OpenRouter) first
            self._has_reasoning_details = True

            for detail in reasoning_details:
                reasoning_delta: str | None = None

                match detail.type:
                    case "reasoning.summary":
                        reasoning_delta = detail.summary
                    case "reasoning.text":
                        reasoning_delta = detail.text
                    case "reasoning.encrypted":
                        # Close existing reasoning cleanly (preserves accumulated text)
                        if self._reasoning_open:
                            yield from self._close_reasoning()

                        # New redacted block → separate item
                        yield from self._open_reasoning()
                        self._reasoning_encrypted_content = detail.data
                        self._reasoning_redacted = True
                        yield from self._close_reasoning()

                if reasoning_delta is not None:
                    if not self._reasoning_open:
                        yield from self._open_reasoning()
                    if not self._reasoning_summary_part_open:
                        yield from self._open_reasoning_summary_part()

                    # Set signature after open (open resets encrypted_content)
                    if detail.type == "reasoning.text":
                        self._reasoning_encrypted_content = detail.signature  # type: ignore[union-attr]

                    yield from self._on_reasoning_content(reasoning_delta)

        elif reasoning_content and not self._has_reasoning_details:
            # Try reasoning_content or reasoning
            if not self._reasoning_open:
                yield from self._open_reasoning()
            if not self._reasoning_summary_part_open:
                yield from self._open_reasoning_summary_part()

            yield from self._on_reasoning_content(reasoning_content)

        # Output message

        if delta.content or delta.refusal:
            if self._reasoning_open:
                yield from self._close_reasoning()

            if not self._message_open:
                yield from self._open_message()

        if delta.content:
            if not self._text_open:
                yield from self._open_text()

            chunk_logprobs = (
                convert_logprobs(choice.logprobs) if choice.logprobs else None
            )
            yield from self._on_text(delta.content, chunk_logprobs)

        if delta.refusal:
            if self._text_open:
                yield from self._close_text()
            if not self._refusal_open:
                yield from self._open_refusal()

            yield from self._on_refusal(delta.refusal)

        # Tool calls

        if delta.tool_calls:
            if self._reasoning_open:
                yield from self._close_reasoning()
            if self._message_open:
                yield from self._close_message()

            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in self._tool_calls:
                    yield from self._open_tool_call(
                        call_id=tc.id or str(uuid4()),
                        name="",
                        idx=idx,
                    )

                state = self._tool_calls[idx]

                if tc.function:
                    if tc.function.name:
                        state.name += tc.function.name

                    if tc.function.arguments:
                        yield from self._on_tool_call_args(idx, tc.function.arguments)

        if choice.finish_reason:
            self._finish_reason = choice.finish_reason

    # ==== Hooks ====

    def _build_text_citations(self) -> list[Citation]:
        return convert_annotations(self._annotations)  # type: ignore[return-value]
