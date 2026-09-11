"""Stateful converter: LiteLLM ModelResponseStream → LlmEvent stream."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from litellm.types.llms.openai import (
    ChatCompletionAnnotation,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
)
from litellm.types.utils import ChatCompletionDeltaToolCall
from litellm.types.utils import ModelResponseStream as LiteLLMCompletionChunk

from grasp_agents.llm.llm_stream_converter import BaseLlmStreamConverter
from grasp_agents.llm_providers.openai_completions.logprob_converters import (
    convert_logprobs,
)
from grasp_agents.llm_providers.openai_completions.provider_output_to_response import (
    convert_annotations,
    convert_usage,
)
from grasp_agents.types.llm_events import ResponseCompleted
from grasp_agents.types.response import Response

from .utils import tool_call_id_and_fields, validate_chunk

LiteLLMThinkingBlock = ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai.types.completion_usage import CompletionUsage

    from grasp_agents.types.content import Annotation
    from grasp_agents.types.llm_events import LlmEvent


class LiteLLMStreamConverter(BaseLlmStreamConverter[LiteLLMCompletionChunk]):
    """Converts a LiteLLM ModelResponseStream async stream into a LlmEvent stream."""

    def __init__(self) -> None:
        super().__init__()
        self._has_thinking_blocks = False
        self._provider_specific_fields: dict[str, Any] = {}
        self._hidden_params: dict[str, Any] | None = None
        self._response_ms: float | None = None
        self._cost: float | None = None

    # ==== Per-chunk dispatch ====

    def _process_event(self, raw_event: LiteLLMCompletionChunk) -> Iterator[LlmEvent]:
        chunk = raw_event
        validate_chunk(chunk)

        # Chunk-level extras LiteLLM attaches with ``setattr`` (never declared).

        usage: CompletionUsage | None = getattr(chunk, "usage", None)
        if usage is not None:
            self._usage = convert_usage(usage)

        service_tier: str | None = getattr(chunk, "service_tier", None)
        if service_tier:
            self._service_tier = service_tier

        hidden_params: dict[str, Any] | None = getattr(chunk, "_hidden_params", None)
        if hidden_params:
            self._hidden_params = hidden_params
            cost: float | None = hidden_params.get("response_cost")
            if cost is not None:
                self._cost = cost

        response_ms: float | None = getattr(chunk, "_response_ms", None)
        if response_ms is not None:
            self._response_ms = response_ms

        if chunk.provider_specific_fields:
            self._provider_specific_fields.update(chunk.provider_specific_fields)

        if not chunk.choices:
            return

        choice = chunk.choices[0]
        delta = choice.delta

        if delta.provider_specific_fields:
            self._provider_specific_fields.update(delta.provider_specific_fields)

        # LiteLLM deletes an optional Delta field it did not fill rather than
        # leaving it ``None``, so those are read with ``getattr``.

        annotations: list[ChatCompletionAnnotation] | None = getattr(
            delta, "annotations", None
        )
        if annotations:
            self._annotations.extend(annotations)

        if not self._started:
            yield from self._start_response(
                id=chunk.id,
                model=chunk.model or "",
                created_at=float(chunk.created),
            )

        # Thinking blocks

        thinking_blocks: list[LiteLLMThinkingBlock] | None = getattr(
            delta, "thinking_blocks", None
        )
        reasoning_content: str | None = getattr(delta, "reasoning_content", None)

        if thinking_blocks:
            self._has_thinking_blocks = True
            yield from self._process_thinking_blocks(thinking_blocks)

        # Reasoning (only if no thinking_blocks, they carry the same data)

        if reasoning_content and not self._has_thinking_blocks:
            if not self._reasoning_open:
                yield from self._open_reasoning()
            if not self._reasoning_summary_part_open:
                yield from self._open_reasoning_summary_part()
            yield from self._on_reasoning_content(reasoning_content)

        # Output message

        text_content = delta.content
        refusal: str | None = getattr(delta, "refusal", None)

        if text_content or refusal:
            if self._reasoning_open:
                yield from self._close_reasoning()
            if not self._message_open:
                yield from self._open_message()

        if text_content:
            if not self._text_open:
                yield from self._open_text()

            chunk_logprobs = (
                convert_logprobs(choice.logprobs)  # type: ignore[arg-type]
                if choice.logprobs  # type: ignore[reportUnknownMemberType]
                else None
            )
            yield from self._on_text(text_content, chunk_logprobs)

        if refusal:
            if self._text_open:
                yield from self._close_text()
            if not self._refusal_open:
                yield from self._open_refusal()

            yield from self._on_refusal(refusal)

        # Tool calls (custom tool calls have no function and are not supported)

        tool_calls = [
            tc
            for tc in delta.tool_calls or []
            if isinstance(tc, ChatCompletionDeltaToolCall)
        ]
        if tool_calls:
            if self._reasoning_open:
                yield from self._close_reasoning()
            if self._message_open:
                yield from self._close_message()

            for tc in tool_calls:
                idx = tc.index
                if idx not in self._tool_calls:
                    # An extra LiteLLM sets on the call, not a declared field.
                    tc_fields: dict[str, Any] | None = getattr(
                        tc, "provider_specific_fields", None
                    )
                    call_id, fields = tool_call_id_and_fields(
                        tc.id or str(uuid4()), tc_fields
                    )
                    yield from self._open_tool_call(call_id=call_id, name="", idx=idx)
                    self._tool_calls[idx].provider_specific_fields = fields

                state = self._tool_calls[idx]
                if tc.function.name:
                    state.name += tc.function.name
                if tc.function.arguments:
                    yield from self._on_tool_call_args(idx, tc.function.arguments)

        # Gemini signs the first non-thought part. A tool call's signature was
        # taken from the call above; a text answer's arrives in a trailing chunk
        # after the text, while the message is still open. Attaching it here,
        # before the message closes, keeps every OutputItemDone final.

        thought_sigs: list[str] | None = (delta.provider_specific_fields or {}).get(
            "thought_signatures"
        )
        if thought_sigs and not tool_calls:
            self._attach_thought_signatures(thought_sigs)

        if choice.finish_reason:
            self._finish_reason = choice.finish_reason

    def _attach_thought_signatures(self, sigs: list[str]) -> None:
        if self._message_open:
            fields = dict(self._message_provider_specific_fields or {})
            fields["thought_signatures"] = [
                *fields.get("thought_signatures", []),
                *sigs,
            ]
            self._message_provider_specific_fields = fields
        elif self._reasoning_open:
            self._reasoning_encrypted_content = sigs[-1]

    # ==== Thinking blocks ====

    def _process_thinking_blocks(
        self, blocks: list[LiteLLMThinkingBlock]
    ) -> Iterator[LlmEvent]:
        for block in blocks:
            if block["type"] == "redacted_thinking":
                # Close existing reasoning cleanly (preserves accumulated text)
                if self._reasoning_open:
                    yield from self._close_reasoning()

                # New redacted block → separate item
                yield from self._open_reasoning()
                self._reasoning_encrypted_content = block.get("data")  # type: ignore[reportUnknownMemberType]
                self._reasoning_redacted = True
                yield from self._close_reasoning()

            else:
                # Write deltas into a single summary part
                # (we stream anyway, so no need to split into separate items)

                if not self._reasoning_open:
                    yield from self._open_reasoning()
                if not self._reasoning_summary_part_open:
                    yield from self._open_reasoning_summary_part()

                text = block.get("thinking", "")  # type: ignore[reportUnknownMemberType]

                sig = block.get("signature")  # type: ignore[reportUnknownMemberType]
                if sig:
                    self._reasoning_encrypted_content = sig

                if text:
                    yield from self._on_reasoning_content(text)

    # ==== Hooks ====

    def _build_text_annotations(self) -> list[Annotation]:
        return convert_annotations(self._annotations)

    # ==== Close response ====

    def _build_response_completed(self) -> ResponseCompleted:
        completed = super()._build_response_completed()
        response = completed.response

        # Patch LiteLLM-specific fields onto the response
        usage = response.usage
        if usage and self._cost is not None:
            usage = usage.model_copy(update={"cost": self._cost})

        patched = Response(
            id=response.id,
            created_at=response.created_at,
            model=response.model,
            status=response.status,
            incomplete_details=response.incomplete_details,
            output=response.output,
            usage=usage,
            service_tier=response.service_tier,  # type: ignore[arg-type]
            response_ms=self._response_ms,
            provider_specific_fields=self._provider_specific_fields or None,
            hidden_params=self._hidden_params,
        )

        return ResponseCompleted(
            response=patched, sequence_number=completed.sequence_number
        )
