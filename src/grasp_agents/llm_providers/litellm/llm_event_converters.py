"""
Stateful converter: LiteLLM ModelResponseStream → LlmEvent stream.

Extends the base CompletionsStreamConverter with:
- structured thinking_blocks (signatures, redacted blocks)
- annotations (URL citations from web search)
- provider-specific metadata (hidden_params, response_ms, cost,
  provider_specific_fields)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from openai.types.chat import ChatCompletionChunk

from grasp_agents.llm_providers.openai_completions.llm_event_converters import (
    CompletionsStreamConverter,
)
from grasp_agents.llm_providers.openai_completions.logprob_converters import (
    convert_logprobs,
)
from grasp_agents.llm_providers.openai_completions.provider_output_to_response import (
    convert_usage,
)
from grasp_agents.types.llm_events import ResponseCompleted
from grasp_agents.types.response import Response
from litellm.types.llms.openai import (
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
)
from litellm.types.utils import ModelResponseStream as LiteLLMCompletionChunk

from .utils import patch_thought_signatures, validate_chunk

LiteLLMThinkingBlock = ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock

if TYPE_CHECKING:
    from collections.abc import Iterator

    from grasp_agents.types.llm_events import LlmEvent


class LiteLLMStreamConverter(CompletionsStreamConverter):
    """
    Converts a LiteLLM ModelResponseStream async stream into a LlmEvent stream.

    Extends CompletionsStreamConverter with thinking_blocks handling for models
    that support thinking/reasoning (e.g., Anthropic Claude, DeepSeek), and
    captures LiteLLM-specific metadata (hidden_params, cost, response_ms, etc.).
    """

    def __init__(self) -> None:
        super().__init__()
        self._has_thinking_blocks = False
        self._provider_specific_fields: dict[str, Any] = {}
        self._hidden_params: dict[str, Any] | None = None
        self._response_ms: float | None = None
        self._cost: float | None = None

    # ==== Per-chunk dispatch ====

    def _process_event(
        self, raw_event: ChatCompletionChunk | LiteLLMCompletionChunk
    ) -> Iterator[LlmEvent]:
        chunk = raw_event

        if not isinstance(chunk, LiteLLMCompletionChunk):
            raise TypeError(
                f"Unsupported chunk type: {type(chunk)}. "
                f"Expected LiteLLMCompletionChunk."
            )

        validate_chunk(chunk)

        # Metadata (all dynamic on chunk)

        raw_usage: Any = getattr(chunk, "usage", None)
        if raw_usage is not None:
            self._usage = convert_usage(raw_usage)

        service_tier: str | None = getattr(chunk, "service_tier", None)
        if service_tier:
            self._service_tier = service_tier

        hidden_params: dict[str, Any] | None = getattr(chunk, "_hidden_params", None)
        if hidden_params:
            self._hidden_params = hidden_params
            cost: float | None = hidden_params.get("response_cost")
            if cost is not None:
                self._cost = cost

        response_ms: Any = getattr(chunk, "_response_ms", None)
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

        # Annotations (deleted from Delta when None)

        annotations: list[Any] | None = getattr(delta, "annotations", None)
        if annotations:
            self._annotations.extend(annotations)

        if not self._started:
            yield from self._start_response(
                id=chunk.id,
                model=chunk.model or "",
                created_at=float(chunk.created),
            )

        # Thinking blocks (deleted from Delta when None)

        thinking_blocks: list[LiteLLMThinkingBlock] | None = getattr(
            delta, "thinking_blocks", None
        )
        reasoning_content: str | None = getattr(delta, "reasoning_content", None)

        if thinking_blocks:
            self._has_thinking_blocks = True
            yield from self._process_thinking_blocks(thinking_blocks)

        # Reasoning (deleted from Delta when None —
        # only if no thinking_blocks, they carry the same data)

        if reasoning_content and not self._has_thinking_blocks:
            if not self._reasoning_open:
                yield from self._open_reasoning()
            if not self._reasoning_summary_part_open:
                yield from self._open_reasoning_summary_part()
            yield from self._on_reasoning_content(reasoning_content)

        # Output message (deleted from Delta when None)

        text_content: str | None = getattr(delta, "content", None)
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

        # Tool calls (deleted from Delta when None)

        tool_calls: list[Any] | None = getattr(delta, "tool_calls", None)
        if tool_calls:
            if self._reasoning_open:
                yield from self._close_reasoning()
            if self._message_open:
                yield from self._close_message()

            for tc in tool_calls:
                idx: int = getattr(tc, "index", 0)
                if idx not in self._tool_calls:
                    call_id: str = getattr(tc, "id", None) or str(uuid4())
                    yield from self._open_tool_call(call_id=call_id, name="", idx=idx)

                fn = getattr(tc, "function", None)
                if fn:
                    name: str | None = getattr(fn, "name", None)
                    if name:
                        self._tool_calls[idx].name += name
                    args: str | None = getattr(fn, "arguments", None)
                    if args:
                        yield from self._on_tool_call_args(idx, args)

        if choice.finish_reason:
            self._finish_reason = choice.finish_reason

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

    # ==== Close response ====

    def _close_response(self) -> Iterator[LlmEvent]:
        """Close open items, patch thought signatures, emit ResponseCompleted."""
        thought_sigs = self._provider_specific_fields.get("thought_signatures", [])
        if thought_sigs:
            patch_thought_signatures(thought_sigs, self._items, self._tool_calls)

        yield from super()._close_response()

    def _build_response_completed(self) -> ResponseCompleted:
        completed = super()._build_response_completed()
        response = completed.response

        # Patch LiteLLM-specific fields onto the response
        usage = response.usage_with_cost
        if usage and self._cost is not None:
            usage = usage.model_copy(update={"cost": self._cost})

        patched = Response(
            id=response.id,
            created_at=response.created_at,
            model=response.model,
            status=response.status,
            incomplete_details=response.incomplete_details,
            output_items=response.output_items,
            usage_with_cost=usage,
            service_tier=response.service_tier,  # type: ignore[arg-type]
            response_ms=self._response_ms,
            provider_specific_fields=self._provider_specific_fields or None,
            hidden_params=self._hidden_params,
        )

        return ResponseCompleted(
            response=patched, sequence_number=completed.sequence_number
        )
