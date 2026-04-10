"""
Stateful converter: Gemini GenerateContentResponse stream → LlmEvent stream.

Consumes Gemini's chunk-based streaming protocol (each chunk is a full
GenerateContentResponse with delta parts) and emits OpenResponses
lifecycle events using the shared BaseLlmStreamConverter.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from grasp_agents.llm.llm_stream_converter import BaseLlmStreamConverter
from grasp_agents.types.items import prefixed_id
from grasp_agents.types.response import ResponseUsage

from . import GeminiResponse, encode_thought_signature
from .provider_output_to_response import (
    attach_grounding_annotations,
    extract_url_context_data,
    extract_web_search_data,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from google.genai.types import GroundingMetadata, UrlContextMetadata
    from openai.types.responses import ResponseStatus

    from grasp_agents.types.llm_events import LlmEvent

    from . import GeminiPart


class GeminiStreamConverter(BaseLlmStreamConverter[GeminiResponse]):
    """
    Converts a Gemini GenerateContentResponse async stream into LlmEvents.

    Gemini streaming is chunk-based: each chunk is a full
    GenerateContentResponse with candidates[0].content.parts containing
    delta content. The converter detects state transitions (thinking → text
    → function_call) to emit the correct OpenResponses lifecycle events.
    """

    def __init__(self, *, model: str = "") -> None:
        super().__init__()
        self._model = model
        self._tool_call_idx: int = 0
        self._grounding: GroundingMetadata | None = None
        self._url_context: UrlContextMetadata | None = None

    def _process_event(self, raw_event: GeminiResponse) -> Iterator[LlmEvent]:
        chunk = raw_event

        # Start response on first chunk
        if not self._started:
            created_at = (
                chunk.create_time.timestamp() if chunk.create_time else time.time()
            )
            yield from self._start_response(
                id=chunk.response_id or prefixed_id("resp"),
                model=self._model,
                created_at=created_at,
            )

        candidate = chunk.candidates[0] if chunk.candidates else None

        if candidate and candidate.finish_reason:
            self._finish_reason = candidate.finish_reason.name

        if candidate and candidate.grounding_metadata:
            self._grounding = candidate.grounding_metadata
        if candidate and candidate.url_context_metadata:
            self._url_context = candidate.url_context_metadata

        um = chunk.usage_metadata
        if um is not None:
            self._usage = ResponseUsage(
                input_tokens=um.prompt_token_count or 0,
                output_tokens=um.candidates_token_count or 0,
                total_tokens=um.total_token_count or 0,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=um.cached_content_token_count or 0
                ),
                output_tokens_details=OutputTokensDetails(
                    reasoning_tokens=um.thoughts_token_count or 0
                ),
            )

        if not candidate or not candidate.content or not candidate.content.parts:
            return

        for part in candidate.content.parts:
            yield from self._process_part(part)

    def _process_part(self, part: GeminiPart) -> Iterator[LlmEvent]:
        if part.thought and part.text is not None:
            yield from self._on_thinking(part)

        elif part.text is not None:
            yield from self._on_regular_text(part)

        elif part.function_call:
            yield from self._on_function_call(part)

    def _on_thinking(self, part: GeminiPart) -> Iterator[LlmEvent]:
        # Gemini puts thought_signature on text parts and function call parts,
        # not on thinking parts (weirdly).

        if not self._reasoning_open:
            yield from self._open_reasoning()
        if not self._reasoning_summary_part_open:
            yield from self._open_reasoning_summary_part()

        yield from self._on_reasoning_content(part.text or "")

    def _on_regular_text(self, part: GeminiPart) -> Iterator[LlmEvent]:
        if self._reasoning_open:
            yield from self._close_reasoning()
        if not self._message_open:
            yield from self._open_message()
        if not self._text_open:
            yield from self._open_text()

        if part.thought_signature is not None:
            self._message_provider_specific_fields = {
                "thought_signature": encode_thought_signature(part.thought_signature)
            }

        yield from self._on_text(part.text or "")

    def _on_function_call(self, part: GeminiPart) -> Iterator[LlmEvent]:
        if self._reasoning_open:
            yield from self._close_reasoning()
        if self._message_open:
            yield from self._close_message()

        fc = part.function_call
        if fc is None:
            return

        call_id = fc.id or prefixed_id("call")
        name = fc.name or ""
        idx = self._tool_call_idx
        self._tool_call_idx += 1

        yield from self._open_tool_call(call_id=call_id, name=name, idx=idx)
        if fc.args:
            yield from self._on_tool_call_args(idx, json.dumps(fc.args))

        if part.thought_signature is not None:
            self._tool_calls[idx].provider_specific_fields = {
                "thought_signature": encode_thought_signature(part.thought_signature)
            }

    def _close_response(self) -> Iterator[LlmEvent]:
        if self._reasoning_open:
            yield from self._close_reasoning()
        if self._message_open:
            yield from self._close_message()
        if self._tool_calls:
            yield from self._close_tool_calls()

        if self._grounding:
            attach_grounding_annotations(self._items, self._grounding)

        # NOTE: add citations_metadata

        # Emit url_context items via web search lifecycle events
        for url_item in extract_url_context_data(self._url_context):
            yield from self._open_web_search(url_item.id)
            yield from self._close_web_search(url_item)

        # Emit web search item from grounding metadata
        if self._grounding:
            web_search_item = extract_web_search_data(self._grounding)
            if web_search_item:
                yield from self._open_web_search(web_search_item.id)
                yield from self._close_web_search(web_search_item)

        yield super()._build_response_completed()

    def _map_finish_reason(
        self,
    ) -> tuple[ResponseStatus, IncompleteDetails | None]:
        if self._finish_reason in {"STOP", "FINISH_REASON_STOP"}:
            return "completed", None

        if self._finish_reason in {
            "MAX_TOKENS",
            "FINISH_REASON_MAX_TOKENS",
        }:
            return "incomplete", IncompleteDetails(reason="max_output_tokens")

        if self._finish_reason in {
            "SAFETY",
            "FINISH_REASON_SAFETY",
            "BLOCKLIST",
            "RECITATION",
        }:
            return "incomplete", IncompleteDetails(reason="content_filter")

        return "completed", None
