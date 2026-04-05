"""
Convert Gemini GenerateContentResponse → internal output items.

Handles all Part types (text, thinking, function_call, executable_code,
code_execution_result) and candidate-level metadata (grounding, citations,
logprobs).
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from google.genai.types import (
    CitationMetadata,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    GroundingMetadata,
    LogprobsResult,
    UrlContextMetadata,
    UrlRetrievalStatus,
)
from google.genai.types import (
    FunctionCall as GeminiFunctionCall,
)
from google.genai.types import GenerateContentResponse as GeminiResponse
from openai.types.responses import ResponseStatus
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_output_text import Logprob, LogprobTopLogprob
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from grasp_agents.types.content import OutputMessageText, ReasoningSummary, UrlCitation
from grasp_agents.types.items import (
    FunctionToolCallItem,
    OpenPageAction,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
    SearchAction,
    SearchSource,
    WebSearchCallItem,
    prefixed_id,
)
from grasp_agents.types.response import (
    Response,
    ResponseUsage,
)

from . import encode_thought_signature


@dataclass
class _ThinkingPartAccumulator:
    parts: list[ReasoningSummary] = field(default_factory=list[ReasoningSummary])
    signature: bytes | None = None

    def append(self, text: str, signature: bytes | None = None) -> None:
        self.parts.append(ReasoningSummary(text=text))
        if signature is not None:
            self.signature = signature

    def flush(self) -> None:
        self.parts = []
        self.signature = None


@dataclass
class _TextPartAccumulator:
    parts: list[OutputMessageText] = field(default_factory=list[OutputMessageText])
    signature: bytes | None = None

    def append(self, text: str, signature: bytes | None = None) -> None:
        self.parts.append(OutputMessageText(text=text))
        if signature is not None:
            self.signature = signature

    def flush(self) -> None:
        self.parts = []
        self.signature = None


def _gemini_response_to_items(
    response: GeminiResponse,
) -> list[OutputItem]:
    """
    Convert a Gemini response's parts to output items.

    Consecutive text parts are merged into a single OutputMessageItem
    (one content part per text block). Consecutive thinking parts are merged
    into a single ReasoningItem (one summary part per thinking block).
    Thought signatures are preserved as encrypted_content.
    """
    candidates = response.candidates
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return []

    candidate = candidates[0]
    assert candidate.content is not None
    parts = candidate.content.parts
    assert parts is not None

    items: list[OutputItem] = []
    pending_text = _TextPartAccumulator()
    pending_thinking = _ThinkingPartAccumulator()

    def _flush_text() -> None:
        if pending_text.parts:
            items.append(_merge_text_parts(pending_text))
            pending_text.flush()

    def _flush_thinking() -> None:
        if pending_thinking.parts:
            items.append(_merge_thinking_parts(pending_thinking))
            pending_thinking.flush()

    for part in parts:
        if part.function_call:
            _flush_text()
            _flush_thinking()

            # Gemini does not stream tool arguments,
            # so we can convert tool calls immediately without accumulation.
            items.append(
                _function_call_to_tool_call_item(
                    part.function_call, part.thought_signature
                )
            )

        elif part.thought and part.text is not None:
            _flush_text()
            pending_thinking.append(part.text, part.thought_signature)

        elif part.text is not None:
            _flush_thinking()
            pending_text.append(part.text, part.thought_signature)

    _flush_text()
    _flush_thinking()

    # Attach candidate-level metadata to items
    attach_grounding_annotations(items, candidate.grounding_metadata)
    attach_citation_annotations(items, candidate.citation_metadata)
    attach_logprobs(items, candidate.logprobs_result)

    web_search_item = extract_web_search_data(candidate.grounding_metadata)
    if web_search_item:
        items.append(web_search_item)

    items.extend(extract_url_context_data(candidate.url_context_metadata))

    return items


def _merge_text_parts(acc: _TextPartAccumulator) -> OutputMessageItem:
    sig = acc.signature
    encoded_sig = encode_thought_signature(sig) if sig else None

    return OutputMessageItem(
        status="completed",
        content_parts=list(acc.parts),
        provider_specific_fields=(
            {"thought_signature": encoded_sig} if encoded_sig else None
        ),
    )


def _merge_thinking_parts(acc: _ThinkingPartAccumulator) -> ReasoningItem:
    sig = acc.signature
    encoded_sig = encode_thought_signature(sig) if sig else None

    return ReasoningItem(
        status="completed", summary_parts=acc.parts, encrypted_content=encoded_sig
    )


def _function_call_to_tool_call_item(
    function_call: GeminiFunctionCall, thought_sig: bytes | None = None
) -> FunctionToolCallItem:
    encoded_sig = encode_thought_signature(thought_sig) if thought_sig else None

    return FunctionToolCallItem(
        call_id=function_call.id or prefixed_id("fc"),
        name=function_call.name or "",
        arguments=json.dumps(function_call.args),
        status="completed",
        provider_specific_fields=(
            {"thought_signature": encoded_sig} if encoded_sig else None
        ),
    )


def extract_web_search_data(
    grounding: GroundingMetadata | None,
) -> WebSearchCallItem | None:
    if not grounding:
        return None

    queries = list(grounding.web_search_queries or [])

    sources = [
        SearchSource(url=c.web.uri, title=c.web.title or "")
        for c in grounding.grounding_chunks or []
        if c.web and c.web.uri
    ]

    if not queries and not sources:
        return None

    entry_point_html = (
        grounding.search_entry_point.rendered_content
        if grounding.search_entry_point
        else None
    )

    return WebSearchCallItem(
        status="completed",
        action=SearchAction(queries=queries, sources=sources),
        provider_specific_fields=(
            {"gemini:entry_point_html": entry_point_html} if entry_point_html else None
        ),
    )


def extract_url_context_data(
    url_context: UrlContextMetadata | None,
) -> list[WebSearchCallItem]:
    """Convert UrlContextMetadata → WebSearchCallItem(OpenPageAction) per URL."""
    if not url_context or not url_context.url_metadata:
        return []

    items: list[WebSearchCallItem] = []
    for meta in url_context.url_metadata:
        url = meta.retrieved_url or ""
        status = meta.url_retrieval_status
        failed = status and status != UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS

        item = WebSearchCallItem(
            status="failed" if failed else "completed",
            action=OpenPageAction(url=url),
            provider_specific_fields=(
                {"gemini:url_retrieval_status": str(status)} if failed else None
            ),
        )
        items.append(item)

    return items


# ==== Grounding annotations ====


def _get_full_text(items: list[OutputItem]) -> str:
    """Concatenate all text content parts into a single string."""
    parts: list[str] = []
    for item in items:
        if isinstance(item, OutputMessageItem):
            for part in item.content_parts:
                if isinstance(part, OutputMessageText):
                    parts.append(part.text)
    return "".join(parts)


def attach_grounding_annotations(
    items: list[OutputItem], grounding: GroundingMetadata | None
) -> None:
    if not grounding:
        return

    chunks = grounding.grounding_chunks or []
    supports = grounding.grounding_supports or []

    # Build full text for byte → character offset conversion.
    # Gemini Segment offsets are byte-based (UTF-8).
    full_text = _get_full_text(items)
    full_bytes = full_text.encode("utf-8")

    annotations: list[UrlCitation] = []

    for support in supports:
        if not support.grounding_chunk_indices:
            continue

        segment = support.segment
        byte_start = segment.start_index if segment and segment.start_index else 0
        byte_end = segment.end_index if segment and segment.end_index else 0

        # Convert byte offsets to character offsets
        char_start = len(full_bytes[:byte_start].decode("utf-8", errors="replace"))
        char_end = len(full_bytes[:byte_end].decode("utf-8", errors="replace"))
        grounded_text = segment.text if segment else None

        for chunk_idx in support.grounding_chunk_indices:
            if chunk_idx >= len(chunks):
                continue

            chunk = chunks[chunk_idx]
            web = chunk.web
            if not web or not web.uri:
                continue

            provider_specific_fields = (
                {"gemini:grounded_text": grounded_text} if grounded_text else None
            )

            annotations.append(
                UrlCitation(
                    type="url_citation",
                    url=web.uri,
                    title=web.title or "",
                    start_index=char_start,
                    end_index=char_end,
                    provider_specific_fields=provider_specific_fields,
                )
            )

    if annotations:
        _distribute_annotations(items, annotations)


# ==== Citation annotations ====


def attach_citation_annotations(
    items: list[OutputItem],
    citation_meta: CitationMetadata | None,
) -> None:
    if not citation_meta or not citation_meta.citations:
        return

    annotations: list[UrlCitation] = []
    for citation in citation_meta.citations:
        if not citation.uri:
            continue
        annotations.append(
            UrlCitation(
                type="url_citation",
                url=citation.uri,
                title=citation.title or "",
                start_index=citation.start_index or 0,
                end_index=citation.end_index or 0,
            )
        )

    if annotations:
        _distribute_annotations(items, annotations)


def _distribute_annotations(
    items: list[OutputItem],
    annotations: list[UrlCitation],
) -> None:
    """Attach annotations to the correct text content parts by char offset."""
    text_parts: list[tuple[int, int, OutputMessageText]] = []
    offset = 0
    for item in items:
        if isinstance(item, OutputMessageItem):
            for part in item.content_parts:
                if isinstance(part, OutputMessageText):
                    end = offset + len(part.text)
                    text_parts.append((offset, end, part))
                    offset = end

    if not text_parts:
        return

    for ann in annotations:
        for start, end, part in text_parts:
            if ann.start_index < end and ann.end_index > start:
                adjusted = UrlCitation(
                    type="url_citation",
                    url=ann.url,
                    title=ann.title,
                    start_index=max(0, ann.start_index - start),
                    end_index=min(end - start, ann.end_index - start),
                    provider_specific_fields=ann.provider_specific_fields,
                )
                part.annotations.append(adjusted)
                break


# ==== Logprobs ====


def attach_logprobs(
    items: list[OutputItem],
    logprobs_result: LogprobsResult | None,
) -> None:
    if not logprobs_result:
        return

    chosen = logprobs_result.chosen_candidates or []
    top = logprobs_result.top_candidates or []
    if not chosen:
        return

    logprobs: list[Logprob] = []
    for i, candidate in enumerate(chosen):
        top_logprobs: list[LogprobTopLogprob] | None = None
        if i < len(top) and top[i].candidates:
            top_logprobs = [
                LogprobTopLogprob(
                    token=tc.token or "",
                    logprob=tc.log_probability or 0.0,
                    bytes=[],
                )
                for tc in (top[i].candidates or [])
            ]
        logprobs.append(
            Logprob(
                token=candidate.token or "",
                logprob=candidate.log_probability or 0.0,
                bytes=[],
                top_logprobs=top_logprobs or [],
            )
        )

    # Collect all text parts across output items
    text_parts: list[OutputMessageText] = []
    for item in items:
        if isinstance(item, OutputMessageItem):
            for part in item.content_parts:
                if isinstance(part, OutputMessageText):
                    text_parts.append(part)

    if not text_parts:
        return

    # Distribute logprobs by matching token text against part text.
    # Gemini logprobs cover ALL decoding steps (thinking, text, etc.)
    # so non-matching tokens (thinking, function calls) are skipped.
    _distribute_logprobs(text_parts, logprobs)


def _distribute_logprobs(
    text_parts: list[OutputMessageText], logprobs: list[Logprob]
) -> None:
    token_idx = 0

    for part in text_parts:
        part_logprobs: list[Logprob] = []
        matched_len = 0

        while token_idx < len(logprobs):
            remaining = part.text[matched_len:]
            if not remaining:
                break  # fully matched this part

            token = logprobs[token_idx].token
            if remaining.startswith(token):
                part_logprobs.append(logprobs[token_idx])
                matched_len += len(token)
                token_idx += 1
            elif matched_len:
                # Already matching this part but token doesn't fit —
                # move to next part
                break
            else:
                # Haven't started matching yet — skip (thinking/other)
                token_idx += 1

        if part_logprobs:
            part.logprobs = part_logprobs


def _convert_usage(usage: GenerateContentResponseUsageMetadata | None) -> ResponseUsage:
    if not usage:
        return ResponseUsage()

    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    total_tokens = usage.total_token_count or 0
    cached = usage.cached_content_token_count or 0
    thinking = usage.thoughts_token_count or 0

    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=thinking),
    )


def _map_finish_reason(
    response: GenerateContentResponse,
) -> tuple[ResponseStatus, IncompleteDetails | None]:
    if not response.candidates or not response.candidates[0].finish_reason:
        return "completed", None

    reason = response.candidates[0].finish_reason.name

    if reason in {"STOP", "FINISH_REASON_STOP"}:
        return "completed", None

    if reason in {"MAX_TOKENS", "FINISH_REASON_MAX_TOKENS"}:
        return "incomplete", IncompleteDetails(reason="max_output_tokens")

    if reason in {
        "SAFETY",
        "FINISH_REASON_SAFETY",
        "BLOCKLIST",
        "RECITATION",
    }:
        return "incomplete", IncompleteDetails(reason="content_filter")

    return "completed", None


def provider_output_to_response(provider_output: GenerateContentResponse) -> Response:
    """Convert a Gemini ``GenerateContentResponse`` to a ``Response``."""
    output_items = _gemini_response_to_items(provider_output)

    usage = _convert_usage(provider_output.usage_metadata)
    status, incomplete_details = _map_finish_reason(provider_output)

    created_at: float = (
        provider_output.create_time.timestamp()
        if provider_output.create_time
        else datetime.now(UTC).timestamp()
    )

    return Response(
        id=provider_output.response_id or prefixed_id("resp"),
        created_at=created_at,
        model=provider_output.model_version or "<unknown-model>",
        status=status,
        incomplete_details=incomplete_details,
        output_items=output_items,
        usage_with_cost=usage,
    )
