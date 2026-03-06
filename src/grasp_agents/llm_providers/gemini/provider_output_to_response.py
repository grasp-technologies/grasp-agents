"""Convert Gemini GenerateContentResponse → grasp-agents Response."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from grasp_agents.types.response import Response, ResponseUsage

from .items_extraction import extract_web_search_info, generated_message_to_items

if TYPE_CHECKING:
    from google.genai.types import (
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
    )
    from openai.types.responses import ResponseStatus


def provider_output_to_response(provider_output: GenerateContentResponse) -> Response:
    """Convert a Gemini ``GenerateContentResponse`` to a ``Response``."""
    output_items = generated_message_to_items(provider_output)
    usage = convert_usage(provider_output.usage_metadata)
    status, incomplete_details = _map_finish_reason(provider_output)
    web_search = extract_web_search_info(provider_output)

    model = provider_output.model_version or "<unknown-model>"

    created_at: float = (
        provider_output.create_time.timestamp()
        if provider_output.create_time
        else 0.0
    )

    return Response(
        id=provider_output.response_id or str(uuid4()),
        created_at=created_at,
        model=model,
        status=status,
        incomplete_details=incomplete_details,
        output_items=output_items,
        usage_with_cost=usage,
        web_search=web_search,
    )


def convert_usage(usage: GenerateContentResponseUsageMetadata | None) -> ResponseUsage:
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
