"""Convert Anthropic Message → grasp-agents Response."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from grasp_agents.types.response import Response, ResponseUsage

from .items_extraction import generated_message_to_items

if TYPE_CHECKING:
    from openai.types.responses import ResponseStatus

    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import StopReason as AnthropicStopReason
    from anthropic.types import Usage as AnthropicUsage


def provider_output_to_response(provider_output: AnthropicMessage) -> Response:
    """Convert an Anthropic ``Message`` to a grasp-agents ``Response``."""
    # NOTE: ignored AnthropicMessage fields: `container`, `model``, `stop_sequence`

    output_items, web_search = generated_message_to_items(provider_output)
    usage = convert_usage(provider_output.usage)
    status, incomplete_details = _map_stop_reason(provider_output.stop_reason)

    return Response(
        id=provider_output.id,
        model=provider_output.model,
        status=status,
        incomplete_details=incomplete_details,
        output_items=output_items,
        usage_with_cost=usage,
        web_search=web_search,
    )


def convert_usage(usage: AnthropicUsage) -> ResponseUsage:
    # TODO: more cached token details (extend ResponseUsage?)

    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cached = getattr(usage, "cache_read_input_tokens", None) or 0

    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _map_stop_reason(
    stop_reason: AnthropicStopReason | None,
) -> tuple[ResponseStatus, IncompleteDetails | None]:
    if stop_reason in {"end_turn", "tool_use", "stop_sequence", "pause_turn"}:
        return "completed", None

    if stop_reason == "max_tokens":
        return "incomplete", IncompleteDetails(reason="max_output_tokens")

    if stop_reason == "refusal":
        return "incomplete", IncompleteDetails(reason="content_filter")

    return "completed", None
