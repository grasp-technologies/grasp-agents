"""Convert OpenAI Chat Completions API wire format → internal Response type."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_status import ResponseStatus
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from grasp_agents.types.response import Response, ResponseUsage

from .items_extraction import generated_message_to_items
from .utils import validate_completion

if TYPE_CHECKING:
    from openai.types import CompletionUsage
    from openai.types.chat.chat_completion import ChatCompletion


def convert_usage(raw_usage: CompletionUsage) -> ResponseUsage:
    cached_tokens = 0
    reasoning_tokens = 0

    if raw_usage.prompt_tokens_details is not None:
        cached_tokens = raw_usage.prompt_tokens_details.cached_tokens or 0

    if raw_usage.completion_tokens_details is not None:
        reasoning_tokens = raw_usage.completion_tokens_details.reasoning_tokens or 0

    return ResponseUsage(
        input_tokens=raw_usage.prompt_tokens,
        output_tokens=raw_usage.completion_tokens,
        total_tokens=raw_usage.total_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
    )


def provider_output_to_response(provider_output: ChatCompletion) -> Response:
    """Convert an OpenAI Chat Completion → internal Response."""
    validate_completion(provider_output)

    # Completion-level fields

    _ = provider_output.system_fingerprint

    usage: ResponseUsage | None = None
    raw_usage = provider_output.usage
    if raw_usage:
        usage = convert_usage(raw_usage)

    # Choice-level fields

    raw_choice = provider_output.choices[0]
    finish_reason = raw_choice.finish_reason
    logprobs = raw_choice.logprobs

    incomplete_details: IncompleteDetails | None = None
    status: ResponseStatus = "completed"

    if finish_reason == "length":
        status = "incomplete"
        incomplete_details = IncompleteDetails(reason="max_output_tokens")
    elif finish_reason == "content_filter":
        status = "incomplete"
        incomplete_details = IncompleteDetails(reason="content_filter")

    # Message-level fields

    raw_message = raw_choice.message

    output_items = generated_message_to_items(
        raw_message=raw_message,
        raw_logprobs=logprobs,
        output_message_status=status,
    )

    return Response(
        id=provider_output.id,
        created_at=float(provider_output.created),
        model=provider_output.model,
        output_items=output_items,
        usage_with_cost=usage,
        status=status,
        incomplete_details=incomplete_details,
        service_tier=provider_output.service_tier,
    )
