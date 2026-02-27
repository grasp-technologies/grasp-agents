"""Convert OpenAI Chat Completions API wire format → internal Response type."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_status import ResponseStatus
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from ...errors import CompletionError
from ...typing.response import Response, ResponseUsage
from .item_converters import from_openai_completions_message

if TYPE_CHECKING:
    from . import OpenAICompletion, OpenAIUsage


def _convert_usage(raw_usage: OpenAIUsage) -> ResponseUsage:
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


def from_openai_completion(raw_completion: OpenAICompletion) -> Response:
    """Convert an OpenAI Chat Completion → internal Response."""
    # Completion-level fields

    if raw_completion.choices is None:  # type: ignore[comparison-overlap]
        raise CompletionError(
            f"Completion API error: {getattr(raw_completion, 'error', None)}"
        )
    if not raw_completion.choices:
        raise CompletionError("No choices in completion")

    raw_usage = raw_completion.usage
    usage = _convert_usage(raw_usage) if raw_usage else None

    # Choice-level fields

    raw_choice = raw_completion.choices[0]
    if raw_choice.message is None:  # type: ignore[comparison-overlap]
        raise CompletionError(
            f"API returned None for message, finish_reason: {raw_choice.finish_reason}"
        )

    finish_reason = raw_choice.finish_reason
    logprobs = raw_choice.logprobs

    incomplete_details: IncompleteDetails | None = None
    status: ResponseStatus = "completed"

    if finish_reason in {"stop", "tool_calls"}:
        status = "completed"
    elif finish_reason == "length":
        status = "incomplete"
        incomplete_details = IncompleteDetails(reason="max_output_tokens")
    elif finish_reason == "content_filter":
        status = "incomplete"
        incomplete_details = IncompleteDetails(reason="content_filter")

    # Message-level fields

    raw_message = raw_choice.message

    output_items = from_openai_completions_message(
        raw_message=raw_message,
        raw_logprobs=logprobs,
        refusal=raw_message.refusal,
        output_message_status=status,
    )

    return Response(
        id=raw_completion.id,
        created_at=float(raw_completion.created),
        model=raw_completion.model,
        output_ext=output_items,
        usage_ext=usage,
        status=status,
        incomplete_details=incomplete_details,
        service_tier=raw_completion.service_tier,
    )
