"""Convert LiteLLM completion wire format → internal Response type."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from litellm.types.llms.openai import OpenAIChatCompletionFinishReason
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_status import ResponseStatus
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from grasp_agents.errors import CompletionError
from grasp_agents.typing.response import Response, ResponseUsage

from . import LiteLLMChoice
from .item_converters import from_litellm_completions_message

if TYPE_CHECKING:
    from . import LiteLLMCompletion, LiteLLMUsage


def _convert_usage(raw_usage: LiteLLMUsage) -> ResponseUsage:
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


def from_litellm_completion(raw_completion: LiteLLMCompletion) -> Response:
    # Completion-level fields

    response_id = raw_completion.id or str(uuid4())
    created_at = raw_completion.created or datetime.now(UTC).timestamp()
    model = raw_completion.model or "unspecified-model"
    _ = raw_completion.system_fingerprint

    hidden_params: dict[str, Any] = raw_completion._hidden_params  # type: ignore # noqa: SLF001
    response_headers: dict[str, Any] = raw_completion._response_headers  # type: ignore # noqa: SLF001
    # request_id = raw_completion._request_id

    response_ms: float | None = getattr(raw_completion, "_response_ms", None)

    raw_usage: LiteLLMUsage | None = getattr(raw_completion, "usage", None)
    usage: ResponseUsage | None = None
    if raw_usage:
        usage = _convert_usage(raw_usage)
        usage.cost = hidden_params.get("response_cost")

    # Choice-level fields

    if not raw_completion.choices:
        raise CompletionError("No choices in completion")

    if len(raw_completion.choices) > 1:
        raise CompletionError("Multiple choices are not supported")

    raw_choice = raw_completion.choices[0]
    if not isinstance(raw_choice, LiteLLMChoice):
        raise CompletionError("choice is not a LiteLLM Choice")

    finish_reason: OpenAIChatCompletionFinishReason | None = raw_choice.finish_reason
    logprobs = raw_choice.logprobs
    provider_specific_fields_choice = raw_choice.provider_specific_fields

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
    elif finish_reason == "guardrail_intervened":
        status = "incomplete"

    # Message-level fields

    raw_message = raw_choice.message
    provider_specific_fields_message = raw_message.provider_specific_fields

    refusal: str | None = getattr(raw_message, "refusal", None)
    if refusal is None and finish_reason == "guardrail_intervened":
        refusal = "guardrail_intervened"

    output_items = from_litellm_completions_message(
        raw_message=raw_message,
        raw_logprobs=logprobs,
        refusal=refusal,
        output_message_status=status,
    )

    if provider_specific_fields_choice or provider_specific_fields_message:
        provider_specific_fields = (provider_specific_fields_choice or {}) | (
            provider_specific_fields_message or {}
        )
    else:
        provider_specific_fields = None

    return Response(
        id=response_id,
        created_at=created_at,
        output_ext=output_items,
        usage_ext=usage,
        error=None,
        metadata=None,
        incomplete_details=incomplete_details,
        status=status,
        model=model,
        response_ms=response_ms,
        provider_specific_fields=provider_specific_fields,
        hidden_params=hidden_params,
        response_headers=response_headers,
    )
