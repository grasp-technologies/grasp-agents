"""Convert LiteLLM completion wire format → internal Response type."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from openai.types.chat.chat_completion import ChoiceLogprobs as CompletionLogprobs
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_status import ResponseStatus

from grasp_agents.llm_providers.openai_completions.provider_output_to_response import (
    convert_usage,
)
from grasp_agents.types.response import Response, ResponseUsage
from litellm.types.utils import Choices as LiteLLMChoice

from .items_extraction import generated_message_to_items
from .utils import validate_completion

if TYPE_CHECKING:
    from litellm.types.llms.openai import OpenAIChatCompletionFinishReason
    from litellm.types.utils import ModelResponse as LiteLLMCompletion
    from litellm.types.utils import Usage as LiteLLMUsage


def provider_output_to_response(provider_output: LiteLLMCompletion) -> Response:
    validate_completion(provider_output)

    # Completion-level fields

    response_id = provider_output.id or str(uuid4())
    created_at = provider_output.created or datetime.now(UTC).timestamp()
    model = provider_output.model or "unspecified-model"
    _ = provider_output.system_fingerprint

    hidden_params: dict[str, Any] = provider_output._hidden_params  # type: ignore # noqa: SLF001
    response_headers: dict[str, Any] = provider_output._response_headers  # type: ignore # noqa: SLF001
    # request_id = provider_output._request_id

    response_ms: float | None = getattr(provider_output, "_response_ms", None)

    raw_usage: LiteLLMUsage | None = getattr(provider_output, "usage", None)
    usage: ResponseUsage | None = None
    if raw_usage:
        usage = convert_usage(raw_usage)
        usage.cost = hidden_params.get("response_cost")

    # Choice-level fields

    raw_choice = provider_output.choices[0]
    assert isinstance(
        raw_choice, LiteLLMChoice
    )  # validate_completion should have already checked this
    finish_reason: OpenAIChatCompletionFinishReason | None = raw_choice.finish_reason
    provider_specific_fields_choice = raw_choice.provider_specific_fields

    raw_logprobs = getattr(raw_choice, "logprobs", None)
    if not isinstance(raw_logprobs, CompletionLogprobs):
        raw_logprobs = None

    incomplete_details: IncompleteDetails | None = None
    status: ResponseStatus = "completed"

    if finish_reason == "length":
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

    output_items = generated_message_to_items(
        raw_message=raw_message,
        raw_logprobs=raw_logprobs,
        model=model,
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
        output_items=output_items,
        usage_with_cost=usage,
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
