from typing import cast

from grasp_agents.errors import CompletionError

from ..typing.completion import Completion, Usage
from . import LiteLLMChoice, LiteLLMCompletion, LiteLLMUsage
from .message_converters import from_api_assistant_message


def from_api_completion_usage(api_usage: LiteLLMUsage) -> Usage:
    reasoning_tokens = None
    cached_tokens = None

    if api_usage.completion_tokens_details is not None:
        reasoning_tokens = api_usage.completion_tokens_details.reasoning_tokens
    if api_usage.prompt_tokens_details is not None:
        cached_tokens = api_usage.prompt_tokens_details.cached_tokens

    input_tokens = api_usage.prompt_tokens - (cached_tokens or 0)
    output_tokens = api_usage.completion_tokens  # - (reasoning_tokens or 0)

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
    )


def from_api_completion(
    api_completion: LiteLLMCompletion, name: str | None = None
) -> Completion:
    if len(api_completion.choices) > 1:
        raise CompletionError("Multiple choices are not supported")
    api_choice = cast("LiteLLMChoice", api_completion.choices[0])

    message = from_api_assistant_message(api_choice.message, name=name)

    api_usage = getattr(api_completion, "usage", None)
    usage: Usage | None = None
    if api_usage:
        usage = from_api_completion_usage(cast("LiteLLMUsage", api_usage))
        hidden_params = getattr(api_completion, "_hidden_params", {})
        usage.cost = hidden_params.get("response_cost")

    return Completion(
        id=api_completion.id,
        model=api_completion.model,
        name=name,
        created=api_completion.created,
        system_fingerprint=api_completion.system_fingerprint,
        message=message,
        finish_reason=api_choice.finish_reason,  # type: ignore[assignment, arg-type]
        logprobs=getattr(api_choice, "logprobs", None),
        usage=usage,
        provider_specific_fields=getattr(api_choice, "provider_specific_fields", None),
        hidden_params=api_completion._hidden_params,  # type: ignore[union-attr]
        response_ms=getattr(api_completion, "_response_ms", None),
    )


def to_api_completion(completion: Completion) -> LiteLLMCompletion:
    raise NotImplementedError
