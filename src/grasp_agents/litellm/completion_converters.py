from typing import cast

from ..typing.completion import Completion, CompletionChoice, Usage
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
    choices: list[CompletionChoice] = []
    usage: Usage | None = None

    for api_choice in api_completion.choices:
        assert isinstance(api_choice, LiteLLMChoice)

        message = from_api_assistant_message(api_choice.message, name=name)

        choices.append(
            CompletionChoice(
                index=api_choice.index,
                message=message,
                finish_reason=api_choice.finish_reason,  # type: ignore[assignment, arg-type]
                logprobs=getattr(api_choice, "logprobs", None),
                provider_specific_fields=getattr(
                    api_choice, "provider_specific_fields", None
                ),
            )
        )

    api_usage = getattr(api_completion, "usage", None)
    usage = None
    if api_usage:
        usage = from_api_completion_usage(cast("LiteLLMUsage", api_usage))
        hidden_params = getattr(api_completion, "_hidden_params", {})
        usage.cost = hidden_params.get("response_cost")

    return Completion(
        id=api_completion.id,
        created=api_completion.created,
        usage=usage,
        choices=choices,
        name=name,
        system_fingerprint=api_completion.system_fingerprint,
        model=api_completion.model,
        hidden_params=api_completion._hidden_params,  # type: ignore[union-attr]
        response_ms=getattr(api_completion, "_response_ms", None),
    )


def to_api_completion(completion: Completion) -> LiteLLMCompletion:
    raise NotImplementedError
