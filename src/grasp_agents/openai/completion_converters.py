from ..errors import CompletionError
from ..typing.completion import Completion, Usage
from . import OpenAICompletion, OpenAIUsage
from .message_converters import from_api_assistant_message


def from_api_completion_usage(api_usage: OpenAIUsage) -> Usage:
    reasoning_tokens = None
    cached_tokens = None

    if api_usage.completion_tokens_details is not None:
        reasoning_tokens = api_usage.completion_tokens_details.reasoning_tokens
    if api_usage.prompt_tokens_details is not None:
        cached_tokens = api_usage.prompt_tokens_details.cached_tokens

    input_tokens = api_usage.prompt_tokens - (cached_tokens or 0)
    output_tokens = api_usage.completion_tokens - (reasoning_tokens or 0)

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
    )


def from_api_completion(
    api_completion: OpenAICompletion, name: str | None = None
) -> Completion:
    # Some providers return None for the choices when there is an error
    # TODO: add custom error types
    if api_completion.choices is None:  # type: ignore
        raise CompletionError(
            f"Completion API error: {getattr(api_completion, 'error', None)}"
        )

    if len(api_completion.choices) > 1:
        raise CompletionError("Multiple choices are not supported")
    api_choice = api_completion.choices[0]

    # Some providers return None for the message when finish_reason is other than "stop"
    finish_reason = api_choice.finish_reason
    if api_choice.message is None:  # type: ignore
        raise CompletionError(
            f"API returned None for message with finish_reason: {finish_reason}"
        )

    message = from_api_assistant_message(api_choice.message, name=name)
    usage = (
        from_api_completion_usage(api_completion.usage)
        if api_completion.usage
        else None
    )

    return Completion(
        id=api_completion.id,
        created=api_completion.created,
        system_fingerprint=api_completion.system_fingerprint,
        message=message,
        finish_reason=finish_reason,
        logprobs=api_choice.logprobs,
        usage=usage,
        model=api_completion.model,
        name=name,
    )


def to_api_completion(completion: Completion) -> OpenAICompletion:
    raise NotImplementedError
