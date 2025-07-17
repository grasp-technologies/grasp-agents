from ..errors import CompletionError
from ..typing.completion import Completion, CompletionChoice, Usage
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
    choices: list[CompletionChoice] = []

    if api_completion.choices is None:  # type: ignore
        # Some providers return None for the choices when there is an error
        # TODO: add custom error types
        raise CompletionError(
            f"Completion API error: {getattr(api_completion, 'error', None)}"
        )
    for api_choice in api_completion.choices:
        finish_reason = api_choice.finish_reason

        # Some providers return None for the message when finish_reason is other than "stop"
        if api_choice.message is None:  # type: ignore
            raise CompletionError(
                f"API returned None for message with finish_reason: {finish_reason}"
            )

        message = from_api_assistant_message(api_choice.message, name=name)

        choices.append(
            CompletionChoice(
                index=api_choice.index,
                message=message,
                finish_reason=finish_reason,
                logprobs=api_choice.logprobs,
            )
        )

    usage = (
        from_api_completion_usage(api_completion.usage)
        if api_completion.usage
        else None
    )

    return Completion(
        id=api_completion.id,
        created=api_completion.created,
        system_fingerprint=api_completion.system_fingerprint,
        usage=usage,
        choices=choices,
        model=api_completion.model,
        name=name,
    )


def to_api_completion(completion: Completion) -> OpenAICompletion:
    raise NotImplementedError
