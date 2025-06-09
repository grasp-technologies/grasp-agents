from collections.abc import AsyncIterator

from ..typing.completion import Completion, CompletionChoice, CompletionChunk
from . import (
    OpenAIAsyncStream,  # type: ignore[import]
    OpenAICompletion,
    OpenAICompletionChunk,
)
from .message_converters import from_api_assistant_message


def from_api_completion(
    api_completion: OpenAICompletion, model_id: str | None = None
) -> Completion:
    choices: list[CompletionChoice] = []
    if api_completion.choices is None:  # type: ignore
        # Some providers return None for the choices when there is an error
        # TODO: add custom error types
        raise RuntimeError(
            f"Completion API error: {getattr(api_completion, 'error', None)}"
        )
    for idx, api_choice in enumerate(api_completion.choices):
        # TODO: currently no way to assign individual message usages when len(choices) > 1
        finish_reason = api_choice.finish_reason
        # Some providers return None for the message when finish_reason is other than "stop"
        if api_choice.message is None:  # type: ignore
            raise RuntimeError(
                f"API returned None for message with finish_reason: {finish_reason}"
            )
        message = from_api_assistant_message(
            api_choice.message, api_completion.usage, model_id=model_id
        )
        choices.append(
            CompletionChoice(
                index=idx,
                message=message,
                finish_reason=finish_reason,
                logprobs=api_choice.logprobs,
            )
        )

    return Completion(
        id=api_completion.id,
        created=api_completion.created,
        usage=api_completion.usage,
        choices=choices,
        model_id=model_id,
    )


def to_api_completion(completion: Completion) -> OpenAICompletion:
    raise NotImplementedError


def from_api_completion_chunk(
    api_completion_chunk: OpenAICompletionChunk, model_id: str | None = None
) -> CompletionChunk:
    return CompletionChunk(
        id=api_completion_chunk.id,
        created=api_completion_chunk.created,
        delta=api_completion_chunk.choices[0].delta.content,
        model_id=model_id,
    )


async def from_api_completion_chunk_iterator(
    api_completion_chunk_iterator: OpenAIAsyncStream[OpenAICompletionChunk],
    model_id: str | None = None,
) -> AsyncIterator[CompletionChunk]:
    async for api_chunk in api_completion_chunk_iterator:
        yield from_api_completion_chunk(api_chunk, model_id=model_id)
