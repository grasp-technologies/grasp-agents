from collections.abc import AsyncIterator

from ..typing.completion import Completion, CompletionChoice, CompletionChunk
from . import (
    ChatCompletion,
    ChatCompletionAsyncStream,  # type: ignore[import]
    ChatCompletionChunk,
)
from .message_converters import from_api_assistant_message


def from_api_completion(
    api_completion: ChatCompletion, model_id: str | None = None
) -> Completion:
    choices: list[CompletionChoice] = []
    if api_completion.choices is None:  # type: ignore
        # Choices can sometimes be None for some providers using the OpenAI API
        # TODO: add custom error types
        raise RuntimeError(
            f"Completion API error: {getattr(api_completion, 'error', None)}"
        )
    for api_choice in api_completion.choices:
        # TODO: currently no way to assign individual message usages when len(choices) > 1
        message = from_api_assistant_message(
            api_choice.message, api_completion.usage, model_id=model_id
        )
        finish_reason = api_choice.finish_reason
        choices.append(CompletionChoice(message=message, finish_reason=finish_reason))

    return Completion(choices=choices, model_id=model_id)


def to_api_completion(completion: Completion) -> ChatCompletion:
    raise NotImplementedError


def from_api_completion_chunk(
    api_completion_chunk: ChatCompletionChunk, model_id: str | None = None
) -> CompletionChunk:
    delta = api_completion_chunk.choices[0].delta.content

    return CompletionChunk(delta=delta, model_id=model_id)


async def from_api_completion_chunk_iterator(
    api_completion_chunk_iterator: ChatCompletionAsyncStream[ChatCompletionChunk],
    model_id: str | None = None,
) -> AsyncIterator[CompletionChunk]:
    async for api_chunk in api_completion_chunk_iterator:
        yield from_api_completion_chunk(api_chunk, model_id=model_id)
