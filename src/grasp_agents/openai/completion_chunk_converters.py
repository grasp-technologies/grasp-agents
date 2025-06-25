from ..errors import CompletionError
from ..typing.completion_chunk import (
    CompletionChunk,
    CompletionChunkChoice,
    CompletionChunkChoiceDelta,
    CompletionChunkDeltaToolCall,
)
from . import OpenAICompletionChunk  # type: ignore[import]
from .completion_converters import from_api_completion_usage


def from_api_completion_chunk(
    api_completion_chunk: OpenAICompletionChunk, name: str | None = None
) -> CompletionChunk:
    if api_completion_chunk.choices is None:  # type: ignore
        raise CompletionError(
            f"Completion chunk API error: "
            f"{getattr(api_completion_chunk, 'error', None)}"
        )

    choices: list[CompletionChunkChoice] = []

    for api_choice in api_completion_chunk.choices:
        index = api_choice.index
        finish_reason = api_choice.finish_reason

        if api_choice.delta is None:  # type: ignore
            raise CompletionError(
                "API returned None for delta content in completion chunk "
                f"with finish_reason: {finish_reason}."
            )
        # if api_choice.delta.content is None:
        #     raise CompletionError(
        #         "API returned None for delta content in completion chunk "
        #         f"with finish_reason: {finish_reason}."
        #     )

        delta = CompletionChunkChoiceDelta(
            content=api_choice.delta.content,
            refusal=api_choice.delta.refusal,
            role=api_choice.delta.role,
            tool_calls=[
                CompletionChunkDeltaToolCall(
                    id=tool_call.id,
                    index=tool_call.index,
                    tool_name=tool_call.function.name,
                    tool_arguments=tool_call.function.arguments,
                )
                for tool_call in (api_choice.delta.tool_calls or [])
                if tool_call.function
            ],
        )

        choice = CompletionChunkChoice(
            index=index,
            delta=delta,
            finish_reason=finish_reason,
            # OpenAI logprobs have identical, but separately defined, types for
            # completion chunks and completions
            logprobs=api_choice.logprobs,
        )

        choices.append(choice)

    usage = (
        from_api_completion_usage(api_completion_chunk.usage)
        if api_completion_chunk.usage
        else None
    )

    return CompletionChunk(
        id=api_completion_chunk.id,
        model=api_completion_chunk.model,
        name=name,
        created=api_completion_chunk.created,
        system_fingerprint=api_completion_chunk.system_fingerprint,
        choices=choices,
        usage=usage,
    )
