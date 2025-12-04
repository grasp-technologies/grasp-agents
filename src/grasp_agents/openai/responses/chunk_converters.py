from uuid import uuid4

from openai.types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from grasp_agents.typing.completion_chunk import (
    CompletionChunk,
    CompletionChunkDelta,
    CompletionChunkDeltaToolCall,
)
from grasp_agents.typing.message import Role

from .converters import ResponseApiChunk


def from_api_completion_chunk(
    api_chunk: ResponseApiChunk, name: str | None = None
) -> CompletionChunk:
    if isinstance(api_chunk, ResponseReasoningSummaryTextDeltaEvent):
        delta = CompletionChunkDelta(
            reasoning_content=api_chunk.delta, role=Role.ASSISTANT
        )
        return CompletionChunk(
            item_id=api_chunk.item_id,
            model=None,
            name=name,
            system_fingerprint=None,
            usage=None,
            delta=delta,
            finish_reason=None,
            logprobs=None,
        )
    if isinstance(api_chunk, ResponseFunctionCallArgumentsDeltaEvent):
        delta = CompletionChunkDelta(
            tool_calls=[
                CompletionChunkDeltaToolCall(
                    id=None,
                    index=api_chunk.output_index,
                    tool_arguments=api_chunk.delta,
                    tool_name=None,
                )
            ],
            role=Role.ASSISTANT,
        )
        return CompletionChunk(
            item_id=api_chunk.item_id,
            model=None,
            name=name,
            system_fingerprint=None,
            usage=None,
            delta=delta,
            finish_reason=None,
            logprobs=None,
        )
    if isinstance(api_chunk, ResponseOutputItemAddedEvent) and isinstance(
        api_chunk.item, ResponseFunctionToolCall
    ):
        item = api_chunk.item
        func_delta = CompletionChunkDelta(
            tool_calls=[
                CompletionChunkDeltaToolCall(
                    id=item.call_id,
                    index=api_chunk.output_index,
                    tool_arguments=item.arguments,
                    tool_name=item.name,
                )
            ],
            role=Role.ASSISTANT,
        )
        return CompletionChunk(
            item_id=item.id,
            model=None,
            name=name,
            system_fingerprint=None,
            usage=None,
            delta=func_delta,
            finish_reason=None,
            logprobs=None,
        )
    if isinstance(api_chunk, ResponseTextDeltaEvent):
        delta = CompletionChunkDelta(content=api_chunk.delta)

        return CompletionChunk(
            id=api_chunk.item_id,
            model=None,
            name=name,
            system_fingerprint=None,
            delta=delta,
            finish_reason=None,
            logprobs=api_chunk.logprobs,
            usage=None,
        )

    raise TypeError(f"Unsupported chunk event: {type(api_chunk)}")


def to_completion_chunk(completion_chunk: CompletionChunk) -> ResponseStreamEvent:
    raise NotImplementedError
