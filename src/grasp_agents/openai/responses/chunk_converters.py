from uuid import uuid4

from openai.types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from ...typing.completion_chunk import (
    CompletionChunk,
    CompletionChunkDelta,
    CompletionChunkDeltaToolCall,
)
from ...typing.message import Role


def from_api_completion_chunk(
    event: ResponseStreamEvent, name: str | None = None
) -> CompletionChunk:
    if isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
        delta = CompletionChunkDelta(reasoning_content=event.delta, role=Role.ASSISTANT)
        return CompletionChunk(
            id=event.item_id,
            model=None,
            name=name,
            system_fingerprint=None,
            usage=None,
            delta=delta,
            finish_reason=None,
            logprobs=None,
        )
    if isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
        delta = CompletionChunkDelta(
            tool_calls=[
                CompletionChunkDeltaToolCall(
                    id=None,
                    index=event.output_index,
                    tool_arguments=event.delta,
                    tool_name=None,
                )
            ],
            role=Role.ASSISTANT,
        )
        return CompletionChunk(
            id=event.item_id,
            model=None,
            name=name,
            system_fingerprint=None,
            usage=None,
            delta=delta,
            finish_reason=None,
            logprobs=None,
        )
    if isinstance(event, ResponseOutputItemAddedEvent) and isinstance(
        event.item, ResponseFunctionToolCall
    ):
        item = event.item
        func_delta = CompletionChunkDelta(
            tool_calls=[
                CompletionChunkDeltaToolCall(
                    id=item.call_id,
                    index=event.output_index,
                    tool_arguments=item.arguments,
                    tool_name=item.name,
                )
            ],
            role=Role.ASSISTANT,
        )
        return CompletionChunk(
            id=item.id or str(uuid4())[:8],
            model=None,
            name=name,
            system_fingerprint=None,
            usage=None,
            delta=func_delta,
            finish_reason=None,
            logprobs=None,
        )
    if isinstance(event, ResponseTextDeltaEvent):
        delta = CompletionChunkDelta(content=event.delta)

        return CompletionChunk(
            id=event.item_id,
            model=None,
            name=name,
            system_fingerprint=None,
            delta=delta,
            finish_reason=None,
            logprobs=event.logprobs,
            usage=None,
        )

    raise TypeError(f"Unsupported chunk event: {type(event)}")


def to_completion_chunk(completion_chunk: CompletionChunk) -> ResponseStreamEvent:
    raise NotImplementedError
