from typing import Any

from pydantic import BaseModel

from ...typing.completion_chunk import (
    CompletionChunk,
    CompletionChunkDelta,
    CompletionChunkDeltaToolCall,
)
from ...typing.message import Role
from .. import OpenAIResponseStreamEvent

# Narrow imports to the delta events we transform into our internal chunks
from openai.types.responses.response_text_delta_event import (
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_reasoning_text_delta_event import (
    ResponseReasoningTextDeltaEvent,
)
from openai.types.responses.response_reasoning_summary_text_delta_event import (
    ResponseReasoningSummaryTextDeltaEvent,
)
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_output_item_done_event import (
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_output_item import ResponseFunctionToolCall
from openai.types.responses.response_text_done_event import ResponseTextDoneEvent
from openai.types.responses.response_reasoning_summary_text_done_event import (
    ResponseReasoningSummaryTextDoneEvent,
)
from openai.types.responses.response_function_call_arguments_done_event import (
    ResponseFunctionCallArgumentsDoneEvent,
)


def from_api_response_stream_event(
    event: OpenAIResponseStreamEvent, name: str | None = None
) -> CompletionChunk | None:
    """
    Convert OpenAI Responses API streaming events into our internal CompletionChunk.

    We only translate incremental delta events that map cleanly to our chunk model:
    - response.output_text.delta -> Response text chunk
    - response.reasoning_text.delta -> Thinking chunk
    - response.function_call_arguments.delta -> Tool call arguments chunk

    Other event types (created, done, completed, etc.) are handled implicitly by
    our higher-level post-processing and final combination, so we return None to
    indicate no chunk should be emitted for them.
    """
    # Reasoning summary text done -> ThinkingChunk with full summary
    if isinstance(event, ResponseReasoningSummaryTextDoneEvent):
        delta = CompletionChunkDelta(reasoning_content=event.text, role=Role.ASSISTANT)
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

    # Finalized function call item -> emit a complete ToolCallChunk
    if isinstance(event, ResponseOutputItemDoneEvent):
        item = event.item
        if getattr(item, "type", "") == "function_call":
            # item is ResponseFunctionToolCall
            item_fc = item  # type: ignore[assignment]
            if isinstance(item_fc, ResponseFunctionToolCall):
                delta = CompletionChunkDelta(
                    tool_calls=[
                        CompletionChunkDeltaToolCall(
                            # Use call_id so downstream can reply correctly
                            id=item_fc.call_id or item_fc.id,
                            index=0,
                            tool_name=item_fc.name,
                            tool_arguments=item_fc.arguments,
                        )
                    ],
                    role=Role.ASSISTANT,
                )
                return CompletionChunk(
                    # The chunk id can be any stable id; prefer call_id
                    id=item_fc.call_id or item_fc.id,
                    model=None,
                    name=name,
                    system_fingerprint=None,
                    usage=None,
                    delta=delta,
                    finish_reason=None,
                    logprobs=None,
                )
    return None


def is_supported_stream_event(event: OpenAIResponseStreamEvent) -> bool:
    return isinstance(
        event,
        (
            ResponseReasoningSummaryTextDoneEvent,
            ResponseOutputItemDoneEvent,
        ),
    )
