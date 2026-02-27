from typing import Annotated, Literal, TypeAlias

from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseQueuedEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseRefusalDeltaEvent,
    ResponseRefusalDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from pydantic import Field

# --- OpenResponses core events ---


# Delta events


class TextDelta(ResponseTextDeltaEvent):
    """Incremental text token from the model."""

    type: Literal["response.output_text.delta"] = "response.output_text.delta"


class TextDone(ResponseTextDoneEvent):
    """Final text content part with the complete text."""

    type: Literal["response.output_text.done"] = "response.output_text.done"


class FunctionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaEvent):
    """Incremental token of a tool call's JSON arguments."""

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )


# Item lifecycle events


class OutputItemAdded(ResponseOutputItemAddedEvent):
    """A new output item (message, tool call, reasoning) started."""

    type: Literal["response.output_item.added"] = "response.output_item.added"


class OutputItemDone(ResponseOutputItemDoneEvent):
    """An output item is complete with its final content."""

    type: Literal["response.output_item.done"] = "response.output_item.done"


class ContentPartAdded(ResponseContentPartAddedEvent):
    """A new content part started within an output message."""

    type: Literal["response.content_part.added"] = "response.content_part.added"


class ContentPartDone(ResponseContentPartDoneEvent):
    """A content part is complete with its final content."""

    type: Literal["response.content_part.done"] = "response.content_part.done"


# Response lifecycle events


class ResponseQueued(ResponseQueuedEvent):
    """Response is queued and waiting to be processed."""

    type: Literal["response.queued"] = "response.queued"


class ResponseInProgress(ResponseInProgressEvent):
    """Response generation has started."""

    type: Literal["response.in_progress"] = "response.in_progress"


class ResponseCompleted(ResponseCompletedEvent):
    """Response finished successfully. Contains the final Response object."""

    type: Literal["response.completed"] = "response.completed"


class ResponseFailed(ResponseFailedEvent):
    """Response failed due to an error. Contains error details."""

    type: Literal["response.failed"] = "response.failed"


# --- OpenAI Responses API extensions (not in OpenResponses core) ---


class FunctionCallArgumentsDone(ResponseFunctionCallArgumentsDoneEvent):
    """Complete JSON arguments string for a tool call."""

    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )


class ReasoningDelta(ResponseReasoningTextDeltaEvent):
    """Incremental reasoning/thinking token."""

    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"


class ReasoningDone(ResponseReasoningTextDoneEvent):
    """Final reasoning content with the complete text."""

    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"


class ReasoningSummaryDelta(ResponseReasoningSummaryTextDeltaEvent):
    """Incremental reasoning summary token."""

    type: Literal["response.reasoning_summary_text.delta"] = (
        "response.reasoning_summary_text.delta"
    )


class ReasoningSummaryDone(ResponseReasoningSummaryTextDoneEvent):
    """Final reasoning summary with the complete text."""

    type: Literal["response.reasoning_summary_text.done"] = (
        "response.reasoning_summary_text.done"
    )


class ReasoningSummaryPartAdded(ResponseReasoningSummaryPartAddedEvent):
    """A new reasoning summary part started."""

    type: Literal["response.reasoning_summary_part.added"] = (
        "response.reasoning_summary_part.added"
    )


class ReasoningSummaryPartDone(ResponseReasoningSummaryPartDoneEvent):
    """A reasoning summary part is complete."""

    type: Literal["response.reasoning_summary_part.done"] = (
        "response.reasoning_summary_part.done"
    )


class RefusalDelta(ResponseRefusalDeltaEvent):
    """Incremental refusal token."""

    type: Literal["response.refusal.delta"] = "response.refusal.delta"


class RefusalDone(ResponseRefusalDoneEvent):
    """Final refusal content with the complete refusal text."""

    type: Literal["response.refusal.done"] = "response.refusal.done"


class AnnotationAdded(ResponseOutputTextAnnotationAddedEvent):
    """An annotation (e.g. URL citation) was added to text content."""

    type: Literal["response.output_text.annotation.added"] = (
        "response.output_text.annotation.added"
    )


class ErrorEvent(ResponseErrorEvent):
    """An error occurred during streaming."""

    type: Literal["error"] = "error"


class ResponseCreated(ResponseCreatedEvent):
    """Response object was created (before generation starts)."""

    type: Literal["response.created"] = "response.created"


class ResponseIncomplete(ResponseIncompleteEvent):
    """Response ended early (e.g. max tokens or content filter)."""

    type: Literal["response.incomplete"] = "response.incomplete"


# --- Union types ---


StreamEvent: TypeAlias = Annotated[
    # Core
    TextDelta
    | TextDone
    | FunctionCallArgumentsDelta
    | OutputItemAdded
    | OutputItemDone
    | ContentPartAdded
    | ContentPartDone
    | ResponseQueued
    | ResponseInProgress
    | ResponseCompleted
    | ResponseFailed
    # Extensions
    | FunctionCallArgumentsDone
    | ReasoningDelta
    | ReasoningDone
    | ReasoningSummaryDelta
    | ReasoningSummaryDone
    | ReasoningSummaryPartAdded
    | ReasoningSummaryPartDone
    | RefusalDelta
    | RefusalDone
    | AnnotationAdded
    | ErrorEvent
    | ResponseCreated
    | ResponseIncomplete,
    Field(discriminator="type"),
]
