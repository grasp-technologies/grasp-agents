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
from openai.types.responses.response_text_delta_event import Logprob as DeltaLogprob
from openai.types.responses.response_text_done_event import Logprob as DoneLogprob
from pydantic import BaseModel, Field

from .content import OutputContentPart, ReasoningSummary
from .items import OutputItem
from .response import Response

# Response lifecycle events


class ResponseCreated(ResponseCreatedEvent):
    """Response object was created (before generation starts)."""

    type: Literal["response.created"] = "response.created"
    sequence_number: int
    response: Response  # type: ignore[assignment]


class ResponseQueued(ResponseQueuedEvent):
    """Response is queued and waiting to be processed."""

    type: Literal["response.queued"] = "response.queued"
    sequence_number: int
    response: Response  # type: ignore[assignment]


class ResponseInProgress(ResponseInProgressEvent):
    """Response generation has started."""

    type: Literal["response.in_progress"] = "response.in_progress"
    sequence_number: int
    response: Response  # type: ignore[assignment]


class ResponseCompleted(ResponseCompletedEvent):
    """Response finished successfully. Contains the final Response object."""

    type: Literal["response.completed"] = "response.completed"
    sequence_number: int
    response: Response  # type: ignore[assignment]


class ResponseIncomplete(ResponseIncompleteEvent):
    """Response ended early (e.g. max tokens or content filter)."""

    type: Literal["response.incomplete"] = "response.incomplete"
    sequence_number: int
    response: Response  # type: ignore[assignment]


class ResponseFailed(ResponseFailedEvent):
    """Response failed due to an error."""

    type: Literal["response.failed"] = "response.failed"
    sequence_number: int
    response: Response  # type: ignore[assignment]


# Output item events


class OutputItemAdded(ResponseOutputItemAddedEvent):
    """A new output item (reasoning, message, tool call) started."""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    sequence_number: int
    item: OutputItem  # type: ignore[assignment]


class OutputItemDone(ResponseOutputItemDoneEvent):
    """An output item is complete with its final content."""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    sequence_number: int
    item: OutputItem  # type: ignore[assignment]


# Content part events


class OutputContentPartAdded(ResponseContentPartAddedEvent):
    """A new content part started within an output message."""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    content_index: int
    output_index: int
    sequence_number: int
    item_id: str
    part: OutputContentPart  # type: ignore[assignment]


class OutputContentPartDone(ResponseContentPartDoneEvent):
    """A content part is complete with its final content."""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    part: OutputContentPart  # type: ignore[assignment]


# Output text events


class OutputMessageTextPartTextDelta(ResponseTextDeltaEvent):
    """Incremental text token from the model."""

    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    delta: str
    logprobs: list[DeltaLogprob] = Field(default_factory=list[DeltaLogprob])


class OutputMessageTextPartTextDone(ResponseTextDoneEvent):
    """Final text content part with the complete text."""

    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    text: str
    logprobs: list[DoneLogprob] = Field(default_factory=list[DoneLogprob])


# Refusal events


class OutputMessageRefusalPartDelta(ResponseRefusalDeltaEvent):
    """Incremental refusal token."""

    type: Literal["response.refusal.delta"] = "response.refusal.delta"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    delta: str


class OutputMessageRefusalPartDone(ResponseRefusalDoneEvent):
    """Final refusal content with the complete refusal text."""

    type: Literal["response.refusal.done"] = "response.refusal.done"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    refusal: str


# Reasoning content events


class ReasoningContentPartTextDelta(ResponseReasoningTextDeltaEvent):
    """Incremental reasoning/thinking token."""

    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    delta: str


class ReasoningContentPartTextDone(ResponseReasoningTextDoneEvent):
    """Final reasoning content with the complete text."""

    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    text: str


# Annotation events


class AnnotationAdded(ResponseOutputTextAnnotationAddedEvent):
    """An annotation (e.g. URL citation) was added to text content."""

    type: Literal["response.output_text.annotation.added"] = (
        "response.output_text.annotation.added"
    )
    item_id: str
    annotation_index: int
    content_index: int
    output_index: int
    sequence_number: int
    annotation: object


# Reasoning summary events


class ReasoningSummaryPartAdded(ResponseReasoningSummaryPartAddedEvent):
    """A new reasoning summary part started."""

    type: Literal["response.reasoning_summary_part.added"] = (
        "response.reasoning_summary_part.added"
    )
    item_id: str
    summary_index: int
    output_index: int
    sequence_number: int
    part: ReasoningSummary  # type: ignore[assignment]


class ReasoningSummaryPartDone(ResponseReasoningSummaryPartDoneEvent):
    """A reasoning summary part is complete."""

    type: Literal["response.reasoning_summary_part.done"] = (
        "response.reasoning_summary_part.done"
    )
    item_id: str
    summary_index: int
    output_index: int
    sequence_number: int
    part: ReasoningSummary  # type: ignore[assignment]


class ReasoningSummaryPartTextDelta(ResponseReasoningSummaryTextDeltaEvent):
    """Incremental reasoning summary token."""

    type: Literal["response.reasoning_summary_text.delta"] = (
        "response.reasoning_summary_text.delta"
    )
    item_id: str
    summary_index: int
    output_index: int
    sequence_number: int
    delta: str


class ReasoningSummaryPartTextDone(ResponseReasoningSummaryTextDoneEvent):
    """Final reasoning summary with the complete text."""

    type: Literal["response.reasoning_summary_text.done"] = (
        "response.reasoning_summary_text.done"
    )
    item_id: str
    summary_index: int
    output_index: int
    sequence_number: int
    text: str


# Tool call events


class FunctionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaEvent):
    """Incremental token of a tool call's JSON arguments."""

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    delta: str
    item_id: str
    output_index: int
    sequence_number: int


class FunctionCallArgumentsDone(ResponseFunctionCallArgumentsDoneEvent):
    """Complete JSON arguments string for a tool call."""

    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    arguments: str
    item_id: str
    name: str = ""
    output_index: int
    sequence_number: int


# Web search events


class WebSearchCallInProgress(BaseModel):
    """A web search call has been initiated."""

    type: Literal["response.web_search_call.in_progress"] = (
        "response.web_search_call.in_progress"
    )
    item_id: str
    output_index: int
    sequence_number: int


class WebSearchCallSearching(BaseModel):
    """A web search call is executing."""

    type: Literal["response.web_search_call.searching"] = (
        "response.web_search_call.searching"
    )
    item_id: str
    output_index: int
    sequence_number: int


class WebSearchCallCompleted(BaseModel):
    """A web search call has completed."""

    type: Literal["response.web_search_call.completed"] = (
        "response.web_search_call.completed"
    )
    item_id: str
    output_index: int
    sequence_number: int


# Error events


class LlmError(ResponseErrorEvent):
    """An error occurred during streaming."""

    type: Literal["error"] = "error"
    code: str | None = None
    message: str
    param: str | None = None
    sequence_number: int


class ResponseRetrying(BaseModel):
    """
    A response attempt failed validation; a retry will follow.

    Signals to stream consumers that the preceding events belong to a failed
    attempt and should be discarded (e.g. clear partial output in the UI).
    """

    type: Literal["response.retrying"] = "response.retrying"
    sequence_number: int = 0
    attempt: int
    """Which retry is about to start (1 = first retry, 2 = second, ...)."""
    error: str
    """Description of the validation failure that triggered the retry."""


class ResponseFallback(BaseModel):
    """Emitted when falling back to another model during streaming."""

    type: Literal["response.fallback"] = "response.fallback"
    sequence_number: int = 0
    failed_model: str
    fallback_model: str
    error_type: str
    attempt: int


# --- Union types ---


LlmEvent: TypeAlias = Annotated[
    ResponseCreated
    | ResponseIncomplete
    | ResponseRetrying
    | ResponseFallback
    | ResponseQueued
    | ResponseInProgress
    | ResponseCompleted
    | ResponseFailed
    | OutputItemAdded
    | OutputItemDone
    | OutputContentPartAdded
    | OutputContentPartDone
    | OutputMessageTextPartTextDelta
    | OutputMessageTextPartTextDone
    | OutputMessageRefusalPartDelta
    | OutputMessageRefusalPartDone
    | AnnotationAdded
    | ReasoningContentPartTextDelta
    | ReasoningContentPartTextDone
    | ReasoningSummaryPartAdded
    | ReasoningSummaryPartDone
    | ReasoningSummaryPartTextDelta
    | ReasoningSummaryPartTextDone
    | FunctionCallArgumentsDelta
    | FunctionCallArgumentsDone
    | WebSearchCallInProgress
    | WebSearchCallSearching
    | WebSearchCallCompleted
    | LlmError,
    Field(discriminator="type"),
]
