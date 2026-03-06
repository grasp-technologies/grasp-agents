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
from openai.types.responses.response_output_text import Logprob
from openai.types.responses.response_text_delta_event import Logprob as DeltaLogprob
from openai.types.responses.response_text_delta_event import (
    LogprobTopLogprob as DeltaTopLogprob,
)
from openai.types.responses.response_text_done_event import Logprob as DoneLogprob
from openai.types.responses.response_text_done_event import (
    LogprobTopLogprob as DoneTopLogprob,
)
from pydantic import Field

from .content import OutputContentPart, ReasoningSummaryPart
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


class ContentPartAdded(ResponseContentPartAddedEvent):
    """A new content part started within an output message."""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    content_index: int
    output_index: int
    sequence_number: int
    item_id: str
    part: OutputContentPart  # type: ignore[assignment]


class ContentPartDone(ResponseContentPartDoneEvent):
    """A content part is complete with its final content."""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    part: OutputContentPart  # type: ignore[assignment]


# Reasoning events


class ReasoningSummaryPartAdded(ResponseReasoningSummaryPartAddedEvent):
    """A new reasoning summary part started."""

    type: Literal["response.reasoning_summary_part.added"] = (
        "response.reasoning_summary_part.added"
    )
    item_id: str
    summary_index: int
    output_index: int
    sequence_number: int
    part: ReasoningSummaryPart  # type: ignore[assignment]


class ReasoningSummaryDelta(ResponseReasoningSummaryTextDeltaEvent):
    """Incremental reasoning summary token."""

    type: Literal["response.reasoning_summary_text.delta"] = (
        "response.reasoning_summary_text.delta"
    )
    item_id: str
    summary_index: int
    output_index: int
    sequence_number: int
    delta: str


class ReasoningSummaryTextDone(ResponseReasoningSummaryTextDoneEvent):
    """Final reasoning summary with the complete text."""

    type: Literal["response.reasoning_summary_text.done"] = (
        "response.reasoning_summary_text.done"
    )
    item_id: str
    summary_index: int
    output_index: int
    sequence_number: int
    text: str


class ReasoningSummaryPartDone(ResponseReasoningSummaryPartDoneEvent):
    """A reasoning summary part is complete."""

    type: Literal["response.reasoning_summary_part.done"] = (
        "response.reasoning_summary_part.done"
    )
    item_id: str
    summary_index: int
    output_index: int
    sequence_number: int
    part: ReasoningSummaryPart  # type: ignore[assignment]


class ReasoningDelta(ResponseReasoningTextDeltaEvent):
    """Incremental reasoning/thinking token."""

    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    delta: str


class ReasoningTextDone(ResponseReasoningTextDoneEvent):
    """Final reasoning content with the complete text."""

    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    text: str


# Output text events


class TextDelta(ResponseTextDeltaEvent):
    """Incremental text token from the model."""

    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    delta: str
    logprobs: list[DeltaLogprob] = Field(default_factory=list[DeltaLogprob])


class TextDone(ResponseTextDoneEvent):
    """Final text content part with the complete text."""

    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    text: str
    logprobs: list[DoneLogprob] = Field(default_factory=list[DoneLogprob])


def output_to_delta_logprobs(logprobs: list[Logprob]) -> list[DeltaLogprob]:
    """Convert by skipping the 'bytes' field which is not needed for delta logprobs."""
    return [
        DeltaLogprob(
            token=lp.token,
            logprob=lp.logprob,
            top_logprobs=[
                DeltaTopLogprob(token=tlp.token, logprob=tlp.logprob)
                for tlp in lp.top_logprobs
            ]
            if lp.top_logprobs
            else None,
        )
        for lp in logprobs
    ]


def output_to_done_logprobs(logprobs: list[Logprob]) -> list[DoneLogprob]:
    """Convert by skipping the 'bytes' field which is not needed for done logprobs."""
    return [
        DoneLogprob(
            token=lp.token,
            logprob=lp.logprob,
            top_logprobs=[
                DoneTopLogprob(token=tlp.token, logprob=tlp.logprob)
                for tlp in lp.top_logprobs
            ]
            if lp.top_logprobs
            else None,
        )
        for lp in logprobs
    ]


# Refusal events


class RefusalDelta(ResponseRefusalDeltaEvent):
    """Incremental refusal token."""

    type: Literal["response.refusal.delta"] = "response.refusal.delta"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    delta: str


class RefusalDone(ResponseRefusalDoneEvent):
    """Final refusal content with the complete refusal text."""

    type: Literal["response.refusal.done"] = "response.refusal.done"
    item_id: str
    content_index: int
    output_index: int
    sequence_number: int
    refusal: str


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


class ErrorEvent(ResponseErrorEvent):
    """An error occurred during streaming."""

    type: Literal["error"] = "error"
    code: str | None = None
    message: str
    param: str | None = None
    sequence_number: int


# --- Union types ---


LlmEvent: TypeAlias = Annotated[
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
    | FunctionCallArgumentsDone
    | ReasoningDelta
    | ReasoningTextDone
    | ReasoningSummaryDelta
    | ReasoningSummaryTextDone
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
