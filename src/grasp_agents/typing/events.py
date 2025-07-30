import time
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar, get_args
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..packet import Packet
from .completion import Completion
from .completion_chunk import (
    AnnotationsChunk,
    CompletionChunk,
    RefusalChunk,
    ResponseChunk,
    ThinkingChunk,
    ToolCallChunk,
)
from .message import (
    AssistantMessage,
    MessageBase,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class EventSourceType(StrEnum):
    LLM = "llm"
    AGENT = "agent"
    USER = "user"
    TOOL = "tool"
    PROC = "processor"
    WORKFLOW = "workflow"
    RUN = "run"


class EventType(StrEnum):
    SYS_MSG = "system_message"
    USR_MSG = "user_message"
    TOOL_MSG = "tool_message"
    TOOL_CALL = "tool_call"
    GEN_MSG = "gen_message"

    COMP = "completion"
    COMP_START = "completion_start"
    COMP_END = "completion_end"

    COMP_CHUNK = "completion_chunk"
    THINK_CHUNK = "thinking_chunk"
    RESP_CHUNK = "response_chunk"
    TOOL_CALL_CHUNK = "tool_call_chunk"
    ANNOT_CHUNK = "annotations_chunk"
    REFUSAL_CHUNK = "refusal_chunk"

    RESP_START = "response_start"
    RESP_END = "response_end"
    THINK_START = "thinking_start"
    THINK_END = "thinking_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    ANNOT_START = "annotations_start"
    ANNOT_END = "annotations_end"

    LLM_ERR = "llm_error"

    PACKET_OUT = "packet_output"
    PAYLOAD_OUT = "payload_output"
    PROC_FINISH = "processor_finish"
    PROC_ERR = "processor_error"
    PROC_START = "processor_start"
    PROC_END = "processor_end"

    WORKFLOW_RES = "workflow_result"
    RUN_RES = "run_result"


_T_co = TypeVar("_T_co", covariant=True)
_M_co = TypeVar("_M_co", covariant=True, bound=MessageBase)
_C_co = TypeVar("_C_co", covariant=True, bound=CompletionChunk)


class Event(BaseModel, Generic[_T_co], frozen=True):
    type: EventType
    source: EventSourceType
    id: str = Field(default_factory=lambda: str(uuid4()))
    created: int = Field(default_factory=lambda: int(time.time()))
    proc_name: str | None = None
    call_id: str | None = None
    data: _T_co


class DummyEvent(Event[Any], frozen=True):
    type: Literal[EventType.PAYLOAD_OUT] = EventType.PAYLOAD_OUT
    source: Literal[EventSourceType.PROC] = EventSourceType.PROC
    data: Any = None


# Non-streamed completion events


class CompletionEvent(Event[Completion], frozen=True):
    type: Literal[EventType.COMP] = EventType.COMP
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class MessageEvent(Event[_M_co], Generic[_M_co], frozen=True):
    pass


class GenMessageEvent(MessageEvent[AssistantMessage], frozen=True):
    type: Literal[EventType.GEN_MSG] = EventType.GEN_MSG
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class ToolMessageEvent(MessageEvent[ToolMessage], frozen=True):
    type: Literal[EventType.TOOL_MSG] = EventType.TOOL_MSG
    source: Literal[EventSourceType.TOOL] = EventSourceType.TOOL


class UserMessageEvent(MessageEvent[UserMessage], frozen=True):
    type: Literal[EventType.USR_MSG] = EventType.USR_MSG
    source: Literal[EventSourceType.USER] = EventSourceType.USER


class SystemMessageEvent(MessageEvent[SystemMessage], frozen=True):
    type: Literal[EventType.SYS_MSG] = EventType.SYS_MSG
    source: Literal[EventSourceType.AGENT] = EventSourceType.AGENT


class ToolCallEvent(Event[ToolCall], frozen=True):
    type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    source: Literal[EventSourceType.AGENT] = EventSourceType.AGENT


# Streamed completion events

StreamedCompletionEventTypes = Literal[
    EventType.COMP_CHUNK,
    EventType.COMP_START,
    EventType.COMP_END,
    EventType.RESP_CHUNK,
    EventType.RESP_START,
    EventType.RESP_END,
    EventType.THINK_CHUNK,
    EventType.THINK_START,
    EventType.THINK_END,
    EventType.TOOL_CALL_CHUNK,
    EventType.TOOL_CALL_START,
    EventType.TOOL_CALL_END,
    EventType.ANNOT_CHUNK,
    EventType.ANNOT_START,
    EventType.ANNOT_END,
    EventType.REFUSAL_CHUNK,
]


class CompletionChunkEvent(Event[_C_co], Generic[_C_co], frozen=True):
    type: StreamedCompletionEventTypes = EventType.COMP_CHUNK
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    data: _C_co

    def split_into_specialized(
        self,
    ) -> "list[CompletionChunkEvent[Any]]":
        specialized_events: list[CompletionChunkEvent[Any]] = []

        for sub_chunk in self.data.split_into_specialized():
            if isinstance(sub_chunk, ResponseChunk):
                new_event = self.model_copy(
                    update={"data": sub_chunk, "type": EventType.RESP_CHUNK}
                )
                specialized_events.append(
                    ResponseChunkEvent.model_validate(new_event.model_dump())
                )
            if isinstance(sub_chunk, ThinkingChunk):
                new_event = self.model_copy(
                    update={"data": sub_chunk, "type": EventType.THINK_CHUNK}
                )
                specialized_events.append(
                    ThinkingChunkEvent.model_validate(new_event.model_dump())
                )
            if isinstance(sub_chunk, ToolCallChunk):
                new_event = self.model_copy(
                    update={"data": sub_chunk, "type": EventType.TOOL_CALL_CHUNK}
                )
                specialized_events.append(
                    ToolCallChunkEvent.model_validate(new_event.model_dump())
                )
            if isinstance(sub_chunk, AnnotationsChunk):
                new_event = self.model_copy(
                    update={"data": sub_chunk, "type": EventType.ANNOT_CHUNK}
                )
                specialized_events.append(
                    AnnotationsChunkEvent.model_validate(new_event.model_dump())
                )
            if isinstance(sub_chunk, RefusalChunk):
                new_event = self.model_copy(
                    update={"data": sub_chunk, "type": EventType.REFUSAL_CHUNK}
                )
                specialized_events.append(
                    RefusalChunkEvent.model_validate(new_event.model_dump())
                )

        return specialized_events


class ResponseChunkEvent(CompletionChunkEvent[ResponseChunk], frozen=True):
    type: Literal[EventType.RESP_CHUNK] = EventType.RESP_CHUNK
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class ThinkingChunkEvent(CompletionChunkEvent[ThinkingChunk], frozen=True):
    type: Literal[EventType.THINK_CHUNK] = EventType.THINK_CHUNK
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class ToolCallChunkEvent(CompletionChunkEvent[ToolCallChunk], frozen=True):
    type: Literal[EventType.TOOL_CALL_CHUNK] = EventType.TOOL_CALL_CHUNK
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class AnnotationsChunkEvent(CompletionChunkEvent[AnnotationsChunk], frozen=True):
    type: Literal[EventType.ANNOT_CHUNK] = EventType.ANNOT_CHUNK
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class RefusalChunkEvent(CompletionChunkEvent[RefusalChunk], frozen=True):
    type: Literal[EventType.REFUSAL_CHUNK] = EventType.REFUSAL_CHUNK
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


START_END_MAP: dict[EventType, list[EventType]] = {
    EventType.COMP_CHUNK: [EventType.COMP_START, EventType.COMP_END],
    EventType.RESP_CHUNK: [EventType.RESP_START, EventType.RESP_END],
    EventType.THINK_CHUNK: [EventType.THINK_START, EventType.THINK_END],
    EventType.TOOL_CALL_CHUNK: [EventType.TOOL_CALL_START, EventType.TOOL_CALL_END],
    EventType.ANNOT_CHUNK: [EventType.ANNOT_START, EventType.ANNOT_END],
}


class LLMStateChangeEvent(CompletionChunkEvent[_C_co], Generic[_C_co], frozen=True):
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: bool = True

    @classmethod
    def from_chunk_event(
        cls, event: CompletionChunkEvent[CompletionChunk]
    ) -> "LLMStateChangeEvent[_C_co]":
        _type = get_args(cls.model_fields["type"].annotation)[0]
        return cls(**event.model_copy(update={"type": _type}).model_dump())


class CompletionStartEvent(LLMStateChangeEvent[CompletionChunk], frozen=True):
    type: Literal[EventType.COMP_START] = EventType.COMP_START
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: Literal[True] = True


class CompletionEndEvent(LLMStateChangeEvent[CompletionChunk], frozen=True):
    type: Literal[EventType.COMP_END] = EventType.COMP_END
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: Literal[False] = False


class ResponseStartEvent(LLMStateChangeEvent[ResponseChunk], frozen=True):
    type: Literal[EventType.RESP_START] = EventType.RESP_START
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: Literal[True] = True


class ResponseEndEvent(LLMStateChangeEvent[ResponseChunk], frozen=True):
    type: Literal[EventType.RESP_END] = EventType.RESP_END
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: Literal[False] = False


class ThinkingStartEvent(LLMStateChangeEvent[ThinkingChunk], frozen=True):
    type: Literal[EventType.THINK_START] = EventType.THINK_START
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: Literal[True] = True


class ThinkingEndEvent(LLMStateChangeEvent[ThinkingChunk], frozen=True):
    type: Literal[EventType.THINK_END] = EventType.THINK_END
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: Literal[False] = False


class ToolCallStartEvent(LLMStateChangeEvent[ToolCallChunk], frozen=True):
    type: Literal[EventType.TOOL_CALL_START] = EventType.TOOL_CALL_START
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: Literal[True] = True


class ToolCallEndEvent(LLMStateChangeEvent[ToolCallChunk], frozen=True):
    type: Literal[EventType.TOOL_CALL_END] = EventType.TOOL_CALL_END
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: Literal[False] = False


class AnnotationsStartEvent(LLMStateChangeEvent[AnnotationsChunk], frozen=True):
    type: Literal[EventType.ANNOT_START] = EventType.ANNOT_START
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: Literal[True] = True


class AnnotationsEndEvent(LLMStateChangeEvent[AnnotationsChunk], frozen=True):
    type: Literal[EventType.ANNOT_END] = EventType.ANNOT_END
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM
    start: Literal[False] = False


class LLMStreamingErrorData(BaseModel):
    error: Exception
    model_name: str | None = None
    model_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMStreamingErrorEvent(Event[LLMStreamingErrorData], frozen=True):
    type: Literal[EventType.LLM_ERR] = EventType.LLM_ERR
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


# Processor events


class ProcStartEvent(Event[None], frozen=True):
    type: Literal[EventType.PROC_START] = EventType.PROC_START
    source: Literal[EventSourceType.PROC] = EventSourceType.PROC


class ProcEndEvent(Event[None], frozen=True):
    type: Literal[EventType.PROC_END] = EventType.PROC_END
    source: Literal[EventSourceType.PROC] = EventSourceType.PROC


class ProcPayloadOutputEvent(Event[Any], frozen=True):
    type: Literal[EventType.PAYLOAD_OUT] = EventType.PAYLOAD_OUT
    source: Literal[EventSourceType.PROC] = EventSourceType.PROC


class ProcPacketOutputEvent(Event[Packet[Any]], frozen=True):
    type: Literal[EventType.PACKET_OUT, EventType.WORKFLOW_RES, EventType.RUN_RES] = (
        EventType.PACKET_OUT
    )
    source: Literal[
        EventSourceType.PROC, EventSourceType.WORKFLOW, EventSourceType.RUN
    ] = EventSourceType.PROC


class ProcStreamingErrorData(BaseModel):
    error: Exception
    call_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProcStreamingErrorEvent(Event[ProcStreamingErrorData], frozen=True):
    type: Literal[EventType.PROC_ERR] = EventType.PROC_ERR
    source: Literal[EventSourceType.PROC] = EventSourceType.PROC


# Workflow and run events


class WorkflowResultEvent(ProcPacketOutputEvent, frozen=True):
    type: Literal[EventType.WORKFLOW_RES] = EventType.WORKFLOW_RES
    source: Literal[EventSourceType.WORKFLOW] = EventSourceType.WORKFLOW


class RunResultEvent(ProcPacketOutputEvent, frozen=True):
    type: Literal[EventType.RUN_RES] = EventType.RUN_RES
    source: Literal[EventSourceType.RUN] = EventSourceType.RUN
