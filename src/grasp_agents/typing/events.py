import time
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..packet import Packet
from .completion import Completion
from .completion_chunk import CompletionChunk
from .message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage


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
    COMP_CHUNK = "completion_chunk"
    LLM_ERR = "llm_error"

    PROC_START = "processor_start"
    PACKET_OUT = "packet_output"
    PAYLOAD_OUT = "payload_output"
    PROC_FINISH = "processor_finish"
    PROC_ERR = "processor_error"

    WORKFLOW_RES = "workflow_result"
    RUN_RES = "run_result"

    # COMP_THINK_CHUNK = "completion_thinking_chunk"
    # COMP_RESP_CHUNK = "completion_response_chunk"


_T = TypeVar("_T")


class Event(BaseModel, Generic[_T], frozen=True):
    type: EventType
    source: EventSourceType
    id: str = Field(default_factory=lambda: str(uuid4()))
    created: int = Field(default_factory=lambda: int(time.time()))
    proc_name: str | None = None
    call_id: str | None = None
    data: _T


class CompletionEvent(Event[Completion], frozen=True):
    type: Literal[EventType.COMP] = EventType.COMP
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class CompletionChunkEvent(Event[CompletionChunk], frozen=True):
    type: Literal[EventType.COMP_CHUNK] = EventType.COMP_CHUNK
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class LLMStreamingErrorData(BaseModel):
    error: Exception
    model_name: str | None = None
    model_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMStreamingErrorEvent(Event[LLMStreamingErrorData], frozen=True):
    type: Literal[EventType.LLM_ERR] = EventType.LLM_ERR
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


# class CompletionThinkingChunkEvent(Event[CompletionChunk], frozen=True):
#     type: Literal[EventType.COMP_THINK_CHUNK] = EventType.COMP_THINK_CHUNK
#     source: Literal[EventSourceType.LLM] = EventSourceType.LLM


# class CompletionResponseChunkEvent(Event[CompletionChunk], frozen=True):
#     type: Literal[EventType.COMP_RESP_CHUNK] = EventType.COMP_RESP_CHUNK
#     source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class MessageEvent(Event[_T], Generic[_T], frozen=True):
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


class ProcStartEvent(Event[None], frozen=True):
    type: Literal[EventType.PROC_START] = EventType.PROC_START
    source: Literal[EventSourceType.PROC] = EventSourceType.PROC


class ProcFinishEvent(Event[None], frozen=True):
    type: Literal[EventType.PROC_FINISH] = EventType.PROC_FINISH
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


class WorkflowResultEvent(ProcPacketOutputEvent, frozen=True):
    type: Literal[EventType.WORKFLOW_RES] = EventType.WORKFLOW_RES
    source: Literal[EventSourceType.WORKFLOW] = EventSourceType.WORKFLOW


class RunResultEvent(ProcPacketOutputEvent, frozen=True):
    type: Literal[EventType.RUN_RES] = EventType.RUN_RES
    source: Literal[EventSourceType.RUN] = EventSourceType.RUN


class ProcStreamingErrorData(BaseModel):
    error: Exception
    call_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProcStreamingErrorEvent(Event[ProcStreamingErrorData], frozen=True):
    type: Literal[EventType.PROC_ERR] = EventType.PROC_ERR
    source: Literal[EventSourceType.PROC] = EventSourceType.PROC
