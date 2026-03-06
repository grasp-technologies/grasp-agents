import time
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..packet import Packet
from .items import FunctionToolCallItem, FunctionToolOutputItem, InputMessageItem
from .llm_events import LlmEvent


class EventSourceType(StrEnum):
    LLM = "llm"
    AGENT = "agent"
    USER = "user"
    TOOL = "tool"
    PROC = "processor"
    RUN = "run"


class EventType(StrEnum):
    SYS_MSG = "system_message"
    USR_MSG = "user_message"
    TOOL_MSG = "tool_message"
    TOOL_CALL = "tool_call"
    GEN_MSG = "gen_message"

    LLM_STREAM = "llm_stream"
    LLM_ERR = "llm_error"

    TOOL_OUT = "tool_output"
    PACKET_OUT = "packet_output"
    PAYLOAD_OUT = "payload_output"
    PROC_ERR = "processor_error"
    PROC_START = "processor_start"
    PROC_END = "processor_end"

    RUN_RES = "run_result"


_T_co = TypeVar("_T_co", covariant=True)


class Event(BaseModel, Generic[_T_co], frozen=True):
    type: EventType
    src_type: EventSourceType
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    created: int = Field(default_factory=lambda: int(time.time()))
    src_name: str | None = None
    dst_name: str | None = None
    call_id: str | None = None
    data: _T_co


class DummyEvent(Event[Any], frozen=True):
    type: Literal[EventType.PAYLOAD_OUT] = EventType.PAYLOAD_OUT
    src_type: Literal[EventSourceType.PROC] = EventSourceType.PROC
    data: Any = None


class LLMStreamEvent(Event[LlmEvent], frozen=True):
    type: Literal[EventType.LLM_STREAM] = EventType.LLM_STREAM
    src_type: Literal[EventSourceType.LLM] = EventSourceType.LLM


# Agent events


class ToolCallEvent(Event[FunctionToolCallItem], frozen=True):
    type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    src_type: Literal[EventSourceType.AGENT] = EventSourceType.AGENT


class SystemMessageEvent(Event[InputMessageItem], frozen=True):
    type: Literal[EventType.SYS_MSG] = EventType.SYS_MSG
    src_type: Literal[EventSourceType.AGENT] = EventSourceType.AGENT


# Tool events


class ToolOutputEvent(Event[Any], frozen=True):
    type: Literal[EventType.TOOL_OUT] = EventType.TOOL_OUT
    src_type: Literal[EventSourceType.TOOL] = EventSourceType.TOOL


class ToolMessageEvent(Event[FunctionToolOutputItem], frozen=True):
    type: Literal[EventType.TOOL_MSG] = EventType.TOOL_MSG
    src_type: Literal[EventSourceType.TOOL] = EventSourceType.TOOL


# User events


class UserMessageEvent(Event[InputMessageItem], frozen=True):
    type: Literal[EventType.USR_MSG] = EventType.USR_MSG
    src_type: Literal[EventSourceType.USER] = EventSourceType.USER


# LLM error events


class LLMStreamingErrorData(BaseModel):
    error: Exception
    model_name: str | None = None
    model_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMStreamingErrorEvent(Event[LLMStreamingErrorData], frozen=True):
    type: Literal[EventType.LLM_ERR] = EventType.LLM_ERR
    src_type: Literal[EventSourceType.LLM] = EventSourceType.LLM


# Processor events


class ProcStartEvent(Event[None], frozen=True):
    type: Literal[EventType.PROC_START] = EventType.PROC_START
    src_type: Literal[EventSourceType.PROC] = EventSourceType.PROC


class ProcEndEvent(Event[None], frozen=True):
    type: Literal[EventType.PROC_END] = EventType.PROC_END
    src_type: Literal[EventSourceType.PROC] = EventSourceType.PROC


class ProcPayloadOutEvent(Event[Any], frozen=True):
    type: Literal[EventType.PAYLOAD_OUT] = EventType.PAYLOAD_OUT
    src_type: Literal[EventSourceType.PROC] = EventSourceType.PROC


class ProcPacketOutEvent(Event[Packet[Any]], frozen=True):
    type: Literal[EventType.PACKET_OUT, EventType.RUN_RES] = EventType.PACKET_OUT
    src_type: Literal[EventSourceType.PROC, EventSourceType.RUN] = EventSourceType.PROC


class ProcStreamingErrorData(BaseModel):
    error: Exception
    call_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProcStreamingErrorEvent(Event[ProcStreamingErrorData], frozen=True):
    type: Literal[EventType.PROC_ERR] = EventType.PROC_ERR
    src_type: Literal[EventSourceType.PROC] = EventSourceType.PROC


# Run events


class RunPacketOutEvent(ProcPacketOutEvent, frozen=True):
    type: Literal[EventType.RUN_RES] = EventType.RUN_RES
    src_type: Literal[EventSourceType.RUN] = EventSourceType.RUN
