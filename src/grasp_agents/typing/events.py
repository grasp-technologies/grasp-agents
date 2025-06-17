import time
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from ..packet import Packet
from .completion import Completion
from .completion_chunk import CompletionChunk
from .message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage


class EventSourceType(StrEnum):
    LLM = "llm"
    AGENT = "agent"
    USER = "user"
    TOOL = "tool"
    PROCESSOR = "processor"


class EventType(StrEnum):
    SYS_MSG = "system_message"
    USR_MSG = "user_message"
    TOOL_MSG = "tool_message"
    TOOL_CALL = "tool_call"
    GEN_MSG = "gen_message"
    COMP = "completion"
    COMP_CHUNK = "completion_chunk"
    PACKET = "packet"
    PROC_OUT = "processor_output"


_T = TypeVar("_T")


class Event(BaseModel, Generic[_T], frozen=True):
    type: EventType
    source: EventSourceType
    created: int = Field(default_factory=lambda: int(time.time()))
    name: str | None = None
    data: _T


class CompletionEvent(Event[Completion], frozen=True):
    type: Literal[EventType.COMP] = EventType.COMP
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class CompletionChunkEvent(Event[CompletionChunk], frozen=True):
    type: Literal[EventType.COMP_CHUNK] = EventType.COMP_CHUNK
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class GenMessageEvent(Event[AssistantMessage], frozen=True):
    type: Literal[EventType.GEN_MSG] = EventType.GEN_MSG
    source: Literal[EventSourceType.LLM] = EventSourceType.LLM


class ToolCallEvent(Event[ToolCall], frozen=True):
    type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    source: Literal[EventSourceType.AGENT] = EventSourceType.AGENT


class ToolMessageEvent(Event[ToolMessage], frozen=True):
    type: Literal[EventType.TOOL_MSG] = EventType.TOOL_MSG
    source: Literal[EventSourceType.TOOL] = EventSourceType.TOOL


class UserMessageEvent(Event[UserMessage], frozen=True):
    type: Literal[EventType.USR_MSG] = EventType.USR_MSG
    source: Literal[EventSourceType.USER] = EventSourceType.USER


class SystemMessageEvent(Event[SystemMessage], frozen=True):
    type: Literal[EventType.SYS_MSG] = EventType.SYS_MSG
    source: Literal[EventSourceType.AGENT] = EventSourceType.AGENT


class PacketEvent(Event[Packet[Any]], frozen=True):
    type: Literal[EventType.PACKET] = EventType.PACKET
    source: Literal[EventSourceType.PROCESSOR] = EventSourceType.PROCESSOR


class ProcOutputEvent(Event[Any], frozen=True):
    type: Literal[EventType.PROC_OUT] = EventType.PROC_OUT
    source: Literal[EventSourceType.PROCESSOR] = EventSourceType.PROCESSOR
