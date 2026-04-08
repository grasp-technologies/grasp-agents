from datetime import UTC, datetime
from enum import Enum
from typing import Any, Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from grasp_agents.packet import Packet

from .items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from .llm_events import LlmEvent
from .response import Response

_T_co = TypeVar("_T_co", covariant=True)


class Event(BaseModel, Generic[_T_co], frozen=True):
    type: str
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    created_at: float = Field(default_factory=lambda: datetime.now(UTC).timestamp())
    source: str | None = None
    exec_id: str | None = None
    data: _T_co


class RoutedEvent(Event[_T_co], frozen=True):
    """Event that carries a routing destination (used by Runner/EventBus)."""

    destination: str | None = None


# ── LLM stream ──


class LLMStreamEvent(Event[LlmEvent], frozen=True):
    type: Literal["llm.stream"] = "llm.stream"


# ── Agent item events (promoted from OutputItemDone) ──


class ToolCallItemEvent(Event[FunctionToolCallItem], frozen=True):
    type: Literal["agent.tool_call"] = "agent.tool_call"


class ReasoningItemEvent(Event[ReasoningItem], frozen=True):
    type: Literal["agent.reasoning"] = "agent.reasoning"


class OutputMessageItemEvent(Event[OutputMessageItem], frozen=True):
    type: Literal["agent.message"] = "agent.message"


# ── Tool execution ──


class ToolResultEvent(Event[FunctionToolOutputItem], frozen=True):
    type: Literal["tool.result"] = "tool.result"


class ToolOutputEvent(Event[Any], frozen=True):
    type: Literal["tool.output"] = "tool.output"


class ToolErrorInfo(BaseModel):
    tool_name: str
    error: str
    timed_out: bool = False


class ToolErrorEvent(Event[ToolErrorInfo], frozen=True):
    type: Literal["tool.error"] = "tool.error"


# ── Background tasks ──


class BackgroundTaskInfo(BaseModel):
    task_id: str
    tool_name: str
    tool_call_id: str


class BackgroundTaskLaunchedEvent(Event[BackgroundTaskInfo], frozen=True):
    type: Literal["agent.bg_launched"] = "agent.bg_launched"


class BackgroundTaskCompletedEvent(Event[BackgroundTaskInfo], frozen=True):
    type: Literal["agent.bg_completed"] = "agent.bg_completed"


# ── Lifecycle ──


class TurnInfo(BaseModel):
    turn: int


class StopReason(Enum):
    FINAL_ANSWER = "final_answer"
    MAX_TURNS = "max_turns"
    ERROR = "error"
    TIMEOUT = "timeout"


class TurnEndInfo(BaseModel):
    turn: int
    had_tool_calls: bool
    stop_reason: StopReason | None = None

    model_config = ConfigDict(use_enum_values=True)


class TurnStartEvent(Event[TurnInfo], frozen=True):
    type: Literal["agent.turn_start"] = "agent.turn_start"


class TurnEndEvent(Event[TurnEndInfo], frozen=True):
    type: Literal["agent.turn_end"] = "agent.turn_end"


class GenerationEndEvent(Event[Response], frozen=True):
    type: Literal["agent.generation_end"] = "agent.generation_end"


# ── Messages ──


class UserMessageEvent(Event[InputMessageItem], frozen=True):
    type: Literal["user.message"] = "user.message"


class SystemMessageEvent(Event[InputMessageItem], frozen=True):
    type: Literal["agent.system_message"] = "agent.system_message"


# ── LLM errors ──


class LLMStreamingErrorData(BaseModel):
    error: Exception
    model_name: str | None = None
    model_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMStreamingErrorEvent(Event[LLMStreamingErrorData], frozen=True):
    type: Literal["llm.error"] = "llm.error"


# ── Processor events ──


class ProcStartEvent(Event[None], frozen=True):
    type: Literal["proc.start"] = "proc.start"


class ProcEndEvent(Event[None], frozen=True):
    type: Literal["proc.end"] = "proc.end"


class ProcPayloadOutEvent(Event[Any], frozen=True):
    type: Literal["proc.payload"] = "proc.payload"


class ProcPacketOutEvent(RoutedEvent[Packet[Any]], frozen=True):
    type: Literal["proc.packet"] = "proc.packet"


class ProcStreamingErrorData(BaseModel):
    error: Exception
    exec_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProcStreamingErrorEvent(Event[ProcStreamingErrorData], frozen=True):
    type: Literal["proc.error"] = "proc.error"


# ── Run events ──


class RunPacketOutEvent(RoutedEvent[Packet[Any]], frozen=True):
    type: Literal["run.packet"] = "run.packet"


# ── Backwards compatibility ──


# DummyEvent: used by workflow_processor as a sentinel
class DummyEvent(Event[Any], frozen=True):
    type: Literal["proc.payload"] = "proc.payload"
    data: Any = None


# ToolMessageEvent: alias for ToolResultEvent during migration
ToolMessageEvent = ToolResultEvent

# ToolCallEvent: alias for ToolCallItemEvent during migration
ToolCallEvent = ToolCallItemEvent
