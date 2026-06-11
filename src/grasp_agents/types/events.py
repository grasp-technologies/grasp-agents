from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
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


class Event[T](BaseModel, frozen=True):
    type: str
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    created_at: float = Field(default_factory=lambda: datetime.now(UTC).timestamp())
    source: str | None = None
    exec_id: str | None = None
    data: T


class RoutedEvent[T](Event[T], frozen=True):
    """Event that carries a routing destination (used by Runner/EventBus)."""

    destination: str | None = None


# ── LLM stream ──


class LLMStreamEvent(Event[LlmEvent], frozen=True):
    type: Literal["llm.stream"] = "llm.stream"


# ── Agent item events (promoted from OutputItemDone) ──


class ToolCallItemEvent(Event[FunctionToolCallItem], frozen=True):
    type: Literal["agent.tool_call"] = "agent.tool_call"
    destination: str | None = None


class ReasoningItemEvent(Event[ReasoningItem], frozen=True):
    type: Literal["agent.reasoning"] = "agent.reasoning"


class OutputMessageItemEvent(Event[OutputMessageItem], frozen=True):
    type: Literal["agent.message"] = "agent.message"


# ── Tool execution ──


class ToolOutputItemEvent(Event[FunctionToolOutputItem], frozen=True):
    type: Literal["tool.result"] = "tool.result"
    destination: str | None = None


class ToolOutputEvent(Event[Any], frozen=True):
    type: Literal["tool.output"] = "tool.output"


class ToolStreamEvent(Event[Any], frozen=True):
    """
    Incremental output from a still-running tool, yielded *before* its terminal
    :class:`ToolOutputEvent`.

    ``data`` is any object: generic consumers (the
    :class:`~grasp_agents.agent.background_tasks.BackgroundTaskManager` buffer
    and progress log, consoles) render it with ``str(data)``, so a tool
    can carry a plain ``str`` or a structured object whose ``__str__`` produces
    the LLM-facing text. A tool with structure to expose subclasses this and
    narrows ``data`` (e.g. the shell tools' ``ExecStreamEvent`` carries an
    ``ExecStreamChunk`` with a stdout/stderr ``channel``); structure-aware
    consumers ``isinstance``-check the subclass, everyone else still sees a
    plain ``ToolStreamEvent``.

    ``destination`` is the owning agent, stamped by :meth:`BaseTool.run_stream`
    from the call's :class:`AgentContext` so a UI routes a tool's live output
    (foreground or backgrounded-and-bubbled) to that agent's pane instead of
    guessing from whichever agent generated most recently.
    """

    type: Literal["tool.stream"] = "tool.stream"
    destination: str | None = None


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
    # basename of the task's streamed-output log, when one is written (a file
    # backend is attached); the task's live output is mirrored there rather than
    # placed in the agent's context. ``None`` when no log is written.
    output_name: str | None = None


class BackgroundTaskLaunchedEvent(Event[BackgroundTaskInfo], frozen=True):
    type: Literal["agent.bg_launched"] = "agent.bg_launched"


class BackgroundTaskCompletedEvent(Event[BackgroundTaskInfo], frozen=True):
    type: Literal["agent.bg_completed"] = "agent.bg_completed"


# ── Lifecycle ──


class TurnInfo(BaseModel):
    turn: int


class StopReason(StrEnum):
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
    destination: str | None = None


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


# ToolMessageEvent: alias for ToolOutputItemEvent during migration
ToolMessageEvent = ToolOutputItemEvent

# ToolCallEvent: alias for ToolCallItemEvent during migration
ToolCallEvent = ToolCallItemEvent
