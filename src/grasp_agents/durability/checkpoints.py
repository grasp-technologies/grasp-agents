from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from grasp_agents.packet import Packet
from grasp_agents.types.events import ProcPacketOutEvent
from grasp_agents.types.items import InputItem
from grasp_agents.types.response import ResponseUsage


class AgentCheckpointLocation(Enum):
    """Where in the agent loop a checkpoint was saved."""

    AFTER_INPUT = "after_input"
    AFTER_TOOL_RESULT = "after_tool_result"
    AFTER_FINAL_ANSWER = "after_final_answer"
    AFTER_MAX_TURNS = "after_max_turns"


class ProcessorCheckpoint(BaseModel):
    """Base checkpoint for any resumable processor."""

    session_id: str
    processor_name: str
    checkpoint_number: int = 0
    saved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_metadata: dict[str, Any] = Field(default_factory=dict)


class AgentCheckpoint(ProcessorCheckpoint):
    """Lightweight checkpoint for an agent session. No business state."""

    messages: list[InputItem]
    usage: ResponseUsage | None = None
    step: int | None = None  # caller's delivery step (None = chat / untracked)
    turn: int = 0  # current LLM cycle within the step
    output: str | None = None  # cached final answer (None = step incomplete)
    location: AgentCheckpointLocation = AgentCheckpointLocation.AFTER_INPUT


class WorkflowCheckpoint(ProcessorCheckpoint):
    """
    Checkpoint for SequentialWorkflow / LoopedWorkflow.

    Stores the step cursor and intermediate packet so the workflow
    can skip completed steps on resume.
    """

    completed_step: int  # index of last completed subproc
    iteration: int = 0  # for LoopedWorkflow
    packet: Packet[Any]


class ParallelCheckpoint(ProcessorCheckpoint):
    """
    Checkpoint for ParallelProcessor.

    Stores original inputs as a Packet and a completion map so the
    processor can skip completed copies on resume.
    """

    input_packet: Packet[Any]
    completed: dict[int, Packet[Any]]  # idx -> completed replica output


class RunnerCheckpoint(ProcessorCheckpoint):
    """
    Checkpoint for Runner.

    Stores pending events (events awaiting delivery OR currently being handled)
    and per-processor session IDs so sessions can be restored on resume.
    """

    pending_events: list[ProcPacketOutEvent]
    active_sessions: dict[str, str] = Field(default_factory=dict)
    active_steps: dict[str, int] = Field(default_factory=dict)
