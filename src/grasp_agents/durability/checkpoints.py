from datetime import UTC, datetime
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from grasp_agents.packet import Packet
from grasp_agents.types.events import ProcPacketOutEvent
from grasp_agents.types.items import InputItem
from grasp_agents.types.response import ResponseUsage

CURRENT_SCHEMA_VERSION: int = 1
"""
Version of the persisted checkpoint / task-record schema.

Bump when the shape of any ``ProcessorCheckpoint`` subclass or ``TaskRecord``
changes in a way that older code could not load. Add an entry to
``SCHEMA_VERSION_SUMMARIES`` describing what changed.
"""

SCHEMA_VERSION_SUMMARIES: dict[int, str] = {
    1: "Initial versioned schema for processor checkpoints and task records.",
}
"""
One-line summary per schema version. The current version MUST have an entry.
Kept alongside ``CURRENT_SCHEMA_VERSION`` so that a mismatch on load can tell
the operator what they are missing.
"""


class CheckpointSchemaError(Exception):
    """
    Raised when a persisted checkpoint or task record was written by a newer
    schema version than this process understands. Inherits directly from
    ``Exception`` (not ``ValueError``) so it propagates through Pydantic
    model validators as-is rather than being wrapped in ``ValidationError``,
    and so callers can distinguish it from generic corruption.
    """


class AgentCheckpointLocation(Enum):
    """Where in the agent loop a checkpoint was saved."""

    AFTER_INPUT = "after_input"
    AFTER_TOOL_RESULT = "after_tool_result"
    AFTER_FINAL_ANSWER = "after_final_answer"
    AFTER_MAX_TURNS = "after_max_turns"


class ProcessorCheckpoint(BaseModel):
    """Base checkpoint for any resumable processor."""

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION)
    session_id: str
    processor_name: str
    checkpoint_number: int = 0
    saved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version > CURRENT_SCHEMA_VERSION:
            known = ", ".join(
                f"{v}: {s}" for v, s in sorted(SCHEMA_VERSION_SUMMARIES.items())
            )
            raise CheckpointSchemaError(
                f"Checkpoint schema version {self.schema_version} is newer than "
                f"this process understands (current={CURRENT_SCHEMA_VERSION}). "
                f"Known versions: {known}."
            )
        return self


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

    completed_step: int  # global step counter (iteration * N + idx for looped)
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
