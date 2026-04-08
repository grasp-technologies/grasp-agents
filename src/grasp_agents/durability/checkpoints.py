from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from grasp_agents.types.items import InputItem
from grasp_agents.types.response import ResponseUsage


class ProcessorCheckpoint(BaseModel):
    """Base checkpoint for any resumable processor."""

    session_id: str
    processor_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentCheckpoint(ProcessorCheckpoint):
    """Lightweight checkpoint for an agent session. No business state."""

    messages: list[InputItem]
    turn_number: int = 0
    usage: ResponseUsage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowCheckpoint(ProcessorCheckpoint):
    """
    Checkpoint for SequentialWorkflow / LoopedWorkflow.

    Stores the step cursor and intermediate packet so the workflow
    can skip completed steps on resume.
    """

    completed_step: int  # index of last completed subproc
    iteration: int = 0  # for LoopedWorkflow
    packet: dict[str, Any]  # Packet.model_dump()


class ParallelCheckpoint(ProcessorCheckpoint):
    """
    Checkpoint for ParallelProcessor.

    Stores original inputs as a Packet and a completion map so the
    processor can skip completed copies on resume.
    """

    input_packet: dict[str, Any]  # Packet(payloads=in_args).model_dump()
    completed: dict[int, dict[str, Any]]  # idx -> Packet.model_dump()
