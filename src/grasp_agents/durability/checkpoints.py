from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from grasp_agents.durability.context_serialization import ContextKind
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
    1: "Initial versioned schema for processor checkpoints and task records."
}
"""
One-line summary per schema version. The current version MUST have an entry.
Kept alongside ``CURRENT_SCHEMA_VERSION`` so that a mismatch on load can tell
the operator what they are missing.
"""


class CheckpointKind(StrEnum):
    """Key segment identifying the kind of record persisted at this key."""

    AGENT = "agent"
    WORKFLOW = "workflow"
    PARALLEL = "parallel"
    RUNNER = "runner"
    TASK = "task"


class CheckpointSchemaError(Exception):
    """Raised when a persisted record uses a newer schema version than known."""


class AgentCheckpointLocation(Enum):
    """Where in the agent loop a checkpoint was saved."""

    AFTER_INPUT = "after_input"
    AFTER_TOOL_RESULT = "after_tool_result"
    AFTER_FINAL_ANSWER = "after_final_answer"
    AFTER_MAX_TURNS = "after_max_turns"


class PersistedRecord(BaseModel):
    """Common base for any record persisted in the checkpoint store."""

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION)
    session_key: str
    saved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version > CURRENT_SCHEMA_VERSION:
            known = ", ".join(
                f"{v}: {s}" for v, s in sorted(SCHEMA_VERSION_SUMMARIES.items())
            )
            raise CheckpointSchemaError(
                f"Persisted record schema version {self.schema_version} is "
                f"newer than this process understands "
                f"(current={CURRENT_SCHEMA_VERSION}). Known versions:\n{known}."
            )
        return self


class ProcessorCheckpoint(PersistedRecord):
    """Snapshot of a resumable processor's state at a turn boundary."""

    processor_name: str
    checkpoint_number: int = 0
    session_metadata: dict[str, Any] = Field(default_factory=dict)


class AgentCheckpoint(ProcessorCheckpoint):
    """Lightweight checkpoint for an agent session. No business state."""

    messages: list[InputItem]
    usage: ResponseUsage | None = None
    step: int | None = None  # caller's delivery step (None = chat / untracked)
    turn: int = 0  # current agent turn within the step (resets to 0 on each new step)
    output: str | None = None  # cached final answer (None = step incomplete)
    location: AgentCheckpointLocation = AgentCheckpointLocation.AFTER_INPUT

    # Optional opt-in auto-serialization of ``RunContext.state``. When
    # ``kind == OMITTED`` (default) the framework does not persist state;
    # ``@agent.add_state_builder`` rebuilds it from the app's own source
    # of truth on resume. See ``context_serialization.py`` for the
    # full contract.
    context_kind: ContextKind | None = None
    context_data: Any | None = None

    # Provider-supplied cache key (OpenAI Responses, Anthropic prompt
    # caching, etc.). Persisted so resume can reuse the same key and
    # avoid invalidating the model-side cache. Providers that ignore
    # this field leave it ``None``.
    prompt_cache_key: str | None = None


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
    and per-processor delivery-step counters so deliveries can be restored
    on resume. Per-processor session paths are fixed at construction time
    and don't need persisting.
    """

    pending_events: list[ProcPacketOutEvent]
    active_steps: dict[str, int] = Field(default_factory=dict)
