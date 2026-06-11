from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from grasp_agents.durability.context_serialization import ContextKind
from grasp_agents.packet import Packet
from grasp_agents.types.events import ProcPacketOutEvent, RunPacketOutEvent
from grasp_agents.types.items import InputItem
from grasp_agents.types.response import ResponseUsage

CURRENT_SCHEMA_VERSION: int = 6
"""
Version of the persisted checkpoint / task-record schema.

Bump when the shape of any ``ProcessorCheckpoint`` subclass or ``TaskRecord``
changes in a way that older code could not load. Add an entry to
``SCHEMA_VERSION_SUMMARIES`` describing what changed.
"""

SCHEMA_VERSION_SUMMARIES: dict[int, str] = {
    1: "Initial versioned schema for processor checkpoints and task records.",
    2: (
        "AgentCheckpoint gained the read-before-write ledger "
        "(read_file_state + dotfile_overrides) and fs_snapshot_ref. "
        "v1 records load fine (fields default); v1 code must not resume "
        "v2 sessions — it would silently skip the filesystem restore."
    ),
    3: (
        "AgentCheckpoint gained code_context_id (the RunPython kernel's code "
        "context, captured with fs_snapshot_ref so resume re-attaches the live "
        "kernel). v2 records load fine (field defaults None); v2 code resuming "
        "a v3 session would silently skip the re-attach (resume with a fresh "
        "kernel — variables lost)."
    ),
    4: (
        "TaskRecord gained output_path (the agent-readable .grasp/tasks log file "
        "holding a backgrounded task's full output) and started_at (so a restart "
        "can point the agent at the partial output + report how long it ran). v3 "
        "records load fine (fields default None)."
    ),
    5: (
        "AgentCheckpoint.messages moved to an append-only message log: the head "
        "blob no longer embeds messages and gains message_count (the log's commit "
        "watermark). Pre-v5 single-blob agent checkpoints are NOT loadable — their "
        "transcript lived inside the head and there is no log to read."
    ),
    6: (
        "AgentCheckpoint.code_context_id renamed to exec_context_id: the persisted "
        "value is a code-execution context that holds arbitrary in-memory sandbox "
        "state (E2B), not a Python-kernel id specifically. v5 records load fine "
        "(old field ignored, exec_context_id defaults None); v5 code resuming a v6 "
        "session loses the re-attach and resumes with a fresh kernel."
    ),
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
    """Lightweight checkpoint for an agent session."""

    # The transcript is persisted out-of-band as an append-only message log
    # (one ``InputItem`` per JSONL line), keyed alongside this head blob. It is
    # never written into the head: the head is overwritten every turn, so
    # embedding the growing transcript here would re-serialize it each time.
    # In memory this carries the full transcript; ``model_dump_json`` for the
    # head excludes it.
    messages: list[InputItem] = Field(default_factory=list[InputItem])

    # Commit watermark: how many leading log records this head acknowledges.
    # The log is appended before the head is saved, so a crash between the two
    # can leave uncommitted trailing records; on resume only the first
    # ``message_count`` are trusted (the rest are dropped).
    message_count: int = 0

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

    # Read-before-write ledger (resolved path -> last-observed mtime) +
    # session dotfile overrides. Restored on resume so the staleness
    # guard keeps working instead of refusing every edit until a
    # re-``Read``; files that changed while suspended still trip it.
    read_file_state: dict[str, float] = Field(default_factory=dict)
    dotfile_overrides: list[str] = Field(default_factory=list)

    # Opaque filesystem-snapshot ref from a ``SnapshotCapable``
    # environment (e.g. an E2B snapshot id, later a shadow-git sha).
    # The bytes live with whoever owns the snapshot store — the
    # checkpoint records only the ref. ``None`` = no snapshot taken.
    fs_snapshot_ref: str | None = None

    # A code-execution context id, captured together with ``fs_snapshot_ref``
    # (only meaningful as a pair: the id re-attaches to a context inside the
    # *restored* sandbox). The context holds arbitrary in-memory state — today
    # the RunPython kernel's, but it is not Python-specific. On a backend that
    # preserves running processes in its snapshot (E2B), resume re-binds to this
    # context so in-memory state survives. ``None`` = no persistent context / a
    # backend that can't persist it.
    exec_context_id: str | None = None


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
    # The run's final (END-routed) result, set when the run completes — lets a
    # resume of a completed session return the result instead of failing.
    final_event: RunPacketOutEvent | None = None
