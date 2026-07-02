from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grasp_agents.durability.context_serialization import ContextKind
from grasp_agents.types.events import ProcPacketOutEvent, RunPacketOutEvent
from grasp_agents.types.folds import FoldSpec
from grasp_agents.types.items import InputItem
from grasp_agents.types.packet import Packet
from grasp_agents.types.response import ResponseUsage

CURRENT_SCHEMA_VERSION: int = 12
"""
Version of the persisted checkpoint / task-record schema.

Bump when the shape of any ``ProcessorCheckpoint`` subclass or ``TaskRecord``
changes in a way that older code could not load. Add an entry to
``SCHEMA_VERSION_SUMMARIES`` describing what changed.
"""

MIN_SUPPORTED_SCHEMA_VERSION: int = 9
"""
Oldest persisted schema version this code can still load.

Records below this floor are rejected rather than partially loaded.
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
        "AgentCheckpoint uses an append-only message log: the head blob stores only "
        "message_count (commit watermark) and session metadata; the transcript lives "
        "in a sibling JSONL file. Single-blob agent checkpoints from earlier versions "
        "are NOT loadable."
    ),
    6: (
        "AgentCheckpoint.exec_context_id holds a code-execution context for arbitrary "
        "in-memory sandbox state (E2B). v5 records load fine "
        "(field defaults None); v5 code resuming a v6 "
        "session loses the re-attach and resumes with a fresh kernel."
    ),
    7: (
        "AgentCheckpoint gained log_version: full-history rewrites of the "
        "message log go to a fresh version file so a crash mid-rewrite "
        "cannot pair the old head with a rewritten log. v6 records load fine "
        "(version defaults 0 = the unsuffixed log); v6 code resuming a v7 "
        "session that has rewritten (version >= 1) would read a stale or "
        "missing log."
    ),
    8: (
        "AgentCheckpoint has ipy_exec_context_id (the RunPython kernel's execution "
        "context) and nb_exec_context_id (the RunCell notebook kernel's), both "
        "captured with fs_snapshot_ref. v7 records load fine (both fields default "
        "None); v7 code resuming a v8 session skips kernel re-attaches and falls "
        "back to fresh kernels (the .ipynb stays the notebook's recoverable artifact)."
    ),
    9: (
        "Step rollback. AgentCheckpoint splits the position into ``current`` (the "
        "live StepWatermark — commit watermark + read-before-write ledger and "
        "kernel-context ids in a nested AgentContextState) and ``step_watermarks`` "
        "(a per-step list of rollback points, each the start of a delivery step). "
        "Drives LLMAgent.rollback_to_step. This is the new minimum supported "
        "version — v5-v8 flat-head records are rejected by the schema-version floor."
    ),
    10: (
        "AgentCheckpoint gained ``folds``: summarized spans of the message log "
        "(context compaction) carried in the head so a lossy summary survives "
        "resume without re-running the LLM. Additive — v9 records load fine "
        "(folds defaults empty); v9 code resuming a v10 session ignores the folds "
        "and re-derives the view from the full log."
    ),
    11: (
        "Agent-team durability + record cleanup (folded — none of v11's pieces "
        "shipped separately). Three changes over v10: (a) AgentCheckpointLocation "
        "gained AFTER_RESIDENT_TURN, a resident agent's per-message turn boundary "
        "(its analog of AFTER_FINAL_ANSWER, written only by resident runs); "
        "(b) PersistedRecord carries one audit pair ``created_at`` + ``updated_at`` "
        "(updated_at bumped on each status change), replacing the never-read "
        "``saved_at`` and the duplicated ``TaskRecord.started_at`` (== created_at) / "
        "``MessageRecord.created_at``; (c) MessageRecord nests the message directly "
        "(``message: TeamMessage``) instead of an opaque ``body: str`` plus "
        "redundant top-level ``message_id``/``sender``/``recipient``. v10 records "
        "load fine (new fields default, dropped ones ignored) EXCEPT an in-flight "
        "v10 mailbox inbox record (``body`` string) won't load under v11 — drain a "
        "mailbox before upgrading; acked ``processed/`` records are unaffected "
        "(dedup probes existence, not content)."
    ),
    12: (
        "Session-scoped state moved out of per-processor checkpoints into one "
        "SessionCheckpoint per session_key (kind ``session``): the serialized "
        "``SessionContext.state`` (``context_kind``/``context_data``), the shared "
        "filesystem's ``fs_snapshot_ref``, and ``session_metadata``. "
        "AgentCheckpoint dropped ``context_kind``/``context_data`` (per-step "
        "watermark refs remain for rollback); ProcessorCheckpoint dropped "
        "``session_metadata``. v9-v11 records load fine (dropped fields "
        "ignored), but their persisted ``ctx.state`` and cold-resume filesystem "
        "restore are NOT migrated — a session saved with "
        "``serialize_state=True`` or ``fs_snapshot_policy`` under v11 code resumes "
        "under v12 with caller-built state and the live filesystem."
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
    TEAM = "team"
    TASK = "task"
    SESSION = "session"
    MAILBOX = "mailbox"


class CheckpointSchemaError(Exception):
    """Raised when a persisted record uses a newer schema version than known."""


class AgentCheckpointLocation(Enum):
    """How a checkpoint head came to be — its loop save-point, or a rollback."""

    AFTER_INPUT = "after_input"
    AFTER_TOOL_RESULT = "after_tool_result"
    AFTER_FINAL_ANSWER = "after_final_answer"
    AFTER_MAX_TURNS = "after_max_turns"
    # A resident agent's per-message turn boundary (it consumes a message inbox
    # between turns and produces no terminal answer): the resident analog of
    # AFTER_FINAL_ANSWER. Drives the per-turn checkpoint that persists the reply
    # and releases the consumed inbox message, and (like AFTER_FINAL_ANSWER) an
    # fs-snapshot so a restored transcript and filesystem stay in step.
    AFTER_RESIDENT_TURN = "after_resident_turn"
    # Not a loop save-point: a synthetic head written by rollback_to_step,
    # parked at the start of the rolled-back-to step.
    ROLLED_BACK = "rolled_back"


class AgentContextState(BaseModel):
    """
    Serializable snapshot of the agent-context state paired with a transcript.

    Captured by :meth:`AgentContext.snapshot` and reapplied by
    :meth:`AgentContext.restore` — the agent-context analogue of an
    environment's snapshot/restore. Failed-run rollback reapplies the
    transactional subset (read-before-write ledger, shell cwd, deferred
    background-task flips); a step rollback also re-attaches the live kernels
    (``rebind_kernels=True``) when it is paired with a restored filesystem
    snapshot. Carried inside a :class:`StepWatermark` so a rollback restores
    all of it in one move.
    """

    model_config = ConfigDict(frozen=True)

    read_file_state: dict[str, float] = Field(default_factory=dict[str, float])
    dotfile_overrides: list[str] = Field(default_factory=list[str])
    shell_cwd: str | None = None
    pending_delivered: dict[str, dict[str, Any]] = Field(
        default_factory=dict[str, dict[str, Any]]
    )
    ipy_exec_context_id: str | None = None
    nb_exec_context_id: str | None = None


class StepWatermark(BaseModel):
    """
    A restorable session position — the coordinates needed to rewind here.

    The transcript length to truncate to (``message_count`` against
    ``log_version``), the loop turn, the delivery step, the provider cache key,
    the filesystem snapshot to restore, and the agent-context state to reapply.
    :class:`AgentCheckpoint` holds one of these as its current position and a
    list of them as the per-step rollback points. See
    :meth:`LLMAgent.rollback_to_step`.
    """

    message_count: int = 0
    log_version: int = 0
    turn: int = 0
    step: int | None = None
    prompt_cache_key: str | None = None
    # Filesystem ref (restored via the environment's SnapshotCapable seam); the
    # rest of the session-resource state rides in ``agent_ctx_state``.
    fs_snapshot_ref: str | None = None
    agent_ctx_state: AgentContextState = Field(default_factory=AgentContextState)


class PersistedRecord(BaseModel):
    """Common base for any record persisted in the checkpoint store."""

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION)
    session_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

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
        if self.schema_version < MIN_SUPPORTED_SCHEMA_VERSION:
            raise CheckpointSchemaError(
                f"Persisted record schema version {self.schema_version} is "
                f"older than the oldest this process can load "
                f"(minimum={MIN_SUPPORTED_SCHEMA_VERSION}): "
                f"{SCHEMA_VERSION_SUMMARIES[MIN_SUPPORTED_SCHEMA_VERSION]}"
            )
        return self


class SessionCheckpoint(PersistedRecord):
    """
    The session-scoped half of a persisted session — state shared by every
    processor bound to one :class:`~grasp_agents.session_context.SessionContext`.

    Exactly one per ``session_key`` (kind ``session``, no processor path):
    the optionally serialized ``SessionContext.state`` and the latest filesystem
    snapshot of the shared ``ctx.environment``. Owned by ``SessionContext``
    (``save_checkpoint`` / ``load_checkpoint``) — never by an individual
    processor, whose checkpoints carry only their own working state.
    """

    context_kind: ContextKind | None = None
    context_data: Any | None = None

    # The shared filesystem as of the last session save: the snapshot taken
    # at that checkpoint boundary, or ``None`` when none was taken there —
    # a cold resume then keeps the live filesystem rather than rewinding to
    # a ref that no longer describes the transcript.
    fs_snapshot_ref: str | None = None

    # Operator-facing session labels (``SessionContext.session_metadata``).
    # Write-only: persisted for external inspection, never restored.
    session_metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessorCheckpoint(PersistedRecord):
    """Snapshot of a resumable processor's state at a turn boundary."""

    processor_name: str
    checkpoint_number: int = 0


class AgentCheckpoint(ProcessorCheckpoint):
    """
    Checkpoint for an agent session.

    ``current`` is where the session is *now* — any point in a step
    (``current.message_count`` is the commit watermark). ``step_watermarks``
    are the per-step rollback points: each marks the transcript position a
    delivery step is (re)delivered from, so :meth:`LLMAgent.rollback_to_step`
    rewinds to the *start* of a step (its input not yet present). Both kinds
    carry the read-before-write ledger and kernel-context ids (in their
    ``agent_ctx_state``). ``step_watermarks`` is empty for chat / untracked
    deliveries. ``output`` and ``location`` describe the current head only —
    the step's cached answer and where in the loop it was saved — so they live
    here rather than on the rewind coordinate.
    """

    # The transcript is persisted out-of-band as an append-only message log
    # (one ``InputItem`` per JSONL line), keyed alongside this head blob. It is
    # never written into the head: the head is overwritten every turn, so
    # embedding the growing transcript here would re-serialize it each time.
    # In memory this carries the full transcript; ``model_dump_json`` for the
    # head excludes it.
    messages: list[InputItem] = Field(default_factory=list[InputItem])

    usage: ResponseUsage | None = None

    # The live position. ``current.message_count`` is the commit watermark —
    # how many leading log records this head acknowledges; the log is appended
    # before the head is saved, so a crash between the two can leave uncommitted
    # trailing records and on resume only the first ``message_count`` are
    # trusted (the rest dropped).
    current: StepWatermark = Field(default_factory=StepWatermark)

    # Per-step rollback points (the start of each delivery step). One entry per
    # delivery step, not per turn, so it stays small in the head blob.
    step_watermarks: list[StepWatermark] = Field(default_factory=list[StepWatermark])

    # Compaction folds: summarized spans of the message log, applied to the
    # model-facing view (never the log). Kept in the head — small, and a lossy
    # summary must survive resume without re-running the LLM.
    folds: list[FoldSpec] = Field(default_factory=list[FoldSpec])

    # The current step's cached final answer — the re-delivery short-circuit
    # reads it. ``None`` while a step is incomplete.
    output: str | None = None

    # Where in the agent loop this head was saved (diagnostics only).
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
    and per-processor delivery-step counters so deliveries can be restored
    on resume. Per-processor session paths are fixed at construction time
    and don't need persisting.
    """

    pending_events: list[ProcPacketOutEvent]
    active_steps: dict[str, int] = Field(default_factory=dict)
    # The run's final (END-routed) result, set when the run completes — lets a
    # resume of a completed session return the result instead of failing.
    final_event: RunPacketOutEvent | None = None


class TeamCheckpoint(ProcessorCheckpoint):
    """
    Checkpoint for AgentTeam — the coordinator scalars the team alone owns.

    Deliberately tiny: the in-flight messages live in the durable mailbox
    transport and each member persists its own transcript, so this holds only
    what neither can express — the session-wide hop count. Its *existence* marks
    the session as seeded, so a resume skips re-seeding the entry (mirroring how
    a loaded ``RunnerCheckpoint`` suppresses the runner's start event). The
    terminal stop reason is re-derived on resume from this count (``activations
    >= max_hops``) and a fresh run, not frozen here — so a member error that
    stopped the run is retried, not permanently recorded.
    """

    activations: int = 0
