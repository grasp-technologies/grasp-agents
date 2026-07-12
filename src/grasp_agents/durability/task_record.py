from enum import StrEnum, auto
from typing import NotRequired, TypedDict

from .checkpoints import PersistedRecord


class TaskStatus(StrEnum):
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    DELIVERED = auto()  # Result/error was injected into parent memory


class TaskDeliveredFlip(TypedDict):
    """
    A deferred DELIVERED update for a ``TaskRecord``: the fields
    ``BackgroundTaskManager.flush_flips`` writes alongside the DELIVERED
    status once the transcript carrying the task's note is durable.
    """

    result: NotRequired[str | None]
    error: NotRequired[str | None]
    note_transcript_pos: NotRequired[int]


class TaskCancelledFlip(TypedDict):
    """
    A deferred CANCELLED update for a task cancelled by a transcript rewind,
    written alongside the CANCELLED status once the rewound transcript is
    durable.
    """

    error: str


class TaskRecord(PersistedRecord):
    """
    Lifecycle of a single background tool invocation.

    Tracks RUNNING → (COMPLETED | FAILED) → DELIVERED, or CANCELLED (killed), at
    ``"<session_key>/task/<parent_path>/tc_<call_id>"``. COMPLETED / FAILED is
    the finished-but-not-yet-delivered outcome a crash can leave behind (resume
    re-injects it); DELIVERED means the outcome reached the agent.
    """

    task_id: str

    # Monotonic per-agent launch sequence number. Watermarks record its
    # high-water value (``AgentContextState.task_launch_seq``); a rewind cancels
    # tasks above the watermark, and resume dead-letters records above the
    # restored head's — their launching tool call is not in the transcript.
    launch_seq: int = 0

    tool_call_id: str  # FunctionToolCallItem.call_id that spawned this
    tool_name: str
    tool_call_arguments: str | None = None  # Serialized tool input for resume replay

    status: TaskStatus = TaskStatus.RUNNING

    result: str | None = None
    error: str | None = None

    # 1-based transcript position of this task's completion note (== the
    # transcript length right after the note was appended) — the note's rewind
    # horizon: a rollback that cuts the transcript below it re-injects the
    # note (``BackgroundTaskManager.redeliver_after``). Set when the record
    # flips DELIVERED.
    note_transcript_pos: int | None = None

    # The agent-readable ``.grasp/tasks`` log file holding this task's full
    # streamed output (the single source of truth for it; ``None`` when no file
    # backend is wired). On resume the interrupted notice points the agent here.
    output_path: str | None = None
