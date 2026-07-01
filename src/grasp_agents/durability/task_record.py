from enum import StrEnum, auto

from .checkpoints import PersistedRecord


class TaskStatus(StrEnum):
    PENDING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    DELIVERED = auto()  # Result/error was injected into parent memory


class TaskRecord(PersistedRecord):
    """
    Lifecycle of a single background tool invocation.

    Tracks PENDING → (COMPLETED | FAILED) → DELIVERED, or CANCELLED (killed), at
    ``"<session_key>/task/<parent_path>/tc_<call_id>"``. COMPLETED / FAILED is
    the finished-but-not-yet-delivered outcome a crash can leave behind (resume
    re-injects it); DELIVERED means the outcome reached the agent.
    """

    task_id: str

    tool_call_id: str  # FunctionToolCallItem.call_id that spawned this
    tool_name: str
    tool_call_arguments: str | None = None  # Serialized tool input for resume replay

    status: TaskStatus = TaskStatus.PENDING

    result: str | None = None
    error: str | None = None

    # The agent-readable ``.grasp/tasks`` log file holding this task's full
    # streamed output (the single source of truth for it; ``None`` when no file
    # backend is wired). On resume the interrupted notice points the agent here.
    output_path: str | None = None
