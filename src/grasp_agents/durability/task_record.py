from datetime import UTC, datetime
from enum import StrEnum, auto

from pydantic import Field

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

    Tracks PENDING → COMPLETED → DELIVERED (or FAILED / CANCELLED) at
    ``"<session_key>/<parent_kind>/<parent_path>/tc_<call_id>/lifecycle"``.
    """

    task_id: str

    tool_call_id: str  # FunctionToolCallItem.call_id that spawned this
    tool_name: str
    tool_call_arguments: str | None = None  # Serialized tool input for resume replay

    status: TaskStatus = TaskStatus.PENDING

    result: str | None = None
    error: str | None = None

    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
