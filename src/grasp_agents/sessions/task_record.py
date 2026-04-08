from datetime import UTC, datetime
from enum import StrEnum, auto

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    PENDING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    DELIVERED = auto()  # Result/error was injected into parent memory


class TaskRecord(BaseModel):
    """
    Persisted link between a parent agent session and a background task.

    Created when AgentLoop spawns a background tool, updated on
    completion/failure/cancellation. On session resume, pending records
    indicate interrupted tasks whose results were never delivered.

    Store key: ``task/{parent_session_id}/{task_id}``
    """

    task_id: str
    parent_session_id: str
    tool_call_id: str  # FunctionToolCallItem.call_id that spawned this
    tool_name: str
    status: TaskStatus = TaskStatus.PENDING
    child_session_id: str | None = None  # If set, child agent checkpoints here
    result: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def store_key(self) -> str:
        return f"task/{self.parent_session_id}/{self.task_id}"
