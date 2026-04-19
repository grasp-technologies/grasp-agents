from datetime import UTC, datetime
from enum import StrEnum, auto
from typing import Self

from pydantic import BaseModel, Field, model_validator

from .checkpoints import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_VERSION_SUMMARIES,
    CheckpointSchemaError,
)


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

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION)
    task_id: str
    parent_session_id: str
    tool_call_id: str  # FunctionToolCallItem.call_id that spawned this
    tool_name: str
    tool_call_arguments: str | None = None  # Serialized tool input for resume replay
    status: TaskStatus = TaskStatus.PENDING
    child_session_id: str | None = None  # If set, child agent checkpoints here
    result: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def store_key(self) -> str:
        return f"task/{self.parent_session_id}/{self.task_id}"

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version > CURRENT_SCHEMA_VERSION:
            known = ", ".join(
                f"{v}: {s}" for v, s in sorted(SCHEMA_VERSION_SUMMARIES.items())
            )
            raise CheckpointSchemaError(
                f"TaskRecord schema version {self.schema_version} is newer than "
                f"this process understands (current={CURRENT_SCHEMA_VERSION}). "
                f"Known versions: {known}."
            )
        return self
