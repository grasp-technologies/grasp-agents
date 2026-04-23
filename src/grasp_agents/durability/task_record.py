from datetime import UTC, datetime
from enum import StrEnum, auto
from typing import Self

from pydantic import BaseModel, Field, model_validator

from .checkpoints import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_VERSION_SUMMARIES,
    CheckpointSchemaError,
)
from .keys import task_key


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

    Store key: ``task/{parent_session_key}/{task_id}``
    """

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION)
    task_id: str
    parent_session_key: str
    tool_call_id: str  # FunctionToolCallItem.call_id that spawned this
    tool_name: str
    tool_call_arguments: str | None = None  # Serialized tool input for resume replay
    status: TaskStatus = TaskStatus.PENDING
    # DEPRECATED: retained for back-compat with pre-B2 task records that
    # minted a separate session_key for each child. New records leave
    # this ``None`` and the child runs under the parent's session_key
    # with a derived session-path — see
    # :func:`..keys.background_child_session_subpath`.
    child_session_key: str | None = None
    result: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def store_key(self) -> str:
        return task_key(self.parent_session_key, self.task_id)

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
