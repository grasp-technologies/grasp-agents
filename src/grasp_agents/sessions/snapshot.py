from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from ..types.items import InputItem
from ..types.response import ResponseUsage


class SessionSnapshot(BaseModel):
    """Lightweight checkpoint for an agent session. No business state."""

    session_id: str
    agent_name: str
    messages: list[InputItem]
    turn_number: int = 0
    usage: ResponseUsage | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
