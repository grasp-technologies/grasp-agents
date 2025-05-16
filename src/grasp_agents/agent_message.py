from collections.abc import Sequence
from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .typing.io import AgentID, AgentState

_PayloadT = TypeVar("_PayloadT", covariant=True)  # noqa: PLC0105
_StateT = TypeVar("_StateT", bound=AgentState, covariant=True)  # noqa: PLC0105


class AgentMessage(BaseModel, Generic[_PayloadT, _StateT]):
    payloads: Sequence[_PayloadT]
    sender_id: AgentID
    sender_state: _StateT | None = None
    recipient_ids: Sequence[AgentID] = Field(default_factory=list)

    message_id: str = Field(default_factory=lambda: str(uuid4())[:8])

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __repr__(self) -> str:
        return (
            f"From: {self.sender_id}, To: {', '.join(self.recipient_ids)}, "
            f"Payloads: {len(self.payloads)}"
        )
