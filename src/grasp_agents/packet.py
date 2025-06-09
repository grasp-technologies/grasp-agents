from collections.abc import Sequence
from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .typing.io import ProcessorName

_PayloadT_co = TypeVar("_PayloadT_co", covariant=True)


class Packet(BaseModel, Generic[_PayloadT_co]):
    payloads: Sequence[_PayloadT_co]
    sender: ProcessorName
    recipients: Sequence[ProcessorName] = Field(default_factory=list)
    message_id: str = Field(default_factory=lambda: str(uuid4())[:8])

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __repr__(self) -> str:
        return (
            f"From: {self.sender}, To: {', '.join(self.recipients)}, "
            f"Payloads: {len(self.payloads)}"
        )
