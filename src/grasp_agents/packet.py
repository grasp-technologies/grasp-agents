from collections.abc import Sequence
from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .typing.io import ProcName

_PayloadT_co = TypeVar("_PayloadT_co", covariant=True)


class Packet(BaseModel, Generic[_PayloadT_co]):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    payloads: Sequence[_PayloadT_co]
    sender: ProcName
    recipients: Sequence[ProcName] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"ID: {self.id}\n"
            f"From: {self.sender}\n"
            f"To: {', '.join(self.recipients)}\n"
            f"Payloads: {len(self.payloads)}"
        )
