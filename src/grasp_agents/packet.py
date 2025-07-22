from collections.abc import Sequence
from typing import Annotated, Any, Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from .typing.io import ProcName

START_PROC_NAME: Literal["*START*"] = "*START*"

_PayloadT_co = TypeVar("_PayloadT_co", covariant=True)


class Packet(BaseModel, Generic[_PayloadT_co]):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    payloads: Sequence[_PayloadT_co]
    sender: ProcName
    recipients: Sequence[ProcName] | None = None

    model_config = ConfigDict(extra="forbid")

    def __repr__(self) -> str:
        _to = ", ".join(self.recipients) if self.recipients else "None"
        return (
            f"{self.__class__.__name__}:\n"
            f"ID: {self.id}\n"
            f"From: {self.sender}\n"
            f"To: {_to}\n"
            f"Payloads: {len(self.payloads)}"
        )


def _check_recipients_length(v: Sequence[ProcName] | None) -> Sequence[ProcName] | None:
    if v is not None and len(v) != 1:
        raise ValueError("recipients must contain exactly one item")
    return v


class StartPacket(Packet[_PayloadT_co]):
    chat_inputs: Any | None = "start"
    sender: ProcName = Field(default=START_PROC_NAME, frozen=True)
    payloads: Sequence[_PayloadT_co] = Field(default=(), frozen=True)
    recipients: Annotated[
        Sequence[ProcName] | None,
        AfterValidator(_check_recipients_length),
    ] = None
