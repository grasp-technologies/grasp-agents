from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Generic, Self, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .typing.io import ProcName

PacketRouting = Sequence[Sequence[ProcName]]
_PayloadT_co = TypeVar("_PayloadT_co", covariant=True)


def is_uniform_routing(routing: PacketRouting | None) -> Sequence[ProcName] | None:
    if not routing or len(routing) == 0:
        return None

    first_recipients = routing[0]
    first_set = set(first_recipients)
    for recipients in routing[1:]:
        if len(recipients) != len(first_recipients) or set(recipients) != first_set:
            return None

    return first_recipients


class Packet(BaseModel, Generic[_PayloadT_co]):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    payloads: Sequence[_PayloadT_co]
    sender: ProcName
    routing: PacketRouting | None = None

    @property
    def uniform_routing(self) -> Sequence[ProcName] | None:
        return is_uniform_routing(self.routing)

    @model_validator(mode="before")
    @classmethod
    def _normalize_routing(cls, data: dict[str, Any]) -> dict[str, Any]:
        routing = data.get("routing")
        if (
            routing
            and isinstance(routing, (list, tuple))
            and all(isinstance(r, str) for r in routing)  # type: ignore[misc]
        ):
            payloads = data.get("payloads", [])
            data["routing"] = [routing for _ in range(len(payloads))]
        return data

    @model_validator(mode="after")
    def _validate_routing(self) -> Self:
        if self.routing is not None and len(self.payloads) != len(self.routing):
            raise ValueError(
                "If routing is specified, its length must match the length of payloads"
            )
        return self

    def split_by_recipient(self) -> Sequence["Packet[_PayloadT_co]"] | None:
        if self.routing is None:
            return None

        recipient_to_payloads: defaultdict[ProcName, list[_PayloadT_co]] = defaultdict(
            list
        )
        for payload, recipients in zip(self.payloads, self.routing, strict=True):
            for recipient in recipients:
                recipient_to_payloads[recipient].append(payload)

        single_recipient = len(recipient_to_payloads) == 1
        return [
            Packet(
                id=f"{self.id}/{recipient}" if not single_recipient else self.id,
                payloads=payloads,
                routing=[[recipient] for _ in range(len(payloads))],
                sender=self.sender,
            )
            for recipient, payloads in recipient_to_payloads.items()
        ]

    def split_per_payload(self) -> Sequence["Packet[_PayloadT_co]"] | None:
        if self.routing is None:
            return None

        single_payload = len(self.payloads) == 1

        return [
            Packet(
                id=f"{self.id}/{i}" if not single_payload else self.id,
                payloads=[payload],
                routing=[recipients],
                sender=self.sender,
            )
            for i, (payload, recipients) in enumerate(
                zip(self.payloads, self.routing, strict=False)
            )
        ]

    model_config = ConfigDict(extra="forbid")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"ID: {self.id}\n"
            f"From: {self.sender}\n"
            f"Routing: {self.routing or 'None'}\n"
            f"Payloads: {len(self.payloads)}"
        )
