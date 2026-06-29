"""
The inter-agent message envelope — a routed, multi-recipient mailbox item.

A neutral data type (it depends only on other ``types`` modules), so it is the
unit carried by both a single agent's durable inbox and a multi-agent host's
routing. It is the messaging sibling of :class:`~grasp_agents.types.packet.Packet`:
same ``sender`` / ``payloads`` / ``routing``, plus conversation threading and a
time-ordered id.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

from .content import InputPart, InputText
from .items import InputMessageItem
from .packet import Packet, PacketRouting

# Sender name stamped on the initial input a host seeds into the entry member's
# inbox (members address each other by name; "user" is the human).
USER_SENDER = "user"

_INPUT_PART_ADAPTER: TypeAdapter[Any] = TypeAdapter(InputPart)


def _new_message_id() -> str:
    # Timestamp prefix + short random suffix: lexical id order equals arrival
    # order, so a mailbox keyed by id drains oldest-first.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"{stamp}-{uuid4().hex[:8]}"


class TeamMessage(BaseModel):
    """
    One message between agents.

    A mailbox sibling of :class:`~grasp_agents.types.packet.Packet`: it carries the
    same ``sender`` / ``payloads`` / ``routing`` an inter-processor packet does
    (so a member's output routes onward unchanged), plus conversation threading and
    a time-ordered id. ``routing`` is multi-recipient (per payload); a send is split
    into one single-recipient message per mailbox (see :meth:`split_by_recipient`),
    so a message *at rest* is single-recipient. ``payloads`` are arbitrary — content
    parts for an agent send, structured values for a processor's output. Content
    parts round-trip a durable transport intact; other payloads must be
    JSON-serializable and come back as plain data (not their original type).
    """

    message_id: str = Field(default_factory=_new_message_id)
    sender: str
    routing: PacketRouting
    payloads: Sequence[Any] = Field(default_factory=list)
    # Threads a multi-message exchange / a unit of requested work / a reply target.
    context_id: str | None = None
    task_id: str | None = None
    reply_to: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("payloads", mode="before")
    @classmethod
    def _rehydrate_content_parts(cls, value: Any) -> Any:
        # A content part serialized over a durable transport comes back as a dict
        # with a discriminating ``type``; re-parse it to its InputPart model so
        # ``isinstance`` / ``.text`` keep working. Non-content payloads pass through.
        if not isinstance(value, list):
            return value
        out: list[Any] = []
        for p in value:  # type: ignore[misc]
            if isinstance(p, dict):
                try:
                    out.append(_INPUT_PART_ADAPTER.validate_python(p))
                    continue
                except ValidationError:
                    pass
            out.append(p)
        return out

    @property
    def recipients(self) -> list[str]:
        """Distinct recipients across the routing, in first-seen order."""
        seen: list[str] = []
        for group in self.routing:
            for r in group:
                if r not in seen:
                    seen.append(r)
        return seen

    @property
    def recipient(self) -> str:
        """The single recipient (a message at rest in a mailbox has exactly one)."""
        return self.recipients[0]

    @property
    def text(self) -> str:
        """The text payloads joined by newlines (non-text payloads omitted)."""
        return "\n".join(p.text for p in self.payloads if isinstance(p, InputText))

    def to_packet(self) -> Packet[Any]:
        """Downcast to a bare :class:`Packet` (drops mailbox/threading fields)."""
        return Packet(
            id=self.message_id,
            sender=self.sender,
            payloads=list(self.payloads),
            routing=self.routing,
        )

    def to_input_message(self) -> InputMessageItem:
        """Render this message as a recipient's user-turn input."""
        return InputMessageItem.from_text(format_inbound(self), role="user")

    def split_by_recipient(self) -> "list[TeamMessage]":
        """One single-recipient message per recipient (threading + id preserved)."""
        return [
            self.model_copy(
                update={
                    "routing": list(sub.routing or []),
                    "payloads": list(sub.payloads),
                }
            )
            for sub in (self.to_packet().split_by_recipient() or [])
        ]

    @classmethod
    def of_text(
        cls,
        *,
        sender: str,
        to: str | Sequence[str],
        text: str,
        **kwargs: Any,
    ) -> "TeamMessage":
        """Build a single-payload text message to one or more recipients."""
        recipients = [to] if isinstance(to, str) else list(to)
        return cls(
            sender=sender,
            routing=[recipients],
            payloads=[InputText(text=text)],
            **kwargs,
        )

    @classmethod
    def from_packet(cls, packet: Packet[Any], **threading: Any) -> "TeamMessage":
        """Wrap a processor's output packet as a message (adds id + threading)."""
        return cls(
            sender=packet.sender,
            routing=list(packet.routing or []),
            payloads=list(packet.payloads),
            **threading,
        )


def _render_payload(payload: Any) -> str:
    if isinstance(payload, InputText):
        return payload.text
    if isinstance(payload, BaseModel):
        return payload.model_dump_json(indent=2)
    if isinstance(payload, dict):
        return json.dumps(payload, indent=2)
    return str(payload)


def format_inbound(message: TeamMessage) -> str:
    """Render one inbound message as a user-turn prompt for the recipient."""
    header = f"Message from {message.sender}"
    if message.reply_to:
        header += f" (reply to {message.reply_to})"
    body = "\n".join(_render_payload(p) for p in message.payloads)
    return f"{header}:\n{body}"
