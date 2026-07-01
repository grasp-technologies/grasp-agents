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

from .content import InputFile, InputImage, InputPart, InputText
from .io import LLMPrompt
from .items import InputMessageItem
from .packet import Packet, PacketRouting

# Sender name stamped on the initial input a host seeds into the entry member's
# inbox (members address each other by name; "user" is the human).
USER_SENDER = "user"

# Priority for control-plane mail (human input, self-wakeups): it drains ahead of
# normal peer messages so an interruption is weighed promptly. Default is 0.
CONTROL_PRIORITY = 1

_INPUT_PART_ADAPTER: TypeAdapter[Any] = TypeAdapter(InputPart)


def _new_message_id() -> str:
    # Timestamp prefix + short random suffix: within one priority, lexical id order
    # approximates arrival order, so a mailbox drains oldest-first — best-effort
    # (same-microsecond ties break on the random suffix; wall clocks are not
    # strictly monotonic, and cross-process clocks may skew).
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
    # Correlation handles for composing multi-message patterns — e.g. a buffering
    # processor (or a resident's own loop) grouping the replies of a fan-out before
    # synthesizing: ``context_id`` threads an exchange, ``task_id`` a unit of
    # requested work, ``reply_to`` targets a specific message.
    context_id: str | None = None
    task_id: str | None = None
    reply_to: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Higher drains first; ties break on the time-ordered id (FIFO within a
    # priority). Control-plane mail (human input, wakeups) uses ``CONTROL_PRIORITY``
    # so it preempts queued peer messages.
    priority: int = 0

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
        """
        The sole recipient. A message *at rest* in a mailbox has exactly one (a
        multi-recipient send is split per box first), so accessing this on a
        not-yet-split message is a bug and raises rather than silently returning the
        first recipient.
        """
        recipients = self.recipients
        if len(recipients) != 1:
            raise ValueError(
                f"TeamMessage.recipient requires exactly one recipient; got "
                f"{recipients}. Split the message per recipient first."
            )
        return recipients[0]

    @property
    def text(self) -> str:
        """The text payloads joined by newlines (non-text payloads omitted)."""
        return "\n".join(p.text for p in self.payloads if isinstance(p, InputText))

    @property
    def is_content(self) -> bool:
        """
        Whether every payload is a content part (text / image) — a "raw" message
        rendered as a recipient's user turn. ``False`` if any payload is a structured
        value (a typed body / a processor's output), which the recipient renders
        through its own input pipeline (``build_input_content``) instead. An empty
        message counts as content.
        """
        return all(
            isinstance(p, (InputText, InputImage, InputFile)) for p in self.payloads
        )

    def to_packet(self) -> Packet[Any]:
        """Downcast to a bare :class:`Packet` (drops mailbox/threading fields)."""
        return Packet(
            id=self.message_id,
            sender=self.sender,
            payloads=list(self.payloads),
            routing=self.routing,
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

    def to_chat_inputs(self) -> list[str | InputImage | InputFile]:
        """
        Render as a recipient's ``chat_inputs``: the rendered text header/body first,
        then any image / file payloads carried through as content — so multimodal
        input is delivered to the model, not flattened to text.
        """
        media = [p for p in self.payloads if isinstance(p, (InputImage, InputFile))]
        return [_format_inbound(self), *media]

    def to_input_message(self) -> InputMessageItem:
        """Render this message as a recipient's user-turn input message."""
        parts: list[InputPart] = [
            InputText(text=p) if isinstance(p, str) else p
            for p in self.to_chat_inputs()
        ]
        return InputMessageItem(content_parts=parts, role="user")

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
    def from_text(
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
    def from_input(
        cls,
        *,
        sender: str,
        to: str | Sequence[str],
        chat_inputs: LLMPrompt | Sequence[str | InputImage],
        **kwargs: Any,
    ) -> "TeamMessage":
        """
        Build a message from recipient-style input — text, or a mix of text and
        images — one payload per part (a plain ``str`` becomes a text part). Used for
        human input, which (like an agent's own ``chat_inputs``) may be multimodal.
        """
        recipients = [to] if isinstance(to, str) else list(to)
        if isinstance(chat_inputs, str):
            payloads: list[Any] = [InputText(text=chat_inputs)]
        else:
            payloads = [
                InputText(text=p) if isinstance(p, str) else p for p in chat_inputs
            ]
        return cls(sender=sender, routing=[recipients], payloads=payloads, **kwargs)


def _render_payload(payload: Any) -> str:
    if isinstance(payload, InputText):
        return payload.text
    if isinstance(payload, BaseModel):
        return payload.model_dump_json(indent=2)
    if isinstance(payload, dict):
        return json.dumps(payload, indent=2)
    return str(payload)


def _format_inbound(message: TeamMessage) -> str:
    """
    Render one inbound message's text as a user-turn header/body for the recipient.
    Image / file payloads are excluded here and carried separately as content parts
    (see :meth:`TeamMessage.to_chat_inputs`), so they reach a multimodal model intact.
    """
    header = f"<team_member_message from={message.sender}"
    if message.reply_to:
        header += f" reply_to={message.reply_to}"
    header += ">"
    body_parts = [
        _render_payload(p)
        for p in message.payloads
        if not isinstance(p, (InputImage, InputFile))
    ]
    tail = "</team_member_message>"

    return "\n\n".join(p for p in (header, *body_parts, tail) if p)
