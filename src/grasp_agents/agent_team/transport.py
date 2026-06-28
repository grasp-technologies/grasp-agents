"""
Delivery substrate for inter-member messages.

:class:`MessageTransport` is the seam that decouples :class:`AgentTeam` from
*how* messages move. Two implementations ship:

- :class:`InMemoryMailboxTransport` — ephemeral, process-local; for a
  single-process team.
- :class:`CheckpointMailboxTransport` — durable, over the session
  :class:`CheckpointStore` (the same substrate background tasks persist through);
  cross-process via a shared store (e.g. ``FileCheckpointStore``).

An alternative — e.g. an Agent2Agent (A2A) adapter for members in another process
or framework — implements the same interface; the team and the ``SendMessage``
tool are unchanged.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from grasp_agents.durability.checkpoints import CheckpointKind
from grasp_agents.durability.message_record import MessageRecord
from grasp_agents.durability.store_keys import make_store_key
from grasp_agents.durability.task_record import TaskStatus
from grasp_agents.run_context import DEFAULT_SESSION_KEY
from grasp_agents.runtime import CLOSED, Closed, Transport

from .message import TeamMessage

if TYPE_CHECKING:
    from grasp_agents.durability.checkpoint_store import CheckpointStore
    from grasp_agents.run_context import RunContext


class MessageTransport(ABC):
    """A set of per-recipient mailboxes a sender writes to and a recipient drains."""

    async def send(self, message: TeamMessage) -> None:
        """
        Deposit ``message`` in each recipient's mailbox.

        A multi-recipient send is split into one single-recipient message per
        mailbox (so a message at rest is always single-recipient); subclasses
        only implement :meth:`_deposit`.
        """
        for single in message.split_by_recipient():
            await self._deposit(single)

    @abstractmethod
    async def _deposit(self, message: TeamMessage) -> None:
        """Deposit one single-recipient ``message`` in its recipient's mailbox."""

    @abstractmethod
    async def fetch_next(self, recipient: str) -> TeamMessage | None:
        """
        Return ``recipient``'s oldest undelivered message, or ``None``.

        One coherent group per call: a member consumes one delivered message per
        turn, never the whole mailbox merged. Non-consuming — a later call returns
        the same message until it is :meth:`ack`-ed.
        """

    @abstractmethod
    async def ack(self, recipient: str, message_ids: list[str]) -> None:
        """Mark messages delivered so a later :meth:`fetch` won't return them."""

    @abstractmethod
    async def has_mail(self, recipient: str) -> bool:
        """Whether ``recipient`` has any undelivered messages."""


class InMemoryMailboxTransport(MessageTransport):
    """
    Process-local mailboxes held in memory — no backend required.

    For single-process teams, where every member shares one event loop and this
    one instance. Not visible across processes: a separate-process team must use
    :class:`CheckpointMailboxTransport` over a shared store. Messages preserve
    send order per recipient.
    """

    def __init__(self) -> None:
        self._boxes: dict[str, list[TeamMessage]] = {}

    async def _deposit(self, message: TeamMessage) -> None:
        self._boxes.setdefault(message.recipients[0], []).append(message)

    async def fetch_next(self, recipient: str) -> TeamMessage | None:
        box = self._boxes.get(recipient)
        return box[0] if box else None

    async def ack(self, recipient: str, message_ids: list[str]) -> None:
        ids = set(message_ids)
        box = self._boxes.get(recipient)
        if box is not None:
            self._boxes[recipient] = [m for m in box if m.message_id not in ids]

    async def has_mail(self, recipient: str) -> bool:
        return bool(self._boxes.get(recipient))


class CheckpointMailboxTransport(MessageTransport):
    """
    Mailboxes persisted in the session :class:`CheckpointStore` as
    :class:`~grasp_agents.durability.MessageRecord`s — the same durable substrate
    (store, store-key convention, ``TaskStatus`` lifecycle) that background tasks
    use, so messages and task records live beside each other under one session key
    and resume together. Durable, and cross-process over a shared store
    (``FileCheckpointStore``).

    Keyed at ``"<session_key>/mailbox/<recipient>/inbox|processed/<message_id>"``;
    ack marks a record ``DELIVERED`` and moves it to ``processed/``. Delivery is
    at-least-once (a crash after the processed-write but before the inbox-delete
    redelivers). Messages within one mailbox are ordered by their timestamped id.
    """

    def __init__(
        self, store: CheckpointStore, *, session_key: str = DEFAULT_SESSION_KEY
    ) -> None:
        self._store = store
        self._session_key = session_key

    def _inbox_key(self, recipient: str, message_id: str) -> str:
        return make_store_key(
            self._session_key, CheckpointKind.MAILBOX, [recipient, "inbox", message_id]
        )

    def _processed_key(self, recipient: str, message_id: str) -> str:
        return make_store_key(
            self._session_key,
            CheckpointKind.MAILBOX,
            [recipient, "processed", message_id],
        )

    def _inbox_prefix(self, recipient: str) -> str:
        base = make_store_key(
            self._session_key, CheckpointKind.MAILBOX, [recipient, "inbox"]
        )
        return base + "/"

    async def _deposit(self, message: TeamMessage) -> None:
        recipient = message.recipients[0]
        record = MessageRecord(
            session_key=self._session_key,
            message_id=message.message_id,
            sender=message.sender,
            recipient=recipient,
            body=message.model_dump_json(),
            status=TaskStatus.PENDING,
            created_at=message.created_at,
        )
        await self._store.save(
            self._inbox_key(recipient, message.message_id),
            record.model_dump_json().encode(),
        )

    async def fetch_next(self, recipient: str) -> TeamMessage | None:
        keys = sorted(await self._store.list_keys(self._inbox_prefix(recipient)))
        for key in keys:
            record = await self._store.load_json(
                key, MessageRecord, subject="mailbox message"
            )
            if record is not None:
                return TeamMessage.model_validate_json(record.body)
        return None

    async def ack(self, recipient: str, message_ids: list[str]) -> None:
        for message_id in message_ids:
            inbox_key = self._inbox_key(recipient, message_id)
            data = await self._store.load(inbox_key)
            if data is None:
                continue
            delivered = MessageRecord.model_validate_json(data).model_copy(
                update={"status": TaskStatus.DELIVERED, "updated_at": datetime.now(UTC)}
            )
            await self._store.save(
                self._processed_key(recipient, message_id),
                delivered.model_dump_json().encode(),
            )
            await self._store.delete(inbox_key)

    async def has_mail(self, recipient: str) -> bool:
        return bool(await self._store.list_keys(self._inbox_prefix(recipient)))


class MailboxChannel(Transport[TeamMessage]):
    """
    Bridges a :class:`MessageTransport` (mailbox semantics) to the actor runtime's
    routing interface, so :class:`~grasp_agents.runtime.ActorDriver` can drive a
    team over any mailbox backend.

    A mailbox has no arrival signal, so :meth:`consume` polls
    :meth:`MessageTransport.fetch_next` every ``poll_interval`` seconds, waking
    immediately on :meth:`shutdown`. Because ``fetch_next`` is non-consuming and the
    driver only acks after a successful activation, delivery is at-least-once and an
    activation that is gated off (e.g. by a hop budget) leaves its message pending.
    """

    def __init__(self, mailbox: MessageTransport, *, poll_interval: float = 0.05):
        self._mailbox = mailbox
        self._poll_interval = poll_interval
        self._closed = asyncio.Event()

    def register(self, recipient: str) -> None:
        # A mailbox is created on first deposit; nothing to pre-allocate.
        del recipient

    async def post(self, envelope: TeamMessage) -> None:
        await self._mailbox.send(envelope)

    async def consume(self, recipient: str) -> TeamMessage | Closed:
        while not self._closed.is_set():
            message = await self._mailbox.fetch_next(recipient)
            if message is not None:
                return message
            try:
                await asyncio.wait_for(
                    self._closed.wait(), timeout=self._poll_interval
                )
            except TimeoutError:
                pass
        return CLOSED

    async def ack(self, recipient: str, envelope: TeamMessage) -> None:
        await self._mailbox.ack(recipient, [envelope.message_id])

    async def has_pending(self, recipient: str) -> bool:
        return await self._mailbox.has_mail(recipient)

    async def shutdown(self) -> None:
        self._closed.set()


def default_transport(ctx: RunContext[Any]) -> MessageTransport:
    """
    A durable transport over ``ctx.checkpoint_store``. Raises if no checkpoint
    store is wired — a single-process team uses :class:`InMemoryMailboxTransport`
    instead (the team falls back to it automatically).
    """
    store = ctx.checkpoint_store
    if store is None:
        raise ValueError(
            "Durable AgentTeam messaging requires ctx.checkpoint_store. Wire a "
            "CheckpointStore (e.g. FileCheckpointStore(root=...)), or rely on the "
            "in-memory transport for a single-process team."
        )
    return CheckpointMailboxTransport(store, session_key=ctx.session_key)
