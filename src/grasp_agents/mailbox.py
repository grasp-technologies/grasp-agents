"""
The durable message mailboxes — implementations of the actor runtime's
:class:`~grasp_agents.runtime.Transport` for :class:`TeamMessage` delivery.

There is **one** transport abstraction (``runtime.Transport[E]``); these are its
mailbox implementations, beside the in-process :class:`~grasp_agents.runtime.
InProcessTransport` used for event routing. So a single agent's inbox, a multi-agent
team's driver, and (in future) a networked backend all sit on the same seam — no
adapter, no parallel transport type. A mailbox has no native arrival signal, so
:meth:`consume` blocks by polling :meth:`has_pending` every ``poll_interval`` and
wakes immediately on :meth:`shutdown`; ``fetch`` is non-removing and a consumer
``ack``s after a successful activation, so delivery is at-least-once.

- :class:`InMemoryMailboxTransport` — ephemeral, process-local; single-process.
- :class:`CheckpointMailboxTransport` — durable, over the session
  :class:`CheckpointStore` (the substrate background-task records also persist
  through); cross-process via a shared store (e.g. ``FileCheckpointStore``).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from grasp_agents.durability.checkpoints import CheckpointKind
from grasp_agents.durability.message_record import MessageRecord
from grasp_agents.durability.store_keys import make_store_key
from grasp_agents.durability.task_record import TaskStatus
from grasp_agents.run_context import DEFAULT_SESSION_KEY
from grasp_agents.runtime import CLOSED, Closed, Transport
from grasp_agents.types.message import TeamMessage

if TYPE_CHECKING:
    from grasp_agents.durability.checkpoint_store import CheckpointStore


class InMemoryMailboxTransport(Transport[TeamMessage]):
    """
    Process-local mailboxes held in memory — no backend required.

    For single-process teams, where every member shares one event loop and this
    one instance. Not visible across processes: a separate-process team must use
    :class:`CheckpointMailboxTransport` over a shared store. Messages preserve
    send order per recipient.
    """

    def __init__(self, *, poll_interval: float = 0.05) -> None:
        self._boxes: dict[str, list[TeamMessage]] = {}
        self._poll_interval = poll_interval
        self._closed = asyncio.Event()

    def register(self, recipient: str) -> None:
        self._boxes.setdefault(recipient, [])

    async def post(self, envelope: TeamMessage) -> None:
        # A multi-recipient send is split into one single-recipient message per box.
        for single in envelope.split_by_recipient():
            self._boxes.setdefault(single.recipient, []).append(single)

    async def consume(self, recipient: str) -> TeamMessage | Closed:
        while not self._closed.is_set():
            box = self._boxes.get(recipient)
            if box:
                return box[0]  # non-removing; removed by ack after activation
            try:
                await asyncio.wait_for(self._closed.wait(), self._poll_interval)
            except TimeoutError:
                pass
        return CLOSED

    async def ack(self, recipient: str, envelope: TeamMessage) -> None:
        box = self._boxes.get(recipient)
        if box is not None:
            self._boxes[recipient] = [
                m for m in box if m.message_id != envelope.message_id
            ]

    async def has_pending(self, recipient: str) -> bool:
        return bool(self._boxes.get(recipient))

    async def shutdown(self) -> None:
        self._closed.set()


class CheckpointMailboxTransport(Transport[TeamMessage]):
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
        self,
        store: CheckpointStore,
        *,
        session_key: str = DEFAULT_SESSION_KEY,
        poll_interval: float = 0.05,
    ) -> None:
        self._store = store
        self._session_key = session_key
        self._poll_interval = poll_interval
        self._closed = asyncio.Event()

    def register(self, recipient: str) -> None:
        # Mailboxes are keyed in the store; nothing to pre-allocate.
        del recipient

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

    async def post(self, envelope: TeamMessage) -> None:
        for single in envelope.split_by_recipient():
            recipient = single.recipient
            record = MessageRecord(
                session_key=self._session_key,
                message_id=single.message_id,
                sender=single.sender,
                recipient=recipient,
                body=single.model_dump_json(),
                status=TaskStatus.PENDING,
                created_at=single.created_at,
            )
            await self._store.save(
                self._inbox_key(recipient, single.message_id),
                record.model_dump_json().encode(),
            )

    async def _fetch_next(self, recipient: str) -> TeamMessage | None:
        keys = sorted(await self._store.list_keys(self._inbox_prefix(recipient)))
        for key in keys:
            record = await self._store.load_json(
                key, MessageRecord, subject="mailbox message"
            )
            if record is not None:
                return TeamMessage.model_validate_json(record.body)
        return None

    async def consume(self, recipient: str) -> TeamMessage | Closed:
        while not self._closed.is_set():
            message = await self._fetch_next(recipient)
            if message is not None:
                return message
            try:
                await asyncio.wait_for(self._closed.wait(), self._poll_interval)
            except TimeoutError:
                pass
        return CLOSED

    async def ack(self, recipient: str, envelope: TeamMessage) -> None:
        inbox_key = self._inbox_key(recipient, envelope.message_id)
        data = await self._store.load(inbox_key)
        if data is None:
            return
        delivered = MessageRecord.model_validate_json(data).model_copy(
            update={"status": TaskStatus.DELIVERED, "updated_at": datetime.now(UTC)}
        )
        await self._store.save(
            self._processed_key(recipient, envelope.message_id),
            delivered.model_dump_json().encode(),
        )
        await self._store.delete(inbox_key)

    async def has_pending(self, recipient: str) -> bool:
        return bool(await self._store.list_keys(self._inbox_prefix(recipient)))

    async def shutdown(self) -> None:
        self._closed.set()
