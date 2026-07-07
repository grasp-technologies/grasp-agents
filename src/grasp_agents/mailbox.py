"""
The durable message mailboxes — implementations of the actor runtime's
:class:`~grasp_agents.runtime.Transport` for :class:`TeamMessage` delivery.

There is **one** transport abstraction (``runtime.Transport[E]``); these are its
mailbox implementations, beside the in-process :class:`~grasp_agents.runtime.
InProcessTransport` used for event routing. So a single agent's inbox, a multi-agent
team's driver, and (in future) a networked backend all sit on the same seam — no
adapter, no parallel transport type. A mailbox has no native arrival signal, so
:meth:`consume` blocks by polling :meth:`has_pending` every ``poll_interval``;
``fetch`` is non-removing and a consumer ``ack``s after a successful activation,
so delivery is at-least-once. The transport is session-scoped and outlives any
one run — consumers stop by cancellation, not by closing it.

- :class:`InMemoryMailboxTransport` — ephemeral, process-local; single-process.
- :class:`CheckpointMailboxTransport` — durable, over the session
  :class:`CheckpointStore` (the substrate background-task records also persist
  through); cross-process via a shared store (e.g. ``FileCheckpointStore``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from grasp_agents.durability.checkpoints import CheckpointKind, CheckpointSchemaError
from grasp_agents.durability.message_record import MessageRecord
from grasp_agents.durability.store_keys import key_leaf, make_store_key
from grasp_agents.durability.task_record import TaskStatus
from grasp_agents.runtime import CLOSED, Closed, Transport
from grasp_agents.session_context import DEFAULT_SESSION_KEY
from grasp_agents.types.message import USER_SENDER, TeamMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from datetime import timedelta

    from grasp_agents.durability.checkpoint_store import CheckpointStore

logger = logging.getLogger(__name__)


def _dropped_message_note(message: TeamMessage, *, recipient: str) -> str:
    excerpt = message.text[:200]
    return (
        "<message_dropped>\n"
        f"Your message to '{recipient}' (id {message.message_id}) was "
        "discarded by a conversation rollback and will not be answered"
        f"{': ' + excerpt if excerpt else '.'}\n"
        "Resend it if it is still relevant.\n"
        "</message_dropped>"
    )


async def void_mail_after(
    transport: Transport[TeamMessage],
    recipient: str,
    seq: int,
    *,
    leased: Sequence[TeamMessage] = (),
) -> None:
    """
    The mailbox half of a step rollback: everything ``recipient`` consumed
    after ``seq`` is voided, never re-delivered — a rollback rewrites history,
    so the human supplies new input instead of a replay (their dropped
    messages are discarded silently), while each peer whose message was
    dropped gets a ``<message_dropped>`` note so it can resend what is still
    relevant. ``leased`` takes (absorbed but never acked) are acked first so
    the void sweep covers them. Each note posts BEFORE its message's void is
    made durable, so a crash in between re-notifies on the retried rollback
    (a duplicate note beats a silent drop).
    """
    for message in leased:
        await transport.ack(recipient, message)

    async def notify(message: TeamMessage) -> None:
        if message.sender == USER_SENDER:
            return
        await transport.post(
            TeamMessage.from_text(
                sender=recipient,
                to=message.sender,
                text=_dropped_message_note(message, recipient=recipient),
            )
        )

    await transport.void_processed_after(recipient, seq, on_void=notify)


class InMemoryMailboxTransport(Transport[TeamMessage]):
    """
    Process-local mailboxes held in memory — no backend required.

    For single-process teams, where every member shares one event loop and this
    one instance. Not visible across processes: a separate-process team must use
    :class:`CheckpointMailboxTransport` over a shared store. Messages preserve
    send order per recipient.
    """

    def __init__(self, *, poll_interval: float = 0.05) -> None:
        super().__init__()
        self._boxes: dict[str, list[TeamMessage]] = {}
        # Acked messages whose consumer stamped a consumption ``seq``, retained
        # as the dedup record and for a step rollback to void (the in-memory
        # analog of the durable ``processed/`` records). Untracked acks
        # (``seq == 0``) are dropped outright, as before.
        self._processed: dict[str, list[TeamMessage]] = {}
        # Ids already voided by a rollback, so a repeated rollback over the
        # same range does not re-notify their senders.
        self._voided: set[str] = set()
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
                # Highest priority first, oldest within a priority.
                # Non-removing; removed by ack.
                return min(box, key=lambda m: (-m.priority, m.message_id))
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
        processed = self._processed.setdefault(recipient, [])
        if envelope.seq > 0 and all(
            m.message_id != envelope.message_id for m in processed
        ):
            processed.append(envelope)

    async def has_pending(self, recipient: str) -> bool:
        return bool(self._boxes.get(recipient))

    async def was_processed(self, recipient: str, envelope_id: str) -> bool:
        # The retained-processed list doubles as the dedupe record, so a
        # deterministic-id re-post (an entry seed) is skipped here exactly as
        # it is on the durable transport. Untracked acks (``seq == 0``) are
        # not retained and so not deduped — they have no redelivery source
        # in-process.
        return any(
            m.message_id == envelope_id for m in self._processed.get(recipient, [])
        )

    async def void_processed_after(
        self,
        recipient: str,
        seq: int,
        *,
        on_void: Callable[[TeamMessage], Awaitable[None]] | None = None,
    ) -> list[TeamMessage]:
        voided: list[TeamMessage] = []
        for message in self._processed.get(recipient, []):
            if message.seq > seq and message.message_id not in self._voided:
                if on_void is not None:
                    await on_void(message)
                self._voided.add(message.message_id)
                voided.append(message)
        return voided

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

    Keyed at ``"<session_key>/mailbox/<recipient>/inbox/<lane>/<message_id>"`` (and
    ``processed/<message_id>``); ack marks a record ``DELIVERED`` and moves it to
    ``processed/``. Delivery is at-least-once (a crash after the processed-write but
    before the inbox-delete redelivers). The ``<lane>`` is an inverted-priority
    prefix, so sorted inbox keys yield priority-then-id order (control mail first,
    then *best-effort* FIFO within a priority — ids are time-ordered but not
    strictly monotonic, see ``_new_message_id``) — and the scan stays lazy (no
    per-message priority load).

    **Invariant: one consumer per recipient.** ``ack`` is a read-modify-write
    (load → save ``processed/`` → delete ``inbox/``) and the store takes no
    cross-process lock, so two processes draining the *same* recipient could
    double-deliver or race the ack. The runtime enforces a single drainer per
    actor in-process; a separate-process ``MemberHost`` deployment must give each
    member its own process (its own recipient). An advisory lease would be needed
    only if that constraint is ever relaxed.
    """

    def __init__(
        self,
        store: CheckpointStore,
        *,
        session_key: str = DEFAULT_SESSION_KEY,
        poll_interval: float = 0.05,
    ) -> None:
        super().__init__()
        self._store = store
        self._session_key = session_key
        self._poll_interval = poll_interval
        self._closed = asyncio.Event()

    def register(self, recipient: str) -> None:
        # Mailboxes are keyed in the store; nothing to pre-allocate.
        del recipient

    @staticmethod
    def _lane(priority: int) -> str:
        # Inverted, fixed-width: a higher priority yields a smaller string, so
        # sorted keys put control mail first; ids then order FIFO within a lane.
        return f"{99 - max(0, min(priority, 99)):02d}"

    def _inbox_key(self, recipient: str, message: TeamMessage) -> str:
        return make_store_key(
            self._session_key,
            CheckpointKind.MAILBOX,
            [recipient, "inbox", self._lane(message.priority), message.message_id],
        )

    def _processed_key(self, recipient: str, message_id: str) -> str:
        return make_store_key(
            self._session_key,
            CheckpointKind.MAILBOX,
            [recipient, "processed", message_id],
        )

    def _corrupt_key(self, recipient: str, message_id: str) -> str:
        return make_store_key(
            self._session_key,
            CheckpointKind.MAILBOX,
            [recipient, "corrupt", message_id],
        )

    def _inbox_prefix(self, recipient: str) -> str:
        base = make_store_key(
            self._session_key, CheckpointKind.MAILBOX, [recipient, "inbox"]
        )
        return base + "/"

    def _processed_prefix(self, recipient: str) -> str:
        base = make_store_key(
            self._session_key, CheckpointKind.MAILBOX, [recipient, "processed"]
        )
        return base + "/"

    def _mailbox_prefix(self) -> str:
        base = make_store_key(self._session_key, CheckpointKind.MAILBOX, [])
        return base + "/"

    async def post(self, envelope: TeamMessage) -> None:
        for single in envelope.split_by_recipient():
            record = MessageRecord(
                session_key=self._session_key,
                message=single,
                status=TaskStatus.PENDING,
            )
            await self._store.save(
                self._inbox_key(single.recipient, single),
                record.model_dump_json().encode(),
            )

    async def _fetch_next(self, recipient: str) -> TeamMessage | None:
        keys = sorted(await self._store.list_keys(self._inbox_prefix(recipient)))
        for key in keys:
            data = await self._store.load(key)
            if data is None:
                # Vanished between the list and the load — a concurrent ack
                # removed it. Benign; move on.
                continue
            try:
                record = MessageRecord.model_validate_json(data)
            except CheckpointSchemaError:
                # A version we cannot interpret — a real, loud failure, not
                # corruption. Let it propagate (matches load_json everywhere).
                raise
            except ValueError:
                # Unparseable bytes (a torn write). Silently skipping would leave
                # the key in inbox/ forever — has_pending stays True, so a resident
                # never quiesces and a triggered consumer parks. Dead-letter it so
                # the mailbox unwedges and the bad record is preserved for
                # inspection, then keep scanning.
                await self._dead_letter(recipient, key, data)
                continue
            return record.message
        return None

    async def _dead_letter(self, recipient: str, inbox_key: str, data: bytes) -> None:
        message_id = key_leaf(inbox_key)
        logger.error(
            "Corrupt mailbox record at %s — dead-lettering to corrupt/ so the "
            "mailbox does not wedge. Inspect the preserved record.",
            inbox_key,
        )
        # Preserve before removing (same crash-safe ordering as ack): a crash
        # between the two re-dead-letters the same record next scan — idempotent,
        # never losing it. Wrapped with a timestamp so ``prune_processed`` can age
        # the corrupt/ tree out (the raw bytes are unparseable, so they are stored
        # decoded-with-replacement inside the envelope).
        envelope = json.dumps(
            {
                "dead_lettered_at": datetime.now(UTC).isoformat(),
                "raw": data.decode("utf-8", errors="replace"),
            }
        ).encode()
        await self._store.save(self._corrupt_key(recipient, message_id), envelope)
        await self._store.delete(inbox_key)

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
        inbox_key = self._inbox_key(recipient, envelope)
        data = await self._store.load(inbox_key)
        if data is None:
            return
        processed_key = self._processed_key(recipient, envelope.message_id)
        if await self._store.load(processed_key) is None:
            # First ack wins, storing the acking consumer's envelope — it
            # carries the consumption ``seq`` that :meth:`void_processed_after`
            # compares. A redelivery dedupe re-acks with an unstamped copy and
            # must only clear the lingering inbox key, never clobber the seq.
            delivered = MessageRecord.model_validate_json(data).model_copy(
                update={
                    "status": TaskStatus.DELIVERED,
                    "message": envelope,
                    "updated_at": datetime.now(UTC),
                }
            )
            await self._store.save(processed_key, delivered.model_dump_json().encode())
        await self._store.delete(inbox_key)

    async def has_pending(self, recipient: str) -> bool:
        return bool(await self._store.list_keys(self._inbox_prefix(recipient)))

    async def was_processed(self, recipient: str, envelope_id: str) -> bool:
        # A ``processed/`` record exists iff ``ack`` already moved this message
        # there. Reusing that durable forensic copy as the at-most-once guard:
        # a redelivery (the message lingered in ``inbox/`` because the inbox
        # delete hadn't landed) finds the processed copy and skips re-running.
        key = self._processed_key(recipient, envelope_id)
        return await self._store.load(key) is not None

    async def void_processed_after(
        self,
        recipient: str,
        seq: int,
        *,
        on_void: Callable[[TeamMessage], Awaitable[None]] | None = None,
    ) -> list[TeamMessage]:
        """
        Mark ``processed/`` records with consumption ``seq >`` the watermark
        ``CANCELLED`` — the mailbox half of a step rollback. Voided records
        are never re-delivered (a rollback rewrites history; senders are
        notified via ``on_void`` instead) but stay on record for the
        redelivery dedup (:meth:`was_processed`). Returns the voided
        messages, once: already-CANCELLED records are skipped, so retrying a
        rollback (or a later rollback over the same range) does not re-notify
        senders. ``on_void`` runs BEFORE a record's CANCELLED save — a crash
        in between re-runs it on the retry (at-least-once notification).
        """
        voided: list[TeamMessage] = []
        for key in await self._store.list_keys(self._processed_prefix(recipient)):
            record = await self._store.load_json(
                key, MessageRecord, subject="mailbox processed record"
            )
            if (
                record is None
                or record.message.seq <= seq
                or record.status is TaskStatus.CANCELLED
            ):
                continue
            if on_void is not None:
                await on_void(record.message)
            cancelled = record.model_copy(
                update={
                    "status": TaskStatus.CANCELLED,
                    "updated_at": datetime.now(UTC),
                }
            )
            await self._store.save(key, cancelled.model_dump_json().encode())
            voided.append(record.message)
        return voided

    async def prune_processed(
        self,
        *,
        older_than: timedelta,
        keep: Callable[[str], bool] | None = None,
        corrupt_older_than: timedelta | None = None,
    ) -> int:
        """
        Delete stale ``processed/`` and ``corrupt/`` mailbox records, returning the
        count removed.

        ``ack`` keeps a ``processed/`` copy of every delivered message — the
        at-most-once dedup guard :meth:`was_processed` consults, and a forensic
        record; ``_dead_letter`` keeps a ``corrupt/`` copy of every torn record. Both
        accumulate over a long-running session; this offline sweep reclaims them.

        A ``processed/`` record is removed only when **no longer deliverable** (its
        ``inbox/`` counterpart is gone, so no crash-redelivery can still need its
        dedup guard) *and* older than ``older_than``; ``keep`` pins records whose
        message id it returns ``True`` for (e.g. a permanent entry-seed marker a
        later resume still consults). A ``corrupt/`` record is removed once older than
        ``corrupt_older_than`` (default: ``older_than`` — pass a longer window to keep
        dead-letters around for inspection). A ``processed/`` record whose stored
        schema version this process cannot read is skipped, not raised on — GC is
        best-effort and never re-runs a record. Mirrors
        ``BackgroundTaskManager.prune_delivered``.
        """
        now = datetime.now(UTC)
        processed_cutoff = now - older_than
        corrupt_cutoff = now - (corrupt_older_than or older_than)
        keys = await self._store.list_keys(self._mailbox_prefix())
        pending = {key_leaf(k) for k in keys if "/inbox/" in k}
        pruned = 0
        for key in keys:
            message_id = key_leaf(key)
            if "/processed/" in key:
                if message_id in pending or (keep is not None and keep(message_id)):
                    continue
                try:
                    record = await self._store.load_json(
                        key, MessageRecord, subject="mailbox processed record"
                    )
                except CheckpointSchemaError:
                    continue
                if record is None or record.updated_at >= processed_cutoff:
                    continue
                await self._store.delete(key)
                pruned += 1
            elif "/corrupt/" in key and self._corrupt_is_stale(
                await self._store.load(key), corrupt_cutoff
            ):
                await self._store.delete(key)
                pruned += 1
        return pruned

    @staticmethod
    def _corrupt_is_stale(data: bytes | None, cutoff: datetime) -> bool:
        """
        Whether a ``corrupt/`` dead-letter envelope is old enough to reap. A
        record that vanished mid-sweep is kept; one that is not our timestamped
        envelope (opaque / pre-existing) is reaped (nothing datable to preserve).
        """
        if data is None:
            return False
        try:
            dead_at = datetime.fromisoformat(json.loads(data)["dead_lettered_at"])
        except (ValueError, KeyError, TypeError):
            return True
        return dead_at < cutoff

    async def shutdown(self) -> None:
        self._closed.set()
