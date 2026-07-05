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
from grasp_agents.types.message import TeamMessage

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import timedelta
    from typing import Any

    from grasp_agents.durability.checkpoint_store import CheckpointStore
    from grasp_agents.session_context import SessionContext

logger = logging.getLogger(__name__)


def resolve_session_transport(ctx: SessionContext[Any]) -> Transport[TeamMessage]:
    """
    The session's one mailbox transport (``ctx.transport``), created and
    installed on first use: durable over ``ctx.checkpoint_store`` when the
    session has one, else in-memory (single process). Hosts, resident inboxes,
    and ``SendMessage`` all resolve through here — never a transport argument —
    so every participant shares the one instance (and its live consumption
    counters) across host rebuilds within the session.
    """
    if ctx.transport is None:
        store = ctx.checkpoint_store
        ctx.transport = (
            CheckpointMailboxTransport(store, session_key=ctx.session_key)
            if store is not None
            else InMemoryMailboxTransport()
        )
    return ctx.transport


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
        # so a step rollback can move them back to pending (they are the
        # in-memory analog of the durable ``processed/`` records). Untracked
        # acks (``seq == 0``) are dropped outright, as before.
        self._processed: dict[str, list[TeamMessage]] = {}
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
                # Highest priority first, then oldest (FIFO within a priority).
                # Non-removing; removed by ack after activation.
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

    async def unprocess_after(self, recipient: str, seq: int) -> int:
        processed = self._processed.get(recipient, [])
        moved = [m for m in processed if m.seq > seq]
        if not moved:
            return 0
        self._processed[recipient] = [m for m in processed if m.seq <= seq]
        box = self._boxes.setdefault(recipient, [])
        for message in moved:
            # Back to pending as if never consumed: the seq is re-minted when
            # the recipient absorbs it again.
            message.seq = 0
            box.append(message)
        return len(moved)

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
            # carries the consumption ``seq`` that :meth:`unprocess_after`
            # needs. A redelivery dedupe re-acks with an unstamped copy and
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

    async def unprocess_after(self, recipient: str, seq: int) -> int:
        """
        Move ``processed/`` records with consumption ``seq >`` the watermark back
        to ``inbox/`` (status PENDING, seq cleared for re-minting), returning the
        count moved. The mailbox half of a step rollback.

        Crash-safe the same way the rollback itself is: the pending copy is
        written before the processed copy is deleted, so a crash in between
        leaves both — the redelivery dedup (:meth:`was_processed`) suppresses
        the stray pending copy, matching the not-yet-rolled-back head, and
        retrying the rollback re-moves the record and heals.
        """
        moved = 0
        for key in await self._store.list_keys(self._processed_prefix(recipient)):
            record = await self._store.load_json(
                key, MessageRecord, subject="mailbox processed record"
            )
            if record is None or record.message.seq <= seq:
                continue
            message = record.message.model_copy(update={"seq": 0})
            pending = record.model_copy(
                update={
                    "status": TaskStatus.PENDING,
                    "message": message,
                    "updated_at": datetime.now(UTC),
                }
            )
            await self._store.save(
                self._inbox_key(recipient, message),
                pending.model_dump_json().encode(),
            )
            await self._store.delete(key)
            moved += 1
        return moved

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

        ``older_than`` also bounds the **rollback horizon**: a step rollback
        restores consumed messages from these records (:meth:`unprocess_after`),
        so pruning one silently shrinks how far back a resident's mailbox can
        be rewound. Keep the retention at least as long as the oldest rollback
        boundary you intend to honor, or pin records via ``keep``.
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
                if record is not None and record.updated_at < processed_cutoff:
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
