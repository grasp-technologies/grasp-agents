"""
The routing plane of the actor runtime.

:class:`Transport` is the seam that decouples :class:`~.driver.ActorDriver` from
*where* envelopes live between a sender and a recipient. The driver shuttles
envelopes of an arbitrary type ``E`` through it — it never inspects them — so the
same activation engine runs over process-local queues, durable mailboxes, or (in
future) a network protocol.

Two delivery disciplines exist:

- **process-local queue** (:class:`InProcessTransport`) — a bounded
  :class:`asyncio.Queue` per recipient; ``consume`` blocks on ``get`` and a full
  queue applies backpressure to the sender. This is the discipline ``Runner`` runs
  on (the engine behind its event bus).
- **mailbox** — a per-recipient store a sender deposits into and a recipient drains
  one item at a time, acking after it has been consumed. Lives in the messaging
  frontend (a team's pluggable mailbox), bridged to this interface by an adapter.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Final, Protocol, runtime_checkable

# Queue bound: a publisher awaiting ``put`` on a full queue blocks until the
# consumer catches up (backpressure) instead of growing memory without limit.
MAX_QUEUE_SIZE = 1024


class Closed:
    """
    The sentinel a transport's :meth:`Transport.consume` returns once the transport
    is shut down — a single distinguished value (:data:`CLOSED`) that can never
    collide with a real envelope, so shutdown is unambiguous even for an envelope
    type that itself admits ``None``.
    """

    __slots__ = ()


CLOSED: Final[Closed] = Closed()


def put_sentinel(queue: asyncio.Queue[Any]) -> None:
    """
    Enqueue the :data:`CLOSED` sentinel without blocking.

    A full queue whose consumer is gone (a crashed handler) must not park shutdown
    forever — drop queued items to make room; the runtime is stopping anyway.
    """
    while True:
        try:
            queue.put_nowait(CLOSED)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                continue
        else:
            return


@runtime_checkable
class HasDestination(Protocol):
    """An envelope whose recipient is a single named destination."""

    @property
    def destination(self) -> str | None: ...


class Transport[E](ABC):
    """
    Per-recipient routing plane the driver delivers to and consumes from.

    Implementations decide where envelopes rest (in-memory queue, durable store,
    network) without changing the activation engine. ``E`` is the envelope type the
    frontend routes — an :class:`~grasp_agents.types.events.Event` for the in-process
    case, a mailbox message for a team.
    """

    def __init__(self) -> None:
        # Per-recipient consumption counters (see mint_consumption_seq). Held
        # here — the transport is the session-shared object that survives run
        # boundaries — while the per-run consumer view (an agent's inbox) mints
        # from and seeds them.
        self._consumption_seqs: dict[str, int] = {}

    def mint_consumption_seq(self, recipient: str) -> int:
        """
        Mint ``recipient``'s next consumption seq — called by its sole consumer
        when it absorbs an envelope, so seqs follow consumption order (priority
        mail drains out of arrival order, so arrival order would not do).
        Process-local; the durable copy is the consumer's persisted high-water,
        seeded back on resume via :meth:`seed_consumption_seq`.
        """
        seq = self._consumption_seqs.get(recipient, 0) + 1
        self._consumption_seqs[recipient] = seq
        return seq

    def seed_consumption_seq(self, recipient: str, seq: int) -> None:
        """
        Seed ``recipient``'s counter from a restored watermark. Never lowers
        it: after a rewind the moved-back envelopes' seqs stay burned, so
        re-absorptions mint fresh ones.
        """
        self._consumption_seqs[recipient] = max(
            self._consumption_seqs.get(recipient, 0), seq
        )

    def last_consumption_seq(self, recipient: str) -> int:
        """High-water: every envelope absorbed so far has ``seq <=`` this."""
        return self._consumption_seqs.get(recipient, 0)

    @abstractmethod
    def register(self, recipient: str) -> None:
        """
        Ensure a mailbox exists for ``recipient`` (so a ``post`` before its first
        ``consume`` is not lost).
        """

    @abstractmethod
    async def post(self, envelope: E) -> None:
        """Deliver ``envelope`` to its recipient mailbox(es)."""

    @abstractmethod
    async def consume(self, recipient: str) -> E | Closed:
        """
        Block until ``recipient`` has an envelope, then return it; return
        :data:`CLOSED` once the transport is shut down (the consumer's signal to
        stop). Never return :data:`CLOSED` to mean "nothing available right now" —
        ``consume`` blocks for that case.
        """

    @abstractmethod
    async def ack(self, recipient: str, envelope: E) -> None:
        """Mark ``envelope`` consumed. A no-op where ``consume`` already removed it."""

    @abstractmethod
    async def has_pending(self, recipient: str) -> bool:
        """
        Whether ``recipient`` has an envelope not yet consumed (or consumed but
        not yet acked) — the signal quiescence detection reads.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Unblock every parked :meth:`consume` (they return :data:`CLOSED`).
        A terminal teardown affordance: the runtime itself never calls it —
        the transport is session-scoped and consumers stop by cancellation.
        """

    async def was_processed(self, recipient: str, envelope_id: str) -> bool:
        """
        Whether ``envelope_id`` was already consumed **and** acked for
        ``recipient`` — the at-most-once guard a frontend checks before
        (re)running a redelivered envelope, so an at-least-once redelivery does
        not re-execute work whose effects are already durable.

        Defaults to ``False``: an ephemeral transport loses its state on a
        process crash, so it never redelivers and has nothing to dedupe. A
        durable transport overrides this (e.g. by probing its acked-message
        store). Best-effort — it closes the redelivery window after the ack's
        durable mark, not the window between a handler finishing and that mark.
        """
        del recipient, envelope_id
        return False

    async def unprocess_after(self, recipient: str, seq: int) -> int:
        """
        Move ``recipient``'s acked envelopes with consumption ``seq > seq`` back
        to pending, returning how many moved — the mailbox half of a step
        rollback: the turns that absorbed those envelopes left the transcript,
        so the recipient must receive them again.

        Only envelopes whose consumer stamped a seq are eligible (a triggered
        worker's driver acks without one; its redelivery is the orchestrator's
        job). Defaults to ``0``: a queue transport removes envelopes on consume
        and retains nothing to restore. Mailbox transports override this.
        """
        del recipient, seq
        return 0


class InProcessTransport[E: HasDestination](Transport[E]):
    """
    Process-local routing over one bounded :class:`asyncio.Queue` per recipient.

    Each envelope names a single ``destination``; ``post`` enqueues it there and a
    full queue applies backpressure. ``consume`` blocks on the queue and returns
    :data:`CLOSED` once a shutdown sentinel arrives. ``ack`` is a no-op — ``consume``
    already removed the envelope. This is the engine ``Runner`` runs on.
    """

    def __init__(self, max_queue_size: int = MAX_QUEUE_SIZE) -> None:
        super().__init__()
        self._queues: dict[str, asyncio.Queue[E | Closed]] = {}
        self._max_queue_size = max_queue_size

    @property
    def queues(self) -> dict[str, asyncio.Queue[E | Closed]]:
        """The per-recipient queues (read-only view — for depth introspection)."""
        return self._queues

    def register(self, recipient: str) -> None:
        self._queues.setdefault(
            recipient, asyncio.Queue(maxsize=self._max_queue_size)
        )

    async def post(self, envelope: E) -> None:
        destination = envelope.destination
        if destination is not None:
            await self._queues[destination].put(envelope)

    async def consume(self, recipient: str) -> E | Closed:
        return await self._queues[recipient].get()

    async def ack(self, recipient: str, envelope: E) -> None:
        # ``consume`` already removed the envelope from the queue — nothing to do.
        del recipient, envelope

    async def has_pending(self, recipient: str) -> bool:
        queue = self._queues.get(recipient)
        return queue is not None and not queue.empty()

    async def shutdown(self) -> None:
        for queue in self._queues.values():
            put_sentinel(queue)
