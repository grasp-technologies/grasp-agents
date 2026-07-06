"""
The per-agent message inbox a resident agent loop consumes between turns.

An :class:`AgentInbox` is one agent's **view over a shared
:class:`~grasp_agents.runtime.Transport`** — it binds the transport to this agent's
recipient name and exposes the consume interface a resident loop needs (``post`` /
``poll`` / ``has_pending`` + the parked-idle signal). It is the *same* transport the
team's :class:`~grasp_agents.runtime.ActorDriver` drives triggered members over (and
the same one a networked backend would swap in) — so durability and cross-process
delivery are the transport's concern, not the inbox's, and there is no second
delivery abstraction.

The inbox is the sibling of the agent's background-task manager — a separate per-
agent delivery substrate, never merged with it (a resident loop's idle-wait wakes on
*either*). It lives on the agent's
:class:`~grasp_agents.agent.agent_context.AgentContext` beside ``bg_tasks``; a host
attaches one per member. Lone non-resident agents never attach one. A resident loop
runs until its task is cancelled from outside, not until the inbox is closed.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING

from grasp_agents.mailbox import InMemoryMailboxTransport
from grasp_agents.runtime import Closed

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from grasp_agents.runtime import Transport
    from grasp_agents.types.message import TeamMessage


class AgentInbox:
    def __init__(
        self,
        *,
        transport: Transport[TeamMessage] | None = None,
        recipient: str = "",
        on_take: Callable[[TeamMessage], None] | None = None,
    ) -> None:
        # No transport → an in-memory one (single process). A durable / networked
        # team passes a shared transport (the same instance every member views).
        self._transport: Transport[TeamMessage] = (
            transport if transport is not None else InMemoryMailboxTransport()
        )
        self._recipient = recipient
        # Runs once a message is chosen for delivery, *before* its consumption
        # seq is minted — the seam a resident agent uses to archive a rollback
        # boundary whose high-waters exclude the message being taken.
        self._on_take = on_take
        self._waiting = False
        # Messages taken but not yet acked — released on the next checkpoint
        # (:meth:`flush_acks`) once the turn that consumed them is durable, or
        # dropped un-acked when the turn leaves the transcript (:meth:`rollback`).
        # In-memory and transient: a crash loses the leases, and the un-acked
        # messages are simply re-delivered. A lease must live exactly as long as
        # the live transcript that absorbed its message — which is why an agent
        # keeps ONE inbox across attach/detach cycles (see
        # :meth:`LLMAgent.attach_inbox`) instead of building one per run.
        self._leased: dict[str, TeamMessage] = {}

    @property
    def transport(self) -> Transport[TeamMessage]:
        return self._transport

    async def post(self, message: TeamMessage) -> None:
        """Deliver ``message`` through the transport (routed by its recipients)."""
        await self._transport.post(message)

    async def take(self) -> TeamMessage | None:
        """
        Lease this agent's oldest **unprocessed** message *without* acking, or
        ``None`` if nothing new is takeable.

        The message stays in the mailbox (leased) until :meth:`flush_acks`
        releases it on the next checkpoint — so a crash before then re-delivers it
        rather than dropping it (at-least-once). While a leased message is still
        the frontmost in the mailbox, ``take`` returns ``None`` (``consume`` is
        non-removing, so it would otherwise keep returning the same one) — but a
        *higher-priority* arrival is taken past an existing lease: that is how
        control-plane mail (human steering) enters mid-task, so more than one
        message can be leased at once. A message already marked processed — a
        redelivery whose ack only partly landed before a crash — is acked and
        skipped, since its effects are already durable.

        A taken message is stamped with its consumption ``seq`` (this inbox is
        the recipient's sole consumer, so take order is absorption order); the
        ack persists it, and a step rollback moves messages above a boundary's
        high-water back to pending (:meth:`unprocess_after`). ``on_take`` fires
        once a message is chosen for delivery, *before* its seq is minted.
        """
        while await self._transport.has_pending(self._recipient):
            # ``has_pending`` is true and this agent is the sole consumer of its
            # own mailbox, so ``consume`` returns at once rather than blocking.
            message = await self._transport.consume(self._recipient)
            if isinstance(message, Closed):
                return None
            if message.message_id in self._leased:
                return None
            if await self._transport.was_processed(self._recipient, message.message_id):
                await self._transport.ack(self._recipient, message)
                continue
            if self._on_take is not None:
                self._on_take(message)
            message.seq = self._transport.mint_consumption_seq(self._recipient)
            self._leased[message.message_id] = message
            return message
        return None

    async def ack(self, message: TeamMessage) -> None:
        """Remove a leased message from the inbox, once its turn is durable."""
        self._leased.pop(message.message_id, None)
        await self._transport.ack(self._recipient, message)

    async def flush_acks(self) -> None:
        """
        Release every leased message — the inbox counterpart to
        :meth:`BackgroundTaskManager.flush_delivered`. Called right after the
        checkpoint that persisted the turn which consumed them: the messages are
        now durably absorbed into the log, so removing them from the mailbox can no
        longer strand them (resume re-derives the owed response from the log). A
        crash before this re-delivers them: one whose ack partly landed is
        dedupe-skipped by :meth:`take`, but one never acked genuinely re-runs
        its (already persisted) turn — the at-least-once duplicate this design
        accepts over a lost message.
        """
        for message in list(self._leased.values()):
            await self._transport.ack(self._recipient, message)
            self._leased.pop(message.message_id, None)

    def rollback(self) -> None:
        """
        Drop all leases without acking, so a transcript rewind (step rollback,
        cold reload) re-takes the still-unacked messages from the mailbox
        rather than wedging on them. In-memory only — they were never removed.
        NOT for a failed-run settle: settling keeps the absorbed message's
        turn, and its lease is what prevents a duplicate re-take.
        """
        self._leased.clear()

    @property
    def last_consumption_seq(self) -> int:
        """High-water consumption seq: every message taken so far has ``seq <=``."""
        return self._transport.last_consumption_seq(self._recipient)

    def seed_consumption_seq(self, seq: int) -> None:
        """
        Seed the consumption counter from a restored watermark (never lowers it
        — the inbox sibling of :meth:`BackgroundTaskManager.seed_launch_seq`).
        The counter lives on the shared transport, so it survives the per-run
        inbox; this seeding covers a fresh process, where the restored
        ``AgentContextState`` is the only surviving copy.
        """
        self._transport.seed_consumption_seq(self._recipient, seq)

    async def poll(self) -> TeamMessage | None:
        """Consume + ack this agent's oldest message in one step, or ``None``."""
        message = await self.take()
        if message is not None:
            await self.ack(message)
        return message

    async def has_pending(self) -> bool:
        return await self._transport.has_pending(self._recipient)

    @property
    def is_waiting(self) -> bool:
        """
        Whether a resident consumer is currently parked on this inbox with nothing
        to do. With :meth:`has_pending` false, this is the per-actor "idle" signal a
        supervisor reads to detect quiescence (no progress is possible until
        something is posted).
        """
        return self._waiting

    @contextmanager
    def waiting(self) -> Generator[None]:
        """Mark a consumer parked on this inbox for the duration of the block."""
        self._waiting = True
        try:
            yield
        finally:
            self._waiting = False

    async def wait(self, timeout: float) -> None:  # noqa: ASYNC109
        """
        Sleep up to ``timeout`` between mailbox polls (a plain mailbox has no
        arrival signal). The resident loop re-checks :meth:`has_pending` and its
        own background-task completions after each wait; it is ended from outside
        by cancelling the run's task.
        """
        await asyncio.sleep(timeout)
