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
    from collections.abc import Generator

    from grasp_agents.runtime import Transport
    from grasp_agents.types.message import TeamMessage


class AgentInbox:
    def __init__(
        self,
        *,
        transport: Transport[TeamMessage] | None = None,
        recipient: str = "",
    ) -> None:
        # No transport → an in-memory one (single process). A durable / networked
        # team passes a shared transport (the same instance every member views).
        self._transport: Transport[TeamMessage] = (
            transport if transport is not None else InMemoryMailboxTransport()
        )
        self._recipient = recipient
        self._waiting = False

    @property
    def transport(self) -> Transport[TeamMessage]:
        return self._transport

    async def post(self, message: TeamMessage) -> None:
        """Deliver ``message`` through the transport (routed by its recipients)."""
        await self._transport.post(message)

    async def poll(self) -> TeamMessage | None:
        """Consume this agent's oldest message (fetch + ack), or ``None``."""
        if not await self._transport.has_pending(self._recipient):
            return None
        # ``has_pending`` is true and this agent is the sole consumer of its own
        # mailbox, so ``consume`` returns at once rather than blocking.
        message = await self._transport.consume(self._recipient)
        if isinstance(message, Closed):
            return None
        await self._transport.ack(self._recipient, message)
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
