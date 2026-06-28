"""
The in-process event bus that ``Runner`` orchestrates on.

``EventBus`` is the actor runtime (:class:`~grasp_agents.runtime.ActorDriver`)
specialized to a single host: process-local bounded-queue routing
(:class:`~grasp_agents.runtime.InProcessTransport`), call-and-return termination
(a handler ends the run by calling :meth:`~grasp_agents.runtime.ActorDriver.finalize`
with the run's result), and routed :class:`~grasp_agents.types.events.Event`
envelopes. Registering a per-destination handler and posting routed events is the
same engine a peer-messaging team runs over a mailbox transport.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from grasp_agents.runtime.driver import ActorDriver
from grasp_agents.runtime.transport import MAX_QUEUE_SIZE, Closed, InProcessTransport
from grasp_agents.types.events import Event, RoutedEvent

if TYPE_CHECKING:
    import asyncio

logger = logging.getLogger(__name__)

__all__ = ["MAX_QUEUE_SIZE", "EventBus", "EventHandler"]


class EventHandler[D](Protocol):
    async def __call__(self, event: Event[D], **kwargs: Any) -> None: ...


class EventBus(ActorDriver[RoutedEvent[Any]]):
    def __init__(self) -> None:
        transport = InProcessTransport[RoutedEvent[Any]]()
        super().__init__(transport, termination="terminal")
        self._inproc = transport

    @property
    def _routed_event_queues(
        self,
    ) -> dict[str, asyncio.Queue[RoutedEvent[Any] | Closed]]:
        return self._inproc.queues

    def register_event_handler(
        self, dst_name: str, handler: EventHandler[Any]
    ) -> None:
        self.register_handler(dst_name, handler)
