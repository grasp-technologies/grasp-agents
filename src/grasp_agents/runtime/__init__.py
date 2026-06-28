"""
The actor runtime — one activation engine behind the framework's multi-actor
frontends.

An :class:`ActorDriver` runs named actors over a pluggable
:class:`Transport`: pull an inbound envelope → run the actor → route its output →
ack, one activation per actor at a time, all actors concurrent in one task group.
The orchestration graph runs it over an :class:`InProcessTransport` (bounded
in-memory queues, call-and-return); a peer-messaging team runs it over a mailbox
transport (quiescence or daemon). Termination is a policy, not a fork in the engine.
"""

from __future__ import annotations

from .driver import ActorDriver, Handler, Termination
from .transport import (
    CLOSED,
    MAX_QUEUE_SIZE,
    Closed,
    HasDestination,
    InProcessTransport,
    Transport,
    put_sentinel,
)

__all__ = [
    "CLOSED",
    "MAX_QUEUE_SIZE",
    "ActorDriver",
    "Closed",
    "Handler",
    "HasDestination",
    "InProcessTransport",
    "Termination",
    "Transport",
    "put_sentinel",
]
