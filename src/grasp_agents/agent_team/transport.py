"""
Team-side delivery wiring.

The mailbox transports themselves are implementations of the actor runtime's
:class:`~grasp_agents.runtime.Transport` and live in :mod:`grasp_agents.mailbox`
(a top-level module, so a single agent's inbox and a multi-agent host share one
seam, with no adapter). This module re-exports them and adds the team-only
:func:`default_transport` helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from grasp_agents.mailbox import (
    CheckpointMailboxTransport,
    InMemoryMailboxTransport,
)

if TYPE_CHECKING:
    from grasp_agents.run_context import RunContext
    from grasp_agents.runtime import Transport
    from grasp_agents.types.message import TeamMessage

__all__ = [
    "CheckpointMailboxTransport",
    "InMemoryMailboxTransport",
    "default_transport",
]


def default_transport(ctx: RunContext[Any]) -> Transport[TeamMessage]:
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
