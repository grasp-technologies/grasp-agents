"""External triggers — non-actor producers that send messages into a team."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .message import TeamMessage
    from .transport import MessageTransport


async def run_interval_source(
    transport: MessageTransport,
    make_message: Callable[[], TeamMessage],
    *,
    interval: float,
    count: int | None = None,
) -> None:
    """
    Feed a (daemon) team from outside on a timer.

    Every ``interval`` seconds, build a message with ``make_message()`` and deposit
    it in the team's mailbox — an external trigger, the way a file-watcher or webhook
    would wake an actor. Run it as a concurrent task alongside
    ``team.run_stream(daemon=True)``; it stops after ``count`` sends (``None`` runs
    until cancelled). It is a plain producer — it knows only the transport, not the
    team — so any code can wake a team the same way by calling ``transport.send``.
    """
    sent = 0
    while count is None or sent < count:
        await asyncio.sleep(interval)
        await transport.send(make_message())
        sent += 1
