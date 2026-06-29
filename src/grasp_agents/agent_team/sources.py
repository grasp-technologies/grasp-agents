"""External triggers — non-actor producers that send messages into a team."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from .message import TeamMessage

logger = logging.getLogger(__name__)


@runtime_checkable
class MessageSink(Protocol):
    """
    Anything a source can deposit a message into — a mailbox
    :class:`~grasp_agents.runtime.Transport`, or a team that routes + counts. The
    parameter name matches :meth:`Transport.post` so a transport satisfies this.
    """

    async def post(self, envelope: TeamMessage) -> None: ...


async def run_interval_source(
    sink: MessageSink,
    make_message: Callable[[], TeamMessage],
    *,
    interval: float,
    count: int | None = None,
) -> None:
    """
    Feed a (daemon) team from outside on a timer.

    Every ``interval`` seconds, build a message with ``make_message()`` and deposit
    it into ``sink`` — an external trigger, the way a file-watcher or webhook would
    wake an actor. Run it as a concurrent task alongside
    ``team.run_stream(daemon=True)``; it stops after ``count`` sends (``None`` runs
    until cancelled). It is a plain producer — any code can wake a team the same way
    by calling ``sink.post`` (the team routes the message to its recipient).
    """
    sent = 0
    while count is None or sent < count:
        await asyncio.sleep(interval)
        await sink.post(make_message())
        sent += 1


class WakeupScheduler:
    """
    Deliver future-dated messages into a team's mailbox — the timer source that
    gives a daemon team its own initiative.

    An actor schedules a *self-addressed* wakeup ("revisit this in 10 min") via the
    ``ScheduleWakeup`` tool; after the delay the message is deposited in the actor's
    own mailbox, which (in a daemon team) reactivates it even with no peer traffic.
    This is the lever a triggered actor pulls to act unprompted without holding a
    live loop open between turns — the message, not a resident coroutine, carries the
    initiative across the idle gap.

    A thin producer over a :class:`MessageSink` (a transport, or the team itself),
    like :func:`run_interval_source`, but one-shot and self-scheduled. Timers are
    owned here and cancelled by :meth:`aclose`, so a closed team leaves none firing
    into a torn-down sink. In-process only: a pending wakeup does not survive a
    restart (a durable schedule is future work).
    """

    def __init__(self, sink: MessageSink) -> None:
        self._sink = sink
        self._timers: set[asyncio.Task[None]] = set()

    def schedule(self, message: TeamMessage, *, delay: float) -> None:
        """Deposit ``message`` in its recipient's mailbox ``delay`` seconds from now."""
        timer = asyncio.create_task(self._fire(message, max(delay, 0.0)))
        self._timers.add(timer)
        timer.add_done_callback(self._timers.discard)

    async def _fire(self, message: TeamMessage, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            await self._sink.post(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Best-effort, like any source: a failed delivery must not surface as an
            # unretrieved task exception or take the team down.
            logger.warning("Scheduled wakeup failed to deliver", exc_info=True)

    @property
    def pending(self) -> int:
        """How many scheduled wakeups have not yet fired."""
        return sum(1 for t in self._timers if not t.done())

    async def aclose(self) -> None:
        """Cancel every pending wakeup (session teardown)."""
        timers = list(self._timers)
        for timer in timers:
            timer.cancel()
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
        self._timers.clear()
