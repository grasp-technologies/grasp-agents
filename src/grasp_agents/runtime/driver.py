"""
The activation engine of the actor runtime.

:class:`ActorDriver` runs a fixed set of named **actors** over a pluggable
:class:`~.transport.Transport`. Each actor gets one serial consumer task in a
single :class:`asyncio.TaskGroup`: pull the next inbound envelope → run the actor's
handler (which streams the actor's events and routes its output back through the
transport) → ack. One activation per actor at a time (the single-drainer
invariant); actors run concurrently with one another.

The engine is transport- and frontend-agnostic. A frontend supplies, per actor, a
**handler** that knows how to invoke that actor and route its output; the driver
owns the hard parts — the task group, backpressure, the shared event stream, the
terminal-result future, error propagation, and shutdown.

How a run ends is a **termination policy**:

- ``terminal`` — never stops on its own; a handler ends the run by calling
  :meth:`finalize` (a designated result). Consumers then unblock and the task group
  drains. This is ``Runner``'s call-and-return shape.
- ``quiescence`` — stops once no actor is running and no mailbox has pending work
  (optionally bounded by ``max_activations``). Bounded peer collaboration.
- ``daemon`` — never stops on its own; runs until the stream is cancelled. A
  long-running reactive system.

In ``terminal`` mode the quiescence/activation-count machinery is inert, so the
engine reduces exactly to the in-process event bus the orchestration layer runs on.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from .transport import MAX_QUEUE_SIZE, Closed, Transport, put_sentinel

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

type Handler[E] = Callable[[E], Awaitable[None]]
type Termination = Literal["terminal", "quiescence", "daemon"]


class ActorDriver[E]:
    def __init__(
        self,
        transport: Transport[E],
        *,
        termination: Termination = "terminal",
        max_activations: int | None = None,
    ) -> None:
        self._transport = transport
        self._termination: Termination = termination
        self._max_activations = max_activations

        self._task_group: asyncio.TaskGroup | None = None

        self._streamed_event_queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=MAX_QUEUE_SIZE
        )

        self._handlers: dict[str, Handler[E]] = {}
        self._handler_tasks: dict[str, asyncio.Task[None]] = {}

        # Created lazily (inside a running loop) so the result can be set or awaited
        # even when the driver is never entered — e.g. a resume of an already
        # completed run returns its persisted result without a run.
        self._final_result_fut: asyncio.Future[Any] | None = None

        self._stopping = False
        self._stopped_evt = asyncio.Event()

        # Activations in flight + total so far — read only by quiescence detection.
        self._active = 0
        self._activation_count = 0

    @property
    def activation_count(self) -> int:
        return self._activation_count

    async def is_quiescent(self) -> bool:
        """
        Whether no actor is currently running and no mailbox has pending work.

        A read-only check a frontend can fold into a larger quiescence decision (a
        team that also supervises resident actors outside this driver); the driver's
        own ``quiescence`` termination uses the same condition internally.
        """
        return self._active == 0 and not await self._any_pending()

    def register_handler(self, name: str, handler: Handler[E]) -> None:
        if self._stopping:
            return

        # Prevent two concurrent consumer tasks for the same actor.
        if name in self._handler_tasks and not self._handler_tasks[name].done():
            raise RuntimeError(f"Handler already registered for {name}")

        self._transport.register(name)
        self._handlers[name] = handler
        if self._task_group is not None:
            self._handler_tasks[name] = self._task_group.create_task(
                self._consume_loop(name), name=f"actor:{name}"
            )

    async def post(self, envelope: E) -> None:
        if self._stopping:
            return
        await self._transport.post(envelope)

    async def push_to_stream(self, event: Any) -> None:
        if self._stopping:
            return
        queue = self._streamed_event_queue
        if not queue.full():
            # Fast path (and the only path until the consumer falls behind): enqueue
            # without parking. No await between the check and the put, so the slot
            # can't be taken from under us.
            queue.put_nowait(event)
            return
        # The queue is full — a slow/stopped stream consumer. Park on the put for
        # backpressure, but also wake on shutdown: otherwise a consumer that stops
        # draining without closing the stream would leave this producer parked
        # forever, and a clean teardown (which waits for it) would hang.
        put = asyncio.ensure_future(queue.put(event))
        stopped = asyncio.ensure_future(self._stopped_evt.wait())
        try:
            await asyncio.wait({put, stopped}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            put.cancel()
            stopped.cancel()

    async def stream_events(self) -> AsyncIterator[Any]:
        while True:
            event = await self._streamed_event_queue.get()
            if isinstance(event, Closed):
                break
            yield event

    async def settle(self) -> None:
        """
        Run the termination check once, after the frontend has seeded the run —
        so an empty ``quiescence`` run ends immediately instead of hanging.
        """
        await self._maybe_terminate()

    def _fut(self) -> asyncio.Future[Any]:
        if self._final_result_fut is None:
            self._final_result_fut = asyncio.get_running_loop().create_future()
        return self._final_result_fut

    async def final_result(self) -> Any:
        return await self._fut()

    async def __aenter__(self) -> ActorDriver[E]:
        self._task_group = asyncio.TaskGroup()
        await self._task_group.__aenter__()

        self._fut()

        # Spawn consumers for any handlers registered before entering.
        for name in self._handlers:
            if name not in self._handler_tasks:
                self._handler_tasks[name] = self._task_group.create_task(
                    self._consume_loop(name), name=f"actor:{name}"
                )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        await self.shutdown()

        ret: bool | None = False
        try:
            if self._task_group is not None:
                try:
                    ret = await self._task_group.__aexit__(exc_type, exc, tb)
                finally:
                    self._task_group = None

            # Fallback only: if finalize() already set the result/exception, don't
            # override it.
            fut = self._fut()
            if not fut.done():
                if exc is not None:
                    if isinstance(exc, asyncio.CancelledError):
                        fut.cancel()
                    else:
                        fut.set_exception(exc)
                else:
                    fut.cancel()

            return ret
        finally:
            # A crashing handler sets the error on the final-result future AND
            # re-raises so the TaskGroup propagates it; the caller consumes the
            # latter (run_stream raises before run() awaits the future), leaving the
            # future's exception unretrieved. Retrieve it here so a GC'd future can't
            # trip asyncio's "Future exception was never retrieved" warning in an
            # unrelated later task. Non-destructive: the result / exception stays
            # available to a caller that does await it.
            done_fut = self._final_result_fut
            if done_fut is not None and done_fut.done() and not done_fut.cancelled():
                done_fut.exception()

    async def _consume_loop(self, name: str) -> None:
        handler = self._handlers[name]

        while not self._stopping:
            # Wait for the next envelope, but also wake on this driver's own
            # shutdown: the transport is shared session infrastructure (other
            # drivers / resident inboxes keep using it after this run), so the
            # driver must never close it — it only stops its own waiting
            # consumers. In-flight handlers below still finish their activation.
            consume = asyncio.ensure_future(self._transport.consume(name))
            stopped = asyncio.ensure_future(self._stopped_evt.wait())
            try:
                await asyncio.wait(
                    {consume, stopped}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                stopped.cancel()
            if not consume.done():
                consume.cancel()
                break
            envelope = consume.result()

            if isinstance(envelope, Closed):
                break

            if self._final_result_fut is not None and self._final_result_fut.done():
                break

            if self._over_budget():
                # The activation budget is spent: do not start another. The check
                # and the count bump below have no ``await`` between them, so two
                # consumers cannot both slip past a budget of one. A mailbox
                # transport (the only kind used with a budget) leaves the fetched
                # envelope unacked, so it stays pending — surfacing as "stopped with
                # mail still waiting". (Inert in ``terminal`` mode.)
                break

            self._active += 1
            self._activation_count += 1
            try:
                await handler(envelope)
                await self._transport.ack(name, envelope)

            except asyncio.CancelledError as err:
                # Cooperative cancellation: the whole TaskGroup is being cancelled.
                logger.info("Actor consumer cancelled for %s", name)
                self.set_result(None, err=err)
                raise

            except Exception as err:
                # Unexpected error: only this actor's handler is affected.
                logger.exception("Error in actor handler for %s", name)
                self.set_result(None, err)
                await self.shutdown()
                raise  # let TaskGroup propagate

            finally:
                self._active -= 1

            await self._maybe_terminate()

    def _over_budget(self) -> bool:
        return (
            self._termination == "quiescence"
            and self._max_activations is not None
            and self._activation_count >= self._max_activations
        )

    async def _maybe_terminate(self) -> None:
        if self._termination != "quiescence":
            return
        if self._over_budget():
            await self.shutdown()
            return
        if self._active == 0 and not await self._any_pending():
            await self.shutdown()

    async def _any_pending(self) -> bool:
        for name in self._handlers:
            if await self._transport.has_pending(name):
                return True
        return False

    def set_result(
        self, result: Any, err: Exception | asyncio.CancelledError | None = None
    ) -> None:
        fut = self._fut()
        if not fut.done():
            if err and isinstance(err, asyncio.CancelledError):
                fut.cancel()
            elif err:
                fut.set_exception(err)
            else:
                fut.set_result(result)

    async def finalize(self, result: Any, err: Exception | None = None) -> None:
        self.set_result(result, err)
        await self.shutdown()

    async def shutdown(self) -> None:
        """
        Stop this driver's consumers and close its event stream. Deliberately
        does NOT shut the transport down: the transport outlives the run (it is
        the session's mailbox, shared with resident inboxes and later runs);
        parked consumers wake on ``_stopped_evt`` instead.
        """
        if self._stopping:
            await self._stopped_evt.wait()
            return
        self._stopping = True
        try:
            put_sentinel(self._streamed_event_queue)
        finally:
            self._stopped_evt.set()
