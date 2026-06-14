import asyncio
import logging
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Protocol

from ..types.events import Event, RoutedEvent

logger = logging.getLogger(__name__)

# Queue bound: a publisher awaiting ``put`` on a full queue blocks until the
# consumer catches up (backpressure) instead of growing memory without limit.
MAX_QUEUE_SIZE = 1024


class EventHandler[D](Protocol):
    async def __call__(self, event: Event[D], **kwargs: Any) -> None: ...


def _put_sentinel(queue: asyncio.Queue[Event[Any] | None]) -> None:
    """
    Enqueue the shutdown sentinel without blocking.

    A full queue whose consumer is gone (crashed handler) must not park
    shutdown forever — drop queued events to make room; the bus is
    stopping anyway.
    """
    while True:
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                continue
        else:
            return


class EventBus:
    def __init__(self) -> None:
        self._task_group: asyncio.TaskGroup | None = None

        self._routed_event_queues: dict[str, asyncio.Queue[Event[Any] | None]] = {}
        self._streamed_event_queue: asyncio.Queue[Event[Any] | None] = asyncio.Queue(
            maxsize=MAX_QUEUE_SIZE
        )

        self._event_handlers: dict[str, EventHandler[Any]] = {}
        self._handler_tasks: dict[str, asyncio.Task[None]] = {}

        # Created lazily (inside a running loop) so the result can be set or
        # awaited even when the bus is never entered — e.g. a resume of an
        # already-completed run returns its persisted result without a run.
        self._final_result_fut: asyncio.Future[Any] | None = None

        self._stopping = False
        self._stopped_evt = asyncio.Event()

    def register_event_handler(self, dst_name: str, handler: EventHandler[Any]) -> None:
        if self._stopping:
            return

        # Prevent multiple concurrent handler tasks for the same destination
        if dst_name in self._handler_tasks and not self._handler_tasks[dst_name].done():
            raise RuntimeError(f"Handler already registered for {dst_name}")

        self._routed_event_queues.setdefault(
            dst_name, asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        )

        self._event_handlers[dst_name] = handler
        if self._task_group is not None:
            self._handler_tasks[dst_name] = self._task_group.create_task(
                self._handle_events(dst_name), name=f"event-handler:{dst_name}"
            )

    async def post(self, event: RoutedEvent[Any]) -> None:
        if self._stopping:
            return

        if event.destination is not None:
            queue = self._routed_event_queues[event.destination]
            await queue.put(event)

    async def push_to_stream(self, event: Event[Any]) -> None:
        if self._stopping:
            return

        await self._streamed_event_queue.put(event)

    async def stream_events(self) -> AsyncIterator[Event[Any]]:
        while True:
            event = await self._streamed_event_queue.get()
            if event is None:
                break
            yield event

    def _fut(self) -> asyncio.Future[Any]:
        if self._final_result_fut is None:
            self._final_result_fut = asyncio.get_running_loop().create_future()
        return self._final_result_fut

    async def final_result(self) -> Any:
        return await self._fut()

    async def __aenter__(self) -> "EventBus":
        self._task_group = asyncio.TaskGroup()
        await self._task_group.__aenter__()

        self._fut()

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

            # Fallback only: if finalize() already set the result/exception,
            # don't override it.
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
            # latter (run_stream raises before run() awaits the future), leaving
            # the future's exception unretrieved. Retrieve it here so a GC'd
            # future can't trip asyncio's "Future exception was never retrieved"
            # warning in an unrelated later task. Non-destructive: the result /
            # exception stays available to a caller that does await it.
            done_fut = self._final_result_fut
            if done_fut is not None and done_fut.done() and not done_fut.cancelled():
                done_fut.exception()

    async def _handle_events(self, dst_name: str) -> None:
        handler = self._event_handlers[dst_name]

        while True:
            queue = self._routed_event_queues[dst_name]
            event = await queue.get()

            if event is None:
                break

            if self._final_result_fut is not None and self._final_result_fut.done():
                break

            try:
                await handler(event)

            except asyncio.CancelledError as err:
                # Cooperative cancellation: the whole TaskGroup is being cancelled)
                logger.info("Event handler cancelled for %s", dst_name)
                self.set_result(None, err=err)
                raise

            except Exception as err:
                # Unexpected error: only this handler is affected
                logger.exception("Error handling event for %s", dst_name)
                self.set_result(None, err)
                await self.shutdown()
                raise  # let TaskGroup propagate

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
        if self._stopping:
            await self._stopped_evt.wait()
            return
        self._stopping = True
        try:
            for queue in self._routed_event_queues.values():
                _put_sentinel(queue)
            _put_sentinel(self._streamed_event_queue)
        finally:
            self._stopped_evt.set()
