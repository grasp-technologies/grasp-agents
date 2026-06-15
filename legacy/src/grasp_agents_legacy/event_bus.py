import asyncio
import logging
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Protocol, TypeVar

from .typing.events import Event

logger = logging.getLogger(__name__)


_D_contra = TypeVar("_D_contra", contravariant=True)


class EventHandler(Protocol[_D_contra]):
    async def __call__(self, event: Event[_D_contra], **kwargs: Any) -> None: ...


class EventBus:
    def __init__(self) -> None:
        self._task_group: asyncio.TaskGroup | None = None

        self._routed_event_queues: dict[str, asyncio.Queue[Event[Any] | None]] = {}
        self._streamed_event_queue: asyncio.Queue[Event[Any] | None] = asyncio.Queue()

        self._event_handlers: dict[str, EventHandler[Any]] = {}
        self._handler_tasks: dict[str, asyncio.Task[None]] = {}

        self._final_result_fut: asyncio.Future[Any]

        self._stopping = False
        self._stopped_evt = asyncio.Event()

    def register_event_handler(self, dst_name: str, handler: EventHandler[Any]) -> None:
        if self._stopping:
            return

        # Prevent multiple concurrent handler tasks for the same destination
        if dst_name in self._handler_tasks and not self._handler_tasks[dst_name].done():
            raise RuntimeError(f"Handler already registered for {dst_name}")

        self._routed_event_queues.setdefault(dst_name, asyncio.Queue())

        self._event_handlers[dst_name] = handler
        if self._task_group is not None:
            self._handler_tasks[dst_name] = self._task_group.create_task(
                self._handle_events(dst_name), name=f"event-handler:{dst_name}"
            )

    async def post(self, event: Event[Any]) -> None:
        if self._stopping:
            return

        if event.dst_name is not None:
            queue = self._routed_event_queues[event.dst_name]
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

    async def final_result(self) -> Any:
        return await self._final_result_fut

    async def __aenter__(self) -> "EventBus":
        self._task_group = asyncio.TaskGroup()
        await self._task_group.__aenter__()

        self._final_result_fut = asyncio.get_running_loop().create_future()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        await self.shutdown()

        ret: bool | None = False
        if self._task_group is not None:
            try:
                ret = await self._task_group.__aexit__(exc_type, exc, tb)
            finally:
                self._task_group = None

        # Fallback only: if finalize() already set the result/exception, don't override it.
        if not self._final_result_fut.done():
            if exc is not None:
                if isinstance(exc, asyncio.CancelledError):
                    self._final_result_fut.cancel()
                else:
                    self._final_result_fut.set_exception(exc)
            else:
                self._final_result_fut.cancel()

        return ret

    async def _handle_events(self, dst_name: str) -> None:
        handler = self._event_handlers[dst_name]

        while True:
            queue = self._routed_event_queues[dst_name]
            event = await queue.get()

            if event is None:
                break

            if self._final_result_fut.done():
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
        if not self._final_result_fut.done():
            if err and isinstance(err, asyncio.CancelledError):
                self._final_result_fut.cancel()
            elif err:
                self._final_result_fut.set_exception(err)
            else:
                self._final_result_fut.set_result(result)

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
                await queue.put(None)
            await self._streamed_event_queue.put(None)
        finally:
            self._stopped_evt.set()
