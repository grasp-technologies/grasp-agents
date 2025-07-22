import asyncio
import logging
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Generic, Literal, Protocol, TypeVar

from .packet import Packet
from .run_context import CtxT, RunContext
from .typing.events import Event
from .typing.io import ProcName

logger = logging.getLogger(__name__)


END_PROC_NAME: Literal["*END*"] = "*END*"


_PayloadT_contra = TypeVar("_PayloadT_contra", contravariant=True)


class PacketHandler(Protocol[_PayloadT_contra, CtxT]):
    async def __call__(
        self,
        packet: Packet[_PayloadT_contra],
        ctx: RunContext[CtxT],
        **kwargs: Any,
    ) -> None: ...


class PacketPool(Generic[CtxT]):
    def __init__(self) -> None:
        self._packet_queues: dict[ProcName, asyncio.Queue[Packet[Any] | None]] = {}
        self._packet_handlers: dict[ProcName, PacketHandler[Any, CtxT]] = {}
        self._task_group: asyncio.TaskGroup | None = None

        self._event_queue: asyncio.Queue[Event[Any] | None] = asyncio.Queue()

        self._final_result_fut: asyncio.Future[Packet[Any]] | None = None

        self._stopping = False
        self._stopped_evt = asyncio.Event()

        self._errors: list[Exception] = []

    async def post(self, packet: Packet[Any]) -> None:
        if packet.recipients == [END_PROC_NAME]:
            fut = self._ensure_final_future()
            if not fut.done():
                fut.set_result(packet)
            await self.shutdown()
            return

        for recipient_id in packet.recipients or []:
            queue = self._packet_queues.setdefault(recipient_id, asyncio.Queue())
            await queue.put(packet)

    def _ensure_final_future(self) -> asyncio.Future[Packet[Any]]:
        fut = self._final_result_fut
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            self._final_result_fut = fut
        return fut

    async def final_result(self) -> Packet[Any]:
        fut = self._ensure_final_future()
        try:
            return await fut
        finally:
            await self.shutdown()

    def register_packet_handler(
        self,
        proc_name: ProcName,
        handler: PacketHandler[Any, CtxT],
        ctx: RunContext[CtxT],
        **run_kwargs: Any,
    ) -> None:
        if self._stopping:
            raise RuntimeError("PacketPool is stopping/stopped")

        self._packet_handlers[proc_name] = handler
        self._packet_queues.setdefault(proc_name, asyncio.Queue())

        if self._task_group is not None:
            self._task_group.create_task(
                self._handle_packets(proc_name, ctx=ctx, **run_kwargs),
                name=f"packet-handler:{proc_name}",
            )

    async def push_event(self, event: Event[Any]) -> None:
        await self._event_queue.put(event)

    async def __aenter__(self) -> "PacketPool[CtxT]":
        self._task_group = asyncio.TaskGroup()
        await self._task_group.__aenter__()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        await self.shutdown()

        if self._task_group is not None:
            try:
                return await self._task_group.__aexit__(exc_type, exc, tb)
            finally:
                self._task_group = None

        if self._errors:
            raise ExceptionGroup("PacketPool worker errors", self._errors)

        return False

    async def _handle_packets(
        self, proc_name: ProcName, ctx: RunContext[CtxT], **run_kwargs: Any
    ) -> None:
        queue = self._packet_queues[proc_name]
        handler = self._packet_handlers[proc_name]

        while True:
            packet = await queue.get()
            if packet is None:
                break
            try:
                await handler(packet, ctx=ctx, **run_kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as err:
                logger.exception("Error handling packet for %s", proc_name)
                self._errors.append(err)
                fut = self._final_result_fut
                if fut and not fut.done():
                    fut.set_exception(err)
                await self.shutdown()
                raise

    async def stream_events(self) -> AsyncIterator[Event[Any]]:
        while True:
            event = await self._event_queue.get()
            if event is None:
                break
            yield event

    async def shutdown(self) -> None:
        if self._stopping:
            await self._stopped_evt.wait()
            return
        self._stopping = True
        try:
            await self._event_queue.put(None)
            for queue in self._packet_queues.values():
                await queue.put(None)

        finally:
            self._stopped_evt.set()
