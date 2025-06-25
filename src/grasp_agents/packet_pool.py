import asyncio
import logging
from typing import Any, Generic, Protocol, TypeVar

from .packet import Packet
from .run_context import CtxT, RunContext
from .typing.io import ProcName

logger = logging.getLogger(__name__)


_PayloadT_contra = TypeVar("_PayloadT_contra", contravariant=True)


class PacketHandler(Protocol[_PayloadT_contra, CtxT]):
    async def __call__(
        self,
        packet: Packet[_PayloadT_contra],
        ctx: RunContext[CtxT] | None,
        **kwargs: Any,
    ) -> None: ...


class PacketPool(Generic[CtxT]):
    def __init__(self) -> None:
        self._queues: dict[ProcName, asyncio.Queue[Packet[Any]]] = {}
        self._packet_handlers: dict[ProcName, PacketHandler[Any, CtxT]] = {}
        self._tasks: dict[ProcName, asyncio.Task[None]] = {}

    async def post(self, packet: Packet[Any]) -> None:
        for recipient_id in packet.recipients:
            queue = self._queues.setdefault(recipient_id, asyncio.Queue())
            await queue.put(packet)

    def register_packet_handler(
        self,
        processor_name: ProcName,
        handler: PacketHandler[Any, CtxT],
        ctx: RunContext[CtxT] | None = None,
        **run_kwargs: Any,
    ) -> None:
        self._packet_handlers[processor_name] = handler
        self._queues.setdefault(processor_name, asyncio.Queue())
        if processor_name not in self._tasks:
            self._tasks[processor_name] = asyncio.create_task(
                self._handle_packets(processor_name, ctx=ctx, **run_kwargs)
            )

    async def _handle_packets(
        self,
        processor_name: ProcName,
        ctx: RunContext[CtxT] | None = None,
        **run_kwargs: Any,
    ) -> None:
        queue = self._queues[processor_name]
        while True:
            try:
                packet = await queue.get()
                handler = self._packet_handlers.get(processor_name)
                if handler is None:
                    break

                try:
                    await self._packet_handlers[processor_name](
                        packet, ctx=ctx, **run_kwargs
                    )
                except Exception:
                    logger.exception(f"Error handling packet for {processor_name}")

                queue.task_done()

            except Exception:
                logger.exception(
                    f"Unexpected error in processing loop for {processor_name}"
                )

    async def unregister_packet_handler(self, processor_name: ProcName) -> None:
        if task := self._tasks.get(processor_name):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"{processor_name} exited")

        self._tasks.pop(processor_name, None)
        self._queues.pop(processor_name, None)
        self._packet_handlers.pop(processor_name, None)

    async def stop_all(self) -> None:
        for processor_name in list(self._tasks):
            await self.unregister_packet_handler(processor_name)
