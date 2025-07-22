from collections.abc import AsyncIterator, Sequence
from functools import partial
from typing import Any, Generic

from .errors import RunnerError
from .packet import Packet, StartPacket
from .packet_pool import END_PROC_NAME, PacketPool
from .processor import Processor
from .run_context import CtxT, RunContext
from .typing.events import Event, ProcPacketOutputEvent, RunResultEvent
from .typing.io import OutT


class Runner(Generic[OutT, CtxT]):
    def __init__(
        self,
        entry_proc: Processor[Any, Any, Any, CtxT],
        procs: Sequence[Processor[Any, Any, Any, CtxT]],
        ctx: RunContext[CtxT] | None = None,
    ) -> None:
        if entry_proc not in procs:
            raise RunnerError(
                f"Entry processor {entry_proc.name} must be in the list of processors: "
                f"{', '.join(proc.name for proc in procs)}"
            )
        if sum(1 for proc in procs if END_PROC_NAME in (proc.recipients or [])) != 1:
            raise RunnerError(
                "There must be exactly one processor with recipient 'END'."
            )

        self._entry_proc = entry_proc
        self._procs = procs
        self._ctx = ctx or RunContext[CtxT]()
        self._packet_pool: PacketPool[CtxT] = PacketPool()

    @property
    def ctx(self) -> RunContext[CtxT]:
        return self._ctx

    def _unpack_packet(
        self, packet: Packet[Any] | None
    ) -> tuple[Packet[Any] | None, Any | None]:
        if isinstance(packet, StartPacket):
            return None, packet.chat_inputs
        return packet, None

    async def _packet_handler(
        self,
        proc: Processor[Any, Any, Any, CtxT],
        pool: PacketPool[CtxT],
        packet: Packet[Any],
        ctx: RunContext[CtxT],
        **run_kwargs: Any,
    ) -> None:
        _in_packet, _chat_inputs = self._unpack_packet(packet)
        out_packet = await proc.run(
            chat_inputs=_chat_inputs, in_packet=_in_packet, ctx=ctx, **run_kwargs
        )
        await pool.post(out_packet)

    async def _packet_handler_stream(
        self,
        proc: Processor[Any, Any, Any, CtxT],
        pool: PacketPool[CtxT],
        packet: Packet[Any],
        ctx: RunContext[CtxT],
        **run_kwargs: Any,
    ) -> None:
        _in_packet, _chat_inputs = self._unpack_packet(packet)

        out_packet: Packet[Any] | None = None
        async for event in proc.run_stream(
            chat_inputs=_chat_inputs, in_packet=_in_packet, ctx=ctx, **run_kwargs
        ):
            if isinstance(event, ProcPacketOutputEvent):
                out_packet = event.data
            await pool.push_event(event)

        assert out_packet is not None

        await pool.post(out_packet)

    async def run(
        self,
        chat_input: Any = "start",
        **run_args: Any,
    ) -> Packet[OutT]:
        async with PacketPool[CtxT]() as pool:
            for proc in self._procs:
                pool.register_packet_handler(
                    proc_name=proc.name,
                    handler=partial(self._packet_handler, proc, pool),
                    ctx=self._ctx,
                    **run_args,
                )
            await pool.post(
                StartPacket[Any](
                    recipients=[self._entry_proc.name], chat_inputs=chat_input
                )
            )
            return await pool.final_result()

    async def run_stream(
        self,
        chat_input: Any = "start",
        **run_args: Any,
    ) -> AsyncIterator[Event[Any]]:
        async with PacketPool[CtxT]() as pool:
            for proc in self._procs:
                pool.register_packet_handler(
                    proc_name=proc.name,
                    handler=partial(self._packet_handler_stream, proc, pool),
                    ctx=self._ctx,
                    **run_args,
                )
            await pool.post(
                StartPacket[Any](
                    recipients=[self._entry_proc.name], chat_inputs=chat_input
                )
            )
            async for event in pool.stream_events():
                if isinstance(
                    event, ProcPacketOutputEvent
                ) and event.data.recipients == [END_PROC_NAME]:
                    yield RunResultEvent(
                        data=event.data,
                        proc_name=event.proc_name,
                        call_id=event.call_id,
                    )
                else:
                    yield event
