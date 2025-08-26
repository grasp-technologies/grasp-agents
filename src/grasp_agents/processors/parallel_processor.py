import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, cast

from ..errors import PacketRoutingError
from ..memory import MemT
from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..typing.events import Event, ProcPacketOutputEvent, ProcPayloadOutputEvent
from ..typing.io import InT, OutT
from ..utils import stream_concurrent
from .base_processor import BaseProcessor, with_retry, with_retry_stream

logger = logging.getLogger(__name__)


class ParallelProcessor(
    BaseProcessor[InT, OutT, MemT, CtxT], Generic[InT, OutT, MemT, CtxT]
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        memory: MemT,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> OutT:
        return cast("OutT", in_args)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        memory: MemT,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        output = cast("OutT", in_args)
        yield ProcPayloadOutputEvent(data=output, proc_name=self.name, call_id=call_id)

    def _validate_parallel_recipients(
        self, out_packets: Sequence[Packet[OutT]], call_id: str
    ) -> None:
        recipient_sets = [set(p.recipients or []) for p in out_packets]
        same_recipients = all(rs == recipient_sets[0] for rs in recipient_sets)
        if not same_recipients:
            raise PacketRoutingError(
                proc_name=self.name,
                call_id=call_id,
                message="Parallel runs must return the same recipients "
                f"[proc_name={self.name}; call_id={call_id}]",
            )

    @with_retry
    async def _run_single(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        forgetful: bool = False,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> Packet[OutT]:
        memory = self.memory.model_copy(deep=True) if forgetful else self.memory

        output = await self._process(
            chat_inputs=chat_inputs,
            in_args=in_args,
            memory=memory,
            call_id=call_id,
            ctx=ctx,
        )
        val_output = self._validate_output(output, call_id=call_id)

        recipients = self._select_recipients(output=val_output, ctx=ctx)
        self._validate_recipients(recipients, call_id=call_id)

        return Packet(payloads=[val_output], sender=self.name, recipients=recipients)

    async def _run_parallel(
        self, in_args: list[InT], call_id: str, ctx: RunContext[CtxT]
    ) -> Packet[OutT]:
        tasks = [
            self._run_single(
                in_args=inp, forgetful=True, call_id=f"{call_id}/{idx}", ctx=ctx
            )
            for idx, inp in enumerate(in_args)
        ]
        out_packets = await asyncio.gather(*tasks)
        self._validate_parallel_recipients(out_packets, call_id=call_id)

        return Packet(
            payloads=[out_packet.payloads[0] for out_packet in out_packets],
            sender=self.name,
            recipients=out_packets[0].recipients,
        )

    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        forgetful: bool = False,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT]:
        call_id = self._generate_call_id(call_id)
        ctx = ctx or RunContext[CtxT](state=None)  # type: ignore

        val_in_args = self._validate_inputs(
            call_id=call_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

        if val_in_args and len(val_in_args) > 1:
            return await self._run_parallel(
                in_args=val_in_args, call_id=call_id, ctx=ctx
            )

        return await self._run_single(
            chat_inputs=chat_inputs,
            in_args=val_in_args[0] if val_in_args else None,
            forgetful=forgetful,
            call_id=call_id,
            ctx=ctx,
        )

    @with_retry_stream
    async def _run_single_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        forgetful: bool = False,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        memory = self.memory.model_copy(deep=True) if forgetful else self.memory

        output: OutT | None = None
        async for event in self._process_stream(
            chat_inputs=chat_inputs,
            in_args=in_args,
            memory=memory,
            call_id=call_id,
            ctx=ctx,
        ):
            if isinstance(event, ProcPayloadOutputEvent):
                output = event.data
            yield event

        assert output is not None

        val_output = self._validate_output(output, call_id=call_id)

        recipients = self._select_recipients(output=val_output, ctx=ctx)
        self._validate_recipients(recipients, call_id=call_id)

        out_packet = Packet[OutT](
            payloads=[val_output], sender=self.name, recipients=recipients
        )

        yield ProcPacketOutputEvent(
            data=out_packet, proc_name=self.name, call_id=call_id
        )

    async def _run_parallel_stream(
        self,
        in_args: list[InT],
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        streams = [
            self._run_single_stream(
                in_args=inp, forgetful=True, call_id=f"{call_id}/{idx}", ctx=ctx
            )
            for idx, inp in enumerate(in_args)
        ]

        out_packets_map: dict[int, Packet[OutT]] = {}
        async for idx, event in stream_concurrent(streams):
            if isinstance(event, ProcPacketOutputEvent):
                out_packets_map[idx] = event.data
            else:
                yield event

        self._validate_parallel_recipients(
            out_packets=list(out_packets_map.values()), call_id=call_id
        )

        out_packet = Packet(
            payloads=[
                out_packet.payloads[0]
                for _, out_packet in sorted(out_packets_map.items())
            ],
            sender=self.name,
            recipients=out_packets_map[0].recipients,
        )

        yield ProcPacketOutputEvent(
            data=out_packet, proc_name=self.name, call_id=call_id
        )

    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        forgetful: bool = False,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        call_id = self._generate_call_id(call_id)
        ctx = ctx or RunContext[CtxT](state=None)  # type: ignore

        val_in_args = self._validate_inputs(
            call_id=call_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

        if val_in_args and len(val_in_args) > 1:
            stream = self._run_parallel_stream(
                in_args=val_in_args, call_id=call_id, ctx=ctx
            )
        else:
            stream = self._run_single_stream(
                chat_inputs=chat_inputs,
                in_args=val_in_args[0] if val_in_args else None,
                forgetful=forgetful,
                call_id=call_id,
                ctx=ctx,
            )
        async for event in stream:
            yield event
