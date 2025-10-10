import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, cast

from grasp_agents.tracing_decorators import agent

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
    """Processor that runs multiple inputs in parallel, each producing one output."""

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
        """Process a single input and return a single output."""
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

        recipients = self.select_recipients(output=val_output, ctx=ctx, call_id=call_id)

        return Packet(
            payloads=[val_output],
            sender=self.name,
            routing=[recipients] if recipients is not None else None,
        )

    def _join_payloads(self, packets: Sequence[Packet[OutT]]) -> list[OutT]:
        return [p.payloads[0] for p in packets]

    def _join_routings(
        self, packets: Sequence[Packet[OutT]]
    ) -> Sequence[Sequence[str]] | None:
        if not packets:
            return None
        routings = [p.routing[0] if p.routing is not None else None for p in packets]
        if all(r is None for r in routings):
            joined_routing = None
        else:
            joined_routing = [r or [] for r in routings]

        return joined_routing

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

        return Packet(
            sender=self.name,
            payloads=self._join_payloads(out_packets),
            routing=self._join_routings(out_packets),
        )

    @agent(name="processor")  # type: ignore
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

        recipients = self.select_recipients(output=val_output, ctx=ctx, call_id=call_id)

        out_packet = Packet[OutT](
            payloads=[val_output],
            sender=self.name,
            routing=[recipients] if recipients is not None else None,
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

        out_packet = Packet(
            sender=self.name,
            payloads=self._join_payloads(list(out_packets_map.values())),
            routing=self._join_routings(list(out_packets_map.values())),
        )

        yield ProcPacketOutputEvent(
            data=out_packet, proc_name=self.name, call_id=call_id
        )

    @agent(name="processor")  # type: ignore
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
