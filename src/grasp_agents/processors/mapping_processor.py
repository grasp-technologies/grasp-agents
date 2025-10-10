import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, cast

from grasp_agents.tracing_decorators import agent

from ..memory import MemT
from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..typing.events import Event, ProcPacketOutputEvent, ProcPayloadOutputEvent
from ..typing.io import InT, OutT, ProcName
from .base_processor import BaseProcessor, with_retry, with_retry_stream

logger = logging.getLogger(__name__)


class MappingProcessor(
    BaseProcessor[InT, OutT, MemT, CtxT], Generic[InT, OutT, MemT, CtxT]
):
    """
    Processor that can have different numbers of inputs and outputs, allowing for an
    arbitrary mapping between them.
    """

    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        memory: MemT,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> list[OutT]:
        """
        Process a list of inputs and return a list of outputs. The length of the
        output list can be different from the input list.
        """
        return cast("list[OutT]", in_args)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        memory: MemT,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        outputs = await self._process(
            chat_inputs=chat_inputs,
            in_args=in_args,
            memory=memory,
            call_id=call_id,
            ctx=ctx,
        )
        for output in outputs:
            yield ProcPayloadOutputEvent(
                data=output, proc_name=self.name, call_id=call_id
            )

    def _preprocess(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        forgetful: bool = False,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> tuple[list[InT] | None, MemT]:
        val_in_args = self._validate_inputs(
            call_id=call_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )
        memory = self.memory.model_copy(deep=True) if forgetful else self.memory

        return val_in_args, memory

    def _join_routings(
        self, routings: list[Sequence[ProcName] | None]
    ) -> Sequence[Sequence[ProcName]] | None:
        if all(r is None for r in routings):
            joined_routing = None
        else:
            joined_routing = [r or [] for r in routings]
        return joined_routing

    def _postprocess(
        self, outputs: list[OutT], call_id: str, ctx: RunContext[CtxT]
    ) -> Packet[OutT]:
        payloads: list[OutT] = []
        routings: list[Sequence[ProcName] | None] = []
        for output in outputs:
            val_output = self._validate_output(output, call_id=call_id)
            payloads.append(val_output)

            selected_recipients = self.select_recipients(
                output=val_output, ctx=ctx, call_id=call_id
            )
            routings.append(selected_recipients)

        routing = self._join_routings(routings)

        return Packet(sender=self.name, payloads=payloads, routing=routing)

    @agent(name="processor")  # type: ignore
    @with_retry
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
        ctx = ctx or RunContext[CtxT](state=None)  # type: ignore
        call_id = self._generate_call_id(call_id)

        val_in_args, memory = self._preprocess(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            forgetful=forgetful,
            call_id=call_id,
            ctx=ctx,
        )
        outputs = await self._process(
            chat_inputs=chat_inputs,
            in_args=val_in_args,
            memory=memory,
            call_id=call_id,
            ctx=ctx,
        )

        return self._postprocess(outputs=outputs, call_id=call_id, ctx=ctx)

    @agent(name="processor")  # type: ignore
    @with_retry_stream
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
        ctx = ctx or RunContext[CtxT](state=None)  # type: ignore
        call_id = self._generate_call_id(call_id)

        val_in_args, memory = self._preprocess(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            forgetful=forgetful,
            call_id=call_id,
            ctx=ctx,
        )
        outputs: list[OutT] = []
        async for event in self._process_stream(
            chat_inputs=chat_inputs,
            in_args=val_in_args,
            memory=memory,
            call_id=call_id,
            ctx=ctx,
        ):
            if isinstance(event, ProcPayloadOutputEvent):
                outputs.append(event.data)
            yield event

        out_packet = self._postprocess(outputs=outputs, call_id=call_id, ctx=ctx)

        yield ProcPacketOutputEvent(
            data=out_packet, proc_name=self.name, call_id=call_id
        )
