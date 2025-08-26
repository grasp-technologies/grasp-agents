import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, cast

from ..memory import MemT
from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..typing.events import Event, ProcPacketOutputEvent, ProcPayloadOutputEvent
from ..typing.io import InT, OutT, ProcName
from .base_processor import BaseProcessor, with_retry, with_retry_stream

logger = logging.getLogger(__name__)


class Processor(BaseProcessor[InT, OutT, MemT, CtxT], Generic[InT, OutT, MemT, CtxT]):
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
        call_id: str | None = None,
        ctx: RunContext[CtxT],
    ) -> tuple[list[InT] | None, MemT, str]:
        call_id = self._generate_call_id(call_id)

        val_in_args = self._validate_inputs(
            call_id=call_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

        memory = self.memory.model_copy(deep=True) if forgetful else self.memory

        return val_in_args, memory, call_id

    def _postprocess(
        self, outputs: list[OutT], call_id: str, ctx: RunContext[CtxT]
    ) -> Packet[OutT]:
        payloads: list[OutT] = []
        routing: dict[int, Sequence[ProcName] | None] = {}
        for idx, output in enumerate(outputs):
            val_output = self._validate_output(output, call_id=call_id)
            recipients = self._select_recipients(output=val_output, ctx=ctx)
            self._validate_recipients(recipients, call_id=call_id)

            payloads.append(val_output)
            routing[idx] = recipients

        recipient_sets = [set(r or []) for r in routing.values()]
        if all(r == recipient_sets[0] for r in recipient_sets):
            recipients = routing[0]
        else:
            recipients = routing

        return Packet(payloads=payloads, sender=self.name, recipients=recipients)  # type: ignore[return-value]

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

        val_in_args, memory, call_id = self._preprocess(
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

        val_in_args, memory, call_id = self._preprocess(
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
