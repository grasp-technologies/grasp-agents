import logging
from collections.abc import AsyncIterator, Sequence
from itertools import pairwise
from typing import Any, Generic, cast, final

from ..errors import WorkflowConstructionError
from ..packet_pool import Packet
from ..processors.base_processor import BaseProcessor
from ..run_context import CtxT, RunContext
from ..typing.events import Event, ProcPacketOutputEvent, WorkflowResultEvent
from ..typing.io import InT, OutT, ProcName
from .workflow_processor import WorkflowProcessor

logger = logging.getLogger(__name__)


class SequentialWorkflow(WorkflowProcessor[InT, OutT, CtxT], Generic[InT, OutT, CtxT]):
    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[BaseProcessor[Any, Any, Any, CtxT]],
        recipients: list[ProcName] | None = None,
        max_retries: int = 0,
    ) -> None:
        super().__init__(
            subprocs=subprocs,
            start_proc=subprocs[0],
            end_proc=subprocs[-1],
            name=name,
            recipients=recipients,
            max_retries=max_retries,
        )

        for prev_proc, proc in pairwise(subprocs):
            if prev_proc.out_type != proc.in_type:
                raise WorkflowConstructionError(
                    f"Output type {prev_proc.out_type} of subprocessor {prev_proc.name}"
                    f" does not match input type {proc.in_type} of subprocessor"
                    f" {proc.name}"
                )

    @final
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | Sequence[InT] | None = None,
        forgetful: bool = False,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT]:
        call_id = self._generate_call_id(call_id)

        packet = in_packet
        for subproc in self.subprocs:
            logger.info(f"\n[Running subprocessor {subproc.name}]\n")

            packet = await subproc.run(
                chat_inputs=chat_inputs,
                in_packet=packet,
                in_args=in_args,
                forgetful=forgetful,
                call_id=f"{call_id}/{subproc.name}",
                ctx=ctx,
            )
            chat_inputs = None
            in_args = None

            logger.info(f"\n[Finished running subprocessor {subproc.name}]\n")

        return cast("Packet[OutT]", packet)

    @final
    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | Sequence[InT] | None = None,
        forgetful: bool = False,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        call_id = self._generate_call_id(call_id)

        packet = in_packet
        for subproc in self.subprocs:
            logger.info(f"\n[Running subprocessor {subproc.name}]\n")

            async for event in subproc.run_stream(
                chat_inputs=chat_inputs,
                in_packet=packet,
                in_args=in_args,
                forgetful=forgetful,
                call_id=f"{call_id}/{subproc.name}",
                ctx=ctx,
            ):
                if isinstance(event, ProcPacketOutputEvent):
                    packet = event.data
                yield event

            chat_inputs = None
            in_args = None

            logger.info(f"\n[Finished running subprocessor {subproc.name}]\n")

        yield WorkflowResultEvent(
            data=cast("Packet[OutT]", packet), proc_name=self.name, call_id=call_id
        )
