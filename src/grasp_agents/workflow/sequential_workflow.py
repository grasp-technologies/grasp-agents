import logging
from collections.abc import AsyncIterator, Sequence
from itertools import pairwise
from typing import Any, cast

from ..errors import WorkflowConstructionError
from ..packet import Packet
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..typing.events import Event, ProcPacketOutEvent, ProcPayloadOutEvent
from ..typing.io import InT, OutT, ProcName
from .workflow_processor import WorkflowProcessor

logger = logging.getLogger(__name__)


class SequentialWorkflow(WorkflowProcessor[InT, OutT, CtxT]):
    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, CtxT]],
        recipients: list[ProcName] | None = None,
    ) -> None:
        super().__init__(
            subprocs=subprocs,
            start_proc=subprocs[0],
            end_proc=subprocs[-1],
            name=name,
            recipients=recipients,
        )

        for prev_proc, proc in pairwise(subprocs):
            if prev_proc.out_type != proc.in_type:
                raise WorkflowConstructionError(
                    f"Output type {prev_proc.out_type} of subprocessor {prev_proc.name}"
                    f" does not match input type {proc.in_type} of subprocessor"
                    f" {proc.name}"
                )

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> list[OutT]:
        packet = Packet(sender=self.name, payloads=in_args) if in_args else None

        for subproc in self.subprocs:
            logger.info(f"\n[Running subprocessor {subproc.name}]\n")

            packet = await subproc.run(
                chat_inputs=chat_inputs,
                in_packet=packet,
                call_id=f"{call_id}/{subproc.name}",
                ctx=ctx,
            )
            chat_inputs = None

            logger.info(f"\n[Finished running subprocessor {subproc.name}]\n")

        return list(cast("Packet[OutT]", packet).payloads)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        packet = Packet(sender=self.name, payloads=in_args) if in_args else None

        for subproc in self.subprocs:
            logger.info(f"\n[Running subprocessor {subproc.name}]\n")

            async for event in subproc.run_stream(
                chat_inputs=chat_inputs,
                in_packet=packet,
                call_id=f"{call_id}/{subproc.name}",
                ctx=ctx,
            ):
                yield event
                if (
                    isinstance(event, ProcPacketOutEvent)
                    and event.src_name == subproc.name
                ):
                    packet = event.data

            if subproc is self.end_proc:
                out_packet = cast("Packet[OutT]", packet)
                for p in out_packet.payloads:
                    yield ProcPayloadOutEvent(
                        data=p, src_name=self.name, call_id=call_id
                    )

            chat_inputs = None

            logger.info(f"\n[Finished running subprocessor {subproc.name}]\n")
