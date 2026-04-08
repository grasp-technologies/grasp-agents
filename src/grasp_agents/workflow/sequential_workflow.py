import logging
from collections.abc import AsyncIterator, Sequence
from itertools import pairwise
from typing import Any, cast

from ..packet import Packet
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..types.errors import WorkflowConstructionError
from ..types.events import Event, ProcPacketOutEvent, ProcPayloadOutEvent
from ..types.io import InT, OutT, ProcName
from .workflow_processor import WorkflowProcessor

logger = logging.getLogger(__name__)


class SequentialWorkflow(WorkflowProcessor[InT, OutT, CtxT]):
    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, CtxT]],
        recipients: list[ProcName] | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        super().__init__(
            subprocs=subprocs,
            start_proc=subprocs[0],
            end_proc=subprocs[-1],
            name=name,
            recipients=recipients,
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
        )

        for prev_proc, proc in pairwise(subprocs):
            if prev_proc.out_type != proc.in_type:
                raise WorkflowConstructionError(
                    f"Output type {prev_proc.out_type} of subprocessor {prev_proc.name}"
                    f" does not match input type {proc.in_type} of subprocessor"
                    f" {proc.name}"
                )

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        exec_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        packet = Packet(sender=self.name, payloads=in_args) if in_args else None
        start_step = 0

        checkpoint = await self._load_checkpoint(ctx)
        if checkpoint is not None:
            packet = Packet[Any].model_validate(checkpoint.packet)
            start_step = checkpoint.completed_step + 1

        for idx, subproc in enumerate(self.subprocs):
            if idx < start_step:
                continue

            logger.info(f"\n[Running subprocessor {subproc.name}]\n")

            async for event in subproc.run_stream(
                chat_inputs=chat_inputs,
                in_packet=packet,
                exec_id=f"{exec_id}/{subproc.name}",
                ctx=ctx,
            ):
                yield event
                if (
                    isinstance(event, ProcPacketOutEvent)
                    and event.source == subproc.name
                ):
                    packet = event.data

            await self._save_checkpoint(
                ctx, completed_step=idx, packet=cast("Packet[Any]", packet)
            )

            if subproc is self.end_proc:
                out_packet = cast("Packet[OutT]", packet)
                for p in out_packet.payloads:
                    yield ProcPayloadOutEvent(data=p, source=self.name, exec_id=exec_id)

            chat_inputs = None

            logger.info(f"\n[Finished running subprocessor {subproc.name}]\n")
