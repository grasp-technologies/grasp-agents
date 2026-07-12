import logging
from collections.abc import AsyncIterator, Sequence
from itertools import pairwise
from typing import Any, cast

from grasp_agents.processors.processor import Processor
from grasp_agents.session_context import SessionContext
from grasp_agents.types.errors import WorkflowConstructionError
from grasp_agents.types.events import Event, ProcPacketOutEvent, ProcPayloadOutEvent
from grasp_agents.types.io import ProcName
from grasp_agents.types.packet import Packet

from .workflow_processor import WorkflowProcessor

logger = logging.getLogger(__name__)


class SequentialWorkflow[InT, OutT, CtxT](WorkflowProcessor[InT, OutT, CtxT]):
    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, CtxT]],
        *,
        ctx: SessionContext[CtxT] | None = None,
        recipients: list[ProcName] | None = None,
        path: list[str] | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
        durability_enabled: bool = True,
    ) -> None:
        super().__init__(
            subprocs=subprocs,
            start_proc=subprocs[0],
            end_proc=subprocs[-1],
            name=name,
            ctx=ctx,
            recipients=recipients,
            path=path,
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
            durability_enabled=durability_enabled,
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
        step: int | None = None,  # noqa: ARG002
    ) -> AsyncIterator[Event[Any]]:
        packet = Packet(sender=self.name, payloads=in_args) if in_args else None
        start_step = 0

        checkpoint = await self.load_checkpoint()
        if checkpoint is not None:
            packet = checkpoint.packet
            start_step = checkpoint.completed_step + 1
            # The checkpointed packet supersedes re-delivered inputs — the
            # first resumed subprocessor must not receive both.
            chat_inputs = None

            # All steps completed in a prior run — emit cached final output
            if start_step >= len(self.subprocs):
                out_packet = cast("Packet[OutT]", packet)
                for p in out_packet.payloads:
                    yield ProcPayloadOutEvent(data=p, source=self.name, exec_id=exec_id)
                return

        for idx, subproc in enumerate(self.subprocs):
            if idx < start_step:
                continue

            logger.info(f"\n[Running subprocessor {subproc.name}]\n")

            self._hand_over_session_writer(subproc)

            async for event in subproc.run_stream(
                chat_inputs=chat_inputs,
                in_packet=packet,
                exec_id=f"{exec_id}/{subproc.name}",
                step=0,  # each subproc is called exactly once
            ):
                yield event
                if (
                    isinstance(event, ProcPacketOutEvent)
                    and event.source == subproc.name
                ):
                    packet = event.data

            await self.save_checkpoint(
                completed_step=idx, packet=cast("Packet[Any]", packet)
            )

            if subproc is self.end_proc:
                out_packet = cast("Packet[OutT]", packet)
                for p in out_packet.payloads:
                    yield ProcPayloadOutEvent(data=p, source=self.name, exec_id=exec_id)

            chat_inputs = None

            logger.info(f"\n[Finished running subprocessor {subproc.name}]\n")
