from collections.abc import AsyncIterator, Sequence
from itertools import pairwise
from logging import getLogger
from typing import Any, cast, final

from ..packet import Packet
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..types.errors import WorkflowConstructionError
from ..types.events import Event, ProcPacketOutEvent, ProcPayloadOutEvent
from ..types.hooks import WorkflowLoopTerminator
from ..types.io import InT, OutT, ProcName
from ..utils.callbacks import is_method_overridden
from .workflow_processor import WorkflowProcessor

logger = getLogger(__name__)


class LoopedWorkflow(WorkflowProcessor[InT, OutT, CtxT]):
    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, CtxT]],
        exit_proc: Processor[Any, OutT, CtxT],
        recipients: list[ProcName] | None = None,
        max_iterations: int = 10,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
        session_id: str | None = None,
        session_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            subprocs=subprocs,
            name=name,
            start_proc=subprocs[0],
            end_proc=exit_proc,
            recipients=recipients,
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
            session_id=session_id,
            session_metadata=session_metadata,
        )

        for prev_proc, proc in pairwise(subprocs):
            if prev_proc.out_type != proc.in_type:
                raise WorkflowConstructionError(
                    f"Output type {prev_proc.out_type} of subprocessor "
                    f"{prev_proc.name} does not match input type {proc.in_type} of "
                    f"subprocessor {proc.name}"
                )
        if subprocs[-1].out_type != subprocs[0].in_type:
            raise WorkflowConstructionError(
                "Looped workflow's last subprocessor output type "
                f"{subprocs[-1].out_type} does not match first subprocessor input "
                f"type {subprocs[0].in_type}"
            )
        if exit_proc is not subprocs[-1]:
            raise WorkflowConstructionError(
                "exit_proc must be the last subprocessor in a LoopedWorkflow"
            )

        self._max_iterations = max_iterations

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def terminate_workflow_loop_impl(
        self, out_packet: Packet[OutT], *, ctx: RunContext[CtxT], **kwargs: Any
    ) -> bool:
        raise NotImplementedError

    def add_workflow_loop_terminator(
        self, func: WorkflowLoopTerminator[OutT, CtxT]
    ) -> WorkflowLoopTerminator[OutT, CtxT]:
        self.terminate_workflow_loop_impl = func

        return func

    @final
    def terminate_workflow_loop(
        self, out_packet: Packet[OutT], *, ctx: RunContext[CtxT], **kwargs: Any
    ) -> bool:
        base_cls = LoopedWorkflow[Any, Any, Any]
        if is_method_overridden("terminate_workflow_loop_impl", self, base_cls):
            return self.terminate_workflow_loop_impl(out_packet, ctx=ctx, **kwargs)

        return False

    @final
    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        exec_id: str,
        ctx: RunContext[CtxT],
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        packet = Packet(sender=self.name, payloads=in_args) if in_args else None
        start_iteration = 1
        start_step = 0

        checkpoint = await self.load_checkpoint(ctx)
        if checkpoint is not None:
            packet = checkpoint.packet
            start_iteration = checkpoint.iteration
            start_step = checkpoint.completed_step + 1

            # All steps completed — re-evaluate termination.
            if start_step >= len(self.subprocs):
                exit_packet = cast("Packet[OutT]", packet)
                if (
                    self.terminate_workflow_loop(exit_packet, ctx=ctx)
                    or start_iteration >= self._max_iterations
                ):
                    for p in exit_packet.payloads:
                        yield ProcPayloadOutEvent(
                            data=p, source=self.name, exec_id=exec_id
                        )
                    return
                # Loop continues — advance to next iteration
                start_iteration += 1
                start_step = 0

        n_subprocs = len(self.subprocs)
        for iteration_num in range(start_iteration, self._max_iterations + 1):
            for idx, subproc in enumerate(self.subprocs):
                if iteration_num == start_iteration and idx < start_step:
                    continue

                logger.info(f"\n[Running subprocessor {subproc.name}]\n")

                child_step = (iteration_num - 1) * n_subprocs + idx
                async for event in subproc.run_stream(
                    chat_inputs=chat_inputs,
                    in_packet=packet,
                    exec_id=f"{exec_id}/{subproc.name}/iter_{iteration_num}",
                    ctx=ctx,
                    step=child_step,
                ):
                    yield event
                    if (
                        isinstance(event, ProcPacketOutEvent)
                        and event.source == subproc.name
                    ):
                        packet = event.data

                await self.save_checkpoint(
                    ctx,
                    completed_step=idx,
                    packet=cast("Packet[Any]", packet),
                    iteration=iteration_num,
                )

                logger.info(f"\n[Finished running subprocessor {subproc.name}]\n")

                if subproc is self.end_proc:
                    exit_packet = cast("Packet[OutT]", packet)

                    if self.terminate_workflow_loop(exit_packet, ctx=ctx):
                        for p in exit_packet.payloads:
                            yield ProcPayloadOutEvent(
                                data=p, source=self.name, exec_id=exec_id
                            )
                        return

                    if iteration_num == self._max_iterations:
                        logger.info(
                            f"Max iterations reached ({self._max_iterations}). "
                            "Exiting loop."
                        )
                        for p in exit_packet.payloads:
                            yield ProcPayloadOutEvent(
                                data=p, source=self.name, exec_id=exec_id
                            )
                        return

                chat_inputs = None
