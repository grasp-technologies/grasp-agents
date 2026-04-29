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
        session_path: list[str] | None = None,
        session_metadata: dict[str, Any] | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        super().__init__(
            subprocs=subprocs,
            name=name,
            start_proc=subprocs[0],
            end_proc=exit_proc,
            recipients=recipients,
            session_path=session_path,
            session_metadata=session_metadata,
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
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
        step: int | None = None,  # noqa: ARG002
    ) -> AsyncIterator[Event[Any]]:
        packet = Packet(sender=self.name, payloads=in_args) if in_args else None
        n = len(self.subprocs)
        exit_idx = self.subprocs.index(self.end_proc)
        max_global_steps = self._max_iterations * n
        start_global = 0

        checkpoint = await self.load_checkpoint(ctx)
        if checkpoint is not None:
            packet = checkpoint.packet
            start_global = checkpoint.completed_step + 1

            # Exit proc completed — re-evaluate termination.
            if checkpoint.completed_step % n == exit_idx:
                exit_packet = cast("Packet[OutT]", packet)
                if (
                    self.terminate_workflow_loop(exit_packet, ctx=ctx)
                    or start_global >= max_global_steps
                ):
                    for p in exit_packet.payloads:
                        yield ProcPayloadOutEvent(
                            data=p, source=self.name, exec_id=exec_id
                        )
                    return

        for global_step in range(start_global, max_global_steps):
            iteration = global_step // n
            idx = global_step % n
            subproc = self.subprocs[idx]

            logger.info(f"\n[Running subprocessor {subproc.name}]\n")

            async for event in subproc.run_stream(
                chat_inputs=chat_inputs,
                in_packet=packet,
                exec_id=f"{exec_id}/{subproc.name}/iter_{iteration}",
                ctx=ctx,
                step=iteration,
            ):
                yield event
                if (
                    isinstance(event, ProcPacketOutEvent)
                    and event.source == subproc.name
                ):
                    packet = event.data

            await self.save_checkpoint(
                ctx,
                completed_step=global_step,
                packet=cast("Packet[Any]", packet),
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

                if iteration == self._max_iterations - 1:
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
