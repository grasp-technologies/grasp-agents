from collections.abc import AsyncIterator, Sequence
from itertools import pairwise
from logging import getLogger
from typing import Any, Protocol, TypeVar, cast, final

from ..errors import WorkflowConstructionError
from ..packet import Packet
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..typing.events import Event, ProcPacketOutEvent, ProcPayloadOutEvent
from ..typing.io import InT, OutT, ProcName
from ..utils.callbacks import is_method_overridden
from .workflow_processor import WorkflowProcessor

logger = getLogger(__name__)

_OutT_contra = TypeVar("_OutT_contra", contravariant=True)


class WorkflowLoopTerminator(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self,
        out_packet: Packet[_OutT_contra],
        *,
        ctx: RunContext[CtxT],
        **kwargs: Any,
    ) -> bool: ...


class LoopedWorkflow(WorkflowProcessor[InT, OutT, CtxT]):
    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, CtxT]],
        exit_proc: Processor[Any, OutT, CtxT],
        recipients: list[ProcName] | None = None,
        max_iterations: int = 10,
    ) -> None:
        super().__init__(
            subprocs=subprocs,
            name=name,
            start_proc=subprocs[0],
            end_proc=exit_proc,
            recipients=recipients,
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

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> list[OutT]:
        packet = Packet(sender=self.name, payloads=in_args) if in_args else None

        for iteration_num in range(1, self._max_iterations + 1):
            for subproc in self.subprocs:
                logger.info(f"\n[Running subprocessor {subproc.name}]\n")

                packet = await subproc.run(
                    chat_inputs=chat_inputs,
                    in_packet=packet,
                    call_id=f"{call_id}/{subproc.name}/iter_{iteration_num}",
                    ctx=ctx,
                )

                logger.info(f"\n[Finished running subprocessor {subproc.name}]\n")

                if subproc is self.end_proc:
                    exit_packet = cast("Packet[OutT]", packet)
                    if self.terminate_workflow_loop(exit_packet, ctx=ctx):
                        return list(exit_packet.payloads)
                    if iteration_num == self._max_iterations:
                        logger.info(
                            f"Max iterations reached ({self._max_iterations}). "
                            "Exiting loop."
                        )
                        return list(exit_packet.payloads)

                chat_inputs = None

        raise RuntimeError("Looped workflow did not exit after max iterations.")

    @final
    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        packet = Packet(sender=self.name, payloads=in_args) if in_args else None

        for iteration_num in range(1, self._max_iterations + 1):
            for subproc in self.subprocs:
                logger.info(f"\n[Running subprocessor {subproc.name}]\n")

                async for event in subproc.run_stream(
                    chat_inputs=chat_inputs,
                    in_packet=packet,
                    call_id=f"{call_id}/{subproc.name}/iter_{iteration_num}",
                    ctx=ctx,
                ):
                    yield event
                    if (
                        isinstance(event, ProcPacketOutEvent)
                        and event.src_name == subproc.name
                    ):
                        packet = event.data

                logger.info(f"\n[Finished running subprocessor {subproc.name}]\n")

                if subproc is self.end_proc:
                    exit_packet = cast("Packet[OutT]", packet)

                    if self.terminate_workflow_loop(exit_packet, ctx=ctx):
                        for p in exit_packet.payloads:
                            yield ProcPayloadOutEvent(
                                data=p, src_name=self.name, call_id=call_id
                            )
                        return

                    if iteration_num == self._max_iterations:
                        logger.info(
                            f"Max iterations reached ({self._max_iterations}). "
                            "Exiting loop."
                        )
                        for p in exit_packet.payloads:
                            yield ProcPayloadOutEvent(
                                data=p, src_name=self.name, call_id=call_id
                            )
                        return

                chat_inputs = None
