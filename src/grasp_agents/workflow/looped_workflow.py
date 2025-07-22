from collections.abc import AsyncIterator, Sequence
from itertools import pairwise
from logging import getLogger
from typing import Any, Generic, Protocol, TypeVar, cast, final

from ..errors import WorkflowConstructionError
from ..packet_pool import Packet
from ..processor import Processor
from ..run_context import CtxT, RunContext
from ..typing.events import Event, ProcPacketOutputEvent, WorkflowResultEvent
from ..typing.io import InT, OutT, ProcName
from .workflow_processor import WorkflowProcessor

logger = getLogger(__name__)

_OutT_contra = TypeVar("_OutT_contra", contravariant=True)


class ExitWorkflowLoopHandler(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self,
        out_packet: Packet[_OutT_contra],
        ctx: RunContext[CtxT] | None,
        **kwargs: Any,
    ) -> bool: ...


class LoopedWorkflow(WorkflowProcessor[InT, OutT, CtxT], Generic[InT, OutT, CtxT]):
    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, Any, CtxT]],
        exit_proc: Processor[Any, OutT, Any, CtxT],
        recipients: list[ProcName] | None = None,
        max_retries: int = 0,
        max_iterations: int = 10,
    ) -> None:
        super().__init__(
            subprocs=subprocs,
            name=name,
            start_proc=subprocs[0],
            end_proc=exit_proc,
            recipients=recipients,
            max_retries=max_retries,
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

        self._exit_workflow_loop_impl: ExitWorkflowLoopHandler[OutT, CtxT] | None = None

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def exit_workflow_loop(
        self, func: ExitWorkflowLoopHandler[OutT, CtxT]
    ) -> ExitWorkflowLoopHandler[OutT, CtxT]:
        self._exit_workflow_loop_impl = func

        return func

    def _exit_workflow_loop(
        self,
        out_packet: Packet[OutT],
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> bool:
        if self._exit_workflow_loop_impl:
            return self._exit_workflow_loop_impl(out_packet, ctx=ctx, **kwargs)

        return False

    @final
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | Sequence[InT] | None = None,
        call_id: str | None = None,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT]:
        call_id = self._generate_call_id(call_id)

        packet = in_packet
        num_iterations = 0
        exit_packet: Packet[OutT] | None = None

        while True:
            for subproc in self.subprocs:
                packet = await subproc.run(
                    chat_inputs=chat_inputs,
                    in_packet=packet,
                    in_args=in_args,
                    forgetful=forgetful,
                    call_id=f"{call_id}/{subproc.name}",
                    ctx=ctx,
                )

                if subproc is self._end_proc:
                    num_iterations += 1
                    exit_packet = cast("Packet[OutT]", packet)
                    if self._exit_workflow_loop(exit_packet, ctx=ctx):
                        return exit_packet
                    if num_iterations >= self._max_iterations:
                        logger.info(
                            f"Max iterations reached ({self._max_iterations}). "
                            "Exiting loop."
                        )
                        return exit_packet

                chat_inputs = None
                in_args = None

    @final
    async def run_stream(  # type: ignore[override]
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | Sequence[InT] | None = None,
        call_id: str | None = None,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        call_id = self._generate_call_id(call_id)

        packet = in_packet
        num_iterations = 0
        exit_packet: Packet[OutT] | None = None

        while True:
            for subproc in self.subprocs:
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

                if subproc is self._end_proc:
                    num_iterations += 1
                    exit_packet = cast("Packet[OutT]", packet)
                    if self._exit_workflow_loop(exit_packet, ctx=ctx):
                        yield WorkflowResultEvent(
                            data=exit_packet, proc_name=self.name, call_id=call_id
                        )
                        return
                    if num_iterations >= self._max_iterations:
                        logger.info(
                            f"Max iterations reached ({self._max_iterations}). "
                            "Exiting loop."
                        )
                        yield WorkflowResultEvent(
                            data=exit_packet, proc_name=self.name, call_id=call_id
                        )
                        return

                chat_inputs = None
                in_args = None
