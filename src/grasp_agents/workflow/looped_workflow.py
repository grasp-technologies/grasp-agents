from collections.abc import Sequence
from itertools import pairwise
from logging import getLogger
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, final

from ..errors import WorkflowConstructionError
from ..packet_pool import Packet, PacketPool
from ..processor import Processor
from ..run_context import CtxT, RunContext
from ..typing.io import InT, OutT_co, ProcName
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


class LoopedWorkflow(
    WorkflowProcessor[InT, OutT_co, CtxT], Generic[InT, OutT_co, CtxT]
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, Any, CtxT]],
        exit_proc: Processor[Any, OutT_co, Any, CtxT],
        packet_pool: PacketPool[CtxT] | None = None,
        recipients: list[ProcName] | None = None,
        num_par_run_retries: int = 0,
        max_iterations: int = 10,
    ) -> None:
        super().__init__(
            subprocs=subprocs,
            name=name,
            start_proc=subprocs[0],
            end_proc=exit_proc,
            packet_pool=packet_pool,
            recipients=recipients,
            num_par_run_retries=num_par_run_retries,
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

        self._exit_workflow_loop_impl: ExitWorkflowLoopHandler[OutT_co, CtxT] | None = (
            None
        )

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def exit_workflow_loop(
        self, func: ExitWorkflowLoopHandler[OutT_co, CtxT]
    ) -> ExitWorkflowLoopHandler[OutT_co, CtxT]:
        self._exit_workflow_loop_impl = func

        return func

    def _exit_workflow_loop(
        self,
        out_packet: Packet[OutT_co],
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
        run_id: str | None = None,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co]:
        packet = in_packet
        num_iterations = 0
        exit_packet: Packet[OutT_co] | None = None

        while True:
            for subproc in self.subprocs:
                packet = await subproc.run(
                    chat_inputs=chat_inputs,
                    in_packet=packet,
                    in_args=in_args,
                    forgetful=forgetful,
                    run_id=self._generate_subproc_run_id(run_id, subproc=subproc),
                    ctx=ctx,
                )

                if subproc is self._end_proc:
                    num_iterations += 1
                    exit_packet = cast("Packet[OutT_co]", packet)
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
