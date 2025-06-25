from collections.abc import Sequence
from itertools import pairwise
from typing import Any, ClassVar, Generic, cast, final

from ..errors import WorkflowConstructionError
from ..packet_pool import Packet, PacketPool
from ..processor import Processor
from ..run_context import CtxT, RunContext
from ..typing.io import InT, OutT_co, ProcName
from .workflow_processor import WorkflowProcessor


class SequentialWorkflow(
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
        packet_pool: PacketPool[CtxT] | None = None,
        recipients: list[ProcName] | None = None,
        num_par_run_retries: int = 0,
    ) -> None:
        super().__init__(
            subprocs=subprocs,
            start_proc=subprocs[0],
            end_proc=subprocs[-1],
            name=name,
            packet_pool=packet_pool,
            recipients=recipients,
            num_par_run_retries=num_par_run_retries,
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
        run_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co]:
        packet = in_packet
        for subproc in self.subprocs:
            packet = await subproc.run(
                chat_inputs=chat_inputs,
                in_packet=packet,
                in_args=in_args,
                forgetful=forgetful,
                run_id=self._generate_subproc_run_id(run_id, subproc),
                ctx=ctx,
            )
            chat_inputs = None
            in_args = None

        return cast("Packet[OutT_co]", packet)
