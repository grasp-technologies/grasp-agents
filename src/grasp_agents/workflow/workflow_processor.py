from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Generic

from ..comm_processor import CommProcessor
from ..packet import Packet
from ..packet_pool import PacketPool
from ..processor import Processor
from ..run_context import CtxT, RunContext
from ..typing.io import InT_contra, OutT_co, ProcName


class WorkflowProcessor(
    CommProcessor[InT_contra, OutT_co, Any, CtxT],
    ABC,
    Generic[InT_contra, OutT_co, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, Any, CtxT]],
        start_proc: Processor[InT_contra, Any, Any, CtxT],
        end_proc: Processor[Any, OutT_co, Any, CtxT],
        packet_pool: PacketPool[CtxT] | None = None,
        recipients: list[ProcName] | None = None,
    ) -> None:
        super().__init__(name=name, packet_pool=packet_pool, recipients=recipients)

        if len(subprocs) < 2:
            raise ValueError("At least two subprocessors are required")
        if start_proc not in subprocs:
            raise ValueError("Start subprocessor must be in the subprocessors list")
        if end_proc not in subprocs:
            raise ValueError("End subprocessor must be in the subprocessors list")

        if start_proc.in_type != self.in_type:
            raise ValueError(
                f"Start subprocessor's input type {start_proc.in_type} does not "
                f"match workflow's input type {self._in_type}"
            )
        if end_proc.out_type != self.out_type:
            raise ValueError(
                f"End subprocessor's output type {end_proc.out_type} does not "
                f"match workflow's output type {self._out_type}"
            )

        self._subprocs = subprocs
        self._start_proc = start_proc
        self._end_proc = end_proc

    @property
    def subprocs(self) -> Sequence[Processor[Any, Any, Any, CtxT]]:
        return self._subprocs

    @property
    def start_proc(self) -> Processor[InT_contra, Any, Any, CtxT]:
        return self._start_proc

    @property
    def end_proc(self) -> Processor[Any, OutT_co, Any, CtxT]:
        return self._end_proc

    @abstractmethod
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT_contra] | None = None,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        ctx: RunContext[CtxT] | None = None,
        forgetful: bool = False,
    ) -> Packet[OutT_co]:
        pass
