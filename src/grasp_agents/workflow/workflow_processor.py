from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Generic

from ..comm_processor import CommProcessor
from ..packet import Packet
from ..packet_pool import PacketPool
from ..processor import Processor
from ..run_context import CtxT, RunContext
from ..typing.io import InT_contra, OutT_co, ProcessorName


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
        name: ProcessorName,
        subprocessors: Sequence[Processor[Any, Any, Any, CtxT]],
        start_processor: Processor[InT_contra, Any, Any, CtxT],
        end_processor: Processor[Any, OutT_co, Any, CtxT],
        packet_pool: PacketPool[CtxT] | None = None,
        recipients: list[ProcessorName] | None = None,
    ) -> None:
        super().__init__(name=name, packet_pool=packet_pool, recipients=recipients)

        if len(subprocessors) < 2:
            raise ValueError("At least two subprocessors are required")
        if start_processor not in subprocessors:
            raise ValueError("Start subprocessor must be in the subprocessors list")
        if end_processor not in subprocessors:
            raise ValueError("End subprocessor must be in the subprocessors list")

        if start_processor.in_type != self.in_type:
            raise ValueError(
                f"Start subprocessor's input type {start_processor.in_type} does not "
                f"match workflow's input type {self._in_type}"
            )
        if end_processor.out_type != self.out_type:
            raise ValueError(
                f"End subprocessor's output type {end_processor.out_type} does not "
                f"match workflow's output type {self._out_type}"
            )

        self._subprocessors = subprocessors
        self._start_processor = start_processor
        self._end_processor = end_processor

    @property
    def subprocessors(self) -> Sequence[Processor[Any, Any, Any, CtxT]]:
        return self._subprocessors

    @property
    def start_processor(self) -> Processor[InT_contra, Any, Any, CtxT]:
        return self._start_processor

    @property
    def end_processor(self) -> Processor[Any, OutT_co, Any, CtxT]:
        return self._end_processor

    @abstractmethod
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT_contra] | None = None,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        ctx: RunContext[CtxT] | None = None,
        entry_point: bool = False,
        forgetful: bool = False,
    ) -> Packet[OutT_co]:
        pass
