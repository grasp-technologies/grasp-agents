import asyncio
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Generic, cast

from ..comm_processor import CommProcessor
from ..packet import Packet
from ..packet_pool import PacketPool
from ..processor import Processor
from ..run_context import CtxT
from ..typing.io import InT_contra, OutT_co, ProcName


class ParallelCommProcessor(
    CommProcessor[InT_contra, OutT_co, Any, CtxT],
    Generic[InT_contra, OutT_co, CtxT],
):
    def __init__(
        self,
        name: ProcName,
        processor_type: type[Processor[InT_contra, OutT_co, Any, CtxT]],
        packet_pool: PacketPool[CtxT] | None = None,
        recipients: Sequence[ProcName] | None = None,
        **subproc_init_kwargs: Any,
    ) -> None:
        self._processor_type = processor_type
        self._subproc_init_kwargs = deepcopy(subproc_init_kwargs)

        # NOTE: If the processor is an LLMAgent, the parallel subprocessors will share
        # the same LLM and tools instances. Make sure their state is managed correctly.

        super().__init__(name=name, packet_pool=packet_pool, recipients=recipients)

    @property
    def processor_type(self) -> type[Processor[InT_contra, OutT_co, Any, CtxT]]:
        return self._processor_type

    async def _run_subprocessor(
        self, in_args: InT_contra, name_suffix: str, **subproc_run_kwargs: Any
    ) -> Packet[OutT_co]:
        subproc_name = f"{self.name}_{name_suffix}"
        subproc = self._processor_type(name=subproc_name, **self._subproc_init_kwargs)

        return await subproc.run(in_args=in_args, **subproc_run_kwargs)

    def _validate_par_inputs(
        self,
        chat_inputs: Any | None,
        in_packet: Packet[InT_contra] | None,
        in_args: InT_contra | Sequence[InT_contra] | None,
    ) -> Sequence[InT_contra]:
        if chat_inputs is not None:
            raise ValueError(
                "chat_inputs are not supported in ParallelCommProcessor. "
                "Use in_packet or in_args."
            )
        if in_packet is not None:
            if not in_packet.payloads:
                raise ValueError(
                    "ParallelCommProcessor requires at least one input payload in "
                    "in_packet."
                )
            return in_packet.payloads
        if in_args is not None:
            if not isinstance(in_args, Sequence) or not in_args:
                raise ValueError("in_args must be a non-empty sequence of input data.")
            return cast("Sequence[InT_contra]", in_args)
        raise ValueError(
            "ParallelCommProcessor requires either in_packet or in_args to be provided."
        )

    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT_contra] | None = None,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        **subproc_run_kwargs: Any,
    ) -> Packet[OutT_co]:
        par_inputs = self._validate_par_inputs(
            chat_inputs=chat_inputs, in_packet=in_packet, in_args=in_args
        )
        tasks = [
            self._run_subprocessor(
                in_args=inp, name_suffix=str(n), **subproc_run_kwargs
            )
            for n, inp in enumerate(par_inputs)
        ]
        out_packets = await asyncio.gather(*tasks)

        return Packet(
            payloads=[out_packet.payloads[0] for out_packet in out_packets],
            sender=self.name,
            recipients=(self.recipients or []),
        )
