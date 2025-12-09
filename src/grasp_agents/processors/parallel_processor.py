import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from itertools import chain
from typing import Any

from grasp_agents.utils.streaming import stream_concurrent

from ..errors import ProcInputValidationError
from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..typing.events import Event, ProcPacketOutEvent, ProcPayloadOutEvent
from ..typing.io import InT, OutT, ProcName
from .processor import Processor

logger = logging.getLogger(__name__)


class ParallelProcessor(Processor[InT, OutT, CtxT]):
    def __init__(
        self, subproc: Processor[InT, OutT, CtxT], drop_failed: bool = False
    ) -> None:
        super().__init__(
            name=subproc.name + "_par",
            recipients=subproc.recipients,
            max_retries=0,
            tracing_enabled=subproc.tracing_enabled,
            tracing_exclude_input_fields=subproc.tracing_exclude_input_fields,
        )

        self._in_type = subproc.in_type
        self._out_type = subproc.out_type
        self._subproc = subproc

        self._drop_failed = drop_failed

        # This disables recipient selection in the subprocessor,
        # but preserves subproc.select_recipients_impl
        subproc.recipients = None

    @property
    def drop_failed(self) -> bool:
        return self._drop_failed

    def select_recipients_impl(
        self, output: OutT, *, ctx: RunContext[CtxT], call_id: str
    ) -> Sequence[ProcName]:
        # Move recipient selection to the outer ParallelProcessor
        return self._subproc.select_recipients_impl(
            output=output, ctx=ctx, call_id=call_id
        )

    @property
    def subproc(self) -> Processor[InT, OutT, CtxT]:
        return self._subproc

    def _validate_in_args(
        self,
        chat_inputs: Any | None = None,
        in_args: list[InT] | None = None,
        *,
        call_id: str,
    ) -> list[InT]:
        err_kwargs = {"proc_name": self.name, "call_id": call_id}
        if chat_inputs is not None:
            raise ProcInputValidationError(
                message=f"ParallelProcessor {self.name} does not support chat_inputs",
                **err_kwargs,
            )
        if in_args is None:
            raise ProcInputValidationError(
                message=f"ParallelProcessor {self.name} requires in_args to be provided",
                **err_kwargs,
            )

        return in_args

    def _join_payloads_from_packets(
        self, packets: Sequence[Packet[OutT] | None]
    ) -> list[OutT | None]:
        return list(
            chain.from_iterable(
                p.payloads if p is not None else [None] for p in packets
            )
        )

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> list[OutT]:
        in_args = self._validate_in_args(
            chat_inputs=chat_inputs, in_args=in_args, call_id=call_id
        )
        subproc_replicas = [self._subproc.copy() for _ in in_args]
        corouts = [
            proc.run(in_args=inp, call_id=f"{call_id}/{idx}", ctx=ctx)
            for idx, (inp, proc) in enumerate(
                zip(in_args, subproc_replicas, strict=True)
            )
        ]
        out_packets = await asyncio.gather(*corouts)

        return self._join_payloads_from_packets(out_packets)  # type: ignore[return-value]

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        in_args = self._validate_in_args(
            chat_inputs=chat_inputs, in_args=in_args, call_id=call_id
        )
        subproc_replicas = [self._subproc.copy() for _ in in_args]

        streams = [
            proc.run_stream(in_args=inp, call_id=f"{call_id}/{idx}", ctx=ctx)
            for idx, (inp, proc) in enumerate(
                zip(in_args, subproc_replicas, strict=True)
            )
        ]
        out_packets_map: dict[int, Packet[OutT] | None] = dict.fromkeys(
            range(len(in_args)), None
        )
        async for idx, event in stream_concurrent(streams):
            if (
                isinstance(event, ProcPacketOutEvent)
                and event.src_name == self._subproc.name
            ):
                out_packets_map[idx] = event.data
            else:
                yield event

        out_packets = [out_packets_map[idx] for idx in sorted(out_packets_map)]
        if self.drop_failed:
            out_packets = [p for p in out_packets if p is not None]

        # Need to emit ProcPayloadOutputEvent in the order of in_args,
        # thus we filter them out first, then yield in order.

        for p in self._join_payloads_from_packets(out_packets):
            yield ProcPayloadOutEvent(data=p, src_name=self.name, call_id=call_id)
