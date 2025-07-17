import asyncio
import logging
from abc import ABC
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, cast, final
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from .errors import ProcInputValidationError, ProcOutputValidationError
from .generics_utils import AutoInstanceAttributesMixin
from .memory import DummyMemory, MemT
from .packet import Packet
from .run_context import CtxT, RunContext
from .typing.events import (
    Event,
    ProcPacketOutputEvent,
    ProcPayloadOutputEvent,
    ProcStreamingErrorData,
    ProcStreamingErrorEvent,
)
from .typing.io import InT, OutT_co, ProcName
from .typing.tool import BaseTool
from .utils import stream_concurrent

logger = logging.getLogger(__name__)


class Processor(AutoInstanceAttributesMixin, ABC, Generic[InT, OutT_co, MemT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(self, name: ProcName, max_retries: int = 0, **kwargs: Any) -> None:
        self._in_type: type[InT]
        self._out_type: type[OutT_co]

        super().__init__()

        self._name: ProcName = name
        self._memory: MemT = cast("MemT", DummyMemory())
        self._max_retries: int = max_retries

    @property
    def in_type(self) -> type[InT]:
        return self._in_type

    @property
    def out_type(self) -> type[OutT_co]:
        return self._out_type

    @property
    def name(self) -> ProcName:
        return self._name

    @property
    def memory(self) -> MemT:
        return self._memory

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def _generate_call_id(self, call_id: str | None) -> str:
        if call_id is None:
            return str(uuid4())[:6] + "_" + self.name
        return call_id

    def _validate_and_resolve_single_input(
        self,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT] | None = None,
        in_args: InT | None = None,
    ) -> InT | None:
        multiple_inputs_err_message = (
            "Only one of chat_inputs, in_args, or in_message must be provided."
        )
        if chat_inputs is not None and in_args is not None:
            raise ProcInputValidationError(multiple_inputs_err_message)
        if chat_inputs is not None and in_packet is not None:
            raise ProcInputValidationError(multiple_inputs_err_message)
        if in_args is not None and in_packet is not None:
            raise ProcInputValidationError(multiple_inputs_err_message)

        if in_packet is not None:
            if len(in_packet.payloads) != 1:
                raise ProcInputValidationError(
                    "Single input runs require exactly one payload in in_packet."
                )
            return in_packet.payloads[0]

        return in_args

    def _validate_and_resolve_parallel_inputs(
        self,
        chat_inputs: Any | None,
        in_packet: Packet[InT] | None,
        in_args: Sequence[InT] | None,
    ) -> Sequence[InT]:
        if chat_inputs is not None:
            raise ProcInputValidationError(
                "chat_inputs are not supported in parallel runs. "
                "Use in_packet or in_args."
            )
        if in_packet is not None:
            if not in_packet.payloads:
                raise ProcInputValidationError(
                    "Parallel runs require at least one input payload in in_packet."
                )
            return in_packet.payloads
        if in_args is not None:
            return in_args
        raise ProcInputValidationError(
            "Parallel runs require either in_packet or in_args to be provided."
        )

    def _validate_outputs(self, out_payloads: Sequence[OutT_co]) -> Sequence[OutT_co]:
        try:
            return [
                TypeAdapter(self._out_type).validate_python(payload)
                for payload in out_payloads
            ]
        except PydanticValidationError as err:
            raise ProcOutputValidationError(
                f"Output validation failed for processor {self.name}:\n{err}"
            ) from err

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        memory: MemT,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[OutT_co]:
        if in_args is None:
            raise ProcInputValidationError(
                "Default implementation of _process requires in_args"
            )

        return cast("Sequence[OutT_co]", in_args)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        memory: MemT,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        if in_args is None:
            raise ProcInputValidationError(
                "Default implementation of _process_stream requires in_args"
            )
        outputs = cast("Sequence[OutT_co]", in_args)
        for out in outputs:
            yield ProcPayloadOutputEvent(data=out, proc_name=self.name, call_id=call_id)

    async def _run_single_once(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | None = None,
        forgetful: bool = False,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co]:
        resolved_in_args = self._validate_and_resolve_single_input(
            chat_inputs=chat_inputs, in_packet=in_packet, in_args=in_args
        )
        _memory = self.memory.model_copy(deep=True) if forgetful else self.memory

        outputs = await self._process(
            chat_inputs=chat_inputs,
            in_args=resolved_in_args,
            memory=_memory,
            call_id=call_id,
            ctx=ctx,
        )
        val_outputs = self._validate_outputs(outputs)

        return Packet(payloads=val_outputs, sender=self.name)

    async def _run_single(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | None = None,
        forgetful: bool = False,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co] | None:
        n_attempt = 0
        while n_attempt <= self.max_retries:
            try:
                return await self._run_single_once(
                    chat_inputs=chat_inputs,
                    in_packet=in_packet,
                    in_args=in_args,
                    forgetful=forgetful,
                    call_id=call_id,
                    ctx=ctx,
                )
            except Exception as err:
                n_attempt += 1
                if n_attempt > self.max_retries:
                    if n_attempt == 1:
                        logger.warning(f"\nProcessor run failed:\n{err}")
                    if n_attempt > 1:
                        logger.warning(f"\nProcessor run failed after retrying:\n{err}")
                    return None
                logger.warning(
                    f"\nProcessor run failed (retry attempt {n_attempt}):\n{err}"
                )

    async def _run_par(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: Sequence[InT] | None = None,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co]:
        par_inputs = self._validate_and_resolve_parallel_inputs(
            chat_inputs=chat_inputs, in_packet=in_packet, in_args=in_args
        )
        tasks = [
            self._run_single(
                in_args=inp, forgetful=True, call_id=f"{call_id}/{idx}", ctx=ctx
            )
            for idx, inp in enumerate(par_inputs)
        ]
        out_packets = await asyncio.gather(*tasks)

        return Packet(  # type: ignore[return]
            payloads=[
                (out_packet.payloads[0] if out_packet else None)
                for out_packet in out_packets
            ],
            sender=self.name,
        )

    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | Sequence[InT] | None = None,
        forgetful: bool = False,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co]:
        call_id = self._generate_call_id(call_id)

        if (in_args is not None and isinstance(in_args, Sequence)) or (
            in_packet is not None and len(in_packet.payloads) > 1
        ):
            return await self._run_par(
                chat_inputs=chat_inputs,
                in_packet=in_packet,
                in_args=cast("Sequence[InT] | None", in_args),
                call_id=call_id,
                ctx=ctx,
            )
        return await self._run_single(  # type: ignore[return]
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=cast("InT | None", in_args),
            forgetful=forgetful,
            call_id=call_id,
            ctx=ctx,
        )

    async def _run_single_stream_once(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | None = None,
        forgetful: bool = False,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        resolved_in_args = self._validate_and_resolve_single_input(
            chat_inputs=chat_inputs, in_packet=in_packet, in_args=in_args
        )
        _memory = self.memory.model_copy(deep=True) if forgetful else self.memory

        outputs: list[OutT_co] = []
        async for event in self._process_stream(
            chat_inputs=chat_inputs,
            in_args=resolved_in_args,
            memory=_memory,
            call_id=call_id,
            ctx=ctx,
        ):
            if isinstance(event, ProcPayloadOutputEvent):
                outputs.append(event.data)
            yield event

        val_outputs = self._validate_outputs(outputs)
        out_packet = Packet[OutT_co](payloads=val_outputs, sender=self.name)

        yield ProcPacketOutputEvent(
            data=out_packet, proc_name=self.name, call_id=call_id
        )

    async def _run_single_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | None = None,
        forgetful: bool = False,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        n_attempt = 0
        while n_attempt <= self.max_retries:
            try:
                async for event in self._run_single_stream_once(
                    chat_inputs=chat_inputs,
                    in_packet=in_packet,
                    in_args=in_args,
                    forgetful=forgetful,
                    call_id=call_id,
                    ctx=ctx,
                ):
                    yield event

                return

            except Exception as err:
                err_data = ProcStreamingErrorData(error=err, call_id=call_id)
                yield ProcStreamingErrorEvent(
                    data=err_data, proc_name=self.name, call_id=call_id
                )

                n_attempt += 1
                if n_attempt > self.max_retries:
                    if n_attempt == 1:
                        logger.warning(f"\nStreaming processor run failed:\n{err}")
                    if n_attempt > 1:
                        logger.warning(
                            f"\nStreaming processor run failed after retrying:\n{err}"
                        )
                    return

                logger.warning(
                    "\nStreaming processor run failed "
                    f"(retry attempt {n_attempt}):\n{err}"
                )

    async def _run_par_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: Sequence[InT] | None = None,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        par_inputs = self._validate_and_resolve_parallel_inputs(
            chat_inputs=chat_inputs, in_packet=in_packet, in_args=in_args
        )
        streams = [
            self._run_single_stream(
                in_args=inp, forgetful=True, call_id=f"{call_id}/{idx}", ctx=ctx
            )
            for idx, inp in enumerate(par_inputs)
        ]

        out_packets_map: dict[int, Packet[OutT_co] | None] = dict.fromkeys(
            range(len(streams)), None
        )

        async for idx, event in stream_concurrent(streams):
            if isinstance(event, ProcPacketOutputEvent):
                out_packets_map[idx] = event.data
            else:
                yield event

        out_packet = Packet(  # type: ignore[return]
            payloads=[
                (out_packet.payloads[0] if out_packet else None)
                for out_packet in out_packets_map.values()
            ],
            sender=self.name,
        )

        yield ProcPacketOutputEvent(
            data=out_packet, proc_name=self.name, call_id=call_id
        )

    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | Sequence[InT] | None = None,
        forgetful: bool = False,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        call_id = self._generate_call_id(call_id)

        # yield ProcStartEvent(proc_name=self.name, call_id=call_id, data=None)

        if (in_args is not None and isinstance(in_args, Sequence)) or (
            in_packet is not None and len(in_packet.payloads) > 1
        ):
            stream = self._run_par_stream(
                chat_inputs=chat_inputs,
                in_packet=in_packet,
                in_args=cast("Sequence[InT] | None", in_args),
                call_id=call_id,
                ctx=ctx,
            )
        else:
            stream = self._run_single_stream(
                chat_inputs=chat_inputs,
                in_packet=in_packet,
                in_args=cast("InT | None", in_args),
                forgetful=forgetful,
                call_id=call_id,
                ctx=ctx,
            )
        async for event in stream:
            yield event

        # yield ProcFinishEvent(proc_name=self.name, call_id=call_id, data=None)

    @final
    def as_tool(
        self, tool_name: str, tool_description: str
    ) -> BaseTool[InT, OutT_co, Any]:  # type: ignore[override]
        # TODO: stream tools
        processor_instance = self
        in_type = processor_instance.in_type
        out_type = processor_instance.out_type
        if not issubclass(in_type, BaseModel):
            raise TypeError(
                "Cannot create a tool from an agent with "
                f"non-BaseModel input type: {in_type}"
            )

        class ProcessorTool(BaseTool[in_type, out_type, Any]):
            name: str = tool_name
            description: str = tool_description

            async def run(
                self, inp: InT, ctx: RunContext[CtxT] | None = None
            ) -> OutT_co:
                result = await processor_instance.run(
                    in_args=in_type.model_validate(inp), forgetful=True, ctx=ctx
                )

                return result.payloads[0]

        return ProcessorTool()  # type: ignore[return-value]
