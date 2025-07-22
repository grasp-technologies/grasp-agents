import asyncio
import logging
from abc import ABC
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, final
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from .errors import (
    PacketRoutingError,
    ProcInputValidationError,
    ProcOutputValidationError,
    ProcRunError,
)
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
from .typing.io import InT, OutT, ProcName
from .typing.tool import BaseTool
from .utils import stream_concurrent

logger = logging.getLogger(__name__)

_OutT_contra = TypeVar("_OutT_contra", contravariant=True)


class SelectRecipientsHandler(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self, output: _OutT_contra, ctx: RunContext[CtxT] | None
    ) -> list[ProcName] | None: ...


class Processor(AutoInstanceAttributesMixin, ABC, Generic[InT, OutT, MemT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcName,
        max_retries: int = 0,
        recipients: list[ProcName] | None = None,
        **kwargs: Any,
    ) -> None:
        self._in_type: type[InT]
        self._out_type: type[OutT]

        super().__init__()

        self._name: ProcName = name
        self._memory: MemT = cast("MemT", DummyMemory())
        self._max_retries: int = max_retries
        self.recipients = recipients
        self.select_recipients_impl: SelectRecipientsHandler[OutT, CtxT] | None = None

    @property
    def in_type(self) -> type[InT]:
        return self._in_type

    @property
    def out_type(self) -> type[OutT]:
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

    def _validate_inputs(
        self,
        call_id: str,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT] | None = None,
        in_args: InT | Sequence[InT] | None = None,
    ) -> Sequence[InT] | None:
        mult_inputs_err_message = (
            "Only one of chat_inputs, in_args, or in_message must be provided."
        )
        err_kwargs = {"proc_name": self.name, "call_id": call_id}

        if chat_inputs is not None and in_args is not None:
            raise ProcInputValidationError(
                message=mult_inputs_err_message, **err_kwargs
            )
        if chat_inputs is not None and in_packet is not None:
            raise ProcInputValidationError(
                message=mult_inputs_err_message, **err_kwargs
            )
        if in_args is not None and in_packet is not None:
            raise ProcInputValidationError(
                message=mult_inputs_err_message, **err_kwargs
            )

        if in_packet is not None and not in_packet.payloads:
            raise ProcInputValidationError(
                message="in_packet must contain at least one payload.", **err_kwargs
            )
        if in_args is not None and not in_args:
            raise ProcInputValidationError(
                message="in_args must contain at least one argument.", **err_kwargs
            )

        if chat_inputs is not None:
            return None

        resolved_args: Sequence[InT]

        if isinstance(in_args, Sequence):
            _in_args = cast("Sequence[Any]", in_args)
            if all(isinstance(x, self.in_type) for x in _in_args):
                resolved_args = cast("Sequence[InT]", _in_args)
            elif isinstance(_in_args, self.in_type):
                resolved_args = cast("Sequence[InT]", [_in_args])
            else:
                raise ProcInputValidationError(
                    message=f"in_args are neither of type {self.in_type} "
                    f"nor a sequence of {self.in_type}.",
                    **err_kwargs,
                )

        elif in_args is not None:
            resolved_args = cast("Sequence[InT]", [in_args])

        else:
            assert in_packet is not None
            resolved_args = in_packet.payloads

        try:
            for args in resolved_args:
                TypeAdapter(self._in_type).validate_python(args)
        except PydanticValidationError as err:
            raise ProcInputValidationError(message=str(err), **err_kwargs) from err

        return resolved_args

    def _validate_output(self, out_payload: OutT, call_id: str) -> OutT:
        if out_payload is None:
            return out_payload
        try:
            return TypeAdapter(self._out_type).validate_python(out_payload)
        except PydanticValidationError as err:
            raise ProcOutputValidationError(
                schema=self._out_type, proc_name=self.name, call_id=call_id
            ) from err

    def _validate_recipients(
        self, recipients: Sequence[ProcName] | None, call_id: str
    ) -> None:
        for r in recipients or []:
            if r not in (self.recipients or []):
                raise PacketRoutingError(
                    proc_name=self.name,
                    call_id=call_id,
                    selected_recipient=r,
                    allowed_recipients=cast("list[str]", self.recipients),
                )

    def _validate_par_recipients(
        self, out_packets: Sequence[Packet[OutT]], call_id: str
    ) -> None:
        recipient_sets = [set(p.recipients or []) for p in out_packets]
        same_recipients = all(rs == recipient_sets[0] for rs in recipient_sets)
        if not same_recipients:
            raise PacketRoutingError(
                proc_name=self.name,
                call_id=call_id,
                message="Parallel runs must return the same recipients "
                f"[proc_name={self.name}; call_id={call_id}]",
            )

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        memory: MemT,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> OutT:
        return cast("OutT", in_args)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        memory: MemT,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        output = cast("OutT", in_args)
        yield ProcPayloadOutputEvent(data=output, proc_name=self.name, call_id=call_id)

    async def _run_single_once(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        forgetful: bool = False,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT]:
        _memory = self.memory.model_copy(deep=True) if forgetful else self.memory

        output = await self._process(
            chat_inputs=chat_inputs,
            in_args=in_args,
            memory=_memory,
            call_id=call_id,
            ctx=ctx,
        )
        val_output = self._validate_output(output, call_id=call_id)

        recipients = self._select_recipients(output=val_output, ctx=ctx)
        self._validate_recipients(recipients, call_id=call_id)

        return Packet(payloads=[val_output], sender=self.name, recipients=recipients)

    async def _run_single(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        forgetful: bool = False,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT]:
        n_attempt = 0
        while n_attempt <= self.max_retries:
            try:
                return await self._run_single_once(
                    chat_inputs=chat_inputs,
                    in_args=in_args,
                    forgetful=forgetful,
                    call_id=call_id,
                    ctx=ctx,
                )
            except Exception as err:
                err_message = (
                    f"\nProcessor run failed [proc_name={self.name}; call_id={call_id}]"
                )
                n_attempt += 1
                if n_attempt > self.max_retries:
                    if n_attempt == 1:
                        logger.warning(f"{err_message}:\n{err}")
                    if n_attempt > 1:
                        logger.warning(f"{err_message} after retrying:\n{err}")
                    raise ProcRunError(proc_name=self.name, call_id=call_id) from err

                logger.warning(f"{err_message} (retry attempt {n_attempt}):\n{err}")

        raise ProcRunError(proc_name=self.name, call_id=call_id)

    async def _run_par(
        self, in_args: Sequence[InT], call_id: str, ctx: RunContext[CtxT] | None = None
    ) -> Packet[OutT]:
        tasks = [
            self._run_single(
                in_args=inp, forgetful=True, call_id=f"{call_id}/{idx}", ctx=ctx
            )
            for idx, inp in enumerate(in_args)
        ]
        out_packets = await asyncio.gather(*tasks)

        self._validate_par_recipients(out_packets, call_id=call_id)

        return Packet(
            payloads=[out_packet.payloads[0] for out_packet in out_packets],
            sender=self.name,
            recipients=out_packets[0].recipients,
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
    ) -> Packet[OutT]:
        call_id = self._generate_call_id(call_id)

        val_in_args = self._validate_inputs(
            call_id=call_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

        if val_in_args and len(val_in_args) > 1:
            return await self._run_par(in_args=val_in_args, call_id=call_id, ctx=ctx)
        return await self._run_single(
            chat_inputs=chat_inputs,
            in_args=val_in_args[0] if val_in_args else None,
            forgetful=forgetful,
            call_id=call_id,
            ctx=ctx,
        )

    async def _run_single_stream_once(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        forgetful: bool = False,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        _memory = self.memory.model_copy(deep=True) if forgetful else self.memory

        output: OutT | None = None
        async for event in self._process_stream(
            chat_inputs=chat_inputs,
            in_args=in_args,
            memory=_memory,
            call_id=call_id,
            ctx=ctx,
        ):
            if isinstance(event, ProcPayloadOutputEvent):
                output = event.data
            yield event

        assert output is not None

        val_output = self._validate_output(output, call_id=call_id)

        recipients = self._select_recipients(output=val_output, ctx=ctx)
        self._validate_recipients(recipients, call_id=call_id)

        out_packet = Packet[OutT](
            payloads=[val_output], sender=self.name, recipients=recipients
        )

        yield ProcPacketOutputEvent(
            data=out_packet, proc_name=self.name, call_id=call_id
        )

    async def _run_single_stream(
        self,
        chat_inputs: Any | None = None,
        *,
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

                err_message = (
                    "\nStreaming processor run failed "
                    f"[proc_name={self.name}; call_id={call_id}]"
                )

                n_attempt += 1
                if n_attempt > self.max_retries:
                    if n_attempt == 1:
                        logger.warning(f"{err_message}:\n{err}")
                    if n_attempt > 1:
                        logger.warning(f"{err_message} after retrying:\n{err}")
                    raise ProcRunError(proc_name=self.name, call_id=call_id) from err

                logger.warning(f"{err_message} (retry attempt {n_attempt}):\n{err}")

    async def _run_par_stream(
        self,
        in_args: Sequence[InT],
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        streams = [
            self._run_single_stream(
                in_args=inp, forgetful=True, call_id=f"{call_id}/{idx}", ctx=ctx
            )
            for idx, inp in enumerate(in_args)
        ]

        out_packets_map: dict[int, Packet[OutT]] = {}
        async for idx, event in stream_concurrent(streams):
            if isinstance(event, ProcPacketOutputEvent):
                out_packets_map[idx] = event.data
            else:
                yield event

        out_packet = Packet(
            payloads=[
                out_packet.payloads[0]
                for _, out_packet in sorted(out_packets_map.items())
            ],
            sender=self.name,
            recipients=out_packets_map[0].recipients,
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

        val_in_args = self._validate_inputs(
            call_id=call_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

        if val_in_args and len(val_in_args) > 1:
            stream = self._run_par_stream(in_args=val_in_args, call_id=call_id, ctx=ctx)
        else:
            stream = self._run_single_stream(
                chat_inputs=chat_inputs,
                in_args=val_in_args[0] if val_in_args else None,
                forgetful=forgetful,
                call_id=call_id,
                ctx=ctx,
            )
        async for event in stream:
            yield event

    def _select_recipients(
        self, output: OutT, ctx: RunContext[CtxT] | None = None
    ) -> list[ProcName] | None:
        if self.select_recipients_impl:
            return self.select_recipients_impl(output=output, ctx=ctx)

        return self.recipients

    def select_recipients(
        self, func: SelectRecipientsHandler[OutT, CtxT]
    ) -> SelectRecipientsHandler[OutT, CtxT]:
        self.select_recipients_impl = func

        return func

    @final
    def as_tool(
        self, tool_name: str, tool_description: str
    ) -> BaseTool[InT, OutT, Any]:  # type: ignore[override]
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

            async def run(self, inp: InT, ctx: RunContext[CtxT] | None = None) -> OutT:
                result = await processor_instance.run(
                    in_args=in_type.model_validate(inp), forgetful=True, ctx=ctx
                )

                return result.payloads[0]

        return ProcessorTool()
