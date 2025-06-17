from abc import ABC
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, cast, final

from pydantic import BaseModel, TypeAdapter

from .generics_utils import AutoInstanceAttributesMixin
from .packet import Packet
from .run_context import CtxT, RunContext
from .typing.events import Event, PacketEvent, ProcOutputEvent
from .typing.io import InT_contra, MemT_co, OutT_co, ProcName
from .typing.tool import BaseTool


class Processor(
    AutoInstanceAttributesMixin, ABC, Generic[InT_contra, OutT_co, MemT_co, CtxT]
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(self, name: ProcName, **kwargs: Any) -> None:
        self._in_type: type[InT_contra]
        self._out_type: type[OutT_co]

        super().__init__()

        self._in_type_adapter: TypeAdapter[InT_contra] = TypeAdapter(self._in_type)
        self._out_type_adapter: TypeAdapter[OutT_co] = TypeAdapter(self._out_type)

        self._name: ProcName = name
        self._memory: MemT_co

    @property
    def in_type(self) -> type[InT_contra]:  # type: ignore[reportInvalidTypeVarUse]
        # Exposing the type of a contravariant variable only, should be type safe
        return self._in_type

    @property
    def out_type(self) -> type[OutT_co]:
        return self._out_type

    @property
    def name(self) -> ProcName:
        return self._name

    @property
    def memory(self) -> MemT_co:
        return self._memory

    def _validate_and_resolve_inputs(
        self,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT_contra] | None = None,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
    ) -> Sequence[InT_contra] | None:
        multiple_inputs_err_message = (
            "Only one of chat_inputs, in_args, or in_message must be provided."
        )
        if chat_inputs is not None and in_args is not None:
            raise ValueError(multiple_inputs_err_message)
        if chat_inputs is not None and in_packet is not None:
            raise ValueError(multiple_inputs_err_message)
        if in_args is not None and in_packet is not None:
            raise ValueError(multiple_inputs_err_message)

        resolved_in_args: Sequence[InT_contra] | None = None
        if in_packet is not None:
            resolved_in_args = in_packet.payloads
        elif isinstance(in_args, self._in_type):
            resolved_in_args = cast("Sequence[InT_contra]", [in_args])
        elif in_args is None:
            resolved_in_args = in_args
        else:
            resolved_in_args = cast("Sequence[InT_contra]", in_args)

        return resolved_in_args

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: Sequence[InT_contra] | None = None,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[OutT_co]:
        assert in_args is not None, (
            "Default implementation of _process requires in_args"
        )

        return cast("Sequence[OutT_co]", in_args)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: Sequence[InT_contra] | None = None,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        assert in_args is not None, (
            "Default implementation of _process requires in_args"
        )
        outputs = cast("Sequence[OutT_co]", in_args)
        for out in outputs:
            yield ProcOutputEvent(data=out, name=self.name)

    def _validate_outputs(self, out_payloads: Sequence[OutT_co]) -> Sequence[OutT_co]:
        return [
            self._out_type_adapter.validate_python(payload) for payload in out_payloads
        ]

    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT_contra] | None = None,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co]:
        resolved_in_args = self._validate_and_resolve_inputs(
            chat_inputs=chat_inputs, in_packet=in_packet, in_args=in_args
        )
        outputs = await self._process(
            chat_inputs=chat_inputs,
            in_args=resolved_in_args,
            forgetful=forgetful,
            ctx=ctx,
        )
        val_outputs = self._validate_outputs(outputs)

        return Packet(payloads=val_outputs, sender=self.name)

    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT_contra] | None = None,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        resolved_in_args = self._validate_and_resolve_inputs(
            chat_inputs=chat_inputs, in_packet=in_packet, in_args=in_args
        )

        outputs: Sequence[OutT_co] = []
        async for output_event in self._process_stream(
            chat_inputs=chat_inputs,
            in_args=resolved_in_args,
            forgetful=forgetful,
            ctx=ctx,
        ):
            if isinstance(output_event, ProcOutputEvent):
                outputs.append(output_event.data)
            else:
                yield output_event

        val_outputs = self._validate_outputs(outputs)
        out_packet = Packet[OutT_co](payloads=val_outputs, sender=self.name)

        yield PacketEvent(data=out_packet, name=self.name)

    @final
    def as_tool(
        self, tool_name: str, tool_description: str
    ) -> BaseTool[InT_contra, OutT_co, Any]:  # type: ignore[override]
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
                self, inp: InT_contra, ctx: RunContext[CtxT] | None = None
            ) -> OutT_co:
                result = await processor_instance.run(
                    in_args=in_type.model_validate(inp), forgetful=True, ctx=ctx
                )

                return result.payloads[0]

        return ProcessorTool()  # type: ignore[return-value]
