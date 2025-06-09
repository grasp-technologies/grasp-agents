from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Generic, cast, final

from pydantic import BaseModel, TypeAdapter

from .generics_utils import AutoInstanceAttributesMixin
from .packet import Packet
from .run_context import CtxT, RunContext
from .typing.io import InT_contra, MemT_co, OutT_co, ProcessorName
from .typing.tool import BaseTool


class Processor(
    AutoInstanceAttributesMixin, ABC, Generic[InT_contra, OutT_co, MemT_co, CtxT]
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    @abstractmethod
    def __init__(self, name: ProcessorName, **kwargs: Any) -> None:
        self._in_type: type[InT_contra]
        self._out_type: type[OutT_co]

        super().__init__()

        self._in_type_adapter: TypeAdapter[InT_contra] = TypeAdapter(self._in_type)
        self._out_type_adapter: TypeAdapter[OutT_co] = TypeAdapter(self._out_type)

        self._name: ProcessorName = name
        self._memory: MemT_co

    @property
    def in_type(self) -> type[InT_contra]:  # type: ignore[reportInvalidTypeVarUse]
        # Exposing the type of a contravariant variable only, should be type safe
        return self._in_type

    @property
    def out_type(self) -> type[OutT_co]:
        return self._out_type

    @property
    def name(self) -> ProcessorName:
        return self._name

    @property
    def memory(self) -> MemT_co:
        return self._memory

    @staticmethod
    def _validate_inputs(
        chat_inputs: Any | None = None,
        in_packet: Packet[InT_contra] | None = None,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        entry_point: bool = False,
    ) -> None:
        multiple_inputs_err_message = (
            "Only one of chat_inputs, in_args, or in_message must be provided."
        )
        if chat_inputs is not None and in_args is not None:
            raise ValueError(multiple_inputs_err_message)
        if chat_inputs is not None and in_packet is not None:
            raise ValueError(multiple_inputs_err_message)
        if in_args is not None and in_packet is not None:
            raise ValueError(multiple_inputs_err_message)

        if entry_point and in_packet is not None:
            raise ValueError(
                "Entry point agent cannot receive packets from other agents."
            )

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        entry_point: bool = False,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[OutT_co]:
        assert in_args is not None, (
            "Default implementation of _process requires in_args"
        )
        outputs: Sequence[OutT_co]
        if not isinstance(in_args, Sequence):
            outputs = cast("Sequence[OutT_co]", [in_args])
        else:
            outputs = cast("Sequence[OutT_co]", in_args)

        return outputs

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
        entry_point: bool = False,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co]:
        self._validate_inputs(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            entry_point=entry_point,
        )
        resolved_in_args = in_packet.payloads if in_packet else in_args
        outputs = await self._process(
            chat_inputs=chat_inputs,
            in_args=resolved_in_args,
            entry_point=entry_point,
            forgetful=forgetful,
            ctx=ctx,
        )
        val_outputs = self._validate_outputs(outputs)

        return Packet(payloads=val_outputs, sender=self.name)

    @final
    def as_tool(
        self, tool_name: str, tool_description: str, tool_strict: bool = True
    ) -> BaseTool[InT_contra, OutT_co, Any]:  # type: ignore[override]
        # Will check if InT is a BaseModel at runtime
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
            strict: bool | None = tool_strict

            async def run(
                self, inp: InT_contra, ctx: RunContext[CtxT] | None = None
            ) -> OutT_co:
                result = await processor_instance.run(
                    in_args=in_type.model_validate(inp),
                    entry_point=False,
                    forgetful=True,
                    ctx=ctx,
                )

                return result.payloads[0]

        return ProcessorTool()  # type: ignore[return-value]
