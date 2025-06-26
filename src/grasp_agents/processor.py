import asyncio
import logging
from abc import ABC
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, cast, final
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from tenacity import RetryCallState, retry, stop_after_attempt, wait_random_exponential

from .errors import InputValidationError
from .generics_utils import AutoInstanceAttributesMixin
from .memory import MemT
from .packet import Packet
from .run_context import CtxT, RunContext
from .typing.events import Event, PacketEvent, ProcOutputEvent
from .typing.io import InT, OutT_co, ProcName
from .typing.tool import BaseTool

logger = logging.getLogger(__name__)


def retry_error_callback(retry_state: RetryCallState) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    if exception:
        if retry_state.attempt_number == 1:
            logger.warning(f"\nParallel run failed:\n{exception}")
        if retry_state.attempt_number > 1:
            logger.warning(f"\nParallel run failed after retrying:\n{exception}")


def retry_before_sleep_callback(retry_state: RetryCallState) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.info(
        f"\nRetrying parallel run (attempt {retry_state.attempt_number}):\n{exception}"
    )


class Processor(AutoInstanceAttributesMixin, ABC, Generic[InT, OutT_co, MemT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self, name: ProcName, num_par_run_retries: int = 0, **kwargs: Any
    ) -> None:
        self._in_type: type[InT]
        self._out_type: type[OutT_co]

        super().__init__()

        self._in_type_adapter: TypeAdapter[InT] = TypeAdapter(self._in_type)
        self._out_type_adapter: TypeAdapter[OutT_co] = TypeAdapter(self._out_type)

        self._name: ProcName = name
        self._memory: MemT
        self._num_par_run_retries: int = num_par_run_retries

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
    def num_par_run_retries(self) -> int:
        return self._num_par_run_retries

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
            raise InputValidationError(multiple_inputs_err_message)
        if chat_inputs is not None and in_packet is not None:
            raise InputValidationError(multiple_inputs_err_message)
        if in_args is not None and in_packet is not None:
            raise InputValidationError(multiple_inputs_err_message)

        if in_packet is not None:
            if len(in_packet.payloads) != 1:
                raise InputValidationError(
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
            raise InputValidationError(
                "chat_inputs are not supported in parallel runs. "
                "Use in_packet or in_args."
            )
        if in_packet is not None:
            if not in_packet.payloads:
                raise InputValidationError(
                    "Parallel runs require at least one input payload in in_packet."
                )
            return in_packet.payloads
        if in_args is not None:
            return in_args
        raise InputValidationError(
            "Parallel runs require either in_packet or in_args to be provided."
        )

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        memory: MemT,
        run_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[OutT_co]:
        if in_args is None:
            raise InputValidationError(
                "Default implementation of _process requires in_args"
            )

        return cast("Sequence[OutT_co]", in_args)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | None = None,
        memory: MemT,
        run_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        if in_args is None:
            raise InputValidationError(
                "Default implementation of _process requires in_args"
            )
        outputs = cast("Sequence[OutT_co]", in_args)
        for out in outputs:
            yield ProcOutputEvent(data=out, name=self.name)

    def _validate_outputs(self, out_payloads: Sequence[OutT_co]) -> Sequence[OutT_co]:
        return [
            self._out_type_adapter.validate_python(payload) for payload in out_payloads
        ]

    def _generate_run_id(self, run_id: str | None) -> str:
        if run_id is None:
            return str(uuid4())[:6] + "_" + self.name
        return run_id

    async def _run_single(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | None = None,
        forgetful: bool = False,
        run_id: str | None = None,
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
            run_id=self._generate_run_id(run_id),
            ctx=ctx,
        )
        val_outputs = self._validate_outputs(outputs)

        return Packet(payloads=val_outputs, sender=self.name)

    def _generate_par_run_id(self, run_id: str | None, idx: int) -> str:
        return f"{self._generate_run_id(run_id)}/{idx}"

    async def _run_par(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: Sequence[InT] | None = None,
        run_id: str | None = None,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co]:
        par_inputs = self._validate_and_resolve_parallel_inputs(
            chat_inputs=chat_inputs, in_packet=in_packet, in_args=in_args
        )

        wrapped_func = retry(
            wait=wait_random_exponential(min=1, max=8),
            stop=stop_after_attempt(self._num_par_run_retries + 1),
            before_sleep=retry_before_sleep_callback,
            retry_error_callback=retry_error_callback,
        )(self._run_single)

        tasks = [
            wrapped_func(
                in_args=inp,
                forgetful=True,
                run_id=self._generate_par_run_id(run_id, idx),
                ctx=ctx,
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
        run_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co]:
        if (in_args is not None and isinstance(in_args, Sequence)) or (
            in_packet is not None and len(in_packet.payloads) > 1
        ):
            return await self._run_par(
                chat_inputs=chat_inputs,
                in_packet=in_packet,
                in_args=cast("Sequence[InT]", in_args),
                run_id=run_id,
                forgetful=forgetful,
                ctx=ctx,
            )
        return await self._run_single(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            forgetful=forgetful,
            run_id=run_id,
            ctx=ctx,
        )

    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | None = None,
        forgetful: bool = False,
        run_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        resolved_in_args = self._validate_and_resolve_single_input(
            chat_inputs=chat_inputs, in_packet=in_packet, in_args=in_args
        )

        _memory = self.memory.model_copy(deep=True) if forgetful else self.memory

        outputs: Sequence[OutT_co] = []
        async for output_event in self._process_stream(
            chat_inputs=chat_inputs,
            in_args=resolved_in_args,
            memory=_memory,
            run_id=self._generate_run_id(run_id),
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
