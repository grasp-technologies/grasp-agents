import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from functools import wraps
from typing import Any, ClassVar, Generic, Self, TypeVar, cast, final
from uuid import uuid4

from pydantic import BaseModel

from ..errors import ProcRunError
from ..generics_utils import AutoInstanceAttributesMixin
from ..memory import DummyMemory, Memory
from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..typing.events import (
    DummyEvent,
    Event,
    ProcPacketOutEvent,
    ProcStreamingErrorData,
    ProcStreamingErrorEvent,
    ToolOutputEvent,
)
from ..typing.io import InT, OutT, ProcName
from ..typing.tool import BaseTool

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., AsyncIterator[Event[Any]]])


def with_retry(func: F) -> F:
    @wraps(func)
    async def wrapper(
        self: "BaseProcessor[Any, Any, Any]", *args: Any, **kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        call_id = kwargs.get("call_id", "unknown")

        n_attempt = 0
        while n_attempt <= self.max_retries:
            try:
                async for event in func(self, *args, **kwargs):
                    yield event
                return

            except Exception as err:
                n_attempt += 1

                err_data = ProcStreamingErrorData(error=err, call_id=call_id)
                yield ProcStreamingErrorEvent(
                    data=err_data, src_name=self.name, call_id=call_id
                )
                err_message = (
                    f"Processor run failed [proc_name={self.name}; call_id={call_id}]"
                )
                if n_attempt > self.max_retries:
                    raise ProcRunError(proc_name=self.name, call_id=call_id) from err

                logger.warning(f"{err_message} (retry attempt {n_attempt}):\n{err}")

    return cast("F", wrapper)


class BaseProcessor(AutoInstanceAttributesMixin, ABC, Generic[InT, OutT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcName,
        max_retries: int = 0,
        memory: Memory | None = None,
        recipients: Sequence[ProcName] | None = None,
        **kwargs: Any,
    ) -> None:
        self._in_type: type[InT]
        self._out_type: type[OutT]

        super().__init__()

        self._name = name
        self._max_retries = max_retries
        self._memory: Memory = memory or DummyMemory()
        self.recipients = recipients

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
    def memory(self) -> Memory:
        return self._memory

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def generate_call_id(self, call_id: str | None) -> str:
        if call_id is None:
            return str(uuid4())[:6] + "_" + self.name
        return call_id

    def copy(self) -> Self:
        return deepcopy(self)

    @abstractmethod
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT]:
        pass

    @abstractmethod
    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        yield DummyEvent()

    @final
    def as_tool(
        self, tool_name: str, tool_description: str, reset_memory_on_run: bool = True
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

            async def run(
                self,
                inp: InT,
                *,
                call_id: str | None = None,
                ctx: RunContext[CtxT] | None = None,
            ) -> OutT:
                if reset_memory_on_run:
                    processor_instance.memory.reset()

                result = await processor_instance.run(
                    in_args=inp, call_id=call_id, ctx=ctx
                )

                return result.payloads[0]

            async def run_stream(
                self,
                inp: InT,
                *,
                call_id: str | None = None,
                ctx: RunContext[CtxT] | None = None,
            ) -> AsyncIterator[Event[Any]]:
                if reset_memory_on_run:
                    processor_instance.memory.reset()

                async for event in processor_instance.run_stream(
                    in_args=inp, call_id=call_id, ctx=ctx
                ):
                    if (
                        isinstance(event, ProcPacketOutEvent)
                        and event.src_name == processor_instance.name
                    ):
                        yield ToolOutputEvent(
                            data=event.data.payloads[0],
                            src_name=processor_instance.name,
                            call_id=call_id,
                        )
                    else:
                        yield event

        return ProcessorTool()
