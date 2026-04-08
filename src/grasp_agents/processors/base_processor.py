import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar, cast, final
from uuid import uuid4

from pydantic import BaseModel

from ..errors import ProcRunError
from ..generics_utils import AutoInstanceAttributesMixin
from ..memory import DummyMemory, Memory
from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..types.events import (
    DummyEvent,
    Event,
    ProcStreamingErrorData,
    ProcStreamingErrorEvent,
)
from ..types.io import InT, OutT, ProcName

if TYPE_CHECKING:
    from .processor_tool import ProcessorTool

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., AsyncIterator[Event[Any]]])


def with_retry(func: F) -> F:
    @wraps(func)
    async def wrapper(
        self: "BaseProcessor[Any, Any, Any]", *args: Any, **kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        exec_id: str | None = kwargs.get("exec_id")

        n_attempt = 0
        while n_attempt <= self.max_retries:
            try:
                async for event in func(self, *args, **kwargs):
                    yield event
                return

            except Exception as err:
                n_attempt += 1

                err_data = ProcStreamingErrorData(error=err, exec_id=exec_id)
                yield ProcStreamingErrorEvent(
                    data=err_data,
                    source=self.name,
                    exec_id=exec_id,
                )
                err_message = (
                    f"Processor run failed [proc_name={self.name}; exec_id={exec_id}]"
                )
                if n_attempt > self.max_retries:
                    raise ProcRunError(
                        proc_name=self.name,
                        exec_id=exec_id,
                        message=err_message + f" after {n_attempt - 1} retries",
                    ) from err

                logger.warning(
                    f"{err_message} -> retrying (attempt {n_attempt}):\n{err}"
                )

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
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        self._in_type: type[InT]
        self._out_type: type[OutT]

        super().__init__()

        self._name = name
        self._max_retries = max_retries
        self._memory: Memory = memory or DummyMemory()
        self.recipients = recipients

        self.tracing_enabled = tracing_enabled
        self.tracing_exclude_input_fields = tracing_exclude_input_fields

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

    def generate_exec_id(self, exec_id: str | None) -> str:
        if exec_id is None:
            return str(uuid4())[:6] + "_" + self.name
        return exec_id

    def copy(self) -> Self:
        return deepcopy(self)

    # --- Session persistence (overridden by LLMAgent) ---

    @property
    def resumable(self) -> bool:
        return False

    def reset_session(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        exec_id: str | None = None,
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
        exec_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        yield DummyEvent()

    async def resume_stream(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        raise NotImplementedError
        yield  # type: ignore[unreachable]  # makes this an async generator

    @final
    def as_tool(
        self,
        tool_name: str,
        tool_description: str,
        reset_memory_on_run: bool = True,
        background: bool = False,
    ) -> "ProcessorTool[InT, OutT, CtxT]":  # type: ignore[return-value]
        from .processor_tool import ProcessorTool as _ProcessorTool

        if not issubclass(self.in_type, BaseModel):
            raise TypeError(
                "Cannot create a tool from an agent with "
                f"non-BaseModel input type: {self.in_type}"
            )

        return _ProcessorTool[InT, OutT, CtxT](  # type: ignore[type-var]
            processor=self,  # InT bound validated above
            name=tool_name,
            description=tool_description,
            background=background,
            reset_memory_on_run=reset_memory_on_run,
        )
