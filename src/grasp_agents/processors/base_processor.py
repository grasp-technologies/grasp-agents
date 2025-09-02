import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from functools import wraps
from typing import (
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    cast,
    final,
)
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from ..errors import (
    PacketRoutingError,
    ProcInputValidationError,
    ProcOutputValidationError,
    ProcRunError,
)
from ..generics_utils import AutoInstanceAttributesMixin
from ..memory import DummyMemory, MemT
from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..typing.events import (
    DummyEvent,
    Event,
    ProcStreamingErrorData,
    ProcStreamingErrorEvent,
)
from ..typing.io import InT, OutT, ProcName
from ..typing.tool import BaseTool

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Packet[Any]]])
F_stream = TypeVar("F_stream", bound=Callable[..., AsyncIterator[Event[Any]]])


def with_retry(func: F) -> F:
    @wraps(func)
    async def wrapper(
        self: "BaseProcessor[Any, Any, Any, Any]", *args: Any, **kwargs: Any
    ) -> Packet[Any]:
        none_packet = Packet(payloads=[None], sender=self.name)
        call_id = kwargs.get("call_id", "unknown")
        for n_attempt in range(self.max_retries + 1):
            try:
                return await func(self, *args, **kwargs)
            except Exception as err:
                err_message = (
                    f"\nProcessor run failed [proc_name={self.name}; call_id={call_id}]"
                )
                if n_attempt == self.max_retries:
                    if self.max_retries == 0:
                        logger.warning(f"{err_message}:\n{err}")
                    else:
                        logger.warning(f"{err_message} after retrying:\n{err}")
                    # raise ProcRunError(proc_name=self.name, call_id=call_id) from err
                    return none_packet

                logger.warning(f"{err_message} (retry attempt {n_attempt + 1}):\n{err}")
        # This part should not be reachable due to the raise in the loop
        # raise ProcRunError(proc_name=self.name, call_id=call_id)
        return none_packet

    return cast("F", wrapper)


def with_retry_stream(func: F_stream) -> F_stream:
    @wraps(func)
    async def wrapper(
        self: "BaseProcessor[Any, Any, Any, Any]", *args: Any, **kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        call_id = kwargs.get("call_id", "unknown")
        for n_attempt in range(self.max_retries + 1):
            try:
                async for event in func(self, *args, **kwargs):
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
                if n_attempt == self.max_retries:
                    if self.max_retries == 0:
                        logger.warning(f"{err_message}:\n{err}")
                    else:
                        logger.warning(f"{err_message} after retrying:\n{err}")
                    raise ProcRunError(proc_name=self.name, call_id=call_id) from err

                logger.warning(f"{err_message} (retry attempt {n_attempt}):\n{err}")

    return cast("F_stream", wrapper)


_OutT_contra = TypeVar("_OutT_contra", contravariant=True)


class RecipientSelector(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self, output: _OutT_contra, *, ctx: RunContext[CtxT]
    ) -> Sequence[ProcName] | None: ...


class BaseProcessor(AutoInstanceAttributesMixin, ABC, Generic[InT, OutT, MemT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcName,
        max_retries: int = 0,
        recipients: Sequence[ProcName] | None = None,
        **kwargs: Any,
    ) -> None:
        self._in_type: type[InT]
        self._out_type: type[OutT]

        super().__init__()

        self._name: ProcName = name
        self._memory: MemT = cast("MemT", DummyMemory())
        self._max_retries: int = max_retries

        self.recipients = recipients

        self.recipient_selector: RecipientSelector[OutT, CtxT] | None
        if not hasattr(type(self), "recipient_selector"):
            self.recipient_selector = None

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
        in_args: InT | list[InT] | None = None,
    ) -> list[InT] | None:
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

        resolved_args: list[InT]

        if isinstance(in_args, list):
            _in_args = cast("list[Any]", in_args)
            if all(isinstance(x, self.in_type) for x in _in_args):
                resolved_args = cast("list[InT]", _in_args)
            elif isinstance(_in_args, self.in_type):
                resolved_args = cast("list[InT]", [_in_args])
            else:
                raise ProcInputValidationError(
                    message=f"in_args are neither of type {self.in_type} "
                    f"nor a sequence of {self.in_type}.",
                    **err_kwargs,
                )

        elif in_args is not None:
            resolved_args = cast("list[InT]", [in_args])

        else:
            assert in_packet is not None
            resolved_args = cast("list[InT]", in_packet.payloads)

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

    @final
    def _select_recipients(
        self, output: OutT, ctx: RunContext[CtxT]
    ) -> Sequence[ProcName] | None:
        if self.recipient_selector:
            return self.recipient_selector(output=output, ctx=ctx)

        return self.recipients

    def add_recipient_selector(
        self, func: RecipientSelector[OutT, CtxT]
    ) -> RecipientSelector[OutT, CtxT]:
        self.recipient_selector = func

        return func

    @abstractmethod
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        forgetful: bool = False,
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
        forgetful: bool = False,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        yield DummyEvent()

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

            async def run(
                self,
                inp: InT,
                *,
                call_id: str | None = None,
                ctx: RunContext[CtxT] | None = None,
            ) -> OutT:
                result = await processor_instance.run(
                    in_args=inp, forgetful=True, call_id=call_id, ctx=ctx
                )

                return result.payloads[0]

        return ProcessorTool()
