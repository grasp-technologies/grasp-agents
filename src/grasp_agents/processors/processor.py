import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, TypeVar, cast, final

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError
from typing_extensions import Protocol

from grasp_agents.errors import (
    PacketRoutingError,
    ProcInputValidationError,
    ProcOutputValidationError,
)
from grasp_agents.tracing_decorators import workflow

from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..typing.events import Event, ProcPacketOutEvent, ProcPayloadOutEvent
from ..typing.io import InT, OutT, ProcName
from ..utils.callbacks import is_method_overridden
from .base_processor import BaseProcessor, with_retry

logger = logging.getLogger(__name__)


_OutT_contra = TypeVar("_OutT_contra", contravariant=True)


class RecipientSelector(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self, output: _OutT_contra, *, ctx: RunContext[CtxT], call_id: str
    ) -> Sequence[ProcName]: ...


class Processor(BaseProcessor[InT, OutT, CtxT], Generic[InT, OutT, CtxT]):
    """
    Processor that can have different numbers of inputs and outputs, allowing for an
    arbitrary mapping between them.
    """

    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def validate_inputs(
        self,
        call_id: str,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
    ) -> list[InT] | None:
        err_kwargs = {"proc_name": self.name, "call_id": call_id}

        num_non_null_inputs = sum(
            x is not None for x in [chat_inputs, in_args, in_packet]
        )

        if num_non_null_inputs > 1:
            raise ProcInputValidationError(
                message="Only one of chat_inputs, in_args, or in_message must be provided",
                **err_kwargs,
            )
        if self.in_type is not type(None) and num_non_null_inputs == 0:
            raise ProcInputValidationError(
                message="One of chat_inputs, in_args, or in_message must be provided",
                **err_kwargs,
            )

        if in_packet is not None and not in_packet.payloads:
            raise ProcInputValidationError(
                message="in_packet must contain at least one payload", **err_kwargs
            )
        if in_args is not None and not in_args:
            raise ProcInputValidationError(
                message="in_args must contain at least one argument", **err_kwargs
            )

        if chat_inputs is not None:
            # 1) chat_inputs are provided -> no need to validate further
            return None

        resolved_args: list[InT]

        if isinstance(in_args, self.in_type):
            # 2) Single in_args of correct type is provided
            resolved_args = [in_args]

        elif isinstance(in_args, list):
            # 3) List of in_args is provided
            resolved_args = cast("list[InT]", in_args)

        elif in_args is not None:
            raise ProcInputValidationError(
                message=f"in_args are neither of type {self.in_type} "
                f"nor a list of {self.in_type}.",
                **err_kwargs,
            )

        else:
            # 4) in_packet is provided
            resolved_args = list(cast("Packet[InT]", in_packet).payloads)

        try:
            for args in resolved_args:
                TypeAdapter(self._in_type).validate_python(args)
        except PydanticValidationError as err:
            raise ProcInputValidationError(message=str(err), **err_kwargs) from err

        return resolved_args

    def validate_output(self, out_payload: OutT, call_id: str) -> OutT:
        if out_payload is None:
            return out_payload
        try:
            return TypeAdapter(self.out_type).validate_python(out_payload)
        except PydanticValidationError as err:
            raise ProcOutputValidationError(
                schema=self.out_type, proc_name=self.name, call_id=call_id
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

    def select_recipients_impl(
        self, output: OutT, *, ctx: RunContext[CtxT], call_id: str
    ) -> Sequence[ProcName]:
        raise NotImplementedError

    def add_recipient_selector(
        self, func: RecipientSelector[OutT, CtxT]
    ) -> RecipientSelector[OutT, CtxT]:
        self.select_recipients_impl = func

        return func

    @final
    def select_recipients(
        self, output: OutT, ctx: RunContext[CtxT], call_id: str
    ) -> Sequence[ProcName]:
        base_cls = BaseProcessor[Any, Any, Any]
        if is_method_overridden("select_recipients_impl", self, base_cls):
            recipients = self.select_recipients_impl(
                output=output, ctx=ctx, call_id=call_id
            )
            self._validate_recipients(recipients, call_id=call_id)
            return recipients

        # self.processor is not None because otherwise we do not call this method
        return cast("list[ProcName]", self.recipients)

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> list[OutT]:
        """
        Process a list of inputs and return a list of outputs. The length of the
        output list can be different from the input list.
        """
        return cast("list[OutT]", in_args)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        call_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        outputs = await self._process(
            chat_inputs=chat_inputs, in_args=in_args, call_id=call_id, ctx=ctx
        )
        for output in outputs:
            yield ProcPayloadOutEvent(data=output, src_name=self.name, call_id=call_id)

    def _build_packet(
        self, outputs: list[OutT], call_id: str, ctx: RunContext[CtxT]
    ) -> Packet[OutT]:
        for output in outputs:
            self.validate_output(output, call_id=call_id)

        routings: list[Sequence[ProcName]] | None = []
        if self.recipients is not None:
            for output in outputs:
                routings.append(
                    self.select_recipients(output=output, ctx=ctx, call_id=call_id)
                )

        joined_routing = [r for r in routings] if routings else None

        return Packet(sender=self.name, payloads=outputs, routing=joined_routing)

    @workflow(name="processor")  # type: ignore
    @final
    @with_retry
    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        ctx = ctx or RunContext[CtxT](state=None)  # type: ignore
        call_id = self.generate_call_id(call_id)

        val_in_args = self.validate_inputs(
            call_id=call_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

        outputs: list[OutT] = []
        async for event in self._process_stream(
            chat_inputs=chat_inputs,
            in_args=val_in_args,
            call_id=call_id,
            ctx=ctx,
        ):
            if isinstance(event, ProcPayloadOutEvent) and event.src_name == self.name:
                # Collect payloads of the processor's own output events
                # This makes sure subprocessor events are yielded but their
                # payloads are not included in the final output packet
                outputs.append(event.data)
            else:
                yield event

        # 1) Combine payloads
        # 2) Obtain recipients per payload
        # 3) Form a packet
        # 4) Yield a ProcPacketOutEvent
        out_packet = self._build_packet(outputs=outputs, call_id=call_id, ctx=ctx)
        yield ProcPacketOutEvent(
            id=out_packet.id, data=out_packet, src_name=self.name, call_id=call_id
        )

    @final
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        call_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT]:
        async for event in self.run_stream(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            call_id=call_id,
            ctx=ctx,
        ):
            if isinstance(event, ProcPacketOutEvent) and event.src_name == self.name:
                return event.data
        raise RuntimeError("Processor run did not yield a ProcPacketOutputEvent")
