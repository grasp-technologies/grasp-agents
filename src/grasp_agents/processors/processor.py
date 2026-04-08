import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, cast, final

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from grasp_agents.tracing_decorators import workflow
from grasp_agents.types.errors import (
    PacketRoutingError,
    ProcInputValidationError,
    ProcOutputValidationError,
)

from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..types.events import Event, ProcPacketOutEvent, ProcPayloadOutEvent
from ..types.hooks import RecipientSelector
from ..types.io import InT, OutT, ProcName
from ..utils.callbacks import is_method_overridden
from .base_processor import BaseProcessor, with_retry

logger = logging.getLogger(__name__)


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
        exec_id: str,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
    ) -> list[InT] | None:
        err_kwargs = {"proc_name": self.name, "exec_id": exec_id}

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

    def validate_output(self, out_payload: OutT, exec_id: str) -> OutT:
        if out_payload is None:
            return out_payload

        try:
            return TypeAdapter(self.out_type).validate_python(out_payload)
        except PydanticValidationError as err:
            raise ProcOutputValidationError(
                schema=self.out_type,
                proc_name=self.name,
                exec_id=exec_id,
            ) from err

    def _validate_recipients(
        self, recipients: Sequence[ProcName] | None, exec_id: str
    ) -> None:
        for r in recipients or []:
            if r not in (self.recipients or []):
                raise PacketRoutingError(
                    proc_name=self.name,
                    exec_id=exec_id,
                    selected_recipient=r,
                    allowed_recipients=cast("list[str]", self.recipients),
                )

    def select_recipients_impl(
        self, output: OutT, *, ctx: RunContext[CtxT], exec_id: str
    ) -> Sequence[ProcName]:
        raise NotImplementedError

    def add_recipient_selector(
        self, func: RecipientSelector[OutT, CtxT]
    ) -> RecipientSelector[OutT, CtxT]:
        self.select_recipients_impl = func

        return func

    @final
    def select_recipients(
        self, output: OutT, ctx: RunContext[CtxT], exec_id: str
    ) -> Sequence[ProcName]:
        base_cls = BaseProcessor[Any, Any, Any]
        if is_method_overridden("select_recipients_impl", self, base_cls):
            recipients = self.select_recipients_impl(
                output=output, ctx=ctx, exec_id=exec_id
            )
            self._validate_recipients(recipients, exec_id=exec_id)
            return recipients

        return cast("list[ProcName]", self.recipients)

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        exec_id: str,
        ctx: RunContext[CtxT],
    ) -> list[OutT]:
        """
        Process a list of inputs and return a list of outputs. The length of
        the output list can be different from the input list.
        """
        return cast("list[OutT]", in_args)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        exec_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        outputs = await self._process(
            chat_inputs=chat_inputs,
            in_args=in_args,
            exec_id=exec_id,
            ctx=ctx,
        )
        for output in outputs:
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)

    def _build_packet(
        self,
        outputs: list[OutT],
        exec_id: str,
        ctx: RunContext[CtxT],
    ) -> Packet[OutT]:
        for output in outputs:
            self.validate_output(output, exec_id=exec_id)

        routings: list[Sequence[ProcName]] | None = []
        if self.recipients is not None:
            for output in outputs:
                routings.append(
                    self.select_recipients(output=output, ctx=ctx, exec_id=exec_id)
                )

        joined_routing = [r for r in routings] if routings else None

        return Packet(sender=self.name, payloads=outputs, routing=joined_routing)

    @final
    @workflow(name="processor")  # type: ignore
    @with_retry
    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        exec_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        ctx = ctx or RunContext[CtxT](state=None)  # type: ignore
        exec_id = self.generate_exec_id(exec_id)

        val_in_args = self.validate_inputs(
            exec_id=exec_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

        outputs: list[OutT] = []
        async for event in self._process_stream(
            chat_inputs=chat_inputs,
            in_args=val_in_args,
            exec_id=exec_id,
            ctx=ctx,
        ):
            if isinstance(event, ProcPayloadOutEvent) and event.source == self.name:
                outputs.append(event.data)
            else:
                yield event

        out_packet = self._build_packet(outputs=outputs, exec_id=exec_id, ctx=ctx)
        yield ProcPacketOutEvent(
            id=out_packet.id,
            data=out_packet,
            source=self.name,
            exec_id=exec_id,
        )

    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        exec_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT]:
        result = None

        async for event in self.run_stream(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            exec_id=exec_id,
            ctx=ctx,
        ):
            if result is not None:
                continue

            if isinstance(event, ProcPacketOutEvent) and event.source == self.name:
                result = event.data

        if result is None:
            raise RuntimeError("Processor run did not yield a ProcPacketOutputEvent")

        return result
