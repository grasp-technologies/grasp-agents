import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast

from .errors import PacketRoutingError
from .memory import MemT
from .packet import Packet
from .packet_pool import PacketPool
from .processor import Processor
from .run_context import CtxT, RunContext
from .typing.events import Event, ProcPacketOutputEvent, RunResultEvent
from .typing.io import InT, OutT_co, ProcName

logger = logging.getLogger(__name__)


_OutT_contra = TypeVar("_OutT_contra", contravariant=True)


class ExitCommunicationHandler(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self,
        out_packet: Packet[_OutT_contra],
        ctx: RunContext[CtxT],
    ) -> bool: ...


class SetRecipientsHandler(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self, out_packet: Packet[_OutT_contra], ctx: RunContext[CtxT]
    ) -> None: ...


class CommProcessor(
    Processor[InT, OutT_co, MemT, CtxT],
    Generic[InT, OutT_co, MemT, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcName,
        *,
        recipients: Sequence[ProcName] | None = None,
        packet_pool: PacketPool[CtxT] | None = None,
        max_retries: int = 0,
    ) -> None:
        super().__init__(name=name, max_retries=max_retries)

        self.recipients = recipients or []
        self._packet_pool = packet_pool
        self._is_listening = False

        self._exit_communication_impl: (
            ExitCommunicationHandler[OutT_co, CtxT] | None
        ) = None
        self._set_recipients_impl: SetRecipientsHandler[OutT_co, CtxT] | None = None

    @property
    def packet_pool(self) -> PacketPool[CtxT] | None:
        return self._packet_pool

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    def _set_recipients(
        self, out_packet: Packet[OutT_co], ctx: RunContext[CtxT]
    ) -> None:
        if self._set_recipients_impl:
            self._set_recipients_impl(out_packet=out_packet, ctx=ctx)
            return

        out_packet.recipients = self.recipients

    def _validate_routing(self, recipients: Sequence[ProcName]) -> Sequence[ProcName]:
        for r in recipients:
            if r not in self.recipients:
                raise PacketRoutingError(
                    selected_recipient=r,
                    allowed_recipients=cast("list[str]", self.recipients),
                )

        return self.recipients

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

        out_packet = await super().run(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            forgetful=forgetful,
            call_id=call_id,
            ctx=ctx,
        )

        if self._packet_pool is not None:
            if ctx is None:
                raise ValueError("RunContext must be provided when using PacketPool")
            if self._exit_communication(out_packet=out_packet, ctx=ctx):
                ctx.result = out_packet
                await self._packet_pool.stop_all()
                return out_packet

            self._set_recipients(out_packet=out_packet, ctx=ctx)
            out_packet.recipients = self._validate_routing(out_packet.recipients)

            await self._packet_pool.post(out_packet)

        return out_packet

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

        out_packet: Packet[OutT_co] | None = None
        async for event in super().run_stream(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            forgetful=forgetful,
            call_id=call_id,
            ctx=ctx,
        ):
            if isinstance(event, ProcPacketOutputEvent):
                out_packet = event.data
            else:
                yield event

        if out_packet is None:
            return

        if self._packet_pool is not None:
            if ctx is None:
                raise ValueError("RunContext must be provided when using PacketPool")
            if self._exit_communication(out_packet=out_packet, ctx=ctx):
                ctx.result = out_packet
                yield RunResultEvent(
                    data=out_packet, proc_name=self.name, call_id=call_id
                )
                await self._packet_pool.stop_all()
                return

            self._set_recipients(out_packet=out_packet, ctx=ctx)
            out_packet.recipients = self._validate_routing(out_packet.recipients)

            await self._packet_pool.post(out_packet)

        yield ProcPacketOutputEvent(
            data=out_packet, proc_name=self.name, call_id=call_id
        )

    def start_listening(self, ctx: RunContext[CtxT], **run_kwargs: Any) -> None:
        if self._packet_pool is None:
            raise RuntimeError("Packet pool must be initialized before listening")

        if self._is_listening:
            return
        self._is_listening = True

        self._packet_pool.register_packet_handler(
            processor_name=self.name,
            handler=self.run_stream if ctx.is_streaming else self.run,  # type: ignore[call-arg]
            ctx=ctx,
            **run_kwargs,
        )

    def _exit_communication(
        self, out_packet: Packet[OutT_co], ctx: RunContext[CtxT]
    ) -> bool:
        if self._exit_communication_impl:
            return self._exit_communication_impl(out_packet=out_packet, ctx=ctx)

        return False

    def exit_communication(
        self, func: ExitCommunicationHandler[OutT_co, CtxT]
    ) -> ExitCommunicationHandler[OutT_co, CtxT]:
        self._exit_communication_impl = func

        return func

    def set_recipients(
        self, func: SetRecipientsHandler[OutT_co, CtxT]
    ) -> SetRecipientsHandler[OutT_co, CtxT]:
        self._select_recipients_impl = func

        return func

    # async def stop_listening(self) -> None:
    #     assert self._packet_pool is not None
    #     self._is_listening = False
    #     await self._packet_pool.unregister_packet_handler(self.name)
