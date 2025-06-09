import logging
from collections.abc import Sequence
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast

from pydantic import BaseModel
from pydantic.json_schema import SkipJsonSchema

from .packet import Packet
from .packet_pool import PacketPool
from .processor import Processor
from .run_context import CtxT, RunContext
from .typing.io import InT_contra, MemT_co, OutT_co, ProcessorName

logger = logging.getLogger(__name__)


class DynCommPayload(BaseModel):
    selected_recipients: SkipJsonSchema[Sequence[ProcessorName]]


_OutT_contra = TypeVar("_OutT_contra", contravariant=True)


class ExitCommunicationHandler(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self,
        out_packet: Packet[_OutT_contra],
        ctx: RunContext[CtxT] | None,
    ) -> bool: ...


class CommProcessor(
    Processor[InT_contra, OutT_co, MemT_co, CtxT],
    Generic[InT_contra, OutT_co, MemT_co, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcessorName,
        *,
        recipients: Sequence[ProcessorName] | None = None,
        packet_pool: PacketPool[CtxT] | None = None,
    ) -> None:
        super().__init__(name=name)

        self.recipients = recipients or []

        self._packet_pool = packet_pool or PacketPool()
        self._is_listening = False
        self._exit_communication_impl: (
            ExitCommunicationHandler[OutT_co, CtxT] | None
        ) = None

    def _validate_routing(self, payloads: Sequence[OutT_co]) -> Sequence[ProcessorName]:
        if all(isinstance(p, DynCommPayload) for p in payloads):
            payloads_ = cast("Sequence[DynCommPayload]", payloads)
            selected_recipients_per_payload = [
                set(p.selected_recipients or []) for p in payloads_
            ]
            assert all(
                x == selected_recipients_per_payload[0]
                for x in selected_recipients_per_payload
            ), "All payloads must have the same recipient IDs for dynamic routing"

            assert payloads_[0].selected_recipients is not None
            selected_recipients = payloads_[0].selected_recipients

            assert all(rid in self.recipients for rid in selected_recipients), (
                "Dynamic routing is enabled, but recipient IDs are not in "
                "the allowed agent's recipient IDs"
            )

            return selected_recipients

        if all((not isinstance(p, DynCommPayload)) for p in payloads):
            return self.recipients

        raise ValueError(
            "All payloads must be either DCommAgentPayload or not DCommAgentPayload"
        )

    async def post_packet(self, packet: Packet[OutT_co]) -> None:
        self._validate_routing(packet.payloads)

        await self._packet_pool.post(packet)

    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT_contra] | None = None,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        entry_point: bool = False,
        forgetful: bool = True,
        ctx: RunContext[CtxT] | None = None,
    ) -> Packet[OutT_co]:
        out_packet = await super().run(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            entry_point=entry_point,
            ctx=ctx,
        )
        recipients = self._validate_routing(out_packet.payloads)

        return Packet(
            payloads=out_packet.payloads, sender=self.name, recipients=recipients
        )

    async def run_and_post(
        self, ctx: RunContext[CtxT] | None = None, **kwargs: Any
    ) -> None:
        out_packet = await self.run(ctx=ctx, in_packet=None, entry_point=True, **kwargs)
        await self.post_packet(out_packet)

    def exit_communication(
        self, func: ExitCommunicationHandler[OutT_co, CtxT]
    ) -> ExitCommunicationHandler[OutT_co, CtxT]:
        self._exit_communication_impl = func

        return func

    def _exit_communication_fn(
        self, out_packet: Packet[OutT_co], ctx: RunContext[CtxT] | None
    ) -> bool:
        if self._exit_communication_impl:
            return self._exit_communication_impl(out_packet=out_packet, ctx=ctx)

        return False

    async def _packet_handler(
        self,
        packet: Packet[Any],
        ctx: RunContext[CtxT] | None = None,
        **run_kwargs: Any,
    ) -> None:
        in_packet = cast("Packet[InT_contra]", packet)
        out_packet = await self.run(ctx=ctx, in_packet=in_packet, **run_kwargs)

        if self._exit_communication_fn(out_packet=out_packet, ctx=ctx):
            await self._packet_pool.stop_all()
            return

        if self.recipients:
            await self.post_packet(out_packet)

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    async def start_listening(
        self, ctx: RunContext[CtxT] | None = None, **run_kwargs: Any
    ) -> None:
        if self._is_listening:
            return

        self._is_listening = True
        self._packet_pool.register_packet_handler(
            processor_name=self.name,
            handler=self._packet_handler,
            ctx=ctx,
            **run_kwargs,
        )

    async def stop_listening(self) -> None:
        self._is_listening = False
        await self._packet_pool.unregister_packet_handler(self.name)
