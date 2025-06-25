import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast

from pydantic import BaseModel
from pydantic.json_schema import SkipJsonSchema

from .memory import MemT
from .packet import Packet
from .packet_pool import PacketPool
from .processor import Processor
from .run_context import CtxT, RunContext
from .typing.events import Event, PacketEvent
from .typing.io import InT, OutT_co, ProcName

logger = logging.getLogger(__name__)


class DynCommPayload(BaseModel):
    selected_recipients: SkipJsonSchema[Sequence[ProcName]]


_OutT_contra = TypeVar("_OutT_contra", contravariant=True)


class ExitCommunicationHandler(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self,
        out_packet: Packet[_OutT_contra],
        ctx: RunContext[CtxT] | None,
    ) -> bool: ...


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
        num_par_run_retries: int = 0,
    ) -> None:
        super().__init__(name=name, num_par_run_retries=num_par_run_retries)

        self.recipients = recipients or []

        self._packet_pool = packet_pool
        self._is_listening = False
        self._exit_communication_impl: (
            ExitCommunicationHandler[OutT_co, CtxT] | None
        ) = None

    @property
    def packet_pool(self) -> PacketPool[CtxT] | None:
        return self._packet_pool

    def _validate_routing(self, payloads: Sequence[OutT_co]) -> Sequence[ProcName]:
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
        out_packet = await super().run(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            forgetful=forgetful,
            run_id=run_id,
            ctx=ctx,
        )
        recipients = self._validate_routing(out_packet.payloads)
        routed_out_packet = Packet(
            payloads=out_packet.payloads, sender=self.name, recipients=recipients
        )
        if self._packet_pool is not None and in_packet is None and in_args is None:
            # If no input packet or args, we assume this is the first run.
            await self._packet_pool.post(routed_out_packet)

        return routed_out_packet

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
        out_packet: Packet[OutT_co] | None = None
        async for event in super().run_stream(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            forgetful=forgetful,
            run_id=run_id,
            ctx=ctx,
        ):
            if isinstance(event, PacketEvent):
                out_packet = event.data
            else:
                yield event

        if out_packet is None:
            raise RuntimeError("No output packet generated during stream run")

        recipients = self._validate_routing(out_packet.payloads)
        routed_out_packet = Packet(
            payloads=out_packet.payloads, sender=self.name, recipients=recipients
        )
        if self._packet_pool is not None and in_packet is None and in_args is None:
            # If no input packet or args, we assume this is the first run.
            await self._packet_pool.post(routed_out_packet)

        yield PacketEvent(data=routed_out_packet, name=self.name)

    def exit_communication(
        self, func: ExitCommunicationHandler[OutT_co, CtxT]
    ) -> ExitCommunicationHandler[OutT_co, CtxT]:
        self._exit_communication_impl = func

        return func

    def _exit_communication(
        self, out_packet: Packet[OutT_co], ctx: RunContext[CtxT] | None
    ) -> bool:
        if self._exit_communication_impl:
            return self._exit_communication_impl(out_packet=out_packet, ctx=ctx)

        return False

    async def _packet_handler(
        self,
        packet: Packet[InT],
        ctx: RunContext[CtxT] | None = None,
        **run_kwargs: Any,
    ) -> None:
        assert self._packet_pool is not None, "Packet pool must be initialized"

        out_packet = await self.run(ctx=ctx, in_packet=packet, **run_kwargs)

        if self._exit_communication(out_packet=out_packet, ctx=ctx):
            await self._packet_pool.stop_all()
            return

        await self._packet_pool.post(out_packet)

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    async def start_listening(
        self, ctx: RunContext[CtxT] | None = None, **run_kwargs: Any
    ) -> None:
        assert self._packet_pool is not None, "Packet pool must be initialized"

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
        assert self._packet_pool is not None, "Packet pool must be initialized"

        self._is_listening = False
        await self._packet_pool.unregister_packet_handler(self.name)
