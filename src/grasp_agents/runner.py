import logging
from collections.abc import AsyncIterator, Sequence
from functools import partial
from typing import Any, Generic, Literal
from uuid import uuid4

from grasp_agents.tracing_decorators import workflow

from .errors import RunnerError
from .event_bus import EventBus
from .packet import Packet
from .processors.base_processor import BaseProcessor
from .run_context import CtxT, RunContext
from .typing.events import Event, ProcPacketOutEvent, RunPacketOutEvent
from .typing.io import OutT

logger = logging.getLogger(__name__)

START_PROC_NAME: Literal["*START*"] = "*START*"
END_PROC_NAME: Literal["*END*"] = "*END*"


class Runner(Generic[OutT, CtxT]):
    def __init__(
        self,
        entry_proc: BaseProcessor[Any, Any, CtxT],
        procs: Sequence[BaseProcessor[Any, Any, CtxT]],
        ctx: RunContext[CtxT] | None = None,
        name: str | None = None,
    ) -> None:
        if entry_proc not in procs:
            raise RunnerError(
                f"Entry processor {entry_proc.name} must be in the list of processors: "
                f"{', '.join(proc.name for proc in procs)}"
            )
        if sum(1 for proc in procs if END_PROC_NAME in (proc.recipients or [])) != 1:
            raise RunnerError(
                "There must be exactly one processor with recipient 'END'."
            )

        self._name = name or str(uuid4())[:6]

        self._entry_proc = entry_proc
        self._procs = procs

        self._event_bus = EventBus()

        self._ctx = ctx or RunContext[CtxT](state=None)  # type: ignore

    @property
    def name(self) -> str:
        return self._name

    @property
    def ctx(self) -> RunContext[CtxT]:
        return self._ctx

    def _generate_call_id(self, proc: BaseProcessor[Any, Any, CtxT]) -> str | None:
        return self._name + "/" + proc.generate_call_id(call_id=None)

    def _make_start_event(self, chat_inputs: Any) -> ProcPacketOutEvent:
        start_packet = Packet[Any](
            sender=START_PROC_NAME,
            routing=[[self._entry_proc.name]],
            payloads=[chat_inputs],
        )
        return ProcPacketOutEvent(
            id=start_packet.id,
            data=start_packet,
            src_name=START_PROC_NAME,
            dst_name=self._entry_proc.name,
            call_id=None,
        )

    def _unpack_packet(
        self, packet: Packet[Any]
    ) -> tuple[Packet[Any] | None, Any | None]:
        if packet.sender == START_PROC_NAME:
            return None, packet.payloads[0]
        return packet, None

    async def _event_handler(
        self,
        in_event: Event[Any],
        *,
        proc: BaseProcessor[Any, Any, CtxT],
        **run_kwargs: Any,
    ) -> None:
        if not (isinstance(in_event, ProcPacketOutEvent)):
            # Currently, we only handle ProcResultEvent in the runner
            # TODO: User handoffs and MCP tool calls
            return

        logger.info(f"\n[Running processor {proc.name}]\n")

        in_packet, chat_inputs = self._unpack_packet(in_event.data)
        call_id = self._generate_call_id(proc)
        out_packet: Packet[Any] | None = None

        async for out_event in proc.run_stream(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            ctx=self._ctx,
            call_id=call_id,
            **run_kwargs,
        ):
            if (
                isinstance(out_event, ProcPacketOutEvent)
                and out_event.src_name == proc.name
            ):
                out_packet = out_event.data

                if out_packet.routing == [[END_PROC_NAME]]:
                    final_event = RunPacketOutEvent(
                        id=out_packet.id,
                        data=out_packet,
                        src_name=out_packet.sender,
                        dst_name=END_PROC_NAME,
                        call_id=call_id,
                    )
                    await self._event_bus.push_to_stream(final_event)
                    await self._event_bus.finalize(final_event.data)
                    break

                for sub_out_packet in out_packet.split_by_recipient() or []:
                    if not sub_out_packet.routing or not sub_out_packet.routing[0]:
                        continue

                    dst_name = sub_out_packet.routing[0][0]
                    sub_out_event = ProcPacketOutEvent(
                        id=sub_out_packet.id,
                        data=sub_out_packet,
                        src_name=sub_out_packet.sender,
                        dst_name=dst_name,
                        call_id=call_id,
                    )
                    await self._event_bus.push_to_stream(sub_out_event)
                    await self._event_bus.post(sub_out_event)

            else:
                await self._event_bus.push_to_stream(out_event)

        if out_packet is None:
            # Cannot happen, for type checking purposes
            return

        route = out_packet.uniform_routing or out_packet.routing
        logger.info(
            f"\n[Finished running processor {proc.name}]\n"
            f"Posted output packet to recipients: {route}\n"
        )

    @workflow(name="runner_run")  # type: ignore
    async def run_stream(
        self, chat_inputs: Any = "start", **run_kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        async with self._event_bus:
            for proc in self._procs:
                self._event_bus.register_event_handler(
                    dst_name=proc.name,
                    handler=partial(self._event_handler, proc=proc, **run_kwargs),
                )

            await self._event_bus.post(self._make_start_event(chat_inputs))

            async for event in self._event_bus.stream_events():
                yield event

    async def run(self, chat_inputs: Any = "start", **run_kwargs: Any) -> Packet[OutT]:
        async for _ in self.run_stream(chat_inputs=chat_inputs, **run_kwargs):
            pass
        return await self._event_bus.final_result()

    async def shutdown(self) -> None:
        await self._event_bus.shutdown()
