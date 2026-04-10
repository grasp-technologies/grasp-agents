import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from functools import partial
from typing import Any, Generic, Literal
from uuid import uuid4

from grasp_agents.tracing_decorators import workflow

from ..durability.checkpoints import RunnerCheckpoint
from ..packet import Packet
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..types.errors import RunnerError
from ..types.events import Event, ProcPacketOutEvent, RunPacketOutEvent
from ..types.io import OutT
from .event_bus import EventBus

logger = logging.getLogger(__name__)

START_PROC_NAME: Literal["*START*"] = "*START*"
END_PROC_NAME: Literal["*END*"] = "*END*"


class Runner(Generic[OutT, CtxT]):
    def __init__(
        self,
        entry_proc: Processor[Any, Any, CtxT],
        procs: Sequence[Processor[Any, Any, CtxT]],
        ctx: RunContext[CtxT] | None = None,
        name: str | None = None,
        session_id: str | None = None,
        session_metadata: dict[str, Any] | None = None,
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

        valid_destinations = {proc.name for proc in procs} | {END_PROC_NAME}
        for proc in procs:
            for r in proc.recipients or []:
                if r not in valid_destinations:
                    raise RunnerError(
                        f"Processor '{proc.name}' references unknown recipient '{r}'. "
                        f"Valid destinations: {', '.join(sorted(valid_destinations))}"
                    )

        self._name = name or str(uuid4())[:6]

        self._entry_proc = entry_proc
        self._procs = procs
        self._procs_by_name: dict[str, Processor[Any, Any, CtxT]] = {
            proc.name: proc for proc in procs
        }

        self._event_bus = EventBus()

        self._ctx = ctx or RunContext[CtxT](state=None)  # type: ignore

        # Session persistence
        self._session_id: str | None = None
        self._checkpoint_number: int = 0
        self._session_metadata: dict[str, Any] = session_metadata or {}
        self._pending_events: dict[str, ProcPacketOutEvent] = {}
        self._active_sessions: dict[str, str] = {}  # proc_name -> session_id
        self._checkpoint_lock = asyncio.Lock()
        self._resume_event_ids: set[str] = set()  # events from checkpoint

        if session_id:
            self.setup_session(session_id)

    @property
    def name(self) -> str:
        return self._name

    @property
    def ctx(self) -> RunContext[CtxT]:
        return self._ctx

    # --- Session persistence ---

    def setup_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._checkpoint_number = 0
        self._active_sessions = {}
        for proc in self._procs:
            proc_session = f"{session_id}/{proc.name}"
            proc.setup_session(proc_session)
            self._active_sessions[proc.name] = proc_session

    @property
    def _checkpoint_store_key(self) -> str | None:
        if self._session_id is None:
            return None
        return f"runner/{self._session_id}"

    async def _load_checkpoint(self) -> RunnerCheckpoint | None:
        store = self._ctx.store
        if store is None or self._checkpoint_store_key is None:
            return None
        data = await store.load(self._checkpoint_store_key)
        if data is None:
            return None
        try:
            checkpoint = RunnerCheckpoint.model_validate_json(data)
        except Exception:
            logger.warning(
                "Corrupt runner checkpoint %s, starting fresh",
                self._session_id,
                exc_info=True,
            )
            return None
        self._checkpoint_number = checkpoint.checkpoint_number
        logger.info(
            "Loaded runner checkpoint %s (%d pending events)",
            self._session_id,
            len(checkpoint.pending_events),
        )
        return checkpoint

    async def _save_checkpoint(
        self,
        pending_events: dict[str, ProcPacketOutEvent] | None = None,
    ) -> None:
        store = self._ctx.store
        if store is None or self._session_id is None:
            return
        assert self._checkpoint_store_key is not None
        self._checkpoint_number += 1
        events = pending_events if pending_events is not None else self._pending_events
        checkpoint = RunnerCheckpoint(
            session_id=self._session_id,
            processor_name=self._name,
            checkpoint_number=self._checkpoint_number,
            pending_events=list(events.values()),
            active_sessions=dict(self._active_sessions),
        )
        await store.save(
            self._checkpoint_store_key,
            checkpoint.model_dump_json().encode("utf-8"),
        )

    # --- Execution helpers ---

    def _generate_exec_id(self, proc: Processor[Any, Any, CtxT]) -> str | None:
        return self._name + "/" + proc.generate_exec_id(exec_id=None)

    def _make_start_event(self, chat_inputs: Any) -> ProcPacketOutEvent:
        start_packet = Packet[Any](
            sender=START_PROC_NAME,
            routing=[[self._entry_proc.name]],
            payloads=[chat_inputs],
        )
        return ProcPacketOutEvent(
            id=start_packet.id,
            data=start_packet,
            source=START_PROC_NAME,
            destination=self._entry_proc.name,
        )

    def _unpack_packet(
        self, packet: Packet[Any]
    ) -> tuple[Packet[Any] | None, Any | None]:
        if packet.sender == START_PROC_NAME:
            return None, packet.payloads[0]
        return packet, None

    async def _route_output(
        self, out_packet: Packet[Any], *, exec_id: str | None
    ) -> list[ProcPacketOutEvent]:
        """Route output packet. Returns sub-events (empty list for END)."""
        uniform = out_packet.uniform_routing
        if uniform is not None and list(uniform) == [END_PROC_NAME]:
            final_event = RunPacketOutEvent(
                id=out_packet.id,
                data=out_packet,
                source=out_packet.sender,
                destination=END_PROC_NAME,
                exec_id=exec_id,
            )
            await self._event_bus.push_to_stream(final_event)
            await self._event_bus.finalize(final_event.data)
            return []

        sub_events: list[ProcPacketOutEvent] = []
        for sub_out_packet in out_packet.split_by_recipient() or []:
            if not sub_out_packet.routing or not sub_out_packet.routing[0]:
                continue

            dst_name = sub_out_packet.routing[0][0]
            sub_out_event = ProcPacketOutEvent(
                id=sub_out_packet.id,
                data=sub_out_packet,
                source=sub_out_packet.sender,
                destination=dst_name,
                exec_id=exec_id,
            )
            sub_events.append(sub_out_event)
            await self._event_bus.push_to_stream(sub_out_event)
        return sub_events

    async def _checkpoint_transition(
        self, consumed_id: str, produced: list[ProcPacketOutEvent] | None = None
    ) -> None:
        """Atomically update pending events and save checkpoint."""
        async with self._checkpoint_lock:
            new_pending = dict(self._pending_events)
            new_pending.pop(consumed_id, None)
            for e in produced or []:
                new_pending[e.id] = e
            await self._save_checkpoint(new_pending)
            self._pending_events = new_pending

    async def _event_handler(
        self,
        in_event: Event[Any],
        *,
        proc: Processor[Any, Any, CtxT],
        **run_kwargs: Any,
    ) -> None:
        if not (isinstance(in_event, ProcPacketOutEvent)):
            return

        logger.info(f"\n[Running processor {proc.name}]\n")

        in_packet, chat_inputs = self._unpack_packet(in_event.data)
        exec_id = self._generate_exec_id(proc)

        out_packet: Packet[Any] | None = None

        finalized: bool = False

        resume = in_event.id in self._resume_event_ids
        self._resume_event_ids.discard(in_event.id)

        async for out_event in proc.run_stream(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            ctx=self._ctx,
            exec_id=exec_id,
            resume=resume,
            **run_kwargs,
        ):
            if finalized:
                # Need to drain the async generator for OTel to work properly
                continue

            if (
                isinstance(out_event, ProcPacketOutEvent)
                and out_event.source == proc.name
            ):
                out_packet = out_event.data
                sub_events = await self._route_output(out_packet, exec_id=exec_id)
                await self._checkpoint_transition(in_event.id, sub_events)

                if not sub_events:
                    finalized = True
                    continue

                # Post to bus AFTER checkpoint save
                for e in sub_events:
                    await self._event_bus.post(e)

            else:
                await self._event_bus.push_to_stream(out_event)

        if out_packet is None:
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
        # Load checkpoint or create initial pending events
        checkpoint = await self._load_checkpoint()

        if checkpoint is not None:
            self._pending_events = {e.id: e for e in checkpoint.pending_events}
            self._resume_event_ids = set(self._pending_events.keys())
            # Restore proc sessions from checkpoint
            for proc_name, session_id in checkpoint.active_sessions.items():
                proc = self._procs_by_name.get(proc_name)
                if proc is not None:
                    proc.setup_session(session_id)

        else:
            start_event = self._make_start_event(chat_inputs)
            self._pending_events = {start_event.id: start_event}
            await self._save_checkpoint()

        initial_events = list(self._pending_events.values())

        if not initial_events:
            logger.info(
                "Runner %s: no pending events, run already completed", self._name
            )
            return

        async with self._event_bus:
            for proc in self._procs:
                self._event_bus.register_event_handler(
                    dst_name=proc.name,
                    handler=partial(self._event_handler, proc=proc, **run_kwargs),
                )

            for event in initial_events:
                await self._event_bus.post(event)

            async for event in self._event_bus.stream_events():
                yield event

    async def run(self, chat_inputs: Any = "start", **run_kwargs: Any) -> Packet[OutT]:
        async for _ in self.run_stream(chat_inputs=chat_inputs, **run_kwargs):
            pass
        return await self._event_bus.final_result()

    async def shutdown(self) -> None:
        await self._event_bus.shutdown()
