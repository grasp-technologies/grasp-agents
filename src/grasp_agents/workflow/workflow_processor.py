import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from ..durability.checkpoints import WorkflowCheckpoint
from ..packet import Packet
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..types.errors import WorkflowConstructionError
from ..types.events import DummyEvent, Event, ProcPayloadOutEvent
from ..types.io import InT, OutT, ProcName

logger = logging.getLogger(__name__)


class WorkflowProcessor(Processor[InT, OutT, CtxT], ABC):
    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, CtxT]],
        start_proc: Processor[InT, Any, CtxT],
        end_proc: Processor[Any, OutT, CtxT],
        recipients: Sequence[ProcName] | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            recipients=(recipients or end_proc.recipients),
            max_retries=0,
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
        )

        self._in_type = start_proc.in_type
        self._out_type = end_proc.out_type

        if len(subprocs) < 2:
            raise WorkflowConstructionError("At least two subprocessors are required")
        if start_proc not in subprocs:
            raise WorkflowConstructionError(
                "Start subprocessor must be in the subprocessors list"
            )
        if end_proc not in subprocs:
            raise WorkflowConstructionError(
                "End subprocessor must be in the subprocessors list"
            )

        self._subprocs = subprocs
        self._start_proc = start_proc
        self._end_proc = end_proc

        for subproc in subprocs:
            subproc.recipients = None

        # Session persistence (set via reset_session)
        self._session_id: str | None = None

    # --- Session persistence ---

    @property
    def resumable(self) -> bool:
        return True

    def reset_session(self, session_id: str) -> None:
        self._session_id = session_id

    def validate_inputs(
        self,
        exec_id: str,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
    ) -> list[InT] | None:
        has_input = any(x is not None for x in [chat_inputs, in_args, in_packet])
        if not has_input and self._session_id is not None:
            return None
        return super().validate_inputs(
            exec_id=exec_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

    @property
    def _checkpoint_store_key(self) -> str | None:
        if self._session_id is None:
            return None
        return f"workflow/{self._session_id}"

    async def _load_checkpoint(
        self, ctx: RunContext[CtxT]
    ) -> WorkflowCheckpoint | None:
        store = ctx.store
        if store is None or self._checkpoint_store_key is None:
            return None
        data = await store.load(self._checkpoint_store_key)
        if data is None:
            return None
        checkpoint = WorkflowCheckpoint.model_validate_json(data)
        logger.info(
            "Loaded workflow checkpoint %s (step=%d, iter=%d)",
            self._session_id,
            checkpoint.completed_step,
            checkpoint.iteration,
        )
        return checkpoint

    async def _save_checkpoint(
        self,
        ctx: RunContext[CtxT],
        *,
        completed_step: int,
        packet: Packet[Any],
        iteration: int = 0,
    ) -> None:
        store = ctx.store
        if store is None or self._session_id is None:
            return
        assert self._checkpoint_store_key is not None
        checkpoint = WorkflowCheckpoint(
            session_id=self._session_id,
            processor_name=self.name,
            completed_step=completed_step,
            iteration=iteration,
            packet=packet.model_dump(),
        )
        await store.save(
            self._checkpoint_store_key,
            checkpoint.model_dump_json().encode("utf-8"),
        )

    # --- Routing ---

    def select_recipients_impl(
        self, output: OutT, *, ctx: RunContext[CtxT], exec_id: str
    ) -> Sequence[ProcName]:
        return self._end_proc.select_recipients_impl(
            output=output, ctx=ctx, exec_id=exec_id
        )

    @property
    def subprocs(self) -> Sequence[Processor[Any, Any, CtxT]]:
        return self._subprocs

    @property
    def start_proc(self) -> Processor[InT, Any, CtxT]:
        return self._start_proc

    @property
    def end_proc(self) -> Processor[Any, OutT, CtxT]:
        return self._end_proc

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> list[OutT]:
        outputs: list[OutT] = []
        async for event in self._process_stream(
            chat_inputs=chat_inputs,
            in_args=in_args,
            ctx=ctx,
            exec_id=exec_id,
        ):
            if isinstance(event, ProcPayloadOutEvent) and event.source == self.name:
                outputs.append(event.data)
        return outputs

    @abstractmethod
    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> AsyncIterator[Event[Any]]:
        yield DummyEvent()
