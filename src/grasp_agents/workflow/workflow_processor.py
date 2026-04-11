import logging
from abc import ABC
from collections.abc import Sequence
from typing import Any, cast

from ..durability.checkpoints import WorkflowCheckpoint
from ..packet import Packet
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..telemetry import SpanKind
from ..types.errors import WorkflowConstructionError
from ..types.io import InT, OutT, ProcName
from ..utils.callbacks import is_method_overridden

logger = logging.getLogger(__name__)


class WorkflowProcessor(Processor[InT, OutT, CtxT], ABC):
    _span_kind = SpanKind.WORKFLOW

    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, CtxT]],
        start_proc: Processor[InT, Any, CtxT],
        end_proc: Processor[Any, OutT, CtxT],
        recipients: Sequence[ProcName] | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
        session_id: str | None = None,
        session_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            recipients=(recipients or end_proc.recipients),
            max_retries=0,
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
            session_id=session_id,
            session_metadata=session_metadata,
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

        if self._session_id:
            self.setup_session(self._session_id)

    def setup_session(self, session_id: str) -> None:
        super().setup_session(session_id)
        for subproc in self._subprocs:
            subproc.setup_session(f"{session_id}/{subproc.name}")

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

    async def load_checkpoint(self, ctx: RunContext[CtxT]) -> WorkflowCheckpoint | None:
        checkpoint = await self._deserialize_checkpoint(ctx, WorkflowCheckpoint)
        if checkpoint is not None:
            logger.info(
                "Loaded workflow checkpoint %s (completed_step=%d)",
                self._session_id,
                checkpoint.completed_step,
            )
        return checkpoint

    async def save_checkpoint(
        self,
        ctx: RunContext[CtxT],
        *,
        completed_step: int,
        packet: Packet[Any],
    ) -> None:
        checkpoint = WorkflowCheckpoint(
            session_id=self._session_id or "",
            processor_name=self.name,
            completed_step=completed_step,
            packet=packet,
        )
        await self._serialize_checkpoint(ctx, checkpoint)

    def select_recipients_impl(
        self, output: OutT, *, ctx: RunContext[CtxT], exec_id: str
    ) -> Sequence[ProcName]:
        if is_method_overridden("select_recipients_impl", self._end_proc, Processor):
            return self._end_proc.select_recipients_impl(
                output=output, ctx=ctx, exec_id=exec_id
            )
        return cast("list[ProcName]", self.recipients or [])

    @property
    def subprocs(self) -> Sequence[Processor[Any, Any, CtxT]]:
        return self._subprocs

    @property
    def start_proc(self) -> Processor[InT, Any, CtxT]:
        return self._start_proc

    @property
    def end_proc(self) -> Processor[Any, OutT, CtxT]:
        return self._end_proc
