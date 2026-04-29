import logging
from abc import ABC
from collections.abc import Sequence
from typing import Any, cast

from ..durability.checkpoints import CheckpointKind, WorkflowCheckpoint
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
    _checkpoint_kind = CheckpointKind.WORKFLOW

    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, CtxT]],
        start_proc: Processor[InT, Any, CtxT],
        end_proc: Processor[Any, OutT, CtxT],
        recipients: Sequence[ProcName] | None = None,
        path: list[str] | None = None,
        session_metadata: dict[str, Any] | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
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

        # Need to set _subprocs before __init__ because it
        # executes _propagate_to_children which calls subproc.on_adopted
        self._subprocs = subprocs
        self._start_proc = start_proc
        self._end_proc = end_proc

        super().__init__(
            name=name,
            recipients=(recipients or end_proc.recipients),
            max_retries=0,
            path=path,
            session_metadata=session_metadata,
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
        )

        self._in_type = start_proc.in_type
        self._out_type = end_proc.out_type

        # TODO: Is it still needed?
        for subproc in subprocs:
            subproc.recipients = None

    def _propagate_to_children(self) -> None:
        for subproc in self._subprocs:
            subproc.on_adopted(self._path)

    def validate_inputs(
        self,
        exec_id: str,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> list[InT] | None:
        has_input = any(x is not None for x in [chat_inputs, in_args, in_packet])
        if not has_input and Processor.is_resumable(ctx):
            return None

        return super().validate_inputs(
            exec_id=exec_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            ctx=ctx,
        )

    async def load_checkpoint(self, ctx: RunContext[CtxT]) -> WorkflowCheckpoint | None:
        checkpoint = await self._deserialize_checkpoint(ctx, WorkflowCheckpoint)
        if checkpoint is not None:
            logger.info(
                "Loaded workflow checkpoint %s (completed_step=%d)",
                self._checkpoint_store_key(ctx),
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
            session_key=ctx.session_key,
            processor_name=self.name,
            session_metadata=self._session_metadata,
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
