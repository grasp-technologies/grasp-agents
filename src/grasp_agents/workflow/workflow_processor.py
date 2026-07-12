import logging
from abc import ABC
from collections.abc import Sequence
from typing import Any, cast

from grasp_agents.durability.checkpoints import CheckpointKind, WorkflowCheckpoint
from grasp_agents.processors.processor import Processor
from grasp_agents.session_context import SessionContext
from grasp_agents.telemetry import SpanKind
from grasp_agents.types.errors import WorkflowConstructionError
from grasp_agents.types.io import ProcName
from grasp_agents.types.packet import Packet
from grasp_agents.utils.callbacks import is_method_overridden

logger = logging.getLogger(__name__)


class WorkflowProcessor[InT, OutT, CtxT](Processor[InT, OutT, CtxT], ABC):
    _span_kind = SpanKind.WORKFLOW
    _checkpoint_kind = CheckpointKind.WORKFLOW

    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, CtxT]],
        start_proc: Processor[InT, Any, CtxT],
        end_proc: Processor[Any, OutT, CtxT],
        *,
        ctx: SessionContext[CtxT] | None = None,
        recipients: Sequence[ProcName] | None = None,
        path: list[str] | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
        durability_enabled: bool = True,
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

        # Each subprocessor's output is captured by ``source == subproc.name``,
        # so names must be unique — and none may equal the workflow's own name,
        # which it emits its output under (that would alias the two).
        subproc_names = [s.name for s in subprocs]
        dup_subprocs = sorted({n for n in subproc_names if subproc_names.count(n) > 1})
        if dup_subprocs:
            raise WorkflowConstructionError(
                f"Duplicate subprocessor names: {dup_subprocs}; names must be unique."
            )
        if name in subproc_names:
            raise WorkflowConstructionError(
                f"Subprocessor name '{name}' collides with the workflow's own name; "
                "names must be unique."
            )

        # Need to set _subprocs before __init__ because it
        # executes _propagate_to_children which calls subproc.on_adopted
        self._subprocs = subprocs
        self._start_proc = start_proc
        self._end_proc = end_proc

        super().__init__(
            name=name,
            ctx=ctx,
            recipients=(recipients or end_proc.recipients),
            max_retries=0,
            path=path,
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
            durability_enabled=durability_enabled,
        )

        self._in_type = start_proc.in_type
        self._out_type = end_proc.out_type
        self._handoff_refusal_warned = False

    def _propagate_to_children(self) -> None:
        for subproc in self._subprocs:
            subproc.on_adopted(self)

    def _hand_over_session_writer(self, subproc: Processor[Any, Any, CtxT]) -> None:
        """
        Hand the session-writer role to the node about to run: nodes run one
        at a time, so each node's boundaries are session-wide frontiers while
        it holds the role. Top-level workflows only (a nested one may run
        concurrently with siblings), and never taken from an outside owner —
        only from this workflow's own nodes.
        """
        if self._contained or not self._ctx.session_record_enabled:
            return
        writer = self._ctx.session_writer
        if writer is None or writer in {p.name for p in self._subprocs}:
            self._ctx.session_writer = subproc.name
        elif not self._handoff_refusal_warned:
            self._handoff_refusal_warned = True
            logger.warning(
                "Workflow %s will not persist session state: %r already owns "
                "session persistence and is not one of this workflow's nodes. "
                "Declare SessionContext(session_writer=...) if this workflow "
                "should own it.",
                self.name,
                writer,
            )

    def validate_inputs(
        self,
        exec_id: str,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
    ) -> list[InT] | None:
        has_input = any(x is not None for x in [chat_inputs, in_args, in_packet])
        if not has_input and self.is_resumable:
            return None

        return super().validate_inputs(
            exec_id=exec_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

    async def load_checkpoint(self) -> WorkflowCheckpoint | None:
        checkpoint = await self._deserialize_checkpoint(self._ctx, WorkflowCheckpoint)
        if checkpoint is not None:
            logger.info(
                "Loaded workflow checkpoint %s (completed_step=%d)",
                self._checkpoint_store_key(self._ctx),
                checkpoint.completed_step,
            )
        return checkpoint

    async def save_checkpoint(
        self,
        *,
        completed_step: int,
        packet: Packet[Any],
    ) -> None:
        checkpoint = WorkflowCheckpoint(
            session_key=self._ctx.session_key,
            processor_name=self.name,
            completed_step=completed_step,
            packet=packet,
        )
        await self._serialize_checkpoint(self._ctx, checkpoint)

    def select_recipients_impl(
        self, output: OutT, *, exec_id: str
    ) -> Sequence[ProcName]:
        if is_method_overridden("select_recipients_impl", self._end_proc, Processor):
            return self._end_proc.select_recipients_impl(output=output, exec_id=exec_id)
        return cast("list[ProcName]", self.recipients or [])

    async def aclose(self) -> None:
        """Cascade session teardown to every subprocessor."""
        for subproc in self._subprocs:
            try:
                await subproc.aclose()
            except Exception:
                logger.warning(
                    "Failed to close subprocessor %r during workflow teardown",
                    subproc.name,
                    exc_info=True,
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
