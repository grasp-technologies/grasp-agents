from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from ..errors import WorkflowConstructionError
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..typing.events import DummyEvent, Event
from ..typing.io import InT, OutT, ProcName


class WorkflowProcessor(Processor[InT, OutT, CtxT], ABC):
    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[Processor[Any, Any, CtxT]],
        start_proc: Processor[InT, Any, CtxT],
        end_proc: Processor[Any, OutT, CtxT],
        recipients: Sequence[ProcName] | None = None,
    ) -> None:
        super().__init__(
            name=name, recipients=(recipients or end_proc.recipients), max_retries=0
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

    def select_recipients_impl(
        self, output: OutT, *, ctx: RunContext[CtxT], call_id: str
    ) -> Sequence[ProcName]:
        return self._end_proc.select_recipients_impl(
            output=output, ctx=ctx, call_id=call_id
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

    @abstractmethod
    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> list[OutT]:
        pass

    @abstractmethod
    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> AsyncIterator[Event[Any]]:
        yield DummyEvent()
