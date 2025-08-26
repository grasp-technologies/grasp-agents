from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any, Generic

from ..errors import WorkflowConstructionError
from ..packet import Packet
from ..processors.base_processor import BaseProcessor, RecipientSelector
from ..run_context import CtxT, RunContext
from ..typing.events import DummyEvent, Event
from ..typing.io import InT, OutT, ProcName


class WorkflowProcessor(
    BaseProcessor[InT, OutT, Any, CtxT],
    ABC,
    Generic[InT, OutT, CtxT],
):
    def __init__(
        self,
        name: ProcName,
        subprocs: Sequence[BaseProcessor[Any, Any, Any, CtxT]],
        start_proc: BaseProcessor[InT, Any, Any, CtxT],
        end_proc: BaseProcessor[Any, OutT, Any, CtxT],
        recipients: Sequence[ProcName] | None = None,
        max_retries: int = 0,
    ) -> None:
        super().__init__(name=name, recipients=recipients, max_retries=max_retries)

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

        self.recipients = recipients
        if hasattr(type(self), "recipient_selector"):
            self._end_proc.recipient_selector = self.recipient_selector

    def add_recipient_selector(
        self, func: RecipientSelector[OutT, CtxT]
    ) -> RecipientSelector[OutT, CtxT]:
        self._end_proc.recipient_selector = func
        self.recipient_selector = func

        return func

    @property
    def recipients(self) -> Sequence[ProcName] | None:
        return self._end_proc.recipients

    @recipients.setter
    def recipients(self, value: Sequence[ProcName] | None) -> None:
        if hasattr(self, "_end_proc"):
            self._end_proc.recipients = value

    @property
    def subprocs(self) -> Sequence[BaseProcessor[Any, Any, Any, CtxT]]:
        return self._subprocs

    @property
    def start_proc(self) -> BaseProcessor[InT, Any, Any, CtxT]:
        return self._start_proc

    @property
    def end_proc(self) -> BaseProcessor[Any, OutT, Any, CtxT]:
        return self._end_proc

    def _generate_subproc_call_id(
        self, call_id: str | None, subproc: BaseProcessor[Any, Any, Any, CtxT]
    ) -> str | None:
        return f"{self._generate_call_id(call_id)}/{subproc.name}"

    @abstractmethod
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | Sequence[InT] | None = None,
        ctx: RunContext[CtxT] | None = None,
        forgetful: bool = False,
        call_id: str | None = None,
    ) -> Packet[OutT]:
        pass

    @abstractmethod
    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | Sequence[InT] | None = None,
        ctx: RunContext[CtxT] | None = None,
        forgetful: bool = False,
        call_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        yield DummyEvent()
