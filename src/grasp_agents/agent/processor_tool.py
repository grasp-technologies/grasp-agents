from collections.abc import AsyncIterator
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..types.events import Event, ProcPacketOutEvent, ToolOutputEvent
from ..types.tool import BaseTool, ToolProgressCallback

_InT = TypeVar("_InT", bound=BaseModel)
_OutT = TypeVar("_OutT")


class ProcessorTool(BaseTool[_InT, _OutT, CtxT]):
    """A tool that wraps a processor (or agent) for use inside an agent loop."""

    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        *,
        processor: Processor[_InT, _OutT, CtxT],
        name: str,
        description: str,
        background: bool = False,
        reset_memory_on_run: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            background=background,
        )
        self._processor = processor
        self._reset_memory_on_run = reset_memory_on_run

        # Resolve types from the processor at runtime
        self._in_type = processor.in_type
        self._out_type = processor.out_type

    @property
    def processor(self) -> Processor[_InT, _OutT, CtxT]:
        return self._processor

    @property
    def resumable(self) -> bool:
        return True

    def _resolve_processor(
        self, session_id: str | None
    ) -> Processor[_InT, _OutT, CtxT]:
        """Return the processor to use — a session-configured copy or self."""
        if session_id is not None:
            proc = self._processor.copy()
            proc.setup_session(session_id)
            return proc

        if self._reset_memory_on_run:
            self._processor.memory.reset()

        return self._processor

    async def _run(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> _OutT:
        proc = self._resolve_processor(session_id)
        result = await proc.run(in_args=inp, exec_id=exec_id, ctx=ctx)

        return result.payloads[0]

    async def _run_stream(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        proc = self._resolve_processor(session_id)
        async for event in self._yield_proc_events(
            proc, in_args=inp, ctx=ctx, exec_id=exec_id, step=0
        ):
            yield event

    async def resume_stream(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        proc = self._resolve_processor(session_id)
        async for event in self._yield_proc_events(
            proc, in_args=None, ctx=ctx, exec_id=exec_id, step=0
        ):
            yield event

    async def _yield_proc_events(
        self,
        proc: Processor[_InT, _OutT, CtxT],
        *,
        in_args: _InT | None = None,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        async for event in proc.run_stream(
            in_args=in_args, exec_id=exec_id, ctx=ctx, step=step
        ):
            if isinstance(event, ProcPacketOutEvent) and event.source == proc.name:
                yield ToolOutputEvent(
                    data=event.data.payloads[0], source=proc.name, exec_id=exec_id
                )
            else:
                yield event
