from collections.abc import AsyncIterator
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from ..run_context import CtxT, RunContext
from ..types.events import Event, ProcPacketOutEvent, ToolOutputEvent
from ..types.tool import BaseTool
from .base_processor import BaseProcessor

_ProcToolInT = TypeVar("_ProcToolInT", bound=BaseModel)
_ProcToolOutT = TypeVar("_ProcToolOutT")


class ProcessorTool(BaseTool[_ProcToolInT, _ProcToolOutT, CtxT]):
    """A tool that wraps a processor (or agent) for use inside an agent loop."""

    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        *,
        processor: BaseProcessor[_ProcToolInT, _ProcToolOutT, CtxT],
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
    def processor(self) -> BaseProcessor[_ProcToolInT, _ProcToolOutT, CtxT]:
        return self._processor

    @property
    def resumable(self) -> bool:
        return self._processor.resumable

    async def _run(
        self,
        inp: _ProcToolInT,
        *,
        exec_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
        progress_callback: Any = None,
    ) -> _ProcToolOutT:
        if self._reset_memory_on_run:
            self._processor.memory.reset()

        result = await self._processor.run(
            in_args=inp, exec_id=exec_id, ctx=ctx
        )
        return result.payloads[0]

    async def run_stream(
        self,
        inp: _ProcToolInT,
        *,
        exec_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
        progress_callback: Any = None,
        _validated: bool = False,
    ) -> AsyncIterator[Event[Any]]:
        if self._reset_memory_on_run:
            self._processor.memory.reset()

        async for event in self._processor.run_stream(
            in_args=inp, exec_id=exec_id, ctx=ctx
        ):
            if (
                isinstance(event, ProcPacketOutEvent)
                and event.source == self._processor.name
            ):
                yield ToolOutputEvent(
                    data=event.data.payloads[0],
                    source=self._processor.name,
                    exec_id=exec_id or "",
                )
            else:
                yield event
