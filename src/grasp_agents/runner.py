from collections.abc import AsyncIterator, Sequence
from typing import Any, Generic

from .comm_processor import CommProcessor
from .run_context import CtxT, RunContext
from .typing.events import Event


class Runner(Generic[CtxT]):
    def __init__(
        self,
        start_proc: CommProcessor[Any, Any, Any, CtxT],
        procs: Sequence[CommProcessor[Any, Any, Any, CtxT]],
        ctx: RunContext[CtxT] | None = None,
    ) -> None:
        if start_proc not in procs:
            raise ValueError(
                f"Start processor {start_proc.name} must be in the list of processors: "
                f"{', '.join(proc.name for proc in procs)}"
            )
        self._start_proc = start_proc
        self._procs = procs
        self._ctx = ctx or RunContext[CtxT]()

    @property
    def ctx(self) -> RunContext[CtxT]:
        return self._ctx

    async def run(self, **run_args: Any) -> Any:
        self._ctx.is_streaming = False
        for proc in self._procs:
            proc.start_listening(ctx=self._ctx, **run_args)
        await self._start_proc.run(**run_args, ctx=self._ctx)

        return self._ctx.result

    async def run_stream(self, **run_args: Any) -> AsyncIterator[Event[Any]]:
        self._ctx.is_streaming = True
        for proc in self._procs:
            proc.start_listening(ctx=self._ctx, **run_args)
        async for event in self._start_proc.run_stream(**run_args, ctx=self._ctx):
            yield event
