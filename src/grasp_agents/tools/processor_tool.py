from collections.abc import AsyncIterator
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from ..agent.agent_context import AgentContext
from ..durability.checkpoints import CheckpointKind
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
        auto_background_at: float | None = None,
        blocks_final_answer: bool = True,
        max_inline_result_chars: int | None = None,
        reset_transcript_on_run: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            auto_background_at=auto_background_at,
            blocks_final_answer=blocks_final_answer,
            max_inline_result_chars=max_inline_result_chars,
        )
        self._processor = processor
        self._reset_transcript_on_run = reset_transcript_on_run

        # Resolve types from the processor at runtime
        self._in_type = processor.in_type
        self._out_type = processor.out_type

    @property
    def processor(self) -> Processor[_InT, _OutT, CtxT]:
        return self._processor

    @property
    def resumable(self) -> bool:
        return True

    @property
    def checkpoint_kind(self) -> CheckpointKind | None:
        return self._processor.checkpoint_kind

    def on_adopted(self, parent: Any) -> None:
        """
        Adopt the host's tracing settings and forward adoption onto the wrapped
        template so its path + default session align with the host.

        The template is a prototype, never run directly. Each dispatch clones
        it (:meth:`_resolve_processor`) and **rebinds the clone to that call's
        ``ctx``** — so a single ``ProcessorTool`` shared across several agents
        (manager/worker graphs, LRU-cached agents) is safe: although the
        template's own ctx is last-writer-wins, no call ever uses it, and the
        clone always carries the caller's session.
        """
        super().on_adopted(parent)  # inherit the tool's own tracing settings
        self._processor.on_adopted(parent)

    def _resolve_processor(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        path: list[str] | None = None,
    ) -> Processor[_InT, _OutT, CtxT]:
        """
        Return a fresh copy bound to this call's ``ctx`` and ``path``.

        Binding ``ctx`` per call (not at adoption) is what makes a shared
        ``ProcessorTool`` safe across agents/sessions: the clone resolves to
        the *caller's* session, never a stale one.
        """
        proc = self._processor.copy()
        # parent=self carries the tool's (host-inherited) tracing settings;
        # ctx + path are stamped explicitly for this call. on_adopted treats an
        # explicit ctx as authoritative, overriding whatever the template held.
        proc.on_adopted(parent=self, ctx=ctx, path=path)
        if self._reset_transcript_on_run:
            # Only LLM agents carry a transcript — other processors have no
            # working-state slot to reset.
            from ..agent.llm_agent import LLMAgent  # noqa: PLC0415

            if isinstance(proc, LLMAgent):
                proc.transcript.reset()
        return proc

    async def _run(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> _OutT:
        del progress_callback, agent_ctx
        proc = self._resolve_processor(ctx=ctx, path=path)
        result = await proc.run(in_args=inp, exec_id=exec_id)

        return result.payloads[0]

    async def _run_stream(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> AsyncIterator[Event[Any]]:
        del progress_callback, agent_ctx
        proc = self._resolve_processor(ctx=ctx, path=path)
        async for event in self._yield_proc_events(
            proc, in_args=inp, exec_id=exec_id, step=0
        ):
            yield event

    async def resume_stream(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> AsyncIterator[Event[Any]]:
        del agent_ctx  # proc builds its own
        proc = self._resolve_processor(ctx=ctx, path=path)
        async for event in self._yield_proc_events(
            proc, in_args=None, exec_id=exec_id, step=0
        ):
            yield event

    async def _yield_proc_events(
        self,
        proc: Processor[_InT, _OutT, CtxT],
        *,
        in_args: _InT | None = None,
        exec_id: str | None = None,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        async for event in proc.run_stream(in_args=in_args, exec_id=exec_id, step=step):
            if isinstance(event, ProcPacketOutEvent) and event.source == proc.name:
                yield ToolOutputEvent(
                    data=event.data.payloads[0], source=proc.name, exec_id=exec_id
                )
            else:
                yield event
