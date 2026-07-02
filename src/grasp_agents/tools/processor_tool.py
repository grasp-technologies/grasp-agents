from collections.abc import AsyncIterator
from typing import Any, ClassVar

from pydantic import BaseModel

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.durability.checkpoints import CheckpointKind
from grasp_agents.processors.processor import Processor
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool, ToolProgressCallback
from grasp_agents.types.events import Event, ProcPacketOutEvent, ToolOutputEvent


class ProcessorTool[InT: BaseModel, OutT, CtxT](BaseTool[InT, OutT, CtxT]):
    """A tool that wraps a processor (or agent) for use inside an agent loop."""

    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        *,
        processor: Processor[InT, OutT, CtxT],
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
    def processor(self) -> Processor[InT, OutT, CtxT]:
        return self._processor

    async def aclose(self) -> None:
        # The template is never run, but close it for symmetry/safety (a user
        # may have run the wrapped processor directly before wrapping it).
        # Per-call clones are closed by the dispatch paths below.
        await self._processor.aclose()

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
        ctx: SessionContext[CtxT] | None = None,
        path: list[str] | None = None,
    ) -> Processor[InT, OutT, CtxT]:
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
            from grasp_agents.agent.llm_agent import LLMAgent  # noqa: PLC0415

            if isinstance(proc, LLMAgent):
                proc.transcript.clear()
        return proc

    async def _run(
        self,
        inp: InT,
        *,
        ctx: SessionContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> OutT:
        del progress_callback, agent_ctx
        proc = self._resolve_processor(ctx=ctx, path=path)
        # The clone is unreachable after this call — its session ends here.
        try:
            result = await proc.run(in_args=inp, exec_id=exec_id)
        finally:
            await proc.aclose()

        return result.payloads[0]

    async def _run_stream(
        self,
        inp: InT,
        *,
        ctx: SessionContext[CtxT] | None = None,
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
        ctx: SessionContext[CtxT] | None = None,
        exec_id: str | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
        tool_call_arguments: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        del agent_ctx  # proc builds its own
        proc = self._resolve_processor(ctx=ctx, path=path)
        cold_start = await self._should_cold_start(proc, ctx)
        in_args = (
            self._input_from_arguments(tool_call_arguments) if cold_start else None
        )
        async for event in self._yield_proc_events(
            proc, in_args=in_args, exec_id=exec_id, step=0
        ):
            yield event

    async def _yield_proc_events(
        self,
        proc: Processor[InT, OutT, CtxT],
        *,
        in_args: InT | None = None,
        exec_id: str | None = None,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        # The per-call clone is unreachable after this stream — its session
        # ends with it (also on cancellation/error).
        try:
            async for event in proc.run_stream(
                in_args=in_args, exec_id=exec_id, step=step
            ):
                if isinstance(event, ProcPacketOutEvent) and event.source == proc.name:
                    yield ToolOutputEvent(
                        data=event.data.payloads[0], source=proc.name, exec_id=exec_id
                    )
                else:
                    yield event
        finally:
            await proc.aclose()
