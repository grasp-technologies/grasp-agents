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
        When the host :class:`LLMAgent` is attached under a parent (or
        constructed with a ctx), forward adoption onto the wrapped
        template so its ctx + path align with the host's session.

        ``.copy()`` at dispatch then shares the bound ctx with each
        per-call copy via :meth:`RunContext.__deepcopy__`; the per-call
        path is set separately by :meth:`_resolve_processor` because it
        embeds the tool call id.
        """
        super().on_adopted(parent)  # inherit the tool's own tracing settings
        self._processor.on_adopted(parent)

    def _resolve_processor(
        self,
        *,
        path: list[str] | None = None,
    ) -> Processor[_InT, _OutT, CtxT]:
        """Return a fresh copy adopted under this tool, with ``path`` set."""
        proc = self._processor.copy()
        # Symmetric with AgentTool: adopt the tool's tracing settings and stamp
        # the per-call path. ctx is already shared through the copy
        # (RunContext.__deepcopy__) and the tool carries no ctx/path of its own,
        # so in practice this only re-applies tracing (idempotent on a copy of
        # the already-adopted template) and sets the path (a no-op if ``None``).
        proc.on_adopted(parent=self, path=path)
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
        ctx: RunContext[CtxT] | None = None,  # noqa: ARG002  # bound at adoption
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> _OutT:
        del progress_callback, agent_ctx
        proc = self._resolve_processor(path=path)
        result = await proc.run(in_args=inp, exec_id=exec_id)

        return result.payloads[0]

    async def _run_stream(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,  # noqa: ARG002  # bound at adoption
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> AsyncIterator[Event[Any]]:
        del progress_callback, agent_ctx
        proc = self._resolve_processor(path=path)
        async for event in self._yield_proc_events(
            proc, in_args=inp, exec_id=exec_id, step=0
        ):
            yield event

    async def resume_stream(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,  # noqa: ARG002  # bound at adoption
        exec_id: str | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,  # noqa: ARG002  # proc builds its own
    ) -> AsyncIterator[Event[Any]]:
        proc = self._resolve_processor(path=path)
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
