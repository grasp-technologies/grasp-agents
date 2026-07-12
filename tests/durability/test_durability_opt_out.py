"""
Selective persistence opt-out: ``durability_enabled=False`` on a processor
keeps it (and everything it contains or dispatches) out of the checkpoint
store — e.g. throwaway replicas fanned out by a ParallelProcessor.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.processors.processor import Processor
from grasp_agents.session_context import SessionContext
from grasp_agents.types.events import Event, ProcPayloadOutEvent
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow
from tests._helpers import MockLLM, _text_response


class AppendProcessor(Processor[str, str, None]):
    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for inp in in_args or []:
            yield ProcPayloadOutEvent(
                data=f"{inp}->{self.name}", source=self.name, exec_id=exec_id
            )


def _make_agent(
    name: str,
    responses: list[Any],
    *,
    ctx: SessionContext[None] | None = None,
    **agent_kwargs: Any,
) -> LLMAgent[str, str, None]:
    return LLMAgent[str, str, None](
        name,
        ctx=ctx,
        llm=MockLLM(responses_queue=responses),
        **agent_kwargs,
    )


class TestAgentOptOut:
    @pytest.mark.asyncio
    async def test_disabled_agent_writes_nothing(self) -> None:
        store = InMemoryCheckpointStore()
        ctx: SessionContext[None] = SessionContext(
            checkpoint_store=store, session_key="s1"
        )
        agent = _make_agent(
            "worker", [_text_response("ok")], ctx=ctx, durability_enabled=False
        )
        out = await agent.run("hi")

        assert out.payloads[0] == "ok"
        assert store._data == {}
        assert store._logs == {}

    @pytest.mark.asyncio
    async def test_enabled_agent_writes(self) -> None:
        store = InMemoryCheckpointStore()
        ctx: SessionContext[None] = SessionContext(
            checkpoint_store=store, session_key="s2"
        )
        agent = _make_agent("worker", [_text_response("ok")], ctx=ctx)
        await agent.run("hi")

        assert any("/agent/" in key for key in store._data)

    @pytest.mark.asyncio
    async def test_disabled_agent_does_not_resume(self) -> None:
        """A persisted session is invisible once durability is off."""
        store = InMemoryCheckpointStore()
        ctx1: SessionContext[None] = SessionContext(
            checkpoint_store=store, session_key="s3"
        )
        agent1 = _make_agent("worker", [_text_response("answer")], ctx=ctx1)
        await agent1.run("question")
        assert await agent1.has_checkpoint(ctx1)

        ctx2: SessionContext[None] = SessionContext(
            checkpoint_store=store, session_key="s3"
        )
        agent2 = _make_agent("worker", [], ctx=ctx2, durability_enabled=False)
        assert not agent2.is_resumable
        assert not await agent2.has_checkpoint(ctx2)


class TestParallelFanOut:
    @pytest.mark.asyncio
    async def test_disabled_replicas_write_nothing(self) -> None:
        store = InMemoryCheckpointStore()
        # Replicas share the template's LLM (and so its queue): one response each.
        template = _make_agent(
            "worker", [_text_response("done")] * 3, durability_enabled=False
        )
        par = ParallelProcessor[str, str, None](subproc=template)
        assert par.durability_enabled is False  # mirrored from the subproc

        ctx: SessionContext[None] = SessionContext(
            checkpoint_store=store, session_key="par-off"
        )
        par.on_adopted(ctx=ctx)
        out = await par.run(in_args=["a", "b", "c"])

        assert len(out.payloads) == 3
        assert store._data == {}
        assert store._logs == {}

    @pytest.mark.asyncio
    async def test_enabled_replicas_write(self) -> None:
        store = InMemoryCheckpointStore()
        template = _make_agent("worker", [_text_response("done")] * 2)
        par = ParallelProcessor[str, str, None](subproc=template)
        ctx: SessionContext[None] = SessionContext(
            checkpoint_store=store, session_key="par-on"
        )
        par.on_adopted(ctx=ctx)
        await par.run(in_args=["a", "b"])

        assert any("/parallel/" in key for key in store._data)
        assert any("/agent/" in key for key in store._data)


class TestDownwardCascade:
    def test_container_disables_descendants(self) -> None:
        a = AppendProcessor("A")
        b = AppendProcessor("B")
        wf = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[a, b], durability_enabled=False
        )
        assert wf.durability_enabled is False
        assert a.durability_enabled is False
        assert b.durability_enabled is False

    def test_child_cannot_reenable(self) -> None:
        disabled_parent = AppendProcessor("P", durability_enabled=False)
        child = AppendProcessor("C")
        child.on_adopted(disabled_parent)
        assert child.durability_enabled is False

        enabled_parent = AppendProcessor("P2")
        opted_out_child = AppendProcessor("C2", durability_enabled=False)
        opted_out_child.on_adopted(enabled_parent)
        assert opted_out_child.durability_enabled is False


class _TaskInput(BaseModel):
    text: str


class TestToolConduit:
    def test_as_tool_inherits_host_opt_out(self) -> None:
        sub = LLMAgent[_TaskInput, str, None]("sub", llm=MockLLM(responses_queue=[]))
        tool = sub.as_tool(tool_name="Sub", tool_description="d")
        host = _make_agent("host", [], tools=[tool], durability_enabled=False)

        assert tool.durability_enabled is False
        assert tool.processor.durability_enabled is False
        clone = tool._resolve_processor(ctx=host.ctx, path=["host", "c1", "Sub"])
        assert clone.durability_enabled is False

    def test_disabled_subproc_stays_disabled_under_enabled_host(self) -> None:
        sub = LLMAgent[_TaskInput, str, None](
            "sub", llm=MockLLM(responses_queue=[]), durability_enabled=False
        )
        tool = sub.as_tool(tool_name="Sub", tool_description="d")
        host = _make_agent("host", [], tools=[tool])

        assert host.durability_enabled is True
        assert tool.processor.durability_enabled is False
        clone = tool._resolve_processor(ctx=host.ctx, path=["host", "c1", "Sub"])
        assert clone.durability_enabled is False


class TestTaskRecordGate:
    def test_bg_task_store_key_gated(self) -> None:
        store = InMemoryCheckpointStore()
        ctx: SessionContext[None] = SessionContext(
            checkpoint_store=store, session_key="bg"
        )
        disabled = _make_agent("a", [], ctx=ctx, durability_enabled=False)
        assert disabled.agent_ctx.bg_tasks.durability_enabled is False
        assert disabled.agent_ctx.bg_tasks._task_store_key(ctx, "call_1") is None

        enabled = _make_agent("b", [], ctx=ctx)
        assert enabled.agent_ctx.bg_tasks._task_store_key(ctx, "call_1") is not None
