"""
Session-scoped execution-resource lifecycle.

Shells, kernels, and background tasks survive run boundaries; they are
released only by ``aclose()`` (LLMAgent / workflow / runner cascade), and
ephemeral clones (parallel replicas, per-call sub-agent tools) are closed by
their creators.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.tools.processor_tool import ProcessorTool
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow
from tests.agent.test_agent_loop import EchoTool  # type: ignore[attr-defined]
from tests.durability.test_sessions import (  # type: ignore[attr-defined]
    MockLLM,
    _text_response,
    _tool_call_response,
)
from tests.orchestration.test_runner import ListProcessor  # type: ignore[attr-defined]

pytestmark = pytest.mark.asyncio


def _make_agent(
    responses: list[Any],
    *,
    tools: list[Any] | None = None,
    max_turns: int = 10,
) -> LLMAgent[str, str, None]:
    return LLMAgent[str, str, None](
        name="test_agent",
        ctx=SessionContext[None](),
        llm=MockLLM(responses_queue=responses),
        tools=tools,
        max_turns=max_turns,
        env_info=False,
    )


# ---------- Background tasks survive run boundaries ----------


class TestBackgroundTasksAcrossRuns:
    async def test_task_survives_run_and_delivers_next_run(self) -> None:
        release = asyncio.Event()

        @function_tool(auto_background_at=0, blocks_final_answer=False)
        async def slow_job(text: str) -> str:
            """Run a slow job."""
            await release.wait()
            return f"job done: {text}"

        agent = _make_agent(
            [
                _tool_call_response("slow_job", '{"text":"tests"}', "tc1"),
                _text_response("started, will report"),
                _text_response("second turn answer"),
            ],
            tools=[slow_job],
        )

        # Run 1: the answer ships while the job runs; the run does NOT kill it.
        result = await agent.run("run the tests")
        assert result.payloads[0] == "started, will report"
        assert agent._loop.agent_ctx.bg_tasks.has_live_tasks

        # The job finishes between runs (e.g. while the human is typing).
        release.set()
        await asyncio.sleep(0.05)

        # Run 2: the first PRE-ACT drain delivers the completion note.
        await agent.run("how did it go?")
        transcript_text = str(agent.transcript.messages)
        assert "job done: tests" in transcript_text

        await agent.aclose()
        assert not agent._loop.agent_ctx.bg_tasks.has_live_tasks

    async def test_force_final_answer_keeps_task_running(self) -> None:
        started = asyncio.Event()

        @function_tool(auto_background_at=0, blocks_final_answer=False)
        async def forever_job(text: str) -> str:
            """Run an endless job."""
            started.set()
            await asyncio.Event().wait()
            return text

        agent = _make_agent(
            [
                _tool_call_response("forever_job", '{"text":"x"}', "tc1"),
                _text_response("still thinking"),
                _text_response("forced final"),
            ],
            tools=[forever_job],
            max_turns=1,
        )

        await agent.run("go")
        await started.wait()
        # MAX_TURNS did not kill the deliberately backgrounded job.
        assert agent._loop.agent_ctx.bg_tasks.has_live_tasks

        await agent.aclose()
        assert not agent._loop.agent_ctx.bg_tasks.has_live_tasks


# ---------- Holders survive runs; aclose closes them ----------


class _FakeProcess:
    def __init__(self) -> None:
        self.closed = False
        self.context_id: str | None = None  # mirrors the real kernel interface

    async def close(self) -> None:
        self.closed = True


class TestHoldersAcrossRuns:
    async def test_holders_survive_run_and_close_on_aclose(self) -> None:
        agent = _make_agent([_text_response("one"), _text_response("two")])

        # Simulate a kernel/shell opened during a run.
        fake_kernel = _FakeProcess()
        agent._loop.agent_ctx.nb_kernel_holder._kernel = fake_kernel  # type: ignore[assignment]

        await agent.run("first turn")
        # The run did not close the kernel — state survives to the next turn.
        assert not fake_kernel.closed

        await agent.run("second turn")
        assert not fake_kernel.closed

        await agent.aclose()
        assert fake_kernel.closed


# ---------- Ephemeral clones are closed by their creators ----------


class _CloseSpyAgent(LLMAgent[str, str, None]):
    # Class-level — shared across deepcopies.
    closed_names: ClassVar[list[str]] = []

    async def aclose(self) -> None:
        type(self).closed_names.append(self.name)
        await super().aclose()


class TestEphemeralCloneTeardown:
    async def test_parallel_replicas_closed_after_run(self) -> None:
        _CloseSpyAgent.closed_names = []
        template = _CloseSpyAgent(
            name="worker",
            ctx=SessionContext[None](),
            llm=MockLLM(responses_queue=[_text_response("a"), _text_response("b")]),
            env_info=False,
        )
        par = ParallelProcessor[str, str, None](subproc=template)
        par.on_adopted(ctx=SessionContext[None]())

        result = await par.run(in_args=["x", "y"], exec_id="e")
        assert sorted(result.payloads) == ["a", "b"]
        # Both per-run replicas were closed (the template itself was not run).
        assert sorted(_CloseSpyAgent.closed_names) == ["worker_0", "worker_1"]

    async def test_processor_tool_clone_closed_after_call(self) -> None:
        _CloseSpyAgent.closed_names = []
        template = _CloseSpyAgent(
            name="sub",
            ctx=SessionContext[None](),
            llm=MockLLM(responses_queue=[_text_response("sub answer")]),
            env_info=False,
        )
        tool = ProcessorTool[Any, str, None](
            processor=template, name="sub", description="d"
        )

        out = await tool._run("hi", ctx=SessionContext[None]())
        assert out == "sub answer"
        assert _CloseSpyAgent.closed_names == ["sub"]


# ---------- Composite cascade ----------


class TestCompositeAclose:
    async def test_workflow_aclose_cascades(self) -> None:
        _CloseSpyAgent.closed_names = []
        shared_ctx = SessionContext[None]()
        a = _CloseSpyAgent(
            name="a",
            ctx=shared_ctx,
            llm=MockLLM(responses_queue=[]),
            env_info=False,
        )
        b = _CloseSpyAgent(
            name="b",
            ctx=shared_ctx,
            llm=MockLLM(responses_queue=[]),
            env_info=False,
        )
        wf = SequentialWorkflow[Any, str, None](name="wf", subprocs=[a, b])
        await wf.aclose()
        assert _CloseSpyAgent.closed_names == ["a", "b"]

    async def test_agent_aclose_cascades_to_processor_tools(self) -> None:
        _CloseSpyAgent.closed_names = []
        inner = _CloseSpyAgent(
            name="inner",
            ctx=SessionContext[None](),
            llm=MockLLM(responses_queue=[]),
            env_info=False,
        )
        tool = ProcessorTool[Any, str, None](
            processor=inner, name="inner", description="d"
        )
        outer = LLMAgent[str, str, None](
            name="outer",
            ctx=SessionContext[None](),
            llm=MockLLM(responses_queue=[]),
            tools=[tool],
            env_info=False,
        )
        await outer.aclose()
        assert _CloseSpyAgent.closed_names == ["inner"]

    async def test_async_context_manager(self) -> None:
        async with _make_agent([_text_response("ok")], tools=[EchoTool()]) as agent:
            result = await agent.run("hi")
            assert result.payloads[0] == "ok"
        # __aexit__ ran aclose — background manager is clean.
        assert not agent._loop.agent_ctx.bg_tasks.has_live_tasks

    async def test_plain_processor_aclose_is_noop(self) -> None:
        proc = ListProcessor(name="p")
        await proc.aclose()  # no raise
