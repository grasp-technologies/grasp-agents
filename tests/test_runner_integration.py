"""
Runner + ParallelProcessor + Workflow + LLMAgent integration tests.

Uses OpenAI Responses API with gpt-5.4-nano.
Skipped by default. Run with:
    uv run pytest -m integration -k test_runner_integration
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import (
    AgentCheckpoint,
    RunnerCheckpoint,
    WorkflowCheckpoint,
)
from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.runner.runner import END_PROC_NAME, Runner
from grasp_agents.types.io import ProcName
from grasp_agents.types.items import FunctionToolCallItem, FunctionToolOutputItem
from grasp_agents.types.tool import BaseTool
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow

# ---------- Schemas ----------


class TopicIdea(BaseModel):
    topic: str = Field(description="A specific topic for a short paragraph")
    audience: str = Field(description="Target audience for the paragraph")


class TopicIdeas(BaseModel):
    ideas: list[TopicIdea]


class Paragraph(BaseModel):
    topic: str = Field(description="The topic this paragraph covers")
    text: str = Field(description="The written paragraph (2-3 sentences)")


# ---------- Helper processors (non-streaming, using _process) ----------


class Passthrough(Processor[str, str, None]):
    """Passes inputs through unchanged."""

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> list[str]:
        if in_args is not None:
            return in_args
        if chat_inputs is not None:
            return [str(chat_inputs)]
        return []


class SplitterProcessor(Processor[str, TopicIdea, None]):
    """Uses an LLMAgent to split a subject into TopicIdeas."""

    def __init__(
        self,
        name: str,
        *,
        agent: LLMAgent[str, TopicIdeas, None],
        recipients: list[ProcName] | None = None,
    ) -> None:
        super().__init__(name=name, recipients=recipients)
        self._agent = agent

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> list[TopicIdea]:
        inputs = in_args or ([str(chat_inputs)] if chat_inputs is not None else [])
        topics: list[TopicIdea] = []
        for inp in inputs:
            result = await self._agent.run(chat_inputs=inp, ctx=ctx, exec_id=exec_id)
            for payload in result.payloads:
                topics.extend(payload.ideas)
        return topics


class CollectorProcessor(Processor[Paragraph, str, None]):
    """Joins paragraphs into a markdown document."""

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[Paragraph] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> list[str]:
        paragraphs = in_args or []
        doc = "\n\n".join(f"## {p.topic}\n{p.text}" for p in paragraphs)
        return [doc]


# ---------- LLM fixture ----------


def _make_llm(
    max_output_tokens: int = 300,
    *,
    structured: bool = False,
) -> Any:
    from grasp_agents.llm_providers.openai_responses.responses_llm import (
        OpenAIResponsesLLM,
    )

    return OpenAIResponsesLLM(
        model_name="gpt-5.4-nano",
        llm_settings={"max_output_tokens": max_output_tokens},
        apply_response_schema_via_provider=structured,
    )


# ---------- Tests ----------


@pytest.mark.integration
class TestRunnerWithAgents:
    """Runner graph with real LLMAgent nodes."""

    @pytest.fixture
    def llm(self, openai_api_key: str) -> Any:  # noqa: ARG002
        return _make_llm()

    @pytest.mark.asyncio
    async def test_linear_agent_pipeline(self, llm: Any) -> None:
        """Agent A picks a topic → Agent B writes about it → END."""
        agent_a = LLMAgent[str, str, None](
            name="topic_picker",
            llm=llm,
            sys_prompt=(
                "You are a topic picker. Given a subject, pick ONE specific "
                "sub-topic. Respond with just the sub-topic name, nothing else."
            ),
            recipients=["writer"],
        )
        agent_b = LLMAgent[str, str, None](
            name="writer",
            llm=llm,
            sys_prompt=(
                "Write exactly one sentence about the given topic. "
                "Respond with just the sentence."
            ),
            recipients=[END_PROC_NAME],
        )

        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](
            entry_proc=agent_a, procs=[agent_a, agent_b], ctx=ctx, name="r"
        )

        result = await runner.run(chat_inputs="Science")
        payloads = list(result.payloads)
        assert len(payloads) == 1
        assert len(payloads[0]) > 10


@pytest.mark.integration
class TestParallelProcessorWithAgents:
    """ParallelProcessor fan-out with real LLMAgent workers."""

    @pytest.fixture
    def llm(self, openai_api_key: str) -> Any:  # noqa: ARG002
        return _make_llm()

    @pytest.mark.asyncio
    async def test_parallel_agent_fanout(self, llm: Any) -> None:
        """Three topics processed concurrently by the same writer agent."""
        writer = LLMAgent[str, str, None](
            name="writer",
            llm=llm,
            sys_prompt=(
                "Write exactly one sentence about the given topic. "
                "Respond with just the sentence."
            ),
        )
        par = ParallelProcessor[str, str, None](subproc=writer)
        ctx: RunContext[None] = RunContext(state=None)

        result = await par.run(
            in_args=["black holes", "photosynthesis", "jazz music"], ctx=ctx
        )
        payloads = list(result.payloads)
        assert len(payloads) == 3
        for p in payloads:
            assert len(p) > 10


@pytest.mark.integration
class TestRunnerParallelFanout:
    """Runner: Splitter → ParallelProcessor(writer) → Collector → END."""

    @pytest.fixture
    def llm(self, openai_api_key: str) -> Any:  # noqa: ARG002
        return _make_llm(500, structured=True)

    @pytest.mark.asyncio
    async def test_splitter_parallel_collector(self, llm: Any) -> None:
        """Full pipeline: split subject → write paragraphs in parallel → collect."""
        splitter_agent = LLMAgent[str, TopicIdeas, None](
            name="splitter_agent",
            llm=llm,
            sys_prompt=(
                "Given a subject, generate exactly 2 specific topic ideas. "
                "Each should target 'general audience'."
            ),
            response_schema=TopicIdeas,
        )
        splitter = SplitterProcessor(
            "splitter", agent=splitter_agent, recipients=["placeholder"]
        )

        writer = LLMAgent[TopicIdea, Paragraph, None](
            name="writer",
            llm=llm,
            sys_prompt=(
                "Write a short paragraph (2-3 sentences) about the given topic "
                "for the specified audience. Return structured output."
            ),
            response_schema=Paragraph,
            max_retries=2,
        )
        par = ParallelProcessor[TopicIdea, Paragraph, None](subproc=writer)

        collector = CollectorProcessor("collector", recipients=[END_PROC_NAME])

        splitter.recipients = [par.name]
        par.recipients = ["collector"]

        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](
            entry_proc=splitter,
            procs=[splitter, par, collector],
            ctx=ctx,
            name="content_pipeline",
        )

        result = await runner.run(chat_inputs="Space Exploration")
        payloads = list(result.payloads)
        assert len(payloads) == 1
        doc = payloads[0]
        assert "##" in doc
        assert len(doc) > 50


@pytest.mark.integration
class TestRunnerDurability:
    """Runner checkpoint with real LLM agents."""

    @pytest.fixture
    def llm(self, openai_api_key: str) -> Any:  # noqa: ARG002
        return _make_llm()

    @pytest.mark.asyncio
    async def test_checkpoint_persists(self, llm: Any) -> None:
        """Full run → checkpoint has no pending events, tracks sessions."""
        store = InMemoryCheckpointStore()

        agent_a = LLMAgent[str, str, None](
            name="topic_picker",
            llm=llm,
            sys_prompt="Pick ONE sub-topic. Respond with just the name.",
            recipients=["writer"],
        )
        agent_b = LLMAgent[str, str, None](
            name="writer",
            llm=llm,
            sys_prompt="Write one sentence about the topic. Just the sentence.",
            recipients=[END_PROC_NAME],
        )

        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="int-1"
        )
        runner = Runner[str, None](
            entry_proc=agent_a, procs=[agent_a, agent_b], ctx=ctx, name="r"
        )

        result = await runner.run(chat_inputs="Mathematics")
        assert len(list(result.payloads)) == 1

        raw = await store.load("runner/int-1")
        assert raw is not None
        cp = RunnerCheckpoint.model_validate_json(raw)
        assert len(cp.pending_events) == 0
        assert len(cp.active_steps) == 2


@pytest.mark.integration
class TestSequentialWorkflowInRunner:
    """SequentialWorkflow as a node inside a Runner graph."""

    @pytest.fixture
    def llm(self, openai_api_key: str) -> Any:  # noqa: ARG002
        return _make_llm()

    @pytest.mark.asyncio
    async def test_workflow_as_runner_node(self, llm: Any) -> None:
        """Runner: entry → SequentialWorkflow(draft → refine) → END."""
        drafter = LLMAgent[str, str, None](
            name="drafter",
            llm=llm,
            sys_prompt="Write a one-sentence draft about the topic.",
        )
        refiner = LLMAgent[str, str, None](
            name="refiner",
            llm=llm,
            sys_prompt=(
                "Improve the given draft sentence — make it more vivid. "
                "Respond with just the improved sentence."
            ),
        )
        wf = SequentialWorkflow[str, str, None](
            name="draft_refine",
            subprocs=[drafter, refiner],
            recipients=[END_PROC_NAME],
        )

        entry = Passthrough(name="entry", recipients=["draft_refine"])

        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](
            entry_proc=entry, procs=[entry, wf], ctx=ctx, name="wf_runner"
        )

        result = await runner.run(chat_inputs="The ocean at sunset")
        payloads = list(result.payloads)
        assert len(payloads) == 1
        assert len(payloads[0]) > 10


# ---------- Crash-and-resume tests ----------


class _AddInput(BaseModel):
    a: int = Field(description="First integer")
    b: int = Field(description="Second integer")


class _FailingAddTool(BaseTool[_AddInput, int, None]):
    """Add tool that raises on a specific call number."""

    def __init__(self, *, fail_on_call: int = 999) -> None:
        super().__init__(name="add", description="Add two integers and return the sum.")
        self.call_count = 0
        self.fail_on_call = fail_on_call

    async def _run(
        self,
        inp: _AddInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
    ) -> int:
        self.call_count += 1
        if self.call_count == self.fail_on_call:
            raise RuntimeError("Deliberate tool failure for testing")
        return inp.a + inp.b


class _FailingProcessor(Processor[str, str, None]):
    """Always raises on first call, succeeds on subsequent calls."""

    def __init__(self, name: str, *, recipients: list[ProcName] | None = None) -> None:
        super().__init__(name=name, recipients=recipients)
        self.call_count = 0

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> list[str]:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError(f"{self.name} deliberate failure")
        return in_args or ([str(chat_inputs)] if chat_inputs else [])


@pytest.mark.integration
class TestAgentCrashResume:
    """Agent crashes mid-execution inside Runner, then resumes from checkpoint."""

    @pytest.mark.asyncio
    async def test_agent_checkpoint_survives_downstream_crash(
        self,
        openai_api_key: str,  # noqa: ARG002
    ) -> None:
        """
        Runner: entry → calculator (add tool) → crasher → END.

        First run: agent uses tool, completes. Downstream crasher fails.
        Verify: agent checkpoint has tool call messages, Runner pending = crasher.
        Resume: agent NOT re-invoked, crasher succeeds.
        """
        store = InMemoryCheckpointStore()

        # --- First run: agent succeeds, downstream crashes ---
        add_tool = _FailingAddTool(fail_on_call=999)
        agent1 = LLMAgent[str, str, None](
            name="calculator",
            llm=_make_llm(500),
            sys_prompt=(
                "You are a calculator. Use the add tool to compute 1 + 2. "
                "Return the final number only."
            ),
            tools=[add_tool],
            recipients=["crasher"],
        )
        crasher1 = _FailingProcessor("crasher", recipients=[END_PROC_NAME])

        entry1 = Passthrough(name="entry", recipients=["calculator"])
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="agent-crash-1"
        )
        runner1 = Runner[str, None](
            entry_proc=entry1, procs=[entry1, agent1, crasher1], ctx=ctx1, name="r"
        )

        with pytest.raises(ExceptionGroup):
            await runner1.run(chat_inputs="go")

        # Verify: agent checkpoint has tool call conversation
        agent_key = "agent/agent-crash-1/calculator"
        agent_data = await store.load(agent_key)
        assert agent_data is not None, "Agent checkpoint should exist"
        agent_cp = AgentCheckpoint.model_validate_json(agent_data)
        has_tool_call = any(
            isinstance(m, FunctionToolCallItem) for m in agent_cp.messages
        )
        has_tool_result = any(
            isinstance(m, FunctionToolOutputItem) for m in agent_cp.messages
        )
        assert has_tool_call, "Checkpoint should contain a tool call"
        assert has_tool_result, "Checkpoint should contain a tool result"

        # Verify: Runner checkpoint shows crasher pending
        runner_data = await store.load("runner/agent-crash-1")
        assert runner_data is not None
        runner_cp = RunnerCheckpoint.model_validate_json(runner_data)
        assert len(runner_cp.pending_events) == 1
        assert runner_cp.pending_events[0].destination == "crasher"

        # --- Resume: crasher succeeds this time ---
        add_tool2 = _FailingAddTool(fail_on_call=999)
        agent2 = LLMAgent[str, str, None](
            name="calculator",
            llm=_make_llm(500),
            sys_prompt="You are a calculator.",
            tools=[add_tool2],
            recipients=["crasher"],
        )
        crasher2 = _FailingProcessor("crasher", recipients=[END_PROC_NAME])
        crasher2.call_count = 1  # skip the failure

        entry2 = Passthrough(name="entry", recipients=["calculator"])
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="agent-crash-1"
        )
        runner2 = Runner[str, None](
            entry_proc=entry2, procs=[entry2, agent2, crasher2], ctx=ctx2, name="r"
        )

        result = await runner2.run()
        payloads = list(result.payloads)
        assert len(payloads) == 1
        assert "3" in payloads[0]

        # Agent was NOT re-invoked
        assert add_tool2.call_count == 0

    @pytest.mark.asyncio
    async def test_agent_resumes_mid_tool_loop(
        self,
        openai_api_key: str,  # noqa: ARG002
    ) -> None:
        """
        Runner: entry → calculator (add tool, crashes on turn 2) → END.

        The agent needs 2 tool calls: (1+2), then (result+10).
        A before_llm_hook crashes the agent on turn 2 (after 1st tool call
        is done and checkpointed).
        Resume: agent loads checkpoint with turn 1 conversation, continues
        from turn 2 without re-doing the 1st tool call.
        """
        store = InMemoryCheckpointStore()
        turn_counter: dict[str, int] = {"count": 0}

        # --- First run: crash on turn 2 via before_llm_hook ---
        add_tool1 = _FailingAddTool(fail_on_call=999)
        agent1 = LLMAgent[str, str, None](
            name="calculator",
            llm=_make_llm(500),
            sys_prompt=(
                "You are a calculator. Use the add tool to compute step by step. "
                "First compute 1 + 2, then add 10 to the result. "
                "Return the final number only."
            ),
            tools=[add_tool1],
            recipients=[END_PROC_NAME],
        )

        @agent1.add_before_llm_hook
        async def _crash_on_turn_2(
            *,
            ctx: Any,
            exec_id: Any,
            num_turns: int,
            extra_llm_settings: Any,  # noqa: ARG001
        ) -> None:
            turn_counter["count"] += 1
            if turn_counter["count"] == 2:
                raise RuntimeError("Deliberate crash on turn 2")

        entry1 = Passthrough(name="entry", recipients=["calculator"])
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="mid-agent-1"
        )
        runner1 = Runner[str, None](
            entry_proc=entry1, procs=[entry1, agent1], ctx=ctx1, name="r"
        )

        with pytest.raises(Exception):
            await runner1.run(chat_inputs="go")

        # Verify: the agent made exactly 1 tool call before crashing
        assert add_tool1.call_count == 1

        # Verify: agent checkpoint has the 1st tool call + result
        agent_key = "agent/mid-agent-1/calculator"
        agent_data = await store.load(agent_key)
        assert agent_data is not None
        agent_cp = AgentCheckpoint.model_validate_json(agent_data)
        tool_calls = [
            m for m in agent_cp.messages if isinstance(m, FunctionToolCallItem)
        ]
        tool_results = [
            m for m in agent_cp.messages if isinstance(m, FunctionToolOutputItem)
        ]
        assert len(tool_calls) >= 1
        assert len(tool_results) >= 1

        # Runner checkpoint: calculator's input event is still pending
        runner_data = await store.load("runner/mid-agent-1")
        assert runner_data is not None
        runner_cp = RunnerCheckpoint.model_validate_json(runner_data)
        assert len(runner_cp.pending_events) == 1
        assert runner_cp.pending_events[0].destination == "calculator"

        # --- Resume: no crash hook, agent continues from checkpoint ---
        add_tool2 = _FailingAddTool(fail_on_call=999)
        agent2 = LLMAgent[str, str, None](
            name="calculator",
            llm=_make_llm(500),
            sys_prompt=(
                "You are a calculator. Use the add tool to compute step by step. "
                "First compute 1 + 2, then add 10 to the result. "
                "Return the final number only."
            ),
            tools=[add_tool2],
            recipients=[END_PROC_NAME],
        )
        # No crash hook on agent2

        entry2 = Passthrough(name="entry", recipients=["calculator"])
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="mid-agent-1"
        )
        runner2 = Runner[str, None](
            entry_proc=entry2, procs=[entry2, agent2], ctx=ctx2, name="r"
        )

        result = await runner2.run()
        payloads = list(result.payloads)
        assert len(payloads) == 1
        assert "13" in payloads[0]  # 1+2=3, 3+10=13

        # Agent resumed from checkpoint — should NOT have re-done the 1st call.
        # Only the 2nd add call (3+10) was made on this run.
        assert add_tool2.call_count == 1


@pytest.mark.integration
class TestWorkflowCrashResume:
    """SequentialWorkflow crashes mid-step inside Runner, then resumes."""

    @pytest.mark.asyncio
    async def test_workflow_resumes_after_step_failure(
        self,
        openai_api_key: str,  # noqa: ARG002
    ) -> None:
        """
        Runner: entry → SequentialWorkflow(drafter → refiner) → END.

        First run: drafter completes, refiner crashes → workflow checkpoint at step 0.
        Resume: drafter skipped, refiner runs.
        """
        store = InMemoryCheckpointStore()

        # --- First run: refiner crashes ---
        drafter1 = LLMAgent[str, str, None](
            name="drafter",
            llm=_make_llm(),
            sys_prompt="Write one sentence about the topic.",
        )
        # Use a FailOnCallProcessor-like LLMAgent — actually just use a
        # Processor that fails, since we can't make an LLMAgent fail reliably.

        class FailingRefiner(Processor[str, str, None]):
            async def _process(
                self,
                chat_inputs: Any = None,
                *,
                in_args: Any = None,
                exec_id: str,
                ctx: RunContext[None],
                step: int | None = None,
            ) -> list[str]:
                raise RuntimeError("Refiner crash")

        refiner1 = FailingRefiner(name="refiner")
        wf1 = SequentialWorkflow[str, str, None](
            name="draft_refine",
            subprocs=[drafter1, refiner1],
            recipients=[END_PROC_NAME],
        )

        entry1 = Passthrough(name="entry", recipients=["draft_refine"])
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="wf-crash-1"
        )
        runner1 = Runner[str, None](
            entry_proc=entry1, procs=[entry1, wf1], ctx=ctx1, name="r"
        )

        with pytest.raises(Exception):
            await runner1.run(chat_inputs="The ocean at sunset")

        # Verify: workflow checkpoint saved at step 0 (drafter completed)
        wf_key = "workflow/wf-crash-1/draft_refine"
        wf_data = await store.load(wf_key)
        assert wf_data is not None
        wf_cp = WorkflowCheckpoint.model_validate_json(wf_data)
        assert wf_cp.completed_step == 0  # drafter done

        # --- Resume: working refiner ---
        drafter2 = LLMAgent[str, str, None](
            name="drafter",
            llm=_make_llm(),
            sys_prompt="Write one sentence about the topic.",
        )
        refiner2 = LLMAgent[str, str, None](
            name="refiner",
            llm=_make_llm(),
            sys_prompt=(
                "Improve the given draft — make it more vivid. "
                "Respond with just the improved sentence."
            ),
        )
        wf2 = SequentialWorkflow[str, str, None](
            name="draft_refine",
            subprocs=[drafter2, refiner2],
            recipients=[END_PROC_NAME],
        )

        entry2 = Passthrough(name="entry", recipients=["draft_refine"])
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="wf-crash-1"
        )
        runner2 = Runner[str, None](
            entry_proc=entry2, procs=[entry2, wf2], ctx=ctx2, name="r"
        )

        result = await runner2.run()
        payloads = list(result.payloads)
        assert len(payloads) == 1
        assert len(payloads[0]) > 10
