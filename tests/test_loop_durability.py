"""
Transcript / durability behavior:

* transcript builder fires on fresh init only (multi-turn history survives)
* pure resume (``agent.run()`` with no input) works instead of crashing
* force-final-answer commits items and counts usage exactly once
* a ``final_answer`` tool call (and its siblings) get paired tool results
* a failed run rolls the transcript back (no poisoning of reused instances)
* lenient LLM-layer arg validation is mirrored at dispatch; residual input
  failures become tool results, not crashes
* the append-only message log is rewritten when the persisted prefix changes
* ``validate_inputs`` coerces resumed dict payloads into typed models
* the approval gate propagates cancellation instead of converting it into a
  denial
* a deadline-backgrounded task persists its COMPLETED outcome
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.approval_store import (
    InMemoryApprovalStore,
    build_store_approval,
)
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.task_record import TaskRecord, TaskStatus
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
)
from grasp_agents.types.packet import Packet

from .test_background_tools import EchoInput, SlowTool
from .test_sessions import (  # type: ignore[attr-defined]
    MockLLM,
    _text_response,
    _tool_call_response,
    load_agent_checkpoint,
)


async def _persisted_log(store: InMemoryCheckpointStore, key: str) -> list[Any]:
    """The committed message-log as resume would see it (head's version)."""
    head = await load_agent_checkpoint(store, key)
    assert head is not None
    return list(head.messages)


def _make_agent(
    responses: list[Any],
    *,
    tools: list[BaseTool[Any, Any, Any]] | None = None,
    session_key: str | None = None,
    store: InMemoryCheckpointStore | None = None,
    **agent_kwargs: Any,
) -> tuple[LLMAgent[str, str, None], RunContext[None]]:
    ctx_kwargs: dict[str, Any] = {"checkpoint_store": store}
    if session_key is not None:
        ctx_kwargs["session_key"] = session_key
    ctx: RunContext[None] = RunContext(**ctx_kwargs)
    agent = LLMAgent[str, str, None](
        name="test_agent",
        ctx=ctx,
        llm=MockLLM(responses_queue=responses),
        tools=tools,
        **agent_kwargs,
    )
    return agent, ctx


class _EchoTool(BaseTool[EchoInput, str, Any]):
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes input")

    async def _run(self, inp: EchoInput, **kwargs: Any) -> str:
        return f"echo: {inp.text}"


# ---------------------------------------------------------------------------
# Transcript builder runs on every step (it owns transcript preparation)
# ---------------------------------------------------------------------------


class TestTranscriptBuilderRunsEveryStep:
    @pytest.mark.anyio
    async def test_builder_runs_each_step_and_can_preserve_history(self) -> None:
        agent, _ = _make_agent([_text_response("one"), _text_response("two")])
        calls: list[str] = []

        def prepare(*, instructions: Any = None, in_args: Any = None, exec_id: str):
            del in_args, exec_id
            calls.append("prepare")
            # Seed only a fresh transcript; later steps could prune/truncate.
            if agent.transcript.is_empty:
                agent.transcript.reset(instructions)

        agent.add_transcript_builder(prepare)

        await agent.run("first")
        n_after_turn_1 = len(agent.transcript.messages)
        await agent.run("second")

        assert calls == ["prepare", "prepare"]  # fires on every step
        # This builder preserved history, so turn 2 appended to it.
        assert len(agent.transcript.messages) > n_after_turn_1
        texts = [str(m) for m in agent.transcript.messages]
        assert any("first" in t for t in texts)
        assert any("second" in t for t in texts)

    @pytest.mark.anyio
    async def test_builder_mutation_rewrites_persisted_log(self) -> None:
        # The builder may rewrite history between steps — the persisted
        # message log must follow it, not keep the stale prefix.
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent(
            [_text_response("answer-alpha"), _text_response("answer-bravo")],
            session_key="tb-1",
            store=store,
        )

        def prepare(*, instructions: Any = None, in_args: Any = None, exec_id: str):
            del in_args, exec_id
            if agent.transcript.is_empty:
                agent.transcript.reset(instructions)
            else:
                # Prune the previous turn (the documented context-management
                # use of this hook).
                agent.transcript.messages = [
                    m for m in agent.transcript.messages if "alpha" not in str(m)
                ]

        agent.add_transcript_builder(prepare)

        await agent.run("input-alpha")
        await agent.run("input-bravo")

        persisted = await _persisted_log(store, "tb-1/agent/test_agent")
        assert len(persisted) == len(agent.transcript.messages)
        assert not any("alpha" in str(m) for m in persisted)


# ---------------------------------------------------------------------------
# Pure resume (no new input)
# ---------------------------------------------------------------------------


class TestPureResume:
    @pytest.mark.anyio
    async def test_resume_completed_session_returns_cached_output(self) -> None:
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent(
            [_text_response("answer")], session_key="s1", store=store
        )
        out = await agent.run("question")
        assert out.payloads[0] == "answer"

        # Fresh instance, same session: no inputs at all → resume, not crash.
        agent2, _ = _make_agent([], session_key="s1", store=store)
        out2 = await agent2.run()
        assert out2.payloads[0] == "answer"

    @pytest.mark.anyio
    async def test_no_input_and_nothing_to_resume_raises_cleanly(self) -> None:
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent([], session_key="fresh", store=store)
        with pytest.raises(ProcRunError) as excinfo:
            await agent.run()
        assert "No inputs were provided" in str(excinfo.value.__cause__)

    @pytest.mark.anyio
    async def test_after_tool_result_resumes_at_next_turn(self) -> None:
        """
        A run interrupted after a tool result resumes at the NEXT turn — it does
        not re-run the completed turn's generation (which would re-issue the
        same tool calls, e.g. re-launching background workers).
        """
        store = InMemoryCheckpointStore()
        calls: list[str] = []

        class _CountingEcho(BaseTool[EchoInput, str, Any]):
            def __init__(self) -> None:
                super().__init__(name="echo", description="Echoes input")

            async def _run(self, inp: EchoInput, **kwargs: Any) -> str:
                calls.append(inp.text)
                return f"echo: {inp.text}"

        agent, _ = _make_agent(
            [_tool_call_response("echo", '{"text": "once"}', "c1")],
            tools=[_CountingEcho()],
            session_key="ot1",
            store=store,
        )

        hits = {"n": 0}

        @agent.add_before_llm_hook
        async def crash_after_tool(**kwargs: Any) -> None:
            hits["n"] += 1
            if hits["n"] == 2:  # turn 0 issued the tool call; crash at turn 1
                raise RuntimeError("crash after tool result")

        with pytest.raises(BaseException):
            await agent.run("go")
        assert calls == ["once"]  # tool ran exactly once

        # The AFTER_TOOL_RESULT checkpoint records turn 1 (the resume point),
        # not turn 0 — the just-completed turn is not re-executed.
        head = await load_agent_checkpoint(store, "ot1/agent/test_agent")
        assert head is not None
        assert head.turn == 1, head.turn

        # Resume: the agent answers from the restored tool result without
        # re-calling echo.
        agent2, _ = _make_agent(
            [_text_response("done from tool result")],
            tools=[_CountingEcho()],
            session_key="ot1",
            store=store,
        )
        out = await agent2.run()
        assert out.payloads[0] == "done from tool result"
        assert calls == ["once"]  # echo was NOT re-issued on resume


# ---------------------------------------------------------------------------
# Force-final-answer: single commit, single usage record
# ---------------------------------------------------------------------------


class TestForceFinalAnswerSingleCommit:
    @pytest.mark.anyio
    async def test_no_duplicate_messages_or_usage(self) -> None:
        agent, ctx = _make_agent(
            [
                _tool_call_response("echo", '{"text": "x"}', "c1"),
                _text_response("forced final"),
            ],
            tools=[_EchoTool()],
            max_turns=1,
        )
        out = await agent.run("go")
        assert out.payloads[0] == "forced final"

        finals = [
            m
            for m in agent.transcript.messages
            if isinstance(m, OutputMessageItem) and "forced final" in str(m)
        ]
        assert len(finals) == 1

        # Two LLM responses → exactly two responses' usage (10 input tokens
        # each); the forced call must not be double-counted.
        usage = ctx.usage_tracker.total_usage
        assert usage.input_tokens == 20


# ---------------------------------------------------------------------------
# final_answer tool call gets a paired tool result
# ---------------------------------------------------------------------------


class _Answer(BaseModel):
    answer: str


def _make_answer_agent(
    responses: list[Any], **agent_kwargs: Any
) -> LLMAgent[str, _Answer, None]:
    ctx: RunContext[None] = RunContext()
    return LLMAgent[str, _Answer, None](
        name="test_agent",
        ctx=ctx,
        llm=MockLLM(responses_queue=responses),
        tools=[_EchoTool()],
        final_answer_as_tool_call=True,
        **agent_kwargs,
    )


class TestFinalAnswerToolCallClosure:
    @pytest.mark.anyio
    async def test_voluntary_final_answer_call_is_closed(self) -> None:
        agent = _make_answer_agent(
            [_tool_call_response("final_answer", '{"answer": "42"}', "fa_1")]
        )
        await agent.run("question")

        outputs = [
            m
            for m in agent.transcript.messages
            if isinstance(m, FunctionToolOutputItem) and m.call_id == "fa_1"
        ]
        assert len(outputs) == 1
        # The pairing invariant holds for the next run on this transcript.
        agent.transcript.validate_tool_call_pairing()

    @pytest.mark.anyio
    async def test_forced_final_answer_call_is_closed(self) -> None:
        agent = _make_answer_agent(
            [
                _tool_call_response("echo", '{"text": "x"}', "c1"),
                _tool_call_response("final_answer", '{"answer": "done"}', "fa_2"),
            ],
            max_turns=1,
        )
        await agent.run("question")

        outputs = [
            m
            for m in agent.transcript.messages
            if isinstance(m, FunctionToolOutputItem) and m.call_id == "fa_2"
        ]
        assert len(outputs) == 1
        agent.transcript.validate_tool_call_pairing()


# ---------------------------------------------------------------------------
# Failed-run rollback
# ---------------------------------------------------------------------------


class TestFailedRunRollback:
    @pytest.mark.anyio
    async def test_failed_run_restores_transcript(self) -> None:
        agent, _ = _make_agent([_text_response("bad"), _text_response("good")])
        agent_ctx = agent._loop.agent_ctx

        fail = True

        @agent.add_output_parser
        def parse(final_answer: str, *, in_args: Any = None, exec_id: str) -> str:
            del in_args, exec_id
            if fail:
                # Simulate mid-run context mutations that must roll back with
                # the transcript (a Read's ledger entry, a Bash `cd`).
                agent_ctx.file_edit_state.import_state({"/tmp/x.py": 1.0}, [])
                agent_ctx.shell_state.cwd = "/mutated"
                raise ValueError("parser boom")
            return final_answer

        before = list(agent.transcript.messages)
        with pytest.raises(ProcRunError) as excinfo:
            await agent.run("first try")
        assert "boom" in str(excinfo.value.__cause__)
        assert list(agent.transcript.messages) == before
        # The transcript-paired context state rolled back with it.
        assert agent_ctx.file_edit_state.read_file_state == {}
        assert agent_ctx.shell_state.cwd is None

        # The same (reused) instance runs cleanly afterwards — no dangling
        # input message, no pairing violation.
        fail = False
        out = await agent.run("second try")
        assert out.payloads[0] == "good"
        texts = [str(m) for m in agent.transcript.messages]
        assert sum("first try" in t for t in texts) == 0
        assert sum("second try" in t for t in texts) == 1


# ---------------------------------------------------------------------------
# Lenient validation ↔ dispatch parity
# ---------------------------------------------------------------------------


class TestLenientArgsDispatch:
    @pytest.mark.anyio
    async def test_python_literal_args_dispatch(self) -> None:
        # Single-quoted (Python-literal) args pass the LLM layer's lenient
        # validation; dispatch must accept them too instead of crashing.
        agent, _ = _make_agent(
            [
                _tool_call_response("echo", "{'text': 'hi'}", "c1"),
                _text_response("done"),
            ],
            tools=[_EchoTool()],
        )
        out = await agent.run("go")
        assert out.payloads[0] == "done"
        tool_outputs = [
            str(m)
            for m in agent.transcript.messages
            if isinstance(m, FunctionToolOutputItem)
        ]
        assert any("echo: hi" in t for t in tool_outputs)

    @pytest.mark.anyio
    async def test_input_converter_failure_becomes_tool_result(self) -> None:
        agent, _ = _make_agent(
            [
                _tool_call_response("echo", '{"text": "hi"}', "c1"),
                _text_response("recovered"),
            ],
            tools=[_EchoTool()],
        )

        @agent.add_tool_input_converter(tool_name="echo")
        async def convert(llm_args: Any, *, exec_id: str) -> Any:
            del exec_id
            raise RuntimeError("converter boom")

        out = await agent.run("go")
        assert out.payloads[0] == "recovered"
        tool_outputs = [
            str(m)
            for m in agent.transcript.messages
            if isinstance(m, FunctionToolOutputItem) and m.call_id == "c1"
        ]
        assert len(tool_outputs) == 1
        assert "input is invalid" in tool_outputs[0]


# ---------------------------------------------------------------------------
# Append-log divergence detection
# ---------------------------------------------------------------------------


class TestAppendLogPruneRewrite:
    @pytest.mark.anyio
    async def test_pruned_transcript_rewrites_log(self) -> None:
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent(
            [_text_response("answer-alpha"), _text_response("answer-bravo")],
            session_key="s1",
            store=store,
        )
        await agent.run("input-alpha")

        # The documented context-management pattern: prune messages in place
        # between turns (drop the first turn's user+assistant pair).
        kept = [
            m
            for m in agent.transcript.messages
            if "alpha" not in str(m)
            or not isinstance(m, (InputMessageItem, OutputMessageItem))
        ]
        agent.transcript.messages = kept

        await agent.run("input-bravo")

        persisted = await _persisted_log(store, "s1/agent/test_agent")
        assert len(persisted) == len(agent.transcript.messages)
        texts = [str(m) for m in persisted]
        assert not any("alpha" in t for t in texts)  # no stale prefix restored
        assert any("answer-bravo" in t for t in texts)


# ---------------------------------------------------------------------------
# validate_inputs coercion
# ---------------------------------------------------------------------------


class _PointInput(BaseModel):
    x: int
    y: int


class _PointProc(Processor[_PointInput, str, None]):
    pass


class TestValidateInputsCoercion:
    def test_packet_dict_payloads_coerced(self) -> None:
        proc = _PointProc(name="p")
        packet: Packet[Any] = Packet(
            payloads=[{"x": 1, "y": 2}], sender="other"
        )
        result = proc.validate_inputs(exec_id="e", in_packet=packet)
        assert result is not None
        assert isinstance(result[0], _PointInput)
        assert result[0].x == 1

    def test_raw_dict_in_args_coerced(self) -> None:
        proc = _PointProc(name="p")
        result = proc.validate_inputs(exec_id="e", in_args={"x": 3, "y": 4})
        assert result is not None
        assert isinstance(result[0], _PointInput)
        assert result[0].y == 4


# ---------------------------------------------------------------------------
# Approval gate cancellation
# ---------------------------------------------------------------------------


class TestApprovalCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_run_propagates_at_approval_gate(self) -> None:
        ctx: RunContext[None] = RunContext(approval_store=InMemoryApprovalStore())
        hook = build_store_approval()
        call = FunctionToolCallItem(call_id="c1", name="t", arguments="{}")

        task = asyncio.create_task(
            hook(tool_calls=[call], ctx=ctx, exec_id="e")
        )
        await asyncio.sleep(0.05)  # let the hook park on the pending future
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()


# ---------------------------------------------------------------------------
# Deadline-backgrounded task persists COMPLETED
# ---------------------------------------------------------------------------


class TestDeadlineBackgroundedOutcomePersisted:
    @pytest.mark.asyncio
    async def test_sidelined_task_record_reaches_completed(self) -> None:
        store = InMemoryCheckpointStore()
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="s1"
        )
        transcript = LLMAgentTranscript()
        transcript.reset(instructions="sys")
        mgr = BackgroundTaskManager[None](
            agent_name="t", transcript=transcript, tools={}, path=[]
        )

        tool = SlowTool(delay=0.3)
        tool.auto_background_at = 0.05  # sideline after 50ms, finish at 300ms
        call = FunctionToolCallItem(
            call_id="c1", name="slow", arguments='{"text": "x"}'
        )
        note, _ = await mgr.run_backgroundable(
            call, tool, EchoInput(text="x"), ctx=ctx, exec_id="e"
        )
        assert "background" in str(note)

        # Wait for the sidelined task to finish, without draining.
        pending = list(mgr._tasks.values())
        assert pending
        await asyncio.gather(*(pt.task for pt in pending))

        keys = await store.list_keys("s1/task/")
        assert keys
        records = [
            TaskRecord.model_validate_json(await store.load(k) or b"")
            for k in keys
        ]
        assert all(r.status == TaskStatus.COMPLETED for r in records)
        assert any(r.result and "slow: x" in r.result for r in records)
