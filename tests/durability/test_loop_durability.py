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
    LocalApprovalStore,
    build_store_approval,
)
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import AgentCheckpointLocation
from grasp_agents.durability.task_record import TaskRecord, TaskStatus
from grasp_agents.processors.processor import Processor
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.events import StopReason, TurnEndEvent, TurnStartEvent
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
)
from grasp_agents.types.packet import Packet
from tests._helpers import MockLLM, _text_response, _tool_call_response
from tests.agent.test_background_tools import EchoInput, SlowTool
from tests.durability.test_sessions import load_agent_checkpoint


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
) -> tuple[LLMAgent[str, str, None], SessionContext[None]]:
    ctx_kwargs: dict[str, Any] = {"checkpoint_store": store}
    if session_key is not None:
        ctx_kwargs["session_key"] = session_key
    ctx: SessionContext[None] = SessionContext(**ctx_kwargs)
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
# Pure resume (no new input)
# ---------------------------------------------------------------------------


class TestPureResume:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_no_input_and_nothing_to_resume_raises_cleanly(self) -> None:
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent([], session_key="fresh", store=store)
        with pytest.raises(ProcRunError) as excinfo:
            await agent.run()
        assert "No inputs were provided" in str(excinfo.value.__cause__)

    @pytest.mark.asyncio
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
        assert head.current.turn == 1, head.current.turn

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
# Turn boundary events are the post-durability lane
# ---------------------------------------------------------------------------


class TestTurnEndAfterCheckpoint:
    @pytest.mark.asyncio
    async def test_tool_round_turn_end_follows_checkpoint_commit(self) -> None:
        """
        When a consumer receives the tool round's ``TurnEndEvent``, the round is
        already committed to the checkpoint store — a side effect keyed on the
        event's ``tool_outputs`` can never double across a crash-resume.
        (Item events are the opposite, at-least-once lane.)
        """
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent(
            [
                _tool_call_response("echo", '{"text": "x"}', "c1"),
                _text_response("done"),
            ],
            tools=[_EchoTool()],
            session_key="te1",
            store=store,
        )

        turn_end = None
        async for event in agent.run_stream("go", step=5):
            if isinstance(event, TurnStartEvent) and event.data.turn == 0:
                assert [
                    m.text
                    for m in event.data.input_messages
                    if isinstance(m, InputMessageItem)
                ] == ["go"]
            if isinstance(event, TurnEndEvent) and event.data.tool_outputs:
                turn_end = event
                # Observed mid-stream: the store must already hold the round.
                persisted = await _persisted_log(store, "te1/agent/test_agent")
                assert any(
                    isinstance(m, FunctionToolOutputItem) and m.call_id == "c1"
                    for m in persisted
                )

        assert turn_end is not None
        assert [o.call_id for o in turn_end.data.tool_outputs] == ["c1"]


# ---------------------------------------------------------------------------
# Force-final-answer: single commit, single usage record
# ---------------------------------------------------------------------------


class TestForceFinalAnswerSingleCommit:
    @pytest.mark.asyncio
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
# Persisted head records how the run ended
# ---------------------------------------------------------------------------


class TestHeadStopReason:
    @pytest.mark.asyncio
    async def test_clean_stop_persists_final_answer_reason(self) -> None:
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent(
            [_text_response("answer")], session_key="sr1", store=store
        )
        await agent.run("question")

        head = await load_agent_checkpoint(store, "sr1/agent/test_agent")
        assert head is not None
        assert head.location is AgentCheckpointLocation.AFTER_FINAL_ANSWER
        assert head.stop_reason is StopReason.FINAL_ANSWER

    @pytest.mark.asyncio
    async def test_forced_stop_persists_max_turns_reason(self) -> None:
        store = InMemoryCheckpointStore()
        # The over-budget turn must NOT be a final answer (a clean stop wins
        # then): a second tool call forces the answer via a third LLM call.
        agent, _ = _make_agent(
            [
                _tool_call_response("echo", '{"text": "x"}', "c1"),
                _tool_call_response("echo", '{"text": "y"}', "c2"),
                _text_response("forced final"),
            ],
            tools=[_EchoTool()],
            max_turns=1,
            session_key="sr2",
            store=store,
        )
        await agent.run("go")

        head = await load_agent_checkpoint(store, "sr2/agent/test_agent")
        assert head is not None
        assert head.location is AgentCheckpointLocation.AFTER_FINAL_ANSWER
        assert head.stop_reason is StopReason.MAX_TURNS

    @pytest.mark.asyncio
    async def test_mid_run_head_has_no_stop_reason(self) -> None:
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent(
            [
                _tool_call_response("echo", '{"text": "x"}', "c1"),
                _text_response("answer"),
            ],
            tools=[_EchoTool()],
            session_key="sr3",
            store=store,
        )

        seen: list[tuple[AgentCheckpointLocation, StopReason | None]] = []

        async for event in agent.run_stream("go"):
            if isinstance(event, TurnEndEvent):
                head = await load_agent_checkpoint(store, "sr3/agent/test_agent")
                assert head is not None
                seen.append((head.location, head.stop_reason))

        assert (AgentCheckpointLocation.AFTER_TOOL_RESULT, None) in seen
        assert seen[-1] == (
            AgentCheckpointLocation.AFTER_FINAL_ANSWER,
            StopReason.FINAL_ANSWER,
        )


# ---------------------------------------------------------------------------
# final_answer tool call gets a paired tool result
# ---------------------------------------------------------------------------


class _Answer(BaseModel):
    answer: str


def _make_answer_agent(
    responses: list[Any], **agent_kwargs: Any
) -> LLMAgent[str, _Answer, None]:
    ctx: SessionContext[None] = SessionContext()
    return LLMAgent[str, _Answer, None](
        name="test_agent",
        ctx=ctx,
        llm=MockLLM(responses_queue=responses),
        tools=[_EchoTool()],
        final_answer_as_tool_call=True,
        **agent_kwargs,
    )


class TestFinalAnswerToolCallClosure:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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


class TestFailedRunSettle:
    @pytest.mark.asyncio
    async def test_failed_run_settles_to_last_closed_turn(self) -> None:
        agent, _ = _make_agent([_text_response("bad"), _text_response("good")])
        agent_ctx = agent._loop.agent_ctx

        fail = True

        @agent.add_output_parser
        def parse(final_answer: str, *, in_args: Any = None, exec_id: str) -> str:
            del in_args, exec_id
            if fail:
                # Simulate mutations after the last checkpoint boundary that
                # must roll back with the pruned answer (a Read's ledger
                # entry, a Bash `cd`).
                agent_ctx.file_edit_state.import_state({"/tmp/x.py": 1.0}, [])
                agent_ctx.shell_state.cwd = "/mutated"
                raise ValueError("parser boom")
            return final_answer

        with pytest.raises(ProcRunError) as excinfo:
            await agent.run("first try")
        assert "boom" in str(excinfo.value.__cause__)
        # Settled, not reverted: the input stays; the unparseable answer —
        # not a closed turn — is pruned so a retry regenerates it.
        texts = [str(m) for m in agent.transcript.messages]
        assert sum("first try" in t for t in texts) == 1
        assert sum("bad" in t for t in texts) == 0
        # The context state rolled back to the last checkpoint boundary.
        assert agent_ctx.file_edit_state.read_file_state == {}
        assert agent_ctx.shell_state.cwd is None

        # The same (reused) instance runs cleanly afterwards — no pairing
        # violation, and the new input lands once.
        fail = False
        out = await agent.run("second try")
        assert out.payloads[0] == "good"
        agent.transcript.validate_tool_call_pairing()
        texts = [str(m) for m in agent.transcript.messages]
        assert sum("second try" in t for t in texts) == 1

    @pytest.mark.asyncio
    async def test_first_round_crash_storeless_settles_from_committed_head(
        self,
    ) -> None:
        """
        Crash in round 1 with NO checkpoint store and no completed round: the
        only prior save is the AFTER_INPUT checkpoint at stream entry, which
        maintains the in-memory head (``_committed``) even without a store.
        The settle's context restore hinges on that head existing — pin it
        directly, so the wiring (unconditional saves + AFTER_INPUT before the
        first LLM call) is load-bearing rather than incidental.
        """
        agent, _ = _make_agent(
            [_tool_call_response("echo", '{"text": "hi"}', "c1")],
            tools=[_EchoTool()],
        )
        agent_ctx = agent._loop.agent_ctx
        assert agent._committed is None  # nothing saved before the first run

        @agent.add_before_tool_hook
        async def crash(*, tool_calls: Any, ctx: Any, exec_id: str) -> None:
            del tool_calls, ctx, exec_id
            # Mutations after the AFTER_INPUT boundary that must roll back
            # with the pruned (dangling) tool-call round.
            agent_ctx.shell_state.cwd = "/mutated"
            raise RuntimeError("boom in round 1")

        with pytest.raises(ProcRunError):
            await agent.run("first try")

        # The AFTER_INPUT save left a committed head despite the missing
        # store, so the settle restored the paired context state...
        assert agent._committed is not None
        assert agent_ctx.shell_state.cwd is None
        # ...and pruned the dangling tool-call round, keeping the input.
        texts = [str(m) for m in agent.transcript.messages]
        assert sum("first try" in t for t in texts) == 1
        assert all("echo" not in t for t in texts)
        agent.transcript.validate_tool_call_pairing()


# ---------------------------------------------------------------------------
# Lenient validation ↔ dispatch parity
# ---------------------------------------------------------------------------


class TestLenientArgsDispatch:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
# View projection leaves the durable log intact
# ---------------------------------------------------------------------------


class TestViewProjectionLeavesLogIntact:
    @pytest.mark.asyncio
    async def test_compacting_projector_does_not_rewrite_log(self) -> None:
        # E0: context management lives in the view projector, which compacts
        # what the LLM sees without touching the durable log. The persisted
        # conversation keeps full fidelity (no rewrite), so rollback / resume
        # still see everything.
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent(
            [_text_response("answer-alpha"), _text_response("answer-bravo")],
            session_key="s1",
            store=store,
        )

        @agent.add_view_projector
        async def compact(messages: Any, *, exec_id: str, input_tokens: int) -> Any:
            del exec_id, input_tokens
            return messages[-1:]  # show the LLM only the latest message

        await agent.run("input-alpha")
        await agent.run("input-bravo")

        # Compaction never rewrote the log; the full conversation is persisted.
        assert agent._log_version == 0
        persisted = await _persisted_log(store, "s1/agent/test_agent")
        assert len(persisted) == len(agent.transcript.messages)
        texts = [str(m) for m in persisted]
        assert any("answer-alpha" in t for t in texts)  # full history retained
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
        packet: Packet[Any] = Packet(payloads=[{"x": 1, "y": 2}], sender="other")
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
        ctx: SessionContext[None] = SessionContext(approval_store=LocalApprovalStore())
        hook = build_store_approval()
        call = FunctionToolCallItem(call_id="c1", name="t", arguments="{}")

        task = asyncio.create_task(hook(tool_calls=[call], ctx=ctx, exec_id="e"))
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
        ctx: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="s1"
        )
        transcript = LLMAgentTranscript()
        transcript.messages = [InputMessageItem.from_text("sys", role="system")]
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
        await asyncio.gather(*(pt.consumer for pt in pending))

        keys = await store.list_keys("s1/task/")
        assert keys
        records = [
            TaskRecord.model_validate_json(await store.load(k) or b"") for k in keys
        ]
        assert all(r.status == TaskStatus.COMPLETED for r in records)
        assert any(r.result and "slow: x" in r.result for r in records)
