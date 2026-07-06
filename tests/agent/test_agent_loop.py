"""
Agent-loop behavior: final-answer tool registration and forced synthesis,
tool-result pairing, closure-event streaming, approval-pending cleanup,
MCP-client construction, and ``None`` input handling.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

import pytest
from pydantic import BaseModel

from grasp_agents.agent.agent_loop import AgentLoop
from grasp_agents.agent.approval_store import (
    ApprovalAllow,
    LocalApprovalStore,
    build_store_approval,
)
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.agent.tool_decision import RejectToolContent
from grasp_agents.context.prompt_builder import PromptBuilder
from grasp_agents.context.untrusted_content import UNTRUSTED_CONTENT_SECTION_NAME
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.events import (
    Event,
    ToolOutputEvent,
    ToolOutputItemEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
)
from grasp_agents.types.response import Response
from tests._helpers import (
    MockLLM,
    _make_agent_loop,
    _make_usage,
    _text_response,
    _tool_call_response,
)

# ---------- Infrastructure ----------


class EchoInput(BaseModel):
    text: str


class EchoTool(BaseTool[EchoInput, Any, Any]):
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes input")

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        return f"echo: {inp.text}"


class NestedEventTool(BaseTool[EchoInput, Any, Any]):
    """
    Simulates a foreground sub-agent: bubbles an *internal* tool_result
    event (stamped with the sub-agent as destination) before its own
    terminal output.
    """

    def __init__(self) -> None:
        super().__init__(name="nested", description="Bubbles nested events")

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        return "outer result"

    async def run_stream(
        self,
        inp: EchoInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
        path: Any = None,
        agent_ctx: Any = None,
    ) -> AsyncIterator[Event[Any]]:
        inner_msg = FunctionToolOutputItem.from_tool_result(
            call_id="inner-call-1", output="inner tool result"
        )
        yield ToolOutputItemEvent(
            source="inner_tool",
            destination="inner_agent",
            exec_id=exec_id,
            data=inner_msg,
        )
        yield ToolOutputEvent(data="outer result", source=self.name, exec_id=exec_id)


def _make_loop(
    responses: list[Response],
    *,
    tools: list[BaseTool[Any, Any, Any]] | None = None,
    max_turns: int = 10,
    final_answer_as_tool_call: bool = False,
    final_answer_type: type[BaseModel] = BaseModel,
    retry_policy: Any = None,
) -> tuple[AgentLoop[None], LLMAgentTranscript]:
    llm = MockLLM(responses_queue=responses, retry_policy=retry_policy)
    transcript = LLMAgentTranscript()
    transcript.messages = [InputMessageItem.from_text("sys", role="system")]
    transcript.update([InputMessageItem.from_text("go", role="user")])

    loop = _make_agent_loop(
        agent_name="test",
        llm=llm,
        transcript=transcript,
        tools=tools,
        ctx=SessionContext[None](state=None),
        max_turns=max_turns,
        final_answer_as_tool_call=final_answer_as_tool_call,
        final_answer_type=final_answer_type,
        stream_llm=False,
    )
    return loop, transcript


async def _drain(loop: AgentLoop[None]) -> list[Event[Any]]:
    events: list[Event[Any]] = []
    async for event in loop.execute_stream(exec_id="t"):
        events.append(event)
    return events


def _tool_outputs_for(
    transcript: LLMAgentTranscript, call_id: str
) -> list[FunctionToolOutputItem]:
    return [
        m
        for m in transcript.messages
        if isinstance(m, FunctionToolOutputItem) and m.call_id == call_id
    ]


# ---------- Item 1: final_answer tool with empty tools ----------


class _Answer(BaseModel):
    answer: str


class TestFinalAnswerToolRegistration:
    def test_registered_without_other_tools(self) -> None:
        loop, _ = _make_loop(
            [],
            tools=None,
            final_answer_as_tool_call=True,
            final_answer_type=_Answer,
        )
        assert "final_answer" in loop.agent_ctx.tools

    @pytest.mark.asyncio
    async def test_loop_stops_on_final_answer_call(self) -> None:
        loop, _ = _make_loop(
            [_tool_call_response("final_answer", '{"answer": "42"}', "fa1")],
            tools=None,
            final_answer_as_tool_call=True,
            final_answer_type=_Answer,
        )
        await _drain(loop)
        assert loop.final_answer == '{"answer": "42"}'


# ---------- Item 2: forced final answer runs LLM hooks ----------


class TestForcedFinalAnswerHooks:
    @pytest.mark.asyncio
    async def test_after_llm_hook_fires_on_forced_call(self) -> None:
        responses = [
            _tool_call_response("echo", '{"text":"x"}', "tc1"),
            _text_response("forced"),
        ]
        loop, _ = _make_loop(responses, tools=[EchoTool()], max_turns=0)

        seen: list[str] = []

        async def after_hook(response: Response, *, exec_id: str, turn: int) -> None:
            seen.append(response.output_text or "<tool call>")

        loop.after_llm_hooks = [after_hook]  # type: ignore[assignment]

        await _drain(loop)

        # Once for the regular ACT call, once for the forced final answer.
        assert len(seen) == 2
        assert seen[-1] == "forced"


# ---------- Items 3 + 8: no duplicate tool_results for skip-marked calls ----------


class TestNoDuplicateToolResults:
    @pytest.mark.asyncio
    async def test_reject_decision_on_validation_skipped_call(self) -> None:
        """
        A call whose result was synthesized at the LLM validation layer
        must not get a second tool_result from a Reject decision.
        """
        responses = [
            _tool_call_response("echo", '{"wrong_field": 1}', "tc1"),
            _text_response("done"),
        ]
        loop, transcript = _make_loop(responses, tools=[EchoTool()])

        async def reject_all(
            *,
            tool_calls: Sequence[FunctionToolCallItem],
            ctx: Any,
            exec_id: str,
        ) -> Mapping[str, Any]:
            return {c.call_id: RejectToolContent(content="denied") for c in tool_calls}

        loop.before_tool_hooks = [reject_all]  # type: ignore[assignment]

        await _drain(loop)

        assert len(_tool_outputs_for(transcript, "tc1")) == 1
        transcript.validate_tool_call_pairing()

    @pytest.mark.asyncio
    async def test_max_turns_after_validation_synthesis(self) -> None:
        """
        Budget exhaustion right after validation synthesis must not pair
        the failed call a second time via _close_dangling_tool_calls.
        """
        responses = [
            _tool_call_response("echo", '{"wrong_field": 1}', "tc1"),
            _text_response("forced"),
        ]
        loop, transcript = _make_loop(responses, tools=[EchoTool()], max_turns=0)

        await _drain(loop)

        assert loop.final_answer == "forced"
        assert len(_tool_outputs_for(transcript, "tc1")) == 1
        transcript.validate_tool_call_pairing()

    @pytest.mark.asyncio
    async def test_stop_with_validation_synthesized_sibling(self) -> None:
        """
        A stopping response whose sibling call already has a synthesized
        result only closes the unanswered calls.
        """
        bad_call = FunctionToolCallItem(
            call_id="bad1", name="echo", arguments='{"wrong_field": 1}'
        )
        fa_call = FunctionToolCallItem(
            call_id="fa1", name="final_answer", arguments='{"answer": "done"}'
        )
        response = Response(
            model="mock",
            output=[bad_call, fa_call],
            usage=_make_usage(),
        )
        loop, transcript = _make_loop(
            [response],
            tools=[EchoTool()],
            final_answer_as_tool_call=True,
            final_answer_type=_Answer,
        )

        await _drain(loop)

        assert loop.final_answer == '{"answer": "done"}'
        assert len(_tool_outputs_for(transcript, "bad1")) == 1
        assert len(_tool_outputs_for(transcript, "fa1")) == 1
        transcript.validate_tool_call_pairing()


# ---------- Synthetic closure outputs are announced on the event stream ----------


def _closure_output_events(
    events: list[Event[Any]],
) -> dict[str, ToolOutputItemEvent]:
    return {e.data.call_id: e for e in events if isinstance(e, ToolOutputItemEvent)}


class TestClosureEventsStreamed:
    @pytest.mark.asyncio
    async def test_dangling_cancellation_streams_on_max_turns(self) -> None:
        """
        The turn-limit cancellation reaches the event stream, not just
        the transcript — event-driven consumers must not diverge.
        """
        responses = [
            _tool_call_response("echo", '{"text":"x"}', "tc1"),
            _text_response("forced"),
        ]
        # max_turns=0: the call dangles (never executes) and is closed by
        # the forced-final-answer path.
        loop, transcript = _make_loop(responses, tools=[EchoTool()], max_turns=0)

        events = await _drain(loop)

        by_call = _closure_output_events(events)
        assert "tc1" in by_call
        cancellation = by_call["tc1"]
        assert "turn limit" in str(cancellation.data.output)
        assert cancellation.source == "echo"
        assert cancellation.destination == "test"
        # Event/transcript parity: the streamed item IS the persisted one.
        assert cancellation.data in transcript.messages

    @pytest.mark.asyncio
    async def test_stop_closures_stream(self) -> None:
        """
        Both stop closures stream: the final_answer acknowledgement and
        the cancelled sibling call.
        """
        sibling = FunctionToolCallItem(
            call_id="tc1", name="echo", arguments='{"text":"x"}'
        )
        fa_call = FunctionToolCallItem(
            call_id="fa1", name="final_answer", arguments='{"answer": "done"}'
        )
        response = Response(
            model="mock",
            output=[sibling, fa_call],
            usage=_make_usage(),
        )
        loop, _ = _make_loop(
            [response],
            tools=[EchoTool()],
            final_answer_as_tool_call=True,
            final_answer_type=_Answer,
        )

        events = await _drain(loop)

        by_call = _closure_output_events(events)
        assert "Final answer recorded." in str(by_call["fa1"].data.output)
        assert "stopped with a final answer" in str(by_call["tc1"].data.output)

    @pytest.mark.asyncio
    async def test_closures_not_fed_to_after_tool_hook(self) -> None:
        """
        Closures are bookkeeping, not executions — the after-tool hook
        must not receive them.
        """
        responses = [
            # Turn 0: executes (hook fires with its real result).
            _tool_call_response("echo", '{"text":"ran"}', "tc1"),
            # Turn 1: dangles — max_turns reached before it can run.
            _tool_call_response("echo", '{"text":"never"}', "tc2"),
            _text_response("forced"),
        ]
        loop, _ = _make_loop(responses, tools=[EchoTool()], max_turns=1)

        received: list[str] = []

        async def after_tool(
            *,
            tool_calls: Sequence[FunctionToolCallItem],
            tool_messages: Sequence[FunctionToolOutputItem],
            exec_id: str,
        ) -> None:
            received.extend(str(m.output) for m in tool_messages)

        loop.after_tool_hooks = [after_tool]  # type: ignore[assignment]

        events = await _drain(loop)

        # The dangling call's cancellation streamed…
        assert "tc2" in _closure_output_events(events)
        # …but the hook saw only the real execution's result.
        assert any("ran" in r for r in received)
        assert not any("turn limit" in r for r in received)


# ---------- Item 4: after-tool hook sees only this agent's results ----------


class TestAfterToolHookPairing:
    @pytest.mark.asyncio
    async def test_nested_subagent_results_not_collected(self) -> None:
        responses = [
            _tool_call_response("nested", '{"text":"x"}', "tc1"),
            _text_response("done"),
        ]
        loop, _ = _make_loop(responses, tools=[NestedEventTool()])

        received: list[Sequence[FunctionToolOutputItem]] = []

        async def after_tool(
            *,
            tool_calls: Sequence[FunctionToolCallItem],
            tool_messages: Sequence[FunctionToolOutputItem],
            exec_id: str,
        ) -> None:
            received.append(tool_messages)

        loop.after_tool_hooks = [after_tool]  # type: ignore[assignment]

        await _drain(loop)

        assert len(received) == 1
        (messages,) = received
        # Only the call's own result — not the bubbled inner-agent result.
        assert [m.call_id for m in messages] == ["tc1"]


# ---------- Item 5: approval timeouts clear pending entries ----------


class TestApprovalPendingCleanup:
    @pytest.mark.asyncio
    async def test_timeout_clears_pending_and_late_resolve_fails(self) -> None:
        store = LocalApprovalStore()
        hook = build_store_approval(timeout=0.05)
        ctx = SessionContext[None](approval_store=store, session_key="s1")
        call = FunctionToolCallItem(call_id="c1", name="t", arguments="{}")

        decisions = await hook(tool_calls=[call], ctx=ctx, exec_id="e")

        assert decisions is not None
        assert isinstance(decisions["c1"], RejectToolContent)
        assert await store.list_pending("s1") == []
        # The phantom entry is gone, so a late resolve reports failure.
        assert await store.resolve("s1", "c1", ApprovalAllow()) is False

    @pytest.mark.asyncio
    async def test_cancellation_clears_pending(self) -> None:
        store = LocalApprovalStore()
        hook = build_store_approval(timeout=None)
        ctx = SessionContext[None](approval_store=store, session_key="s1")
        call = FunctionToolCallItem(call_id="c1", name="t", arguments="{}")

        task = asyncio.create_task(hook(tool_calls=[call], ctx=ctx, exec_id="e"))
        # Let the hook park on the pending future, then cancel the run.
        for _ in range(100):
            if await store.list_pending("s1"):
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert await store.list_pending("s1") == []


# ---------- Item 6: mcp_clients= ctor path (section + filter + collisions) ------


class _StubMCPClient:
    instructions: str | None = None

    def __init__(self, tools: list[BaseTool[Any, Any, Any]]) -> None:
        self._tools = tools

    def tools(self) -> list[BaseTool[Any, Any, Any]]:
        return self._tools


def _section_names(agent: LLMAgent[Any, Any, Any]) -> list[str]:
    return [
        s.name
        for s in agent._prompt_builder.system_prompt_sections  # pyright: ignore[reportPrivateUsage]
    ]


def _agent_with_mcp(clients: list[Any]) -> LLMAgent[str, str, None]:
    return LLMAgent[str, str, None](
        name="t",
        llm=MockLLM(responses_queue=[]),
        mcp_clients=clients,
        stream_llm=True,
        env_info=False,
    )


class TestMcpClientsCtor:
    def test_registers_untrusted_content_section(self) -> None:
        bare = LLMAgent[str, str, None](
            name="t", llm=MockLLM(responses_queue=[]), stream_llm=True, env_info=False
        )
        assert UNTRUSTED_CONTENT_SECTION_NAME not in _section_names(bare)

        agent = _agent_with_mcp([cast("Any", _StubMCPClient([EchoTool()]))])
        assert UNTRUSTED_CONTENT_SECTION_NAME in _section_names(agent)
        assert "echo" in agent.tools

    def test_include_exclude_filter(self) -> None:
        from grasp_agents.mcp.spec import MCPClientSpec

        client = _StubMCPClient([EchoTool(), NestedEventTool()])
        agent = _agent_with_mcp(
            [cast("Any", MCPClientSpec(client=cast("Any", client), include={"echo"}))]
        )
        assert "echo" in agent.tools
        assert "nested" not in agent.tools

    def test_explicit_tool_wins_over_mcp_collision(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # MCP tools are auto-sourced (the server names them, not the user), so a
        # clash with an explicit tool is resolved in the explicit tool's favour:
        # the MCP one is skipped with a warning, never a construction error.
        native = EchoTool()
        with caplog.at_level(logging.WARNING, logger="grasp_agents.agent.llm_agent"):
            agent = LLMAgent[str, str, None](
                name="t",
                llm=MockLLM(responses_queue=[]),
                tools=[native],
                mcp_clients=[cast("Any", _StubMCPClient([EchoTool()]))],
                stream_llm=True,
                env_info=False,
            )
        assert agent.tools["echo"] is native  # the explicit instance wins
        assert "shadows an existing tool" in caplog.text

    def test_cross_client_collision_keeps_first(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Two auto-sourced MCP tools with the same name: the first registered
        # wins, the rest are skipped with a warning (no construction error).
        first = EchoTool()
        with caplog.at_level(logging.WARNING, logger="grasp_agents.agent.llm_agent"):
            agent = _agent_with_mcp(
                [
                    cast("Any", _StubMCPClient([first])),
                    cast("Any", _StubMCPClient([EchoTool()])),
                ]
            )
        assert agent.tools["echo"] is first
        assert "shadows an existing tool" in caplog.text


# ---------- Item 7: InT=None produces no junk input message ----------


class TestNoneInputType:
    def test_no_message_for_none_in_type(self) -> None:
        builder = PromptBuilder[type(None), None](  # type: ignore[type-arg]
            agent_name="t", sys_prompt=None, in_prompt=None
        )
        assert builder.build_input_message(None, in_args=None, exec_id="e") is None

    def test_chat_inputs_still_build_message(self) -> None:
        builder = PromptBuilder[type(None), None](  # type: ignore[type-arg]
            agent_name="t", sys_prompt=None, in_prompt=None
        )
        msg = builder.build_input_message("hello", in_args=None, exec_id="e")
        assert msg is not None
