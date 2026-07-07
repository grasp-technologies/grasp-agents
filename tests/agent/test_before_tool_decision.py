"""
Tests for :class:`BeforeToolHook` decision returns.

Verifies that the hook's return value controls per-call tool execution:
* ``None`` / missing entries → ``AllowTool`` (default, tool runs normally)
* ``RejectToolContent`` → tool skipped, synthetic output reaches the LLM
* ``RaiseToolException`` → batch aborts by raising, no tools run
* Mixed decisions → only ``AllowTool`` calls execute
* Hooks that return nothing keep working
"""

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel

import grasp_agents
from grasp_agents.agent.agent_loop import AgentLoop, ResponseCapture
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.agent.tool_decision import (
    AllowTool,
    RaiseToolException,
    RejectToolContent,
    ToolCallDecision,
)
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
)
from grasp_agents.types.response import Response
from tests._helpers import MockLLM, _make_agent_loop, _make_usage, _text_response

# ---------- Infrastructure ----------


def _tool_call_response(
    calls: Sequence[tuple[str, str, str]],
) -> Response:
    """Build a response containing one or more function tool calls."""
    return Response(
        model="mock",
        output=[
            FunctionToolCallItem(call_id=call_id, name=name, arguments=args)
            for name, args, call_id in calls
        ],
        usage=_make_usage(),
    )


class EchoInput(BaseModel):
    text: str


class EchoTool(BaseTool[EchoInput, Any, Any]):
    """Tracks invocation count so tests can assert execution vs skip."""

    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes input")
        # Pydantic-backed BaseTool rejects unknown attrs; stash on module scope.
        _invocations[self.name] = []

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        _invocations[self.name].append(inp.text)
        return f"echo: {inp.text}"


class ShoutTool(BaseTool[EchoInput, Any, Any]):
    def __init__(self) -> None:
        super().__init__(name="shout", description="Shouts input")
        _invocations[self.name] = []

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        _invocations[self.name].append(inp.text)
        return f"SHOUT: {inp.text.upper()}"


# Module-level registry because BaseTool (Pydantic model) doesn't allow
# arbitrary attributes on instances.
_invocations: dict[str, list[str]] = {}


def _make_executor(
    responses: list[Response],
    *,
    tools: list[BaseTool[Any, Any, Any]] | None = None,
    max_turns: int = 10,
) -> tuple[AgentLoop[None], LLMAgentTranscript, MockLLM]:
    llm = MockLLM(model_name="mock", responses_queue=responses)
    memory = LLMAgentTranscript()
    memory.messages = [InputMessageItem.from_text("sys", role="system")]
    memory.update([InputMessageItem.from_text("go", role="user")])

    ctx = SessionContext[None](state=None)
    executor = _make_agent_loop(
        agent_name="test",
        llm=llm,
        transcript=memory,
        ctx=ctx,
        tools=tools,
        max_turns=max_turns,
        stream_llm=False,
    )
    # Final-answer extractor: stop on text-only responses.
    executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
        response.output_text if response and not response.tool_call_items else None
    )
    return executor, memory, llm


async def _drain(executor: AgentLoop[None], ctx: SessionContext[None]) -> Response:
    executor.ctx = ctx
    stream = ResponseCapture(executor.execute_stream(exec_id="t"))
    async for _ in stream:
        pass
    assert stream.response is not None
    return stream.response


# ---------- Tests ----------


class TestDefaultAllowBehavior:
    """Missing / None decisions leave existing behavior unchanged."""

    @pytest.mark.asyncio
    async def test_none_return_allows_all(self):
        """Hook returning None (current API) → tool runs normally."""
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])

        async def hook(*, tool_calls, ctx, exec_id):
            return None

        executor.before_tool_hooks = [hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)
        assert _invocations["echo"] == ["hi"]

    @pytest.mark.asyncio
    async def test_empty_mapping_allows_all(self):
        """Hook returning an empty dict → all calls default to AllowTool."""
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"ok"}', "tc1")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])

        async def hook(*, tool_calls, ctx, exec_id):
            return {}

        executor.before_tool_hooks = [hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)
        assert _invocations["echo"] == ["ok"]

    @pytest.mark.asyncio
    async def test_allow_decision_runs_tool(self):
        """Explicit AllowTool for a call is identical to not specifying it."""
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])

        async def hook(*, tool_calls, ctx, exec_id):
            return {"tc1": AllowTool()}

        executor.before_tool_hooks = [hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)
        assert _invocations["echo"] == ["hi"]


class TestRejectToolContent:
    """Rejection synthesizes output and skips real execution."""

    @pytest.mark.asyncio
    async def test_rejection_skips_execution(self):
        """
        Rejected call: real tool does NOT run; synthesized output is
        written to memory; loop continues normally.
        """
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool()])

        async def hook(*, tool_calls, ctx, exec_id):
            return {"tc1": RejectToolContent(content="blocked by policy")}

        executor.before_tool_hooks = [hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        # Real tool never ran
        assert _invocations["echo"] == []

        # Synthesized output is in memory with the rejection content
        tool_outputs = [
            m for m in memory.messages if isinstance(m, FunctionToolOutputItem)
        ]
        assert len(tool_outputs) == 1
        assert tool_outputs[0].call_id == "tc1"
        assert "blocked by policy" in tool_outputs[0].text

    @pytest.mark.asyncio
    async def test_after_tool_hook_sees_rejections(self):
        """
        AfterToolHook receives the full tool_messages list including
        synthesized rejections — downstream consumers can't distinguish
        rejected from executed calls by inspecting the messages alone.
        """
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"x"}', "tc1")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])

        async def before(*, tool_calls, ctx, exec_id):
            return {"tc1": RejectToolContent(content="nope")}

        captured: dict[str, Any] = {}

        async def after(*, tool_calls, tool_messages, exec_id):
            captured["num_messages"] = len(tool_messages)
            captured["first_text"] = tool_messages[0].text

        executor.before_tool_hooks = [before]  # type: ignore[assignment]
        executor.after_tool_hooks = [after]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert captured["num_messages"] == 1
        assert "nope" in captured["first_text"]


class TestRaiseToolException:
    """RaiseToolException aborts the whole batch before any tool runs."""

    @pytest.mark.asyncio
    async def test_raise_aborts_batch(self):
        """RaiseToolException propagates; no tools execute."""
        _invocations.clear()
        responses = [
            _tool_call_response(
                [
                    ("echo", '{"text":"a"}', "tc1"),
                    ("shout", '{"text":"b"}', "tc2"),
                ]
            ),
            _text_response("never reached"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool(), ShoutTool()])

        class PolicyBlock(RuntimeError):
            pass

        async def hook(*, tool_calls, ctx, exec_id):
            return {"tc1": RaiseToolException(exception=PolicyBlock("halt"))}

        executor.before_tool_hooks = [hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        with pytest.raises(PolicyBlock, match="halt"):
            await _drain(executor, ctx)

        # Neither tool ran; the first RaiseToolException aborts immediately
        # so no tool in the batch — allowed or rejected — executes.
        assert _invocations["echo"] == []
        assert _invocations["shout"] == []

    @pytest.mark.asyncio
    async def test_raise_preempts_reject(self):
        """
        If one decision raises and another rejects, the raise wins:
        no rejection output is synthesized either.
        """
        _invocations.clear()
        responses = [
            _tool_call_response(
                [
                    ("echo", '{"text":"a"}', "tc1"),
                    ("shout", '{"text":"b"}', "tc2"),
                ]
            ),
            _text_response("never reached"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool(), ShoutTool()])

        async def hook(*, tool_calls, ctx, exec_id):
            return {
                "tc1": RejectToolContent(content="rejected"),
                "tc2": RaiseToolException(exception=RuntimeError("boom")),
            }

        executor.before_tool_hooks = [hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        with pytest.raises(RuntimeError, match="boom"):
            await _drain(executor, ctx)

        # No rejection output leaked into memory
        assert not [m for m in memory.messages if isinstance(m, FunctionToolOutputItem)]


class TestMixedDecisions:
    """Allow + Reject in the same batch: only allowed tools execute."""

    @pytest.mark.asyncio
    async def test_mixed_allow_and_reject(self):
        _invocations.clear()
        responses = [
            _tool_call_response(
                [
                    ("echo", '{"text":"keep"}', "tc1"),
                    ("shout", '{"text":"drop"}', "tc2"),
                ]
            ),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool(), ShoutTool()])

        async def hook(*, tool_calls, ctx, exec_id):
            return {"tc2": RejectToolContent(content="shout disabled")}

        executor.before_tool_hooks = [hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        # Allowed tool ran, rejected tool did not
        assert _invocations["echo"] == ["keep"]
        assert _invocations["shout"] == []

        # Both tool outputs in memory — one real, one synthesized
        outs = [m for m in memory.messages if isinstance(m, FunctionToolOutputItem)]
        by_call = {m.call_id: m.text for m in outs}
        assert "echo: keep" in by_call["tc1"]
        assert "shout disabled" in by_call["tc2"]


class TestLegacyReturnShape:
    """Side-effect-only hooks (implicit ``None`` return) still work."""

    @pytest.mark.asyncio
    async def test_legacy_none_returning_hook(self):
        """
        A side-effect-only hook (logs / mutates ctx / returns nothing)
        works without adjustment — tools execute normally.
        """
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"go"}', "tc1")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])

        calls_seen: list[int] = []

        async def legacy_hook(*, tool_calls, ctx, exec_id):
            calls_seen.append(len(tool_calls))
            # No explicit return — implicit None

        executor.before_tool_hooks = [legacy_hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert calls_seen == [1]
        assert _invocations["echo"] == ["go"]


class TestADTShape:
    """Verify the decision ADT itself."""

    def test_reject_content_is_frozen(self):
        """frozen=True means existing fields can't be reassigned."""
        from dataclasses import FrozenInstanceError

        d = RejectToolContent(content="hello")
        with pytest.raises(FrozenInstanceError):
            d.content = "mutated"  # type: ignore[misc]

    def test_reject_content_carries_content(self):
        d = RejectToolContent(content="hello")
        assert d.content == "hello"

    def test_raise_exception_carries_exception(self):
        err = ValueError("bad")
        d = RaiseToolException(exception=err)
        assert d.exception is err

    def test_union_membership(self):
        """All three variants are valid ToolCallDecision members."""
        decisions: list[ToolCallDecision] = [
            AllowTool(),
            RejectToolContent(content="x"),
            RaiseToolException(exception=RuntimeError("x")),
        ]
        assert len(decisions) == 3


def test_tool_decision_is_public_api():
    """The ADT and its variants are exported from the grasp_agents.agent package."""
    assert grasp_agents.agent.AllowTool is AllowTool
    assert grasp_agents.agent.RejectToolContent is RejectToolContent
    assert grasp_agents.agent.RaiseToolException is RaiseToolException
    assert grasp_agents.agent.ToolCallDecision is ToolCallDecision


class TestStackedBeforeToolHooks:
    """Stacked before-tool hooks merge per call; the most restrictive wins."""

    @pytest.mark.asyncio
    async def test_reject_overrides_allow(self):
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool()])

        async def allow_hook(*, tool_calls, ctx, exec_id):
            return {"tc1": AllowTool()}

        async def reject_hook(*, tool_calls, ctx, exec_id):
            return {"tc1": RejectToolContent(content="blocked by policy")}

        executor.before_tool_hooks = [allow_hook, reject_hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        # reject wins over allow → the tool never ran
        assert _invocations["echo"] == []
        outs = [m for m in memory.messages if isinstance(m, FunctionToolOutputItem)]
        assert "blocked by policy" in outs[0].text

    @pytest.mark.asyncio
    async def test_raise_overrides_reject_regardless_of_order(self):
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("never"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])

        class Halt(RuntimeError):
            pass

        async def raise_hook(*, tool_calls, ctx, exec_id):
            return {"tc1": RaiseToolException(exception=Halt("halt"))}

        async def reject_hook(*, tool_calls, ctx, exec_id):
            return {"tc1": RejectToolContent(content="reject")}

        # raise registered first, reject second — raise must still win
        executor.before_tool_hooks = [raise_hook, reject_hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        with pytest.raises(Halt, match="halt"):
            await _drain(executor, ctx)

        assert _invocations["echo"] == []

    @pytest.mark.asyncio
    async def test_independent_rejections_union(self):
        _invocations.clear()
        responses = [
            _tool_call_response(
                [
                    ("echo", '{"text":"a"}', "tc1"),
                    ("shout", '{"text":"b"}', "tc2"),
                ]
            ),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool(), ShoutTool()])

        async def reject_echo(*, tool_calls, ctx, exec_id):
            return {"tc1": RejectToolContent(content="no echo")}

        async def reject_shout(*, tool_calls, ctx, exec_id):
            return {"tc2": RejectToolContent(content="no shout")}

        executor.before_tool_hooks = [reject_echo, reject_shout]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        # Each hook's rejection applies to its own call (disjoint keys union).
        assert _invocations["echo"] == []
        assert _invocations["shout"] == []
        by_call = {
            m.call_id: m.text
            for m in memory.messages
            if isinstance(m, FunctionToolOutputItem)
        }
        assert "no echo" in by_call["tc1"]
        assert "no shout" in by_call["tc2"]
