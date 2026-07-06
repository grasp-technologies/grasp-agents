"""
Tests for the synchronous tool-call approval guard.

Verifies :func:`build_callback_approval`:

* Approver returning ``True`` → tool runs.
* Approver returning ``False`` → tool skipped; synthesized rejection
  message reaches the LLM.
* ``tool_names`` filter: approver is not asked about unmatched calls.
* ``deny_message`` template is formatted with ``name`` and ``arguments``.
* No denials → hook returns ``None`` (fast path).
* Public API exports.
"""

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel

import grasp_agents
from grasp_agents.agent.agent_loop import AgentLoop, ResponseCapture
from grasp_agents.agent.approval_callback import (
    DEFAULT_DENY_MESSAGE,
    ApprovalCallback,
    build_callback_approval,
)
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
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
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes input")
        _invocations[self.name] = []

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        _invocations[self.name].append(inp.text)
        return f"echo: {inp.text}"


class DeleteTool(BaseTool[EchoInput, Any, Any]):
    def __init__(self) -> None:
        super().__init__(name="delete_file", description="Deletes a file")
        _invocations[self.name] = []

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        _invocations[self.name].append(inp.text)
        return f"deleted {inp.text}"


_invocations: dict[str, list[str]] = {}


def _make_executor(
    responses: list[Response],
    *,
    tools: list[BaseTool[Any, Any, Any]] | None = None,
    max_turns: int = 10,
    ctx: SessionContext[Any] | None = None,
) -> tuple[AgentLoop[None], LLMAgentTranscript, MockLLM]:
    llm = MockLLM(model_name="mock", responses_queue=responses)
    memory = LLMAgentTranscript()
    memory.messages = [InputMessageItem.from_text("sys", role="system")]
    memory.update([InputMessageItem.from_text("go", role="user")])

    ctx = ctx if ctx is not None else SessionContext[None](state=None)
    executor = _make_agent_loop(
        agent_name="test",
        llm=llm,
        transcript=memory,
        tools=tools,
        ctx=ctx,
        max_turns=max_turns,
        stream_llm=False,
    )
    executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
        response.output_text if response and not response.tool_call_items else None
    )
    return executor, memory, llm


async def _drain(executor: AgentLoop[None], ctx: SessionContext[None]) -> Response:
    executor._ctx = ctx
    stream = ResponseCapture(executor.execute_stream(exec_id="t"))
    async for _ in stream:
        pass
    assert stream.response is not None
    return stream.response


# ---------- Tests ----------


class TestAllowPath:
    """Approver returning True runs the tool normally."""

    @pytest.mark.asyncio
    async def test_allow_runs_tool(self):
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])

        calls_seen: list[str] = []

        async def approve(call, *, ctx, exec_id):
            calls_seen.append(call.name)
            return True

        executor.before_tool_hooks = [build_callback_approval(approve)]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert _invocations["echo"] == ["hi"]
        assert calls_seen == ["echo"]

    @pytest.mark.asyncio
    async def test_approver_receives_call_and_ctx(self):
        """The approver sees the actual call and the run context."""
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"x"}', "tc-abc")]),
            _text_response("done"),
        ]
        ctx = SessionContext[str](state="sentinel")
        executor, _, _ = _make_executor(responses, tools=[EchoTool()], ctx=ctx)

        captured: dict[str, Any] = {}

        async def approve(call, *, ctx, exec_id):
            captured["call_id"] = call.call_id
            captured["name"] = call.name
            captured["arguments"] = call.arguments
            captured["exec_id"] = exec_id
            captured["ctx_state"] = ctx.state
            return True

        executor.before_tool_hooks = [build_callback_approval(approve)]  # type: ignore[assignment]

        stream = ResponseCapture(executor.execute_stream(exec_id="run-xyz"))
        async for _ in stream:
            pass

        assert captured["call_id"] == "tc-abc"
        assert captured["name"] == "echo"
        assert captured["arguments"] == '{"text":"x"}'
        assert captured["exec_id"] == "run-xyz"
        assert captured["ctx_state"] == "sentinel"


class TestDenyPath:
    """Approver returning False surfaces a rejection to the LLM."""

    @pytest.mark.asyncio
    async def test_deny_skips_execution(self):
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool()])

        async def approve(call, *, ctx, exec_id):
            return False

        executor.before_tool_hooks = [build_callback_approval(approve)]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        # Tool did not run
        assert _invocations["echo"] == []

        # Default deny message reached memory
        outs = [m for m in memory.messages if isinstance(m, FunctionToolOutputItem)]
        assert len(outs) == 1
        assert "echo" in outs[0].text

    @pytest.mark.asyncio
    async def test_custom_deny_message(self):
        """
        ``deny_message`` template is formatted with ``name`` + ``arguments``.

        The formatted message is a string result, so it reaches the model as
        text verbatim (``from_tool_result`` no longer JSON-re-encodes a string,
        which would have escaped the embedded ``arguments`` quotes).
        """
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"x"}', "tc1")]),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool()])

        async def approve(call, *, ctx, exec_id):
            return False

        executor.before_tool_hooks = [  # type: ignore[assignment]
            build_callback_approval(
                approve,
                deny_message="Blocked {name} with args {arguments}",
            )
        ]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        outs = [m for m in memory.messages if isinstance(m, FunctionToolOutputItem)]
        text = outs[0].text
        # {name} substituted
        assert "Blocked echo with args" in text
        # {arguments} substituted verbatim (no JSON re-escaping of the string)
        assert '{"text":"x"}' in text


class TestToolNameFilter:
    """``tool_names`` restricts gating to a subset of tools."""

    @pytest.mark.asyncio
    async def test_filter_bypasses_approver_for_unmatched(self):
        """
        Only ``delete_file`` is gated; the approver is never called for
        ``echo`` even though both appear in the batch.
        """
        _invocations.clear()
        responses = [
            _tool_call_response(
                [
                    ("echo", '{"text":"keep"}', "tc1"),
                    ("delete_file", '{"text":"data"}', "tc2"),
                ]
            ),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool(), DeleteTool()])

        names_seen: list[str] = []

        async def approve(call, *, ctx, exec_id):
            names_seen.append(call.name)
            return False  # deny whatever gets through

        executor.before_tool_hooks = [  # type: ignore[assignment]
            build_callback_approval(approve, tool_names={"delete_file"})
        ]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        # Only delete_file was gated
        assert names_seen == ["delete_file"]

        # echo ran, delete_file did not
        assert _invocations["echo"] == ["keep"]
        assert _invocations["delete_file"] == []

    @pytest.mark.asyncio
    async def test_none_filter_gates_every_call(self):
        """``tool_names=None`` is the default: every call goes through."""
        _invocations.clear()
        responses = [
            _tool_call_response(
                [
                    ("echo", '{"text":"a"}', "tc1"),
                    ("delete_file", '{"text":"b"}', "tc2"),
                ]
            ),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool(), DeleteTool()])

        names_seen: list[str] = []

        async def approve(call, *, ctx, exec_id):
            names_seen.append(call.name)
            return True

        executor.before_tool_hooks = [build_callback_approval(approve)]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert sorted(names_seen) == ["delete_file", "echo"]


class TestFastPath:
    """When every call is allowed the hook returns None, not an empty map."""

    @pytest.mark.asyncio
    async def test_all_allowed_returns_none(self):
        async def approve(call, *, ctx, exec_id):
            return True

        hook = build_callback_approval(approve)

        calls = [
            FunctionToolCallItem(call_id="tc1", name="echo", arguments="{}"),
            FunctionToolCallItem(call_id="tc2", name="echo", arguments="{}"),
        ]

        result = await hook(
            ctx=SessionContext[None](),
            tool_calls=calls,
            exec_id="t",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_partial_deny_returns_mapping(self):
        """Only denied calls appear in the returned mapping."""

        async def approve(call, *, ctx, exec_id):
            return call.name != "delete_file"

        hook = build_callback_approval(approve)

        calls = [
            FunctionToolCallItem(call_id="tc1", name="echo", arguments="{}"),
            FunctionToolCallItem(call_id="tc2", name="delete_file", arguments="{}"),
        ]

        result = await hook(
            ctx=SessionContext[None](),
            tool_calls=calls,
            exec_id="t",
        )
        assert result is not None
        assert "tc1" not in result  # allowed → absent
        assert "tc2" in result  # denied → present
        from grasp_agents.agent.tool_decision import RejectToolContent

        assert isinstance(result["tc2"], RejectToolContent)


class TestMixedBatch:
    """End-to-end: two calls, one allowed and one denied."""

    @pytest.mark.asyncio
    async def test_allow_one_deny_one(self):
        _invocations.clear()
        responses = [
            _tool_call_response(
                [
                    ("echo", '{"text":"good"}', "tc1"),
                    ("delete_file", '{"text":"bad"}', "tc2"),
                ]
            ),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(
            responses, tools=[EchoTool(), DeleteTool()]
        )

        async def approve(call, *, ctx, exec_id):
            return call.name == "echo"

        executor.before_tool_hooks = [build_callback_approval(approve)]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert _invocations["echo"] == ["good"]
        assert _invocations["delete_file"] == []

        outs = [m for m in memory.messages if isinstance(m, FunctionToolOutputItem)]
        by_call = {m.call_id: m.text for m in outs}
        assert "echo: good" in by_call["tc1"]
        # Default deny message for delete_file
        assert "delete_file" in by_call["tc2"]


def test_approval_guard_public_api():
    """Helpers are exported from the grasp_agents.agent package."""
    assert grasp_agents.agent.build_callback_approval is build_callback_approval
    assert grasp_agents.agent.ApprovalCallback is ApprovalCallback
    assert grasp_agents.agent.DEFAULT_DENY_MESSAGE == DEFAULT_DENY_MESSAGE
