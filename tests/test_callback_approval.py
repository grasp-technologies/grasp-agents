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

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import BaseModel

import grasp_agents
from grasp_agents.agent.agent_loop import AgentLoop, ResponseCapture
from grasp_agents.agent.approval_callback import (
    DEFAULT_DENY_MESSAGE,
    ApprovalCallback,
    build_callback_approval,
)
from grasp_agents.agent.llm_agent_memory import LLMAgentMemory
from grasp_agents.llm.llm import LLM
from grasp_agents.run_context import RunContext
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
)
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
)
from grasp_agents.types.response import Response, ResponseUsage
from grasp_agents.types.tool import BaseTool

# ---------- Infrastructure (mirrors test_before_tool_decision.py) ----------


def _make_usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _text_response(text: str) -> Response:
    return Response(
        model="mock",
        output_items=[
            OutputMessageItem(
                content_parts=[OutputMessageText(text=text)],
                status="completed",
            )
        ],
        usage_with_cost=_make_usage(),
    )


def _tool_call_response(
    calls: Sequence[tuple[str, str, str]],
) -> Response:
    return Response(
        model="mock",
        output_items=[
            FunctionToolCallItem(call_id=call_id, name=name, arguments=args)
            for name, args, call_id in calls
        ],
        usage_with_cost=_make_usage(),
    )


@dataclass(frozen=True)
class MockLLM(LLM):
    responses_queue: list[Response] = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, "_call_count", 0)

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        count = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        assert self.responses_queue, "MockLLM: no more responses"
        return self.responses_queue.pop(0)

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        response = await self._generate_response_once(
            input,
            tools=tools,
            output_schema=output_schema,
            tool_choice=tool_choice,
            **extra_llm_settings,
        )
        seq = 0
        seq += 1
        yield ResponseCreated(response=response, sequence_number=seq)  # type: ignore[arg-type]
        for idx, item in enumerate(response.output):
            seq += 1
            yield OutputItemAdded(item=item, output_index=idx, sequence_number=seq)
            seq += 1
            yield OutputItemDone(item=item, output_index=idx, sequence_number=seq)
        seq += 1
        yield ResponseCompleted(response=response, sequence_number=seq)  # type: ignore[arg-type]


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
) -> tuple[AgentLoop[None], LLMAgentMemory, MockLLM]:
    llm = MockLLM(model_name="mock", responses_queue=responses)
    memory = LLMAgentMemory()
    memory.reset(instructions="sys")
    memory.update([InputMessageItem.from_text("go", role="user")])

    executor = AgentLoop[None](
        agent_name="test",
        llm=llm,
        memory=memory,
        tools=tools,
        max_turns=max_turns,
        stream_llm=False,
    )
    executor.final_answer_extractor = (
        lambda *, ctx, exec_id, response=None, **kw: response.output_text
        if response and not response.tool_call_items
        else None
    )
    return executor, memory, llm


async def _drain(executor: AgentLoop[None], ctx: RunContext[None]) -> Response:
    stream = ResponseCapture(executor.execute_stream(ctx=ctx, exec_id="t"))
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

        executor.before_tool_hook = build_callback_approval(approve)  # type: ignore[assignment]

        ctx = RunContext[None]()
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
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])

        captured: dict[str, Any] = {}

        async def approve(call, *, ctx, exec_id):
            captured["call_id"] = call.call_id
            captured["name"] = call.name
            captured["arguments"] = call.arguments
            captured["exec_id"] = exec_id
            captured["ctx_state"] = ctx.state
            return True

        executor.before_tool_hook = build_callback_approval(approve)  # type: ignore[assignment]

        ctx = RunContext[str](state="sentinel")
        stream = ResponseCapture(executor.execute_stream(ctx=ctx, exec_id="run-xyz"))
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

        executor.before_tool_hook = build_callback_approval(approve)  # type: ignore[assignment]

        ctx = RunContext[None]()
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

        ``FunctionToolOutputItem.from_tool_result`` JSON-encodes the
        string, so ``.text`` will contain escaped quotes; we just check
        that both template placeholders got substituted.
        """
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"x"}', "tc1")]),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool()])

        async def approve(call, *, ctx, exec_id):
            return False

        executor.before_tool_hook = build_callback_approval(  # type: ignore[assignment]
            approve,
            deny_message="Blocked {name} with args {arguments}",
        )

        ctx = RunContext[None]()
        await _drain(executor, ctx)

        outs = [m for m in memory.messages if isinstance(m, FunctionToolOutputItem)]
        text = outs[0].text
        # {name} substituted
        assert "Blocked echo with args" in text
        # {arguments} substituted (content visible through JSON escaping)
        assert "text" in text and '\\"x\\"' in text


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

        executor.before_tool_hook = build_callback_approval(  # type: ignore[assignment]
            approve, tool_names={"delete_file"}
        )

        ctx = RunContext[None]()
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

        executor.before_tool_hook = build_callback_approval(approve)  # type: ignore[assignment]

        ctx = RunContext[None]()
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
            tool_calls=calls,
            ctx=RunContext[None](),
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
            tool_calls=calls,
            ctx=RunContext[None](),
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

        executor.before_tool_hook = build_callback_approval(approve)  # type: ignore[assignment]

        ctx = RunContext[None]()
        await _drain(executor, ctx)

        assert _invocations["echo"] == ["good"]
        assert _invocations["delete_file"] == []

        outs = [m for m in memory.messages if isinstance(m, FunctionToolOutputItem)]
        by_call = {m.call_id: m.text for m in outs}
        assert "echo: good" in by_call["tc1"]
        # Default deny message for delete_file
        assert "delete_file" in by_call["tc2"]


def test_approval_guard_public_api():
    """Helpers are exported from the top-level package."""
    assert grasp_agents.build_callback_approval is build_callback_approval
    assert grasp_agents.ApprovalCallback is ApprovalCallback
    assert grasp_agents.DEFAULT_DENY_MESSAGE == DEFAULT_DENY_MESSAGE
