"""
Tests for :class:`BeforeToolHook` decision returns.

Verifies that the hook's return value controls per-call tool execution:
* ``None`` / missing entries → ``AllowTool`` (default, tool runs normally)
* ``RejectToolContent`` → tool skipped, synthetic output reaches the LLM
* ``RaiseToolException`` → batch aborts by raising, no tools run
* Mixed decisions → only ``AllowTool`` calls execute
* Hooks that return nothing keep working
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
from grasp_agents.agent.llm_agent_memory import LLMAgentMemory
from grasp_agents.agent.tool_decision import (
    AllowTool,
    RaiseToolException,
    RejectToolContent,
    ToolCallDecision,
)
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

# ---------- Infrastructure (mirrors test_agent_loop_hooks.py) ----------


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
    """Build a response containing one or more function tool calls."""
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

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
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
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        response = await self._generate_response_once(
            input,
            tools=tools,
            response_schema=response_schema,
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
        stream_llm_responses=False,
    )
    # Final-answer extractor: stop on text-only responses.
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

        executor.before_tool_hook = hook  # type: ignore[assignment]

        ctx = RunContext[None]()
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

        executor.before_tool_hook = hook  # type: ignore[assignment]

        ctx = RunContext[None]()
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

        executor.before_tool_hook = hook  # type: ignore[assignment]

        ctx = RunContext[None]()
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

        executor.before_tool_hook = hook  # type: ignore[assignment]

        ctx = RunContext[None]()
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

        async def after(*, tool_calls, tool_messages, ctx, exec_id):
            captured["num_messages"] = len(tool_messages)
            captured["first_text"] = tool_messages[0].text

        executor.before_tool_hook = before  # type: ignore[assignment]
        executor.after_tool_hook = after  # type: ignore[assignment]

        ctx = RunContext[None]()
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
        executor, _, _ = _make_executor(
            responses, tools=[EchoTool(), ShoutTool()]
        )

        class PolicyBlock(RuntimeError):
            pass

        async def hook(*, tool_calls, ctx, exec_id):
            return {"tc1": RaiseToolException(exception=PolicyBlock("halt"))}

        executor.before_tool_hook = hook  # type: ignore[assignment]

        ctx = RunContext[None]()
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
        executor, memory, _ = _make_executor(
            responses, tools=[EchoTool(), ShoutTool()]
        )

        async def hook(*, tool_calls, ctx, exec_id):
            return {
                "tc1": RejectToolContent(content="rejected"),
                "tc2": RaiseToolException(exception=RuntimeError("boom")),
            }

        executor.before_tool_hook = hook  # type: ignore[assignment]

        ctx = RunContext[None]()
        with pytest.raises(RuntimeError, match="boom"):
            await _drain(executor, ctx)

        # No rejection output leaked into memory
        assert not [
            m for m in memory.messages if isinstance(m, FunctionToolOutputItem)
        ]


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
        executor, memory, _ = _make_executor(
            responses, tools=[EchoTool(), ShoutTool()]
        )

        async def hook(*, tool_calls, ctx, exec_id):
            return {"tc2": RejectToolContent(content="shout disabled")}

        executor.before_tool_hook = hook  # type: ignore[assignment]

        ctx = RunContext[None]()
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

        executor.before_tool_hook = legacy_hook  # type: ignore[assignment]

        ctx = RunContext[None]()
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
    """The ADT and its variants are exported from the top-level package."""
    assert grasp_agents.AllowTool is AllowTool
    assert grasp_agents.RejectToolContent is RejectToolContent
    assert grasp_agents.RaiseToolException is RaiseToolException
    assert grasp_agents.ToolCallDecision is ToolCallDecision
