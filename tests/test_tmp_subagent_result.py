"""
TEMPORARY — demonstrates how a sub-agent's result reaches the *parent* transcript,
and why a degenerate "echo the TaskOutput" answer shows up as a kv-table in the UI.

A ``str``-output ``AgentTool`` child is driven to ``max_turns`` two ways:

  (A) the forced final-answer turn returns TEXT that happens to be a JSON dump of a
      TaskOutput poll  → the parent records that string as the tool result; the
      renderer peels the JSON and draws a kv-table (what the user saw).
  (B) the forced final-answer turn returns ANOTHER TOOL CALL → the child raises
      AgentFinalAnswerError → the parent records "Tool '...' failed: ..." (plain text).

Run:  pytest tests/test_tmp_subagent_result.py -s
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel
from rich.table import Table

from grasp_agents import AgentTool, LLMAgent, RunContext
from grasp_agents.llm.llm import LLM
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import ToolOutputItemEvent
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
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
from grasp_agents.ui._event_render import _build_result_renderable, _unwrap_json

pytestmark = pytest.mark.asyncio


def _usage() -> ResponseUsage:
    return ResponseUsage(input_tokens=10, output_tokens=5, total_tokens=15)


def _text(text: str) -> Response:
    return Response(
        model="mock",
        output_items=[
            OutputMessageItem(
                content_parts=[OutputMessageText(text=text)], status="completed"
            )
        ],
        usage_with_cost=_usage(),
    )


def _call(name: str, args: str, call_id: str) -> Response:
    return Response(
        model="mock",
        output_items=[FunctionToolCallItem(call_id=call_id, name=name, arguments=args)],
        usage_with_cost=_usage(),
    )


@dataclass(frozen=True)
class MockLLM(LLM):
    model_name: str = "mock"
    responses_queue: list[Response] = field(default_factory=list)

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> Response:
        assert self.responses_queue, "MockLLM: no more responses"
        return self.responses_queue.pop(0)

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        resp = await self._generate_response_once(input, tools=tools)
        seq = 1
        yield ResponseCreated(response=resp, sequence_number=seq)  # type: ignore[arg-type]
        for idx, item in enumerate(resp.output):
            seq += 1
            yield OutputItemAdded(item=item, output_index=idx, sequence_number=seq)
            seq += 1
            yield OutputItemDone(item=item, output_index=idx, sequence_number=seq)
        seq += 1
        yield ResponseCompleted(response=resp, sequence_number=seq)  # type: ignore[arg-type]


class _EchoInput(BaseModel):
    text: str


class _EchoTool(BaseTool[_EchoInput, str, Any]):
    def __init__(self) -> None:
        super().__init__(name="echo", description="echo")

    async def _run(
        self,
        inp: _EchoInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
        path: Any = None,
        agent_ctx: Any = None,
    ) -> str:
        return f"echo: {inp.text}"


# A serialized TaskOutputResult — exactly the shape the real model regurgitated.
_TASKOUTPUT_JSON = json.dumps(
    {
        "task_id": "bg_2",
        "tool_name": "Bash",
        "status": "completed",
        "output": "awk: calling undefined function RAND\n source line number 1\n",
        "result": None,
        "elapsed_s": 6.0,
    }
)


async def _run_parent(child_responses: list[Response]) -> FunctionToolOutputItem:
    """
    Parent (``analyst``) calls the ``worker`` AgentTool once, then says "done".
    Return the FunctionToolOutputItem the parent recorded as the worker's result.
    """
    worker = AgentTool[None](
        name="worker",
        description="a sub-agent",
        llm=MockLLM(responses_queue=list(child_responses)),
        tools=[_EchoTool()],
        sys_prompt="you are a worker",
        max_turns=1,  # turn 0 runs a tool, turn 1 trips the force step
        stream_llm=True,
    )
    ctx = RunContext[None](state=None)
    parent = LLMAgent[str, str, None](
        name="analyst",
        ctx=ctx,
        llm=MockLLM(
            responses_queue=[
                _call("worker", '{"prompt": "do it"}', "fc_p0"),
                _text("done"),
            ]
        ),
        tools=[worker],
        sys_prompt="you are the analyst",
        max_turns=5,
        stream_llm=True,
    )

    captured: list[FunctionToolOutputItem] = []
    async for ev in parent.run_stream("go"):
        if (
            isinstance(ev, ToolOutputItemEvent)
            and ev.source == "worker"
            and ev.destination == "analyst"
        ):
            captured.append(ev.data)

    assert len(captured) == 1, f"expected one worker result, got {len(captured)}"
    # The SAME message is what lives in the analyst's transcript.
    in_transcript = [
        m
        for m in parent.transcript.messages
        if isinstance(m, FunctionToolOutputItem) and m.call_id == "fc_p0"
    ]
    assert in_transcript, "worker result not found in the analyst transcript"
    assert in_transcript[0].output == captured[0].output
    return captured[0]


async def test_subagent_text_answer_becomes_kv_table_in_parent() -> None:
    """
    (A) child's forced answer is TEXT (a JSON TaskOutput echo) → parent records
    that string; the renderer peels it to a dict → kv-table (what the user saw).
    """
    result = await _run_parent(
        [
            _call("echo", '{"text": "a"}', "c0"),
            _call("echo", '{"text": "b"}', "c1"),
            _text(_TASKOUTPUT_JSON),  # forced final answer = JSON echo (free text)
        ]
    )
    parsed = _unwrap_json(result.output)
    rendered = _build_result_renderable(result.output, "white", inline_images=False)
    print(f"\n[A] unwrapped = {parsed!r}")
    print(f"[A] renders as: {type(rendered).__name__}")
    assert isinstance(parsed, dict), "expected the echoed TaskOutput JSON → dict"
    assert parsed["task_id"] == "bg_2"
    assert parsed["tool_name"] == "Bash"
    assert isinstance(rendered, Table), "the kv-table the analyst pane showed"


async def test_subagent_raise_becomes_error_in_parent() -> None:
    """
    (B) child's forced turn is ANOTHER tool call → AgentFinalAnswerError →
    parent records the *worker's* failure (a ToolErrorInfo), NOT a leaked inner
    tool output (the bug) and NOT None.
    """
    result = await _run_parent(
        [
            _call("echo", '{"text": "a"}', "c0"),
            _call("echo", '{"text": "b"}', "c1"),
            _call("echo", '{"text": "c"}', "c2"),  # forced turn is a tool call → raise
        ]
    )
    parsed = _unwrap_json(result.output)
    print(f"\n[B] unwrapped = {parsed!r}")
    assert isinstance(parsed, dict), "expected a ToolErrorInfo, not a bare value/None"
    # the failure is attributed to the worker tool and carries a message
    assert parsed.get("tool_name") == "worker"
    assert parsed.get("error"), "carries the failure message"
    # regression guard: the child's inner tool output must NOT leak through
    assert "task_id" not in parsed, "must NOT be the leaked inner TaskOutput"
    assert parsed.get("status") != "completed"


async def test_inner_tool_sharing_agent_name_does_not_leak() -> None:
    """
    Robustness to name collisions: an AgentTool whose *inner* tool shares its
    name. Capture is by ORDER (the agent's own terminal event is emitted last,
    after its children's), NOT by name — so the parent records the agent's
    failure, never the nested same-named tool's output. (Fails on the pre-fix
    code, which captured the last inner ToolOutputEvent.)
    """

    class _DupTool(BaseTool[_EchoInput, str, Any]):
        def __init__(self) -> None:
            super().__init__(name="dup", description="inner tool; same name as agent")

        async def _run(
            self,
            inp: _EchoInput,
            *,
            ctx: Any = None,
            exec_id: str | None = None,
            progress_callback: Any = None,
            path: Any = None,
            agent_ctx: Any = None,
        ) -> str:
            return "INNER-DUP-OUTPUT"

    worker = AgentTool[None](
        name="dup",  # collides with its inner tool's name
        description="a sub-agent whose inner tool shares its name",
        llm=MockLLM(
            responses_queue=[
                _call("dup", '{"text": "x"}', "c0"),  # runs the inner "dup" tool
                _call("dup", '{"text": "y"}', "c1"),  # forced turn = tool call → raise
            ]
        ),
        tools=[_DupTool()],
        sys_prompt="w",
        max_turns=1,
        stream_llm=True,
    )
    ctx = RunContext[None](state=None)
    parent = LLMAgent[str, str, None](
        name="analyst",
        ctx=ctx,
        llm=MockLLM(
            responses_queue=[
                _call("dup", '{"prompt": "go"}', "fc_p0"),
                _text("done"),
            ]
        ),
        tools=[worker],
        sys_prompt="a",
        max_turns=5,
        stream_llm=True,
    )

    captured: list[FunctionToolOutputItem] = []
    async for ev in parent.run_stream("go"):
        if (
            isinstance(ev, ToolOutputItemEvent)
            and ev.source == "dup"
            and ev.destination == "analyst"
        ):
            captured.append(ev.data)

    assert len(captured) == 1
    parsed = _unwrap_json(captured[0].output)
    print(f"\n[collision] unwrapped = {parsed!r}")
    assert "INNER-DUP-OUTPUT" not in str(parsed), "nested same-named output leaked"
    assert isinstance(parsed, dict) and parsed.get("error"), "agent's own failure"
