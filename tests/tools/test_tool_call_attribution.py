"""
A tool call's events are attributed to that call: its stream events go to the
calling agent, and its terminal event carries its tool call id, so consumers
find a call's outcome by id rather than by position in the stream.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.events import (
    Event,
    ToolErrorEvent,
    ToolErrorInfo,
    ToolOutputEvent,
    ToolStreamEvent,
)
from grasp_agents.types.items import FunctionToolCallItem, FunctionToolOutputItem
from tests._helpers import (
    MockLLM,
    _make_agent_ctx,
    _make_agent_loop,
    _text_response,
    _tool_call_response,
)


class _In(BaseModel):
    text: str


class _OutOfOrderTool(BaseTool[_In, Any, Any]):
    """
    Yields its result before a trailing stream event and a nested tool's
    (already claimed) error — the result is not its last terminal-looking event.
    """

    def __init__(self, *, auto_background_at: float | None = None) -> None:
        super().__init__(
            name="ooo",
            description="Out of order",
            auto_background_at=auto_background_at,
        )

    async def _run(self, inp: _In, *, ctx: Any = None, **kwargs: Any) -> str:
        del ctx, kwargs
        return f"own: {inp.text}"

    async def _run_stream(
        self, inp: _In, *, exec_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        del kwargs
        yield ToolOutputEvent(
            data=f"own: {inp.text}", source=self.name, exec_id=exec_id
        )
        yield ToolStreamEvent(data="trailing", source=self.name, exec_id=exec_id)
        yield ToolErrorEvent(
            data=ToolErrorInfo(tool_name="inner", error="inner boom"),
            source="inner",
            exec_id=exec_id,
            tool_call_id="inner_call",
        )

    async def _resume_stream(
        self, *, exec_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        del kwargs
        yield ToolStreamEvent(data="resumed", source=self.name, exec_id=exec_id)
        yield ToolOutputEvent(data="own: resumed", source=self.name, exec_id=exec_id)


def _terminals(events: list[Event[Any]]) -> list[tuple[Any, str | None]]:
    return [
        (e.data, e.tool_call_id)
        for e in events
        if isinstance(e, (ToolOutputEvent, ToolErrorEvent))
    ]


@pytest.mark.asyncio
async def test_run_stream_claims_the_calls_own_terminal() -> None:
    agent_ctx = _make_agent_ctx(agent_name="caller").for_tool_call("c1")

    events = [
        e
        async for e in _OutOfOrderTool().run_stream(_In(text="x"), agent_ctx=agent_ctx)
    ]

    terminals = _terminals(events)
    assert terminals[0] == ("own: x", "c1")
    # A nested tool's terminal keeps its own call's id.
    assert terminals[1][1] == "inner_call"
    streamed = [e for e in events if isinstance(e, ToolStreamEvent)]
    assert all(e.destination == "caller" for e in streamed)


@pytest.mark.asyncio
async def test_run_stream_without_a_call_claims_nothing() -> None:
    events = [e async for e in _OutOfOrderTool().run_stream(_In(text="x"))]

    assert _terminals(events)[0] == ("own: x", None)


@pytest.mark.asyncio
async def test_resume_stream_is_attributed_like_run_stream() -> None:
    agent_ctx = _make_agent_ctx(agent_name="caller").for_tool_call("c9")

    events = [e async for e in _OutOfOrderTool().resume_stream(agent_ctx=agent_ctx)]

    assert _terminals(events) == [("own: resumed", "c9")]
    streamed = [e for e in events if isinstance(e, ToolStreamEvent)]
    assert [e.destination for e in streamed] == ["caller"]


@pytest.mark.asyncio
async def test_foreground_result_is_the_calls_own_terminal() -> None:
    loop = _make_agent_loop(
        agent_name="agent",
        llm=MockLLM(
            responses_queue=[
                _tool_call_response("ooo", '{"text": "x"}', "c1"),
                _text_response("done"),
            ]
        ),
        ctx=SessionContext[None](state=None),
        tools=[_OutOfOrderTool()],
        max_turns=3,
    )

    _ = [e async for e in loop.execute_stream(exec_id="t")]

    outputs = [m for m in loop.cw.transcript if isinstance(m, FunctionToolOutputItem)]
    assert len(outputs) == 1
    assert "own: x" in str(outputs[0])
    assert "inner boom" not in str(outputs[0])


@pytest.mark.asyncio
async def test_background_outcome_is_the_tasks_own_terminal() -> None:
    ctx = SessionContext[None](state=None)
    mgr = BackgroundTaskManager[None](tools={}, path=[])
    agent_ctx = _make_agent_ctx(agent_name="agent", bg_tasks=mgr)
    call = FunctionToolCallItem(call_id="c1", name="ooo", arguments='{"text": "x"}')

    _note, launched = await mgr.run_backgroundable(
        call,
        _OutOfOrderTool(auto_background_at=0),
        _In(text="x"),
        ctx=ctx,
        exec_id="t",
        agent_ctx=agent_ctx,
    )
    assert launched is not None
    await mgr.wait_idle()
    completions = mgr.pop_completions(ctx=ctx)

    assert len(completions) == 1
    note = completions[0].note.message.text
    assert "<status> completed </status>" in note
    assert "own: x" in note
    assert "inner boom" not in note
