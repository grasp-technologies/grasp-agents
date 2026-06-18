"""
Cancelling a run mid-turn rolls back to the last settled boundary — the last
completed tool round or the submitted message — keeping the work that finished
and dropping only the unfinished tail. The transcript is pruned exactly as on
session resume (``prepare_messages_for_resume``), and the paired agent-context
state (here, the shell cwd) is restored to the same boundary. The TUI's
Esc-interrupt relies on this so the next turn isn't poisoned by a dangling call
or a context that disagrees with the transcript.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.items import FunctionToolCallItem, FunctionToolOutputItem
from tests._helpers import MockLLM, _text_response, _tool_call_response


class _ToolInput(BaseModel):
    pass


class _CwdTool(BaseTool[_ToolInput, str, Any]):
    """Mutates the agent-context shell cwd, then optionally parks until released."""

    def __init__(
        self,
        name: str,
        cwd: str,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        super().__init__(name=name, description="Sets the shell cwd.")
        self._cwd = cwd
        self._started = started
        self._release = release

    async def _run(
        self,
        inp: _ToolInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
        path: Any = None,
        agent_ctx: Any = None,
    ) -> str:
        del inp, ctx, exec_id, progress_callback, path
        if agent_ctx is not None:
            agent_ctx.shell_state.cwd = self._cwd
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            await self._release.wait()
        return "ok"


def _call_ids(agent: LLMAgent[Any, Any, Any]) -> tuple[list[str], list[str]]:
    msgs = agent.transcript.messages
    calls = [m.call_id for m in msgs if isinstance(m, FunctionToolCallItem)]
    outs = [m.call_id for m in msgs if isinstance(m, FunctionToolOutputItem)]
    return calls, outs


async def _cancel(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_first_turn_prunes_and_restores_state() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    agent = LLMAgent[str, str, None](
        name="a",
        llm=MockLLM(
            model_name="mock",
            responses_queue=[
                _tool_call_response("block", "{}", "tc1"),
                _text_response("second"),  # next run
            ],
        ),
        tools=[_CwdTool("block", "/mutated", started, release)],
        sys_prompt="x",
        env_info=False,
    )

    async def consume() -> None:
        async for _ in agent.run_stream("go"):
            pass

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(started.wait(), timeout=2.0)
        assert _call_ids(agent) == (["tc1"], [])  # dangling call committed
        assert agent._loop.agent_ctx.shell_state.cwd == "/mutated"  # tool mutated it
        await _cancel(task)
    finally:
        release.set()

    agent.transcript.validate_tool_call_pairing()
    assert _call_ids(agent) == ([], [])  # cancelled turn pruned, no dangling call
    assert agent._loop.agent_ctx.shell_state.cwd is None  # state restored to boundary

    out = await agent.run("again")  # reusable
    assert out.payloads[0] == "second"


@pytest.mark.asyncio
async def test_cancel_strips_whole_turn_including_completed_round() -> None:
    # A dangling call strips the WHOLE incomplete assistant turn back to the
    # submitted message (resume semantics) — an earlier completed round in the
    # same turn goes too, and its context mutation reverts with it.
    started, release = asyncio.Event(), asyncio.Event()
    agent = LLMAgent[str, str, None](
        name="a",
        llm=MockLLM(
            model_name="mock",
            responses_queue=[
                _tool_call_response("step", "{}", "tc1"),  # turn 0: completes
                _tool_call_response("block", "{}", "tc2"),  # turn 1: parks
                _text_response("second"),  # next run
            ],
        ),
        tools=[
            _CwdTool("step", "/a"),
            _CwdTool("block", "/b", started, release),
        ],
        sys_prompt="x",
        env_info=False,
    )

    async def consume() -> None:
        async for _ in agent.run_stream("go"):
            pass

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(started.wait(), timeout=2.0)
        # turn 0 completed (tc1 + result); turn 1's tc2 is dangling
        assert _call_ids(agent) == (["tc1", "tc2"], ["tc1"])
        await _cancel(task)
    finally:
        release.set()

    agent.transcript.validate_tool_call_pairing()
    # the whole incomplete turn (both rounds) is stripped back to the message
    assert _call_ids(agent) == ([], [])
    # ...and the turn's context mutations revert with it (pre-turn cwd = None)
    assert agent._loop.agent_ctx.shell_state.cwd is None

    out = await agent.run("again")
    assert out.payloads[0] == "second"
