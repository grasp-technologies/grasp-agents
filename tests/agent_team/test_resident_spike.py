"""
Resident-agent spike: one communicator runs its own loop off a session inbox,
consuming peer messages and its own background-task completions across turns
WITHOUT terminating on a final answer. Residency is set by attaching the agent's
session inbox to its loop — ``decide_next_step`` then never stops while the inbox
is open, so the loop ends only when its task is cancelled from outside. A trivial
synthetic seed satisfies the run's input requirement (the resident wait fires
before any generation, so the seed is context, not reacted to). Lone/triggered
agents are unaffected (no inbox attached → original single-answer behavior).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.inbox import AgentInbox
from grasp_agents.run_context import RunContext
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.types.items import InputMessageItem
from grasp_agents.types.message import TeamMessage
from tests._helpers import MockLLM, _text_response, _tool_call_response

pytestmark = pytest.mark.asyncio


async def _until(pred: Callable[[], bool]) -> None:
    """Poll ``pred`` until true, bounded to ~3s so the resident loop can't hang."""
    for _ in range(300):
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


async def test_resident_consumes_multiple_messages_in_one_run() -> None:
    agent = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(
            responses_queue=[_text_response("reply 1"), _text_response("reply 2")]
        ),
        ctx=RunContext[None](),
    )
    # Attaching an inbox makes the agent resident: the seeded run_stream then
    # drives the loop off this inbox instead of terminating on a final answer.
    inbox = AgentInbox(recipient="curator")
    agent.inbox = inbox

    run = asyncio.create_task(_collect(agent))
    await inbox.post(
        TeamMessage.of_text(sender="user", to="curator", text="first task")
    )
    try:
        await _until(lambda: agent.llm.call_count == 1)

        # The loop is still the SAME resident run — feed it a second message.
        await inbox.post(
            TeamMessage.of_text(sender="user", to="curator", text="second task")
        )
        await _until(lambda: agent.llm.call_count == 2)

        # Two activations, one run: it did not terminate on the first answer.
        assert not run.done()
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        await agent.aclose()

    texts = [
        part.text
        for m in agent.transcript.messages
        if isinstance(m, InputMessageItem)
        for part in m.content_parts
        if getattr(part, "text", None) is not None
    ]
    # Each peer message is rendered as a user turn ("Message from <sender>: ...").
    assert any("first task" in t for t in texts), texts
    assert any("second task" in t for t in texts), texts
    assert agent.llm.call_count == 2


async def test_resident_survives_past_max_turns() -> None:
    # A resident must NOT fall out of the loop on the turn budget — it runs until
    # cancelled. With max_turns=2 and 4 messages (cumulative turns > 2), the pre-fix
    # `while turn <= max_turns` exited and tripped the final-answer assert.
    agent = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(
            responses_queue=[_text_response(f"reply {i}") for i in range(4)]
        ),
        max_turns=2,
        ctx=RunContext[None](),
    )
    inbox = AgentInbox(recipient="curator")
    agent.inbox = inbox

    run = asyncio.create_task(_collect(agent))
    try:
        for i in range(4):
            await inbox.post(
                TeamMessage.of_text(sender="user", to="curator", text=f"m{i}")
            )
            await _until(lambda i=i: agent.llm.call_count == i + 1)
        # Cumulative turns now exceed max_turns, yet the loop is still alive.
        assert not run.done()
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        await agent.aclose()

    assert agent.llm.call_count == 4


async def test_bg_completion_while_idle_wakes_resident_loop() -> None:
    release = asyncio.Event()

    @function_tool(auto_background_at=0, blocks_final_answer=False)
    async def slow_job(text: str) -> str:
        """A slow background job that finishes only once released."""
        await release.wait()
        return f"job done: {text}"

    agent = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(
            responses_queue=[
                _tool_call_response("slow_job", '{"text":"vault"}', "tc1"),
                _text_response("started; idling"),
                _text_response("completion handled"),
            ]
        ),
        tools=[slow_job],
        ctx=RunContext[None](),
    )
    inbox = AgentInbox(recipient="curator")
    agent.inbox = inbox

    run = asyncio.create_task(_collect(agent))
    await inbox.post(
        TeamMessage.of_text(sender="user", to="curator", text="stash this in the vault")
    )
    try:
        # Round 1: tool call (call 1) then "started; idling" (call 2). The loop is
        # now idle, parked on the inbox while the backgrounded job runs.
        await _until(lambda: agent.llm.call_count == 2)
        assert agent.background_tasks.has_live_tasks

        # Release the job: its completion (NOT a peer message) must wake the loop
        # so the next turn's drain delivers the <task_notification>.
        release.set()
        await _until(lambda: agent.llm.call_count == 3)
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        await agent.aclose()

    transcript_text = str(agent.transcript.messages)
    assert "job done: vault" in transcript_text, transcript_text
    assert "<task_notification>" in transcript_text


async def _collect(agent: LLMAgent[Any, Any, None]) -> None:
    # The synthetic seed satisfies the input requirement; the resident PRE-ACT
    # wait fires before any generation, so it is context, not reacted to — the
    # first real turn is the first inbox message.
    async for _event in agent.run_stream("team started"):
        pass
