"""
The self-scheduled wakeup primitive: the :class:`WakeupScheduler` timer source and
the ``ScheduleWakeup`` tool that lets a daemon member reactivate itself.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent_team.agent_team import AgentTeam
from grasp_agents.agent_team.events import MessageDeliveredEvent
from grasp_agents.agent_team.message import TeamMessage
from grasp_agents.agent_team.sources import WakeupScheduler
from grasp_agents.agent_team.tools import ScheduleWakeupInput, ScheduleWakeupTool
from grasp_agents.agent_team.transport import InMemoryMailboxTransport
from grasp_agents.file_backend.local import LocalFileBackend
from grasp_agents.run_context import RunContext
from grasp_agents.types.response import Response
from tests._helpers import MockLLM, _text_response, _tool_call_response


def _ctx(tmp_path: Path) -> RunContext[None]:
    return RunContext[None](
        state=None, file_backend=LocalFileBackend(allowed_roots=[tmp_path])
    )


def _agent(name: str, responses: list[Response]) -> LLMAgent[Any, Any, None]:
    return LLMAgent[Any, Any, None](name=name, llm=MockLLM(responses_queue=responses))


def _schedule(delay: float, note: str, call_id: str) -> Response:
    return _tool_call_response(
        "ScheduleWakeup", f'{{"delay_seconds": {delay}, "note": "{note}"}}', call_id
    )


async def _until(pred: Callable[[], bool]) -> None:
    """Poll ``pred`` until true, bounded to ~3s so a daemon test can't hang."""
    for _ in range(300):
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


@pytest.mark.asyncio
async def test_scheduler_delivers_after_delay() -> None:
    transport = InMemoryMailboxTransport()
    scheduler = WakeupScheduler(transport)
    scheduler.schedule(
        TeamMessage.of_text(sender="curator", to="curator", text="ping"), delay=0.02
    )
    assert scheduler.pending == 1

    # The timer is removed once it has fired (after the post), so pending == 0
    # implies the message has landed — no race with the has_pending check.
    await _until(lambda: scheduler.pending == 0)
    assert await transport.has_pending("curator")
    got = await transport.consume("curator")
    assert isinstance(got, TeamMessage)
    assert got.text == "ping"


@pytest.mark.asyncio
async def test_scheduler_aclose_cancels_pending() -> None:
    transport = InMemoryMailboxTransport()
    scheduler = WakeupScheduler(transport)
    scheduler.schedule(
        TeamMessage.of_text(sender="c", to="c", text="late"), delay=100
    )
    assert scheduler.pending == 1

    await scheduler.aclose()

    assert scheduler.pending == 0
    assert await transport.has_pending("c") is False


@pytest.mark.asyncio
async def test_schedule_wakeup_tool_self_addresses() -> None:
    transport = InMemoryMailboxTransport()
    scheduler = WakeupScheduler(transport)
    tool = ScheduleWakeupTool(scheduler)

    out = await tool._run(
        ScheduleWakeupInput(delay_seconds=0.02, note="check the vault"),
        agent_ctx=SimpleNamespace(agent_name="curator"),  # type: ignore[arg-type]
    )
    assert "Wake-up scheduled" in out

    await _until(lambda: scheduler.pending == 0)
    msg = await transport.consume("curator")
    assert isinstance(msg, TeamMessage)
    # Self-addressed: the scheduling member is both sender and recipient.
    assert msg.sender == "curator"
    assert msg.recipient == "curator"
    assert "<scheduled_wakeup>" in msg.text
    assert "check the vault" in msg.text


@pytest.mark.asyncio
async def test_schedule_wakeup_requires_agent_ctx() -> None:
    tool = ScheduleWakeupTool(WakeupScheduler(InMemoryMailboxTransport()))
    with pytest.raises(ValueError, match="team member"):
        await tool._run(
            ScheduleWakeupInput(delay_seconds=1, note="x"), agent_ctx=None
        )


@pytest.mark.asyncio
async def test_daemon_member_wakes_itself(tmp_path: Path) -> None:
    # A daemon member with no peer traffic reactivates itself: turn 1 schedules a
    # wakeup and goes idle; the timer fires and delivers a self-addressed message
    # that drives turn 2. This is initiative without a resident loop.
    ctx = _ctx(tmp_path)
    solo = _agent(
        "solo",
        [
            _schedule(0.02, "revisit the goal", "c1"),  # turn 1: schedule
            _text_response("scheduled; going idle"),  # turn 1: after tool result
            _text_response("woke up and acted"),  # turn 2: driven by the wakeup
        ],
    )
    team = AgentTeam([solo], ctx=ctx)

    delivered: list[TeamMessage] = []

    async def _drain() -> None:
        async for ev in team.run_stream("start", daemon=True, poll_interval=0.01):
            if isinstance(ev, MessageDeliveredEvent):
                delivered.append(ev.data)

    consumer = asyncio.create_task(_drain())
    try:
        await _until(lambda: solo.llm.call_count == 3)
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
    await team.aclose()

    wakeups = [m for m in delivered if m.sender == "solo" and m.recipient == "solo"]
    assert wakeups, "member did not reactivate itself via a self-addressed wakeup"
    assert any("revisit the goal" in m.text for m in wakeups)
    assert any("<scheduled_wakeup>" in m.text for m in wakeups)
