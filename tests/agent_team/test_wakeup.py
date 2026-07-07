"""
The self-scheduled wakeup: ``ScheduleWakeup`` is a **background tool** — it sleeps
the delay off-turn, then its completion reactivates the (idle) member with the note,
reusing the durable background-task substrate (no separate timer / scheduler).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent_team.agent_card import MemberCard
from grasp_agents.agent_team.agent_team import AgentTeam
from grasp_agents.agent_team.member import MemberHost
from grasp_agents.agent_team.tools import (
    SCHEDULE_WAKEUP_TOOL_NAME,
    ScheduleWakeupInput,
    ScheduleWakeupTool,
)
from grasp_agents.file_backend.local import LocalFileBackend
from grasp_agents.session_context import SessionContext
from grasp_agents.types.response import Response
from tests._helpers import MockLLM, _text_response, _tool_call_response


def _ctx(tmp_path: Path) -> SessionContext[None]:
    return SessionContext[None](
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


def test_schedule_wakeup_is_a_background_tool() -> None:
    tool = ScheduleWakeupTool()
    # Backgrounds immediately and never blocks the final answer, so the member parks
    # (or keeps working) while the wakeup is pending, and its completion wakes it.
    assert tool.auto_background_at == 0.0
    assert tool.blocks_final_answer is False


@pytest.mark.asyncio
async def test_schedule_wakeup_sleeps_then_returns_note() -> None:
    tool = ScheduleWakeupTool()
    out = await tool._run(  # pyright: ignore[reportPrivateUsage]
        ScheduleWakeupInput(delay_seconds=0.01, note="check the vault")
    )
    assert "<scheduled_wakeup>" in out
    assert "check the vault" in out


@pytest.mark.asyncio
async def test_daemon_member_wakes_itself(tmp_path: Path) -> None:
    # A daemon member with no peer traffic reactivates itself: turn 1 schedules a
    # wakeup (a background sleep) and goes idle; the wakeup's completion is drained as
    # a task notification carrying the note, driving a later turn — initiative with no
    # separate timer.
    ctx = _ctx(tmp_path)
    solo = _agent(
        "solo",
        [
            _schedule(0.02, "revisit the goal", "c1"),  # turn 1: schedule (bg)
            _text_response("scheduled; going idle"),  # turn 2: park
            _text_response("woke up and acted"),  # turn 3: driven by the wakeup
        ],
    )
    team = AgentTeam([solo], cards=[MemberCard(name="solo", wakeups=True)], ctx=ctx)

    async def _drain() -> None:
        async for _ in team.run_stream("start", daemon=True, poll_interval=0.01):
            pass

    consumer = asyncio.create_task(_drain())
    try:
        await _until(lambda: solo.llm.call_count == 3)
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
    await team.aclose()

    # The note arrived as a wakeup notification (a background-task completion), not a
    # self-addressed mailbox message.
    blob = str(solo.transcript.messages)
    assert "revisit the goal" in blob
    assert "<scheduled_wakeup>" in blob


@pytest.mark.asyncio
async def test_wakeups_are_opt_in_per_card(tmp_path: Path) -> None:
    # ScheduleWakeup is opt-in (mail already wakes a resident): only a member
    # carded with wakeups=True carries the tool — in both hosting layers — and
    # only that member's team framing mentions it.
    ctx = _ctx(tmp_path)
    plain = _agent("plain", [_text_response("hi")])
    keeper = _agent("keeper", [_text_response("hi")])
    cards = [MemberCard(name="plain"), MemberCard(name="keeper", wakeups=True)]
    team = AgentTeam([plain, keeper], entry="plain", cards=cards, ctx=ctx)

    def team_text(agent: LLMAgent[Any, Any, None]) -> str:
        sections = agent._prompt_builder.system_prompt_sections
        text = next(s for s in sections if s.name == "team").compute()
        assert isinstance(text, str)
        return text

    assert SCHEDULE_WAKEUP_TOOL_NAME not in plain.tools
    assert SCHEDULE_WAKEUP_TOOL_NAME in keeper.tools
    assert "ScheduleWakeup" not in team_text(plain)
    assert "ScheduleWakeup" in team_text(keeper)
    await team.aclose()

    host_plain = MemberHost(_agent("plain", []), cards=cards)
    host_keeper = MemberHost(_agent("keeper", []), cards=cards)
    assert SCHEDULE_WAKEUP_TOOL_NAME not in host_plain._member.tools  # type: ignore[attr-defined]
    assert SCHEDULE_WAKEUP_TOOL_NAME in host_keeper._member.tools  # type: ignore[attr-defined]
    await host_plain.aclose()
    await host_keeper.aclose()
