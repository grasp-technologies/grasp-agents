"""Headless interactive-mode test: typing a message drives the agent."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import (
    OutputMessageItemEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
)
from grasp_agents.types.items import OutputMessageItem
from grasp_agents.ui.app import GraspAgentsApp, PromptArea


async def _fake_agent(text: str):
    yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
    yield OutputMessageItemEvent(
        data=OutputMessageItem(
            content_parts=[OutputMessageText(text=f"echo: {text}")], status="completed"
        ),
        source="analyst",
    )
    yield TurnEndEvent(
        data=TurnEndInfo(turn=0, had_tool_calls=False, stop_reason="final_answer"),  # type: ignore[arg-type]
        source="analyst",
    )


@pytest.mark.asyncio
async def test_interactive_submit_runs_agent() -> None:
    app = GraspAgentsApp(on_submit=_fake_agent, main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", PromptArea).text = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "analyst" in app._ga_panes
        assert app._ga_status.get("analyst") == "done"


@pytest.mark.asyncio
async def test_newline_keys_insert_newline_and_do_not_submit() -> None:
    app = GraspAgentsApp(on_submit=_fake_agent, main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", PromptArea)
        prompt.insert("hello")
        await pilot.press("shift+enter")  # kitty-protocol newline
        await pilot.press("ctrl+j")  # portable newline (LF)
        await pilot.pause()
        assert prompt.text.count("\n") == 2, repr(prompt.text)
        assert "analyst" not in app._ga_panes  # newline must NOT submit
        await pilot.press("enter")  # enter submits
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "analyst" in app._ga_panes
