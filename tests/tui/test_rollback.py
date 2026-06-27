"""Headless tests for rollback-to-a-previous-message (picker + pane truncation)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("textual")

from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import (
    Event,
    OutputMessageItemEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
)
from grasp_agents.types.items import OutputMessageItem
from grasp_agents.ui._widgets import RollbackScreen
from grasp_agents.ui.app import GraspAgentsApp, PromptArea


async def _fake(text: str) -> AsyncIterator[Event[Any]]:
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


async def _submit(pilot: Any, app: GraspAgentsApp, text: str) -> None:
    app.query_one("#prompt", PromptArea).text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
async def test_do_rollback_truncates_pane_and_calls_host() -> None:
    calls: list[int] = []

    async def on_rollback(index: int) -> None:
        calls.append(index)

    app = GraspAgentsApp(on_submit=_fake, on_rollback=on_rollback, main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        for msg in ("m0", "m1", "m2"):
            await _submit(pilot, app, msg)
        assert app._ga_user_turns == ["m0", "m1", "m2"]
        pane = app._ga_panes["analyst"]
        mark1 = app._ga_turn_marks[1]
        assert len(pane.children) > mark1  # later turns added widgets

        await app._do_rollback(1)
        await pilot.pause()

        assert calls == [1]  # host was asked to rewind to message index 1
        assert app._ga_user_turns == ["m0"]  # turns 1+ dropped
        assert app._ga_turn_marks == [0]
        assert len(pane.children) == mark1  # pane truncated to turn 1's start
        # the rolled-back message is handed back in the prompt for editing
        assert app.query_one("#prompt", PromptArea).text == "m1"


@pytest.mark.asyncio
async def test_slash_rollback_opens_picker_without_sending_a_turn() -> None:
    async def on_rollback(index: int) -> None:
        del index

    app = GraspAgentsApp(on_submit=_fake, on_rollback=on_rollback, main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, app, "m0")
        app.query_one("#prompt", PromptArea).text = "/rollback"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, RollbackScreen)
        assert app._ga_user_turns == ["m0"]  # /rollback was not sent to the agent


@pytest.mark.asyncio
async def test_ctrl_r_opens_picker() -> None:
    async def on_rollback(index: int) -> None:
        del index

    app = GraspAgentsApp(on_submit=_fake, on_rollback=on_rollback, main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, app, "m0")
        await pilot.press("ctrl+r")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, RollbackScreen)


@pytest.mark.asyncio
async def test_rollback_flow_selects_earlier_message_and_rewinds() -> None:
    calls: list[int] = []

    async def on_rollback(index: int) -> None:
        calls.append(index)

    app = GraspAgentsApp(on_submit=_fake, on_rollback=on_rollback, main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, app, "m0")
        await _submit(pilot, app, "m1")
        app.query_one("#prompt", PromptArea).text = "/rollback"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, RollbackScreen)
        await pilot.press("up")  # default highlight is the most recent (m1) → m0
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert calls == [0]
        assert app._ga_user_turns == []
        assert app.query_one("#prompt", PromptArea).text == "m0"  # prefilled to edit


@pytest.mark.asyncio
async def test_rollback_rewinds_the_turn_counter() -> None:
    # The displayed turn count is monotonic across steps; a rollback must rewind
    # it to its value at the rolled-back message, so the next run continues from
    # there instead of climbing from the pre-rollback total.
    async def on_rollback(index: int) -> None:
        del index

    app = GraspAgentsApp(on_submit=_fake, on_rollback=on_rollback, main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        for msg in ("m0", "m1", "m2"):
            await _submit(pilot, app, msg)
        assert app._ga_turns["analyst"] == 3  # one turn per submission

        await app._do_rollback(1)  # rewind to m1's start (after m0's single turn)
        await pilot.pause()
        assert app._ga_turns["analyst"] == 1  # counter rewound, not left at 3

        await _submit(pilot, app, "m1-again")
        assert app._ga_turns["analyst"] == 2  # continues from 1, not from 3 → 4


@pytest.mark.asyncio
async def test_rollback_escape_cancels() -> None:
    calls: list[int] = []

    async def on_rollback(index: int) -> None:
        calls.append(index)

    app = GraspAgentsApp(on_submit=_fake, on_rollback=on_rollback, main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, app, "m0")
        app.query_one("#prompt", PromptArea).text = "/rollback"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, RollbackScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert calls == []
        assert app._ga_user_turns == ["m0"]
