"""Headless test: Esc interrupts an in-flight interactive turn."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")

from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import (
    OutputMessageItemEvent,
    TurnInfo,
    TurnStartEvent,
)
from grasp_agents.types.items import OutputMessageItem
from grasp_agents.ui.app import GraspAgentsApp, PromptArea, SelectableStatic


def _msg(text: str) -> OutputMessageItemEvent:
    return OutputMessageItemEvent(
        data=OutputMessageItem(
            content=[OutputMessageText(text=text)], status="completed"
        ),
        source="ops",
    )


async def _slow_agent(text: str):
    """Stream a line, then park forever — until the turn is interrupted."""
    yield TurnStartEvent(data=TurnInfo(turn=0), source="ops")
    yield _msg("working…")
    await asyncio.sleep(30)
    yield _msg("done")  # never reached: the run is cancelled while parked above


async def _wait(pilot, predicate, *, ticks: int = 30) -> None:
    for _ in range(ticks):
        await pilot.pause()
        if predicate():
            return


@pytest.mark.asyncio
async def test_escape_interrupts_running_turn() -> None:
    app = GraspAgentsApp(on_submit=_slow_agent, main_agent="ops")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", PromptArea).text = "go"
        await pilot.press("enter")
        await _wait(pilot, lambda: app._ga_running and "ops" in app._ga_panes)
        assert app._ga_running is True
        prompt = app.query_one("#prompt", PromptArea)
        assert prompt.disabled is True  # disabled while the turn runs

        await pilot.press("escape")
        await _wait(pilot, lambda: not app._ga_running)

        assert app._ga_running is False  # the turn was cancelled
        assert prompt.disabled is False  # ...and the prompt re-enabled
        # Static keeps its renderable in the name-mangled __content.
        notes = [
            str(getattr(w, "_Static__content", ""))
            for w in app._ga_panes["ops"].query(SelectableStatic)
        ]
        assert any("interrupt" in n.lower() for n in notes), notes


@pytest.mark.asyncio
async def test_escape_is_a_noop_when_idle() -> None:
    # No turn running: Esc must not raise or wedge the app.
    app = GraspAgentsApp(on_submit=_slow_agent, main_agent="ops")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._ga_running is False
        await pilot.press("escape")
        await pilot.pause()
        assert app._ga_running is False
        assert app.query_one("#prompt", PromptArea).disabled is False
