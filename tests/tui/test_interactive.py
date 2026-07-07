"""Headless interactive-mode test: typing a message drives the agent."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from textual.widgets import Static

from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import (
    OutputMessageItemEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import InputMessageItem, OutputMessageItem
from grasp_agents.ui.app import GraspAgentsApp, PromptArea


async def _fake_agent(text: str):
    yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
    yield OutputMessageItemEvent(
        data=OutputMessageItem(
            content=[OutputMessageText(text=f"echo: {text}")], status="completed"
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


@pytest.mark.asyncio
async def test_on_post_queues_submissions_until_taken() -> None:
    # Mailbox mode: submissions post fire-and-forget, the prompt stays enabled
    # (messages can stack), and each is listed as queued until its drained user
    # turn (source = the human sender) renders — then the oldest pops.
    posted: list[str] = []

    async def on_post(text: str) -> None:
        posted.append(text)

    app = GraspAgentsApp(on_post=on_post, main_agent="lead")
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", PromptArea)
        prompt.text = "first"
        await pilot.press("enter")
        prompt.text = "second"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert posted == ["first", "second"]
        assert not prompt.disabled  # never blocked while the member works
        assert [label for _, label in app._ga_queued] == ["first", "second"]
        assert app.query_one("#queued-strip", Static).display

        # The member takes the oldest queued message at its turn boundary.
        msg = InputMessageItem.from_text("first", role="user")
        await app._feed(
            UserMessageEvent(data=msg, source="user", destination="lead", exec_id="e1")
        )
        await pilot.pause()
        assert [label for _, label in app._ga_queued] == ["second"]

        # A peer hand-off (source = a member) must NOT pop the human queue.
        peer = InputMessageItem.from_text("from a peer", role="user")
        await app._feed(
            UserMessageEvent(
                data=peer, source="writer", destination="lead", exec_id="e2"
            )
        )
        await pilot.pause()
        assert [label for _, label in app._ga_queued] == ["second"]


@pytest.mark.asyncio
async def test_on_post_failure_unqueues_and_notifies() -> None:
    async def on_post(text: str) -> None:
        raise RuntimeError("mailbox down")

    app = GraspAgentsApp(on_post=on_post, main_agent="lead")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", PromptArea).text = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._ga_queued == []
        assert not app.query_one("#queued-strip", Static).display


@pytest.mark.asyncio
async def test_roster_pre_creates_idle_panes() -> None:
    """
    Known members get panes at launch (idle) — a resumed session would
    otherwise show nothing until a member speaks; a turn flips it to working.
    """

    async def on_post(text: str) -> None:
        pass

    app = GraspAgentsApp(
        on_post=on_post, main_agent="lead", agents=["lead", "researcher"]
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert set(app._ga_panes) == {"lead", "researcher"}
        assert app._ga_status["lead"] == "idle"
        assert app._ga_status["researcher"] == "idle"

        await app._feed(TurnStartEvent(data=TurnInfo(turn=0), source="lead"))
        await pilot.pause()
        assert app._ga_status["lead"] == "working"
        assert app._ga_status["researcher"] == "idle"
