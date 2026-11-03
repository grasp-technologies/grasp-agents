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
async def test_prompt_grows_with_wrapped_text_up_to_its_cap() -> None:
    app = GraspAgentsApp(on_submit=_fake_agent, main_agent="analyst")
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", PromptArea)
        prompt.insert("word " * 40)
        await pilot.pause()
        assert prompt.document.line_count == 1
        assert prompt.size.height == prompt.wrapped_document.height > 1

        await pilot.resize_terminal(30, 30)
        await pilot.pause()
        assert prompt.size.height == PromptArea._MAX_ROWS


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


class _FailingDelivery:
    """Background work that has finished, and whose delivery turn fails."""

    def __init__(self) -> None:
        self.delivered = False

    @property
    def outstanding(self) -> bool:
        return not self.delivered

    @property
    def finished(self) -> bool:
        return not self.delivered

    async def progress(self):
        return
        yield

    async def deliver(self):
        self.delivered = True
        raise RuntimeError("model unavailable")
        yield


@pytest.mark.asyncio
async def test_a_failed_delivery_turn_is_noted_and_the_app_stays() -> None:
    app = GraspAgentsApp(
        on_submit=_fake_agent, main_agent="analyst", background=_FailingDelivery()
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", PromptArea).text = "hello"
        await pilot.press("enter")
        # The turn, then the background watcher, then the delivery turn it starts.
        for _ in range(3):
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert app.is_running
        notes = [str(note.render()) for note in app.query(".ga-msg")]
        assert any(
            "delivering a finished task failed: model unavailable" in n for n in notes
        )
        assert not app.query_one("#prompt", PromptArea).disabled


@pytest.mark.asyncio
async def test_interrupting_a_turn_before_it_starts_gives_the_prompt_back() -> None:
    app = GraspAgentsApp(on_submit=_fake_agent, main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._dispatch("hello")  # pyright: ignore[reportPrivateUsage]
        await app.action_interrupt()
        await pilot.pause()
        assert not app._ga_running  # pyright: ignore[reportPrivateUsage]
        assert not app.query_one("#prompt", PromptArea).disabled


def test_a_parked_agent_takes_input_before_a_delivery() -> None:
    from types import SimpleNamespace

    from grasp_agents.ui.app import _AgentTurns  # pyright: ignore[reportPrivateUsage]

    bg_tasks = SimpleNamespace(has_undelivered_completions=True)
    agent = SimpleNamespace(
        agent_ctx=SimpleNamespace(bg_tasks=bg_tasks), parked_after_rollback=True
    )
    turns = _AgentTurns(agent)  # type: ignore[arg-type]
    assert not turns.finished
    agent.parked_after_rollback = False
    assert turns.finished
