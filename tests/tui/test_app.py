"""
Headless behavior tests via Textual's Pilot — no manual UI inspection.

Feeds the fixed demo event stream and asserts the tree/panes/status that the
app derives, with zero rendering checks.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from textual.widgets import ContentSwitcher, Tab

from grasp_agents.ui.app import GraspAgentsApp, _pane_id, _slug, _tab_id
from grasp_agents.ui.demo import build_demo_events


def test_slug_distinguishes_colliding_sources() -> None:
    """
    Distinct sources must not collapse to one DOM id (Textual rejects
    duplicate ids, which would kill the feed worker).
    """
    assert _slug("worker 1") != _slug("worker-1")
    assert _pane_id("worker 1") != _pane_id("worker-1")
    assert _tab_id("worker 1") != _tab_id("worker-1")
    # Stable for a given source.
    assert _slug("research-agent") == _slug("research-agent")


async def _drain(app: GraspAgentsApp) -> None:
    await app.wait_for_stream()


@pytest.mark.asyncio
async def test_builds_one_pane_per_agent_and_subagent() -> None:
    app = GraspAgentsApp(build_demo_events(delay=0.0))
    async with app.run_test() as pilot:
        await _drain(app)
        await pilot.pause()
        # …plus a log pane per backgrounded task (the demo indexes sources).
        assert set(app._ga_panes) == {
            "coordinator",
            "researcher",
            "writer",
            "index_sources bg1",
        }


@pytest.mark.asyncio
async def test_subagents_nest_under_caller() -> None:
    app = GraspAgentsApp(build_demo_events(delay=0.0))
    async with app.run_test() as pilot:
        await _drain(app)
        await pilot.pause()
        assert app._ga_parent["researcher"] == "coordinator"
        assert app._ga_parent["writer"] == "coordinator"


@pytest.mark.asyncio
async def test_status_transitions_to_done() -> None:
    app = GraspAgentsApp(build_demo_events(delay=0.0))
    async with app.run_test() as pilot:
        await _drain(app)
        await pilot.pause()
        assert app._ga_status["coordinator"] == "done"
        assert app._ga_status["researcher"] == "done"
        assert app._ga_status["writer"] == "done"


@pytest.mark.asyncio
async def test_active_pane_defaults_to_first_agent() -> None:
    app = GraspAgentsApp(build_demo_events(delay=0.0))
    async with app.run_test() as pilot:
        await _drain(app)
        await pilot.pause()
        switcher = app.query_one("#panes", ContentSwitcher)
        # follow is off by default → the first agent's pane stays active
        assert switcher.current == _pane_id("coordinator")


@pytest.mark.asyncio
async def test_toggle_follow_action() -> None:
    app = GraspAgentsApp(build_demo_events(delay=0.0))
    async with app.run_test() as pilot:
        assert app._ga_follow is False
        await pilot.press("f")
        assert app._ga_follow is True


@pytest.mark.asyncio
async def test_only_one_pane_visible_no_split() -> None:
    # spawning subagents mounts their panes hidden; only the active pane shows
    # (regression: plain mount() left every pane displayed → vertical split)
    app = GraspAgentsApp(build_demo_events(delay=0.0))
    async with app.run_test() as pilot:
        await _drain(app)
        await pilot.pause()
        visible = [src for src, pane in app._ga_panes.items() if pane.display]
        assert visible == ["coordinator"], visible


@pytest.mark.asyncio
async def test_background_task_gets_its_own_log_pane() -> None:
    """A backgrounded task's stamped stream renders in its own pane, not inline."""
    from grasp_agents.types.events import (
        BackgroundTaskCompletedEvent,
        BackgroundTaskInfo,
        BackgroundTaskLaunchedEvent,
        ToolStreamEvent,
        TurnInfo,
        TurnStartEvent,
    )

    info = BackgroundTaskInfo(
        task_id="bg_1", tool_name="Bash", tool_call_id="c1", output_name="c1.log"
    )
    app = GraspAgentsApp()
    async with app.run_test() as pilot:
        await app._feed(TurnStartEvent(data=TurnInfo(turn=0), source="coordinator"))
        await app._feed(BackgroundTaskLaunchedEvent(source="coordinator", data=info))
        for chunk in ("hello ", "world"):
            await app._feed(
                ToolStreamEvent(
                    data=chunk, source="Bash", destination="coordinator", task_id="bg_1"
                )
            )
        await pilot.pause()
        # Its own pane, nested under the launching agent, accumulating the log.
        assert "Bash bg_1" in app._ga_panes
        assert app._ga_parent["Bash bg_1"] == "coordinator"
        assert app._ga_task_log_text["Bash bg_1"] == "hello world"
        # Nothing folded into the agent pane's inline live-output box.
        assert "coordinator" not in app._ga_stream_tool
        # Its tab is tinted as a task tab; an agent's tab is not.
        assert app.query_one(f"#{_tab_id('Bash bg_1')}", Tab).has_class("ga-task")
        assert not app.query_one(f"#{_tab_id('coordinator')}", Tab).has_class("ga-task")

        await app._feed(BackgroundTaskCompletedEvent(source="coordinator", data=info))
        await pilot.pause()
        assert app._ga_status["Bash bg_1"] == "done"
        assert ("coordinator", "bg_1") not in app._ga_task_panes
        # The tint outlives completion — the pane stays a task pane.
        assert app.query_one(f"#{_tab_id('Bash bg_1')}", Tab).has_class("ga-task")


@pytest.mark.asyncio
async def test_unstamped_tool_stream_stays_inline() -> None:
    """A foreground stream (no task_id) renders in the agent pane as before."""
    from grasp_agents.types.events import ToolStreamEvent, TurnInfo, TurnStartEvent

    app = GraspAgentsApp()
    async with app.run_test() as pilot:
        await app._feed(TurnStartEvent(data=TurnInfo(turn=0), source="coordinator"))
        await app._feed(
            ToolStreamEvent(data="x", source="Bash", destination="coordinator")
        )
        await pilot.pause()
        assert "coordinator" in app._ga_stream_tool
        assert not app._ga_task_keys


@pytest.mark.asyncio
async def test_bg_lifecycle_notices_route_to_launching_agent() -> None:
    """
    Launch/completion notices render in the launcher's pane — not in
    whichever member happened to stream last (interleaved team streams).
    """
    from grasp_agents.types.events import (
        BackgroundTaskCompletedEvent,
        BackgroundTaskInfo,
        BackgroundTaskLaunchedEvent,
        TurnInfo,
        TurnStartEvent,
    )

    app = GraspAgentsApp()
    async with app.run_test() as pilot:
        await app._feed(TurnStartEvent(data=TurnInfo(turn=0), source="lead"))
        await app._feed(TurnStartEvent(data=TurnInfo(turn=0), source="researcher"))
        assert app._ga_last_agent == "researcher"

        info = BackgroundTaskInfo(task_id="bg_9", tool_name="poller", tool_call_id="c9")
        await app._feed(BackgroundTaskLaunchedEvent(source="lead", data=info))
        await pilot.pause()
        # The ⧗ notice mounted in the lead's pane (turn headers are deferred,
        # so the notice is its only child); nothing leaked to the researcher.
        assert len(app._ga_panes["lead"].children) == 1
        assert len(app._ga_panes["researcher"].children) == 0

        await app._feed(BackgroundTaskCompletedEvent(source="lead", data=info))
        await pilot.pause()
        assert len(app._ga_panes["lead"].children) == 2
        assert len(app._ga_panes["researcher"].children) == 0
