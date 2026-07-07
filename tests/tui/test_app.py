"""
Headless behavior tests via Textual's Pilot — no manual UI inspection.

Feeds the fixed demo event stream and asserts the tree/panes/status that the
app derives, with zero rendering checks.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from textual.widgets import ContentSwitcher

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
        assert set(app._ga_panes) == {"coordinator", "researcher", "writer"}


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
