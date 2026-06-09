"""
SVG snapshot regression test — captures the whole rendered screen to a
committed baseline and auto-diffs it on every run, so UI changes are caught
without anyone eyeballing individual widgets.

First run (creates the baseline)::

    uv run pytest tests/tui/test_snapshot.py --snapshot-update
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")
pytest.importorskip("pytest_textual_snapshot")

from textual.pilot import Pilot

from grasp_agents.tui.app import GraspAgentsApp
from grasp_agents.tui.demo import build_demo_events


async def _run_before(pilot: Pilot[None]) -> None:
    from textual.containers import VerticalScroll

    # pin the theme so the baseline is independent of the default and of any
    # theme a developer has persisted locally
    pilot.app.theme = "catppuccin-macchiato"
    drain = getattr(pilot.app, "wait_for_stream", None)
    if drain is not None:
        await drain()
    # Pin every pane to the top: with delay=0 the auto-scroll "at bottom" check
    # races, leaving the active pane at a non-deterministic offset (different
    # visible content → different SVG). Scrolling home makes the capture stable.
    for pane in pilot.app.query(VerticalScroll):
        pane.scroll_home(animate=False)
    await pilot.pause()


def test_snapshot(snap_compare) -> None:
    app = GraspAgentsApp(build_demo_events(delay=0.0, with_image=False))
    assert snap_compare(app, terminal_size=(120, 40), run_before=_run_before)
