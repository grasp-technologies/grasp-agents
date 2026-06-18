"""Notebook-helper tests (headless; no manual inspection)."""

from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("textual")

from rich.console import Console

from grasp_agents.types.events import ToolOutputItemEvent
from grasp_agents.types.items import FunctionToolOutputItem
from grasp_agents.ui._event_render import image_path_of
from grasp_agents.ui.demo import build_demo_events, demo_event_list
from grasp_agents.ui.notebook import render_events_inline, screenshot


@pytest.mark.asyncio
async def test_render_events_inline_passthrough_and_renders() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    seen = [
        ev
        async for ev in render_events_inline(
            build_demo_events(delay=0.0), console=console, show_images=False
        )
    ]
    assert len(seen) == len(demo_event_list())
    assert console.file.getvalue()  # something was rendered


@pytest.mark.asyncio
async def test_screenshot_returns_svg() -> None:
    svg = await screenshot(build_demo_events(delay=0.0), size=(100, 30))
    assert isinstance(svg, str)
    assert "<svg" in svg.lower()


def test_image_path_of(tmp_path) -> None:
    from PIL import Image

    png = tmp_path / "x.png"
    Image.new("RGB", (4, 4), "#ffffff").save(png)
    hit = ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(
            call_id="1", output=json.dumps({"image_path": str(png)})
        ),
        source="tool",
        destination="agent",
    )
    assert image_path_of(hit) == str(png)
    miss = ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(call_id="1", output="plain"),
        source="tool",
        destination="agent",
    )
    assert image_path_of(miss) is None
