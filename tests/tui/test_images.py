"""Inline image → click-to-zoom modal behavior (headless Pilot)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("textual")
pytest.importorskip("PIL")

from grasp_agents.types.events import ToolOutputItemEvent
from grasp_agents.types.items import FunctionToolOutputItem
from grasp_agents.ui.app import GraspAgentsApp, _ImageZoomScreen, _ZoomableImage


async def _one_image_event(png_path: str):
    yield ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(
            call_id="1", output=json.dumps({"image_path": png_path})
        ),
        source="make_chart",
        destination="agent",
    )


@pytest.mark.asyncio
async def test_inline_image_zoomable_opens_and_closes_modal(tmp_path) -> None:
    from PIL import Image

    png = tmp_path / "chart.png"
    Image.new("RGB", (16, 10), "#3366cc").save(png)

    app = GraspAgentsApp(_one_image_event(str(png)))
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()

        imgs = list(app.query(_ZoomableImage))
        assert imgs, "no zoomable inline image mounted"

        # clicking requests a zoom (bubbles _ZoomableImage.Zoom → app handler)
        imgs[0].on_click()
        await pilot.pause()
        assert isinstance(app.screen, _ImageZoomScreen)

        # esc (or a click) closes it
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, _ImageZoomScreen)


@pytest.mark.asyncio
async def test_zoom_preserves_aspect_ratio_not_stretched(tmp_path) -> None:
    from PIL import Image

    # a very wide image: zoomed, it must fit within the box (letterboxed),
    # never stretch to fill the full 90% height
    png = tmp_path / "wide.png"
    Image.new("RGB", (320, 40), "#cc6633").save(png)

    app = GraspAgentsApp(_one_image_event(str(png)))
    async with app.run_test(size=(100, 40)) as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        next(iter(app.query(_ZoomableImage))).on_click()
        await pilot.pause()
        assert isinstance(app.screen, _ImageZoomScreen)

        box = app.screen.query_one("#zoom-box")
        img = app.screen.query_one(".zoom-img")
        # fits within the box, aspect-preserved → a wide image is letterboxed
        assert 0 < img.size.height < box.size.height, (img.size, box.size)
        assert img.size.width <= box.size.width, (img.size, box.size)
