"""
Bordered panels / markdown support character-range selection + a visible highlight.

These drive a real mouse drag (not a hand-set ``screen.selections``), because the
bug was entirely in the live path: Rich renderables carried no content offsets, so
Textual could only select the whole widget, and the highlight used the theme's
``screen--selection`` style whose foreground collapses to its background.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from rich.markdown import Markdown
from rich.panel import Panel
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.events import MouseMove

from grasp_agents.ui.app import GRASP_DARK, SelectableStatic

_PANEL_TEXT = "ALPHA BETA GAMMA DELTA"


class _SelApp(App[None]):
    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield SelectableStatic(Panel(_PANEL_TEXT, title="tool"), id="p")
            yield SelectableStatic(Markdown("hello world of markdown text"), id="m")

    def on_mount(self) -> None:
        self.register_theme(GRASP_DARK)
        self.theme = "grasp-dark"


async def _drag(pilot, widget, y: int, x0: int, x1: int) -> None:
    await pilot.mouse_down(widget, offset=(x0, y))
    await pilot._post_mouse_events([MouseMove], widget, offset=(x1, y), button=1)
    await pilot.mouse_up(widget, offset=(x1, y))
    await pilot.pause()


@pytest.mark.asyncio
async def test_panel_click_resolves_to_a_content_offset() -> None:
    # the offset tags are what let the compositor map a click to a character, so
    # selection is a range rather than the whole widget
    app = _SelApp()
    async with app.run_test(size=(60, 16)) as pilot:
        await pilot.pause()
        p = app.query_one("#p", SelectableStatic)
        widget, offset = app.screen.get_widget_and_offset_at(5, 1)
        assert widget is p, widget
        assert offset is not None, offset


@pytest.mark.asyncio
async def test_panel_partial_selection_extracts_substring() -> None:
    app = _SelApp()
    async with app.run_test(size=(60, 16)) as pilot:
        await pilot.pause()
        p = app.query_one("#p", SelectableStatic)
        await _drag(pilot, p, y=1, x0=4, x1=9)
        text = (app.screen.get_selected_text() or "").strip()
        # a few characters of the panel body — NOT the whole panel, NOT empty
        assert text, "nothing was selected"
        assert text in _PANEL_TEXT, repr(text)
        assert text != _PANEL_TEXT, "selected the whole panel instead of a range"


@pytest.mark.asyncio
async def test_selection_highlight_visible_and_text_readable() -> None:
    app = _SelApp()
    async with app.run_test(size=(60, 16)) as pilot:
        await pilot.pause()
        p = app.query_one("#p", SelectableStatic)
        await _drag(pilot, p, y=1, x0=4, x1=12)
        sel_color = app.screen.get_component_rich_style("screen--selection").bgcolor
        row = list(p.render_line(1))
        highlighted = [s for s in row if s.style and s.style.bgcolor == sel_color]
        assert highlighted, "selection highlight background was not painted"
        # readable: highlighted text keeps a foreground distinct from the bg
        assert all(
            s.style is not None
            and s.style.color is not None
            and s.style.color != sel_color
            for s in highlighted
        )
        # partial: part of the row is not highlighted
        assert any(not (s.style and s.style.bgcolor == sel_color) for s in row), (
            "the entire row was highlighted (selection was not partial)"
        )


@pytest.mark.asyncio
async def test_markdown_is_selectable() -> None:
    app = _SelApp()
    async with app.run_test(size=(60, 16)) as pilot:
        await pilot.pause()
        m = app.query_one("#m", SelectableStatic)
        await _drag(pilot, m, y=0, x0=0, x1=10)
        assert (app.screen.get_selected_text() or "").strip()
