"""
Textual UI that monitors (and optionally drives) a grasp-agents run.

One pane per agent/subagent (keyed by ``event.source``), a bottom tab bar to
switch between them (status glyph per tab, ``↳`` marks a subagent), a
follow-latest toggle, and switchable color themes (``ctrl+p`` palette).

Two modes:
- monitor: ``GraspAgentsApp(events)`` consumes a fixed event stream.
- interactive: ``GraspAgentsApp(on_submit=...)`` shows an input box; each
  submitted message runs the agent and streams its events into the panes.

Run the demo::

    python -m grasp_agents.ui.demo --tui
"""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from rich.align import Align
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.theme import Theme
from textual.widgets import (
    ContentSwitcher,
    Footer,
    Header,
    Static,
    Tab,
    Tabs,
    TextArea,
)
from textual.worker import Worker

from ..types.content import InputImage
from ..types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskLaunchedEvent,
    Event,
    GenerationEndEvent,
    LLMStreamEvent,
    LLMStreamingErrorEvent,
    OutputMessageItemEvent,
    ProcStreamingErrorEvent,
    ReasoningItemEvent,
    SystemMessageEvent,
    ToolCallItemEvent,
    ToolErrorEvent,
    ToolOutputItemEvent,
    ToolStreamEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessageEvent,
)
from ..types.llm_events import OutputMessageTextPartTextDelta
from ._event_render import (
    event_images,
    image_to_pil,
    render_event,
    render_image,
    render_input_image,
    render_tool_stream,
    render_turn_rule,
    set_markup_theme,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from rich.console import RenderableType
    from textual.selection import Selection
    from textual.widget import Widget

# Events only an LLM-backed agent emits. A source that emits one of these owns
# an agent pane; tools (RunPython, Bash, …) emit only tool/exec output, so they
# never get a pane of their own — their output renders in the calling agent's.
_AGENT_EVENTS: tuple[type[Event[Any]], ...] = (
    SystemMessageEvent,
    TurnStartEvent,
    TurnEndEvent,
    GenerationEndEvent,
    OutputMessageItemEvent,
    ReasoningItemEvent,
    LLMStreamEvent,
    ToolCallItemEvent,
)

_GLYPH: dict[str, str] = {
    "working": "●",
    "done": "✓",
    "error": "✗",
    "idle": "○",
}

GRASP_DARK = Theme(
    name="grasp-dark",
    primary="#AAACFA",
    secondary="#BEE4F7",
    accent="#BFB53B",
    success="#3BBF69",
    warning="#BFB53B",
    error="#FCA9A9",
    background="#0F0F0F",
    surface="#1A1A1A",
    panel="#222222",
    dark=True,
)
GRASP_LIGHT = Theme(
    name="grasp-light",
    primary="#5B5FD6",
    secondary="#2F7FB0",
    accent="#8A7F10",
    success="#1F8F4D",
    warning="#8A7F10",
    error="#C0392B",
    background="#FAFAFA",
    surface="#EEEEEE",
    panel="#E2E2E2",
    dark=False,
)

_DEFAULT_THEME = "catppuccin-macchiato"


def _theme_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "grasp-agents" / "tui.json"


def _load_saved_theme() -> str | None:
    """The theme the user last selected, persisted across launches."""
    try:
        data: Any = json.loads(_theme_config_path().read_text())
        name = data.get("theme")
    except Exception:
        return None
    return name if isinstance(name, str) else None


def _save_theme(name: str) -> None:
    path = _theme_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"theme": name}))
    except Exception:
        pass


def _prime_cell_size() -> None:
    # textual-image's halfcell/sixel/unicode renderers call get_cell_size() at
    # RENDER time; its `\x1b[16t` probe + stdin read then fights Textual for the
    # terminal once the app owns the screen → blank/garbled images (and stolen
    # keystrokes). The result is cached after the first call, so prime it now,
    # while the terminal is still in normal mode — the render path then returns
    # the cached value without ever probing. Best-effort.
    try:
        from textual_image._terminal import get_cell_size  # noqa: PLC0415, PLC2701

        get_cell_size()
    except Exception:
        pass


def _prime_image_protocol() -> None:
    # The zoom modal renders full-res via textual-image's terminal-graphics
    # widget; its protocol + cell-size detection probes the terminal over stdin,
    # which fights Textual once the app owns the screen. Run it now, while the
    # terminal is still in normal mode — the results are cached for the modal.
    try:
        importlib.import_module("textual_image.widget")
    except Exception:
        pass
    _prime_cell_size()


if TYPE_CHECKING:

    def _zoom_widget(src: InputImage | str) -> Widget: ...

else:

    def _zoom_widget(src):
        # Full-resolution zoom view: textual-image's terminal-graphics Image
        # widget, auto-detecting the terminal's protocol (TGP on Kitty/Ghostty,
        # Sixel/unicode elsewhere). It lives on a dedicated, non-scrolling modal
        # screen — the one place graphics protocols are reliable, since nothing
        # scrolls the cells out from under them. Falls back to the inline
        # symbol-art renderable when the dep or terminal can't do graphics.
        # Untyped dep → behind a runtime branch.
        pil = image_to_pil(src)
        if pil is not None:
            try:
                from textual_image.widget import Image as TImage  # noqa: PLC0415

                return TImage(pil, classes="zoom-img")
            except Exception:
                pass
        rend = (
            render_input_image(src)
            if isinstance(src, InputImage)
            else render_image(src)
        )
        return Static(rend, classes="zoom-img")


def _slug(source: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "-", source).strip("-").lower() or "x"


def _pane_id(source: str) -> str:
    return "pane-" + _slug(source)


def _tab_id(source: str) -> str:
    return "tab-" + _slug(source)


class _PromptArea(TextArea):
    """
    Multi-line prompt: Enter submits; Shift+Enter or Ctrl+J insert a newline; the
    box grows upward with its content (up to ``_MAX_ROWS``).

    Telling Shift+Enter apart from Enter needs a terminal that speaks the Kitty
    keyboard protocol (Ghostty, Kitty, WezTerm, iTerm2 ≥ 3.5); elsewhere it
    arrives as a plain Enter (which submits), so Ctrl+J is the portable newline.
    """

    _MAX_ROWS = 10
    # Keys that insert a newline instead of submitting. "shift+enter" = kitty
    # keyboard protocol; "shift+\r"/"shift+\n" = legacy CSI-u; "ctrl+j" = LF,
    # which works on every terminal. A terminal with no keyboard protocol sends a
    # plain Enter for shift+enter — byte-identical to Enter, impossible to tell
    # apart — so Ctrl+J is the portable newline there.
    _NEWLINE_KEYS = frozenset({"shift+enter", "shift+\r", "shift+\n", "ctrl+j"})

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def on_mount(self) -> None:
        self.sync_height()

    async def _on_key(self, event: events.Key) -> None:
        if event.key in self._NEWLINE_KEYS:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            self.sync_height()
            return
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self.text))
            return
        await super()._on_key(event)
        self.sync_height()

    def sync_height(self) -> None:
        # sits at the bottom of the screen, so growing its height expands upward
        self.styles.height = min(max(self.document.line_count, 1), self._MAX_ROWS)


class _SelectableStatic(Static):
    """
    A Static whose Rich content — Panels, Tables, Markdown — supports
    character-range selection and shows the selection highlight.

    Rich renderables go through RichVisual, which neither tags content offsets
    (so Textual's compositor can't map a click to a character — it would only
    ever select the whole widget) nor paints the selection. We tag each segment
    with its ``(char-x, y)`` content offset to enable range selection, then
    paint a background-only highlight over the selected span — preserving each
    segment's own foreground so the text stays readable (the theme's
    ``screen--selection`` foreground is ``transparent``, which collapses to the
    background if applied as text colour).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # cache of offset-tagged strips keyed by line; invalidated when the base
        # strip object changes (Textual re-renders), so a selection drag — which
        # leaves content unchanged — reuses them instead of re-tagging per frame
        self._offset_cache: dict[int, tuple[Strip, Strip]] = {}

    def render_line(self, y: int) -> Strip:
        base = super().render_line(y)
        cached = self._offset_cache.get(y)
        if cached is not None and cached[0] is base:
            strip = cached[1]
        else:
            strip = self._tag_offsets(base, y)
            self._offset_cache[y] = (base, strip)
        selection = self.text_selection
        if selection is None:
            return strip
        span = selection.get_span(y)
        if span is None:
            return strip
        start, end = span
        if end == -1:
            end = strip.cell_length
        highlight = Style(
            bgcolor=self.screen.get_component_rich_style("screen--selection").bgcolor
        )
        segments = [
            *strip.crop(0, start),
            # post_style (3rd arg) wins over each segment's own style, so the
            # highlight background paints over the panel's background
            *Segment.apply_style(list(strip.crop(start, end)), None, highlight),
            *strip.crop(end, strip.cell_length),
        ]
        return Strip(segments, strip.cell_length)

    def _tag_offsets(self, strip: Strip, y: int) -> Strip:
        segments: list[Segment] = []
        x = 0
        for segment in strip:
            offset = Style.from_meta({"offset": (x, y)})
            style = segment.style + offset if segment.style is not None else offset
            segments.append(Segment(segment.text, style, segment.control))
            x += len(segment.text)
        return Strip(segments, strip.cell_length)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        if not self.size.height:
            return None
        # render_line keeps the text intact (highlight/offsets are style-only)
        text = "\n".join(self.render_line(y).text for y in range(self.size.height))
        return selection.extract(text), "\n"


class _ZoomableImage(Static):
    """Inline symbol-art image; clicking it requests a full-resolution zoom."""

    class Zoom(Message):
        def __init__(self, src: InputImage | str) -> None:
            self.src = src
            super().__init__()

    def __init__(self, src: InputImage | str, renderable: RenderableType) -> None:
        super().__init__(renderable, classes="ga-img")
        self._src = src
        self.tooltip = "Click to zoom"

    def on_click(self) -> None:
        self.post_message(self.Zoom(self._src))


class _ImageZoomScreen(ModalScreen[None]):
    """
    Full-screen overlay showing one image at full terminal-graphics fidelity.

    A dedicated, non-scrolling screen is the one place a graphics protocol (TGP
    on Kitty/Ghostty) renders reliably — nothing scrolls the cells out from
    under it. Dismissed with ``esc``/``q`` or a click anywhere.
    """

    # The image sizes itself (``width/height: auto``) inside a 90% box, so
    # textual-image fits it to the box preserving aspect ratio. Pinning the
    # widget to a fixed box instead would stretch the image to fill it.
    CSS = """
    _ImageZoomScreen { align: center middle; background: $background 85%; }
    _ImageZoomScreen #zoom-box { width: 90%; height: 90%; align: center middle; }
    _ImageZoomScreen .zoom-img { width: auto; height: auto; }
    _ImageZoomScreen #zoom-hint {
        dock: bottom; width: 1fr; text-align: center; color: $text-muted;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "dismiss_zoom", "Close"),
        ("q", "dismiss_zoom", "Close"),
    ]

    def __init__(self, src: InputImage | str) -> None:
        super().__init__()
        self._src = src

    def compose(self) -> ComposeResult:
        with Container(id="zoom-box"):
            yield _zoom_widget(self._src)
        yield Static("esc / click to close", id="zoom-hint")

    def on_click(self) -> None:
        self.dismiss()

    def action_dismiss_zoom(self) -> None:
        self.dismiss()


class GraspAgentsApp(App[None]):
    """Multi-subagent monitor: a tab bar of agents + one transcript pane each."""

    TITLE = "grasp-agents · monitor"

    CSS = """
    #panes { height: 1fr; padding: 0 3; }
    VerticalScroll { height: 1fr; background: $background; }
    Static.ga-msg { margin-top: 1; width: 1fr; }
    Static.ga-turn { margin-top: 1; width: 1fr; }
    Static.ga-usage { width: 1fr; }
    Static.ga-usage-spaced { margin-top: 1; width: 1fr; }
    Static.ga-after-usage { margin-top: 1; width: 1fr; }
    .ga-img { margin-top: 1; width: 1fr; height: auto; }
    #prompt-bar { height: auto; max-height: 12; margin-top: 1; border: round #767C8C; }
    #prompt-arrow { width: 2; height: auto; color: $accent; }
    #prompt {
        width: 1fr; height: 1; max-height: 10; border: none; background: transparent;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Quit"),
        ("f", "toggle_follow", "Follow latest"),
    ]

    def __init__(
        self,
        events: AsyncIterator[Event[Any]] | None = None,
        *,
        on_submit: Callable[[str], AsyncIterator[Event[Any]]] | None = None,
        main_agent: str | None = None,
    ) -> None:
        super().__init__()
        _prime_image_protocol()
        self._ga_events = events
        self._ga_on_submit = on_submit
        self._ga_main = main_agent
        self._ga_last_agent: str | None = None
        self._ga_agents: set[str] = set()
        self._ga_turns: dict[str, int] = {}
        self._ga_last_kind: dict[str, str] = {}
        self._ga_panes: dict[str, VerticalScroll] = {}
        self._ga_tab_source: dict[str, str] = {}
        self._ga_status: dict[str, str] = {}
        self._ga_parent: dict[str, str] = {}
        self._ga_follow = False
        self._ga_worker: Worker[None] | None = None
        # live-streaming widgets per owner (LLM tokens / tool output), finalised
        # by the matching item event; empty when the agent isn't streaming
        self._ga_stream_msg: dict[str, _SelectableStatic] = {}
        self._ga_stream_msg_text: dict[str, str] = {}
        self._ga_stream_tool: dict[str, _SelectableStatic] = {}
        self._ga_stream_tool_text: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield ContentSwitcher(id="panes")
        if self._ga_on_submit is not None:
            with Horizontal(id="prompt-bar"):
                yield Static("❯", id="prompt-arrow")  # noqa: RUF001
                yield _PromptArea(id="prompt")
        yield Tabs(id="agents")
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(GRASP_DARK)
        self.register_theme(GRASP_LIGHT)
        # restore the last-used theme (persisted across launches) instead of
        # resetting every time; default to Catppuccin Macchiato
        saved = _load_saved_theme()
        self.theme = (
            saved if saved and self.get_theme(saved) is not None else _DEFAULT_THEME
        )
        self._apply_markup_theme()
        self.theme_changed_signal.subscribe(self, self._on_theme_changed)
        if self._ga_events is not None:
            self._ga_worker = self._consume()
        if self._ga_on_submit is not None:
            self.query_one("#prompt", _PromptArea).focus()

    def _on_theme_changed(self, theme: Theme) -> None:
        _save_theme(theme.name)  # persist the user's choice across launches
        self._apply_markup_theme()

    def _apply_markup_theme(self) -> None:
        # keep code/markdown highlighting consistent with the active app theme:
        # syntax colours follow the theme name, Markdown element styles its colours
        theme = self.get_theme(self.theme)
        if theme is None:
            return
        accent = theme.accent or theme.primary
        secondary = theme.secondary or theme.primary
        set_markup_theme(
            self.theme,
            {
                "markdown.h1": f"bold {accent}",
                "markdown.h1.border": accent,
                "markdown.h2": f"bold {theme.primary}",
                "markdown.h3": f"bold {theme.primary}",
                "markdown.h4": f"bold {secondary}",
                "markdown.h5": f"bold {secondary}",
                "markdown.h6": f"bold {secondary}",
                "markdown.item.bullet": f"bold {accent}",
                "markdown.item.number": f"bold {accent}",
                "markdown.link": f"underline {accent}",
                "markdown.link_url": secondary,
                "markdown.block_quote": secondary,
                "markdown.code": theme.primary,
            },
            theme.background or "default",
        )

    # ── event consumption ──

    @work(exclusive=False)
    async def _consume(self) -> None:
        if self._ga_events is None:
            return
        async for event in self._ga_events:
            await self._feed(event)

    async def wait_for_stream(self) -> None:
        """Await full consumption of the event stream (tests / screenshots)."""
        if self._ga_worker is not None:
            await self._ga_worker.wait()

    # ── interactive input ──

    @on(_PromptArea.Submitted)
    def _on_prompt_submit(self, event: _PromptArea.Submitted) -> None:
        text = event.value.strip()
        if not text or self._ga_on_submit is None:
            return
        prompt = self.query_one("#prompt", _PromptArea)
        prompt.text = ""
        prompt.sync_height()
        prompt.disabled = True
        self._run_turn(text)

    @work
    async def _run_turn(self, text: str) -> None:
        if self._ga_on_submit is None:
            return
        try:
            async for event in self._ga_on_submit(text):
                await self._feed(event)
        finally:
            prompt = self.query_one("#prompt", _PromptArea)
            prompt.disabled = False
            prompt.focus()

    async def _feed(self, event: Event[Any]) -> None:
        self._record_edge(event)
        owner = self._owner(event)
        pane = await self._ensure(owner)
        # auto-scroll only when already at the bottom, so streaming content never
        # yanks the user down while they read scrolled-up history
        at_bottom = pane.scroll_offset.y >= pane.max_scroll_y - 1

        # live token / tool-output streaming when the agent streams; if it isn't
        # (or once an item completes) fall through to render the finished item
        if not await self._feed_streaming(event, owner, pane, at_bottom):
            await self._feed_item(event, owner, pane, at_bottom)

        self._update_status(event, owner)
        if self._ga_follow:
            self._activate(owner)

    async def _feed_streaming(
        self, event: Event[Any], owner: str, pane: VerticalScroll, at_bottom: bool
    ) -> bool:
        """
        Render live streaming deltas / finalise a streamed widget.

        Returns ``True`` when the event was a stream delta or completed a widget
        that was being streamed — so the normal item render is skipped.
        """
        if isinstance(event, LLMStreamEvent):
            data = event.data
            if isinstance(data, OutputMessageTextPartTextDelta) and data.delta:
                await self._stream_message(owner, pane, data.delta, at_bottom)
            return True
        if isinstance(event, ToolStreamEvent):
            await self._stream_tool(
                owner, pane, event.source or "tool", str(event.data), at_bottom
            )
            return True
        if isinstance(event, OutputMessageItemEvent) and owner in self._ga_stream_msg:
            self._finalize_message(owner, event, pane, at_bottom)
            return True
        if isinstance(event, ToolOutputItemEvent) and owner in self._ga_stream_tool:
            await self._finalize_tool(owner, event, pane, at_bottom)
            return True
        return False

    async def _stream_message(
        self, owner: str, pane: VerticalScroll, delta: str, at_bottom: bool
    ) -> None:
        # accumulate plain text live (cheap); markdown is rendered once on finalise
        text = self._ga_stream_msg_text.get(owner, "") + delta
        self._ga_stream_msg_text[owner] = text
        widget = self._ga_stream_msg.get(owner)
        if widget is None:
            widget = _SelectableStatic(Text(text), classes="ga-msg")
            self._ga_stream_msg[owner] = widget
            self._ga_last_kind[owner] = "text"
            await pane.mount(widget)
        else:
            widget.update(Text(text))
        if at_bottom:
            pane.scroll_end(animate=False)

    async def _stream_tool(
        self, owner: str, pane: VerticalScroll, tool: str, delta: str, at_bottom: bool
    ) -> None:
        text = self._ga_stream_tool_text.get(owner, "") + delta
        self._ga_stream_tool_text[owner] = text
        rend = Align.right(render_tool_stream(owner, tool, text))
        widget = self._ga_stream_tool.get(owner)
        if widget is None:
            widget = _SelectableStatic(rend, classes="ga-msg")
            self._ga_stream_tool[owner] = widget
            self._ga_last_kind[owner] = "box"
            await pane.mount(widget)
        else:
            widget.update(rend)
        if at_bottom:
            pane.scroll_end(animate=False)

    def _finalize_message(
        self, owner: str, event: Event[Any], pane: VerticalScroll, at_bottom: bool
    ) -> None:
        widget = self._ga_stream_msg.pop(owner)
        self._ga_stream_msg_text.pop(owner, None)
        final = render_event(event, inline_images=False)
        if final is not None:  # swap streamed plain text for the rendered markdown
            widget.update(final)
        if at_bottom:
            pane.scroll_end(animate=False)

    async def _finalize_tool(
        self, owner: str, event: Event[Any], pane: VerticalScroll, at_bottom: bool
    ) -> None:
        widget = self._ga_stream_tool.pop(owner)
        self._ga_stream_tool_text.pop(owner, None)
        final = render_event(event, inline_images=False)
        if final is not None:
            widget.update(Align.right(final))
        for img in event_images(event):
            await pane.mount(self._image_widget(img))
        if at_bottom:
            pane.scroll_end(animate=False)

    async def _feed_item(
        self, event: Event[Any], owner: str, pane: VerticalScroll, at_bottom: bool
    ) -> None:
        if isinstance(event, TurnStartEvent):
            # monotonic per-agent turn count — the loop's own turn resets to 0
            # each step, so two consecutive turns can both read turn=0
            self._ga_turns[owner] = self._ga_turns.get(owner, 0) + 1
            text = render_turn_rule(owner, self._ga_turns[owner])
        else:
            text = render_event(event, inline_images=False)
        images = event_images(event)
        if text is not None:
            last = self._ga_last_kind.get(owner)
            if isinstance(event, OutputMessageItemEvent):
                kind = "text"  # borderless (markdown) generated message
            elif isinstance(event, GenerationEndEvent):
                kind = "usage"
            elif isinstance(event, TurnStartEvent):
                kind = "turn"
            else:
                kind = "box"  # rendered as a bordered panel
            if kind == "usage":
                # hug a boxed message (its border is the gap), but add a little
                # space below a borderless generated message
                cls = "ga-usage-spaced" if last == "text" else "ga-usage"
            elif kind == "turn":
                cls = "ga-turn"
            elif last == "usage":
                cls = "ga-after-usage"
            else:
                cls = "ga-msg"
            self._ga_last_kind[owner] = kind
            if isinstance(
                event, (UserMessageEvent, SystemMessageEvent, ToolOutputItemEvent)
            ):
                # border to the right; the text inside the panel stays left
                text = Align.right(text)
            await pane.mount(_SelectableStatic(text, classes=cls))
        for img in images:
            await pane.mount(self._image_widget(img))
        if (text is not None or images) and at_bottom:
            pane.scroll_end(animate=False)

    def _image_widget(self, src: InputImage | str) -> Widget:
        # Inline symbol-art (sharp, scroll-safe); click it to open the full-res
        # zoom modal.
        rend = (
            render_input_image(src)
            if isinstance(src, InputImage)
            else render_image(src)
        )
        # right-aligned to sit with the (right-aligned) tool output it came from
        return _ZoomableImage(src, Align.right(rend))

    def _owner(self, event: Event[Any]) -> str:
        # An agent pane belongs to whoever emits "generation" events. Only
        # LLM-backed agents do; tools (RunPython, Bash, …) emit just tool/exec
        # output. So a source seen emitting one of these IS an agent and owns a
        # pane — and a tool never does, whatever its source field says.
        if isinstance(event, _AGENT_EVENTS) and event.source:
            self._ga_agents.add(event.source)
            self._ga_last_agent = event.source
            return event.source
        # user input is addressed to an agent (its destination)
        if isinstance(event, UserMessageEvent):
            return (
                event.destination or self._ga_main or self._ga_last_agent or "session"
            )
        # tool / exec output (source = tool name) renders in the calling agent's
        # pane: its destination when that's a known agent, else the current one
        dest = getattr(event, "destination", None)
        if isinstance(dest, str) and dest in self._ga_agents:
            return dest
        return self._ga_last_agent or self._ga_main or "agent"

    def _record_edge(self, event: Event[Any]) -> None:
        child = parent = None
        if isinstance(event, ToolCallItemEvent):
            child, parent = event.data.name, event.source
        elif isinstance(event, BackgroundTaskLaunchedEvent):
            child, parent = event.data.tool_name, event.source
        if child and parent and child != parent:
            self._ga_parent.setdefault(child, parent)

    async def _ensure(self, source: str) -> VerticalScroll:
        if source in self._ga_panes:
            return self._ga_panes[source]

        pane = VerticalScroll(id=_pane_id(source))
        switcher = self.query_one("#panes", ContentSwitcher)
        # add_content mounts the pane hidden; only the first becomes current and
        # visible. (Plain mount() leaves new panes displayed, so every spawned
        # subagent would stack on top of the active pane — the "split" bug.)
        await switcher.add_content(pane, set_current=switcher.current is None)
        self._ga_panes[source] = pane

        tab_id = _tab_id(source)
        self._ga_tab_source[tab_id] = source
        self._ga_status[source] = "working"
        await self.query_one("#agents", Tabs).add_tab(
            Tab(self._tab_label(source), id=tab_id)
        )
        return pane

    def _activate(self, source: str) -> None:
        tabs = self.query_one("#agents", Tabs)
        if tabs.active != _tab_id(source):
            tabs.active = _tab_id(source)
        self.query_one("#panes", ContentSwitcher).current = _pane_id(source)

    @on(Tabs.TabActivated)
    def _on_tab_activated(self, event: Tabs.TabActivated) -> None:
        source = self._ga_tab_source.get(event.tab.id or "")
        if source is not None:
            self.query_one("#panes", ContentSwitcher).current = _pane_id(source)

    @on(_ZoomableImage.Zoom)
    def _on_image_zoom(self, event: _ZoomableImage.Zoom) -> None:
        self.push_screen(_ImageZoomScreen(event.src))

    # ── status ──

    def _tab_label(self, source: str) -> str:
        glyph = _GLYPH[self._ga_status.get(source, "idle")]
        prefix = "↳ " if source in self._ga_parent else ""
        return f"{glyph} {prefix}{source}"

    def _update_status(self, event: Event[Any], owner: str) -> None:
        if isinstance(
            event, (ToolErrorEvent, LLMStreamingErrorEvent, ProcStreamingErrorEvent)
        ):
            self._set_status(owner, "error")
        elif isinstance(event, TurnEndEvent):
            reason = event.data.stop_reason
            if str(getattr(reason, "value", reason)) == "final_answer":
                self._set_status(owner, "done")
        elif isinstance(event, BackgroundTaskCompletedEvent):
            child = event.data.tool_name
            if child in self._ga_panes:
                self._set_status(child, "done")

    def _set_status(self, source: str, status: str) -> None:
        self._ga_status[source] = status
        if source not in self._ga_panes:
            return
        self.query_one(f"#{_tab_id(source)}", Tab).label = self._tab_label(source)

    # ── actions ──

    def action_toggle_follow(self) -> None:
        self._ga_follow = not self._ga_follow
        self.notify(f"Follow latest: {'on' if self._ga_follow else 'off'}")


def run_tui(events: AsyncIterator[Event[Any]]) -> None:
    """Launch the TUI over an event stream (blocks until the user quits)."""
    GraspAgentsApp(events).run()


def run_tui_interactive(
    on_submit: Callable[[str], AsyncIterator[Event[Any]]],
    *,
    main_agent: str | None = None,
) -> None:
    """Interactive TUI: type a message, the agent runs, events stream into panes."""
    GraspAgentsApp(on_submit=on_submit, main_agent=main_agent).run()
