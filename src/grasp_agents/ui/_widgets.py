"""
Reusable Textual widgets for the agent monitor.

The prompt box and skill palette (interactive input), a selection-aware Static
for Rich content, and the clickable inline image. Each is self-contained and
posts plain Textual messages the app handles — none reach back into the app —
so they live apart from :mod:`.app`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual import events, on
from textual.binding import Binding, BindingType
from textual.css.query import NoMatches
from textual.fuzzy import Matcher
from textual.message import Message
from textual.strip import Strip
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from rich.console import RenderableType
    from textual.selection import Selection

    from grasp_agents.skills import Skill
    from grasp_agents.types.content import InputImage


def _skill_row(name: str, description: str) -> RenderableType:
    """One palette row: the skill name on the left, its description on the right."""
    row = Table.grid(expand=True, padding=(0, 1))
    row.add_column(justify="left", no_wrap=True)
    row.add_column(justify="right", ratio=1, no_wrap=True)
    row.add_row(
        Text(f"/{name}", style="bold"),
        Text(description, style="dim", overflow="ellipsis", no_wrap=True),
    )
    return row


class SkillPalette(OptionList):
    """
    Dropdown of available skills, shown when the prompt starts with ``/``.

    Each row is the skill name (left) and its one-line description (right).
    Typing after the ``/`` fuzzy-filters by name; ↑/↓ move the highlight and
    Enter selects — all routed from the prompt, which keeps focus for typing
    (so the palette never grabs it: ``can_focus = False``).
    """

    can_focus = False

    def __init__(self, skills: list[Skill]) -> None:
        super().__init__(id="skill-palette")
        self._skills = skills
        self.display = False

    def filter_to(self, query: str) -> bool:
        """Repopulate with skills matching ``query`` by name; True if any remain."""
        self.clear_options()
        if query:
            matcher = Matcher(query)
            skills = [s for s in self._skills if matcher.match(s.name) > 0]
            skills.sort(key=lambda s: matcher.match(s.name), reverse=True)
        else:
            skills = list(self._skills)
        self.add_options(
            [Option(_skill_row(s.name, s.description), id=s.name) for s in skills]
        )
        if skills:
            self.highlighted = 0
        return bool(skills)

    def highlighted_skill(self) -> str | None:
        """Name (option id) of the highlighted skill, or ``None`` if empty."""
        if self.highlighted is None:
            return None
        return self.get_option_at_index(self.highlighted).id


class PromptArea(TextArea):
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

    # macOS-style word editing (TextArea already binds ctrl+←/→ and ctrl+w); the
    # alt variants only reach us on terminals that emit them distinctly.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("alt+left", "cursor_word_left", "Word left", show=False),
        Binding("alt+right", "cursor_word_right", "Word right", show=False),
        Binding("alt+backspace", "delete_word_left", "Delete word", show=False),
    ]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    class SkillChosen(Message):
        """Enter pressed while the skill palette is open — invoke this skill."""

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

    def on_mount(self) -> None:
        self.sync_height()

    def _skill_palette(self) -> SkillPalette | None:
        try:
            return self.screen.query_one("#skill-palette", SkillPalette)
        except NoMatches:
            return None

    @on(TextArea.Changed)
    def _resize_on_change(self) -> None:
        # every edit (type, newline, paste, delete) posts Changed; resize here so
        # all of them grow/shrink the box — overriding _on_paste/_on_key to resize
        # would double-run TextArea's own handler (it's re-dispatched via the MRO)
        self.sync_height()

    async def _on_key(self, event: events.Key) -> None:
        # When the skill palette is open, ↑/↓ move its highlight, Enter selects,
        # and Esc closes it — the prompt keeps focus so other keys still type
        # (filtering the list). Everything else falls through to normal editing.
        palette = self._skill_palette()
        if palette is not None and palette.display:
            if event.key == "down":
                event.prevent_default()
                event.stop()
                palette.action_cursor_down()
                return
            if event.key == "up":
                event.prevent_default()
                event.stop()
                palette.action_cursor_up()
                return
            if event.key == "escape":
                event.prevent_default()
                event.stop()
                palette.display = False
                return
            if event.key == "enter":
                name = palette.highlighted_skill()
                if name is not None:
                    event.prevent_default()
                    event.stop()
                    self.post_message(self.SkillChosen(name))
                    return
        if event.key in self._NEWLINE_KEYS:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self.text))
            return
        await super()._on_key(event)

    def sync_height(self) -> None:
        # sits at the bottom of the screen, so growing its height expands upward
        self.styles.height = min(max(self.document.line_count, 1), self._MAX_ROWS)


class SelectableStatic(Static):
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


class ZoomableImage(Static):
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
