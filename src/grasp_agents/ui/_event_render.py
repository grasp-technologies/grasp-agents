"""
Event → Rich renderable builders, shared by any UI surface.

Textual-free on purpose: both the light ``console.EventConsole`` and the
``tui`` Textual app can build the same panels/tables from these helpers
without pulling Textual into the dependency graph.
"""

from __future__ import annotations

import base64
import io
import json
import pathlib
import re
from typing import TYPE_CHECKING, Any, ClassVar, cast

from rich.box import HORIZONTALS, ROUNDED
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.markdown import CodeBlock, Markdown, MarkdownElement
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from ..types.content import InputImage, InputText
from ..types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskLaunchedEvent,
    Event,
    GenerationEndEvent,
    LLMStreamingErrorEvent,
    OutputMessageItemEvent,
    ProcStreamingErrorEvent,
    ReasoningItemEvent,
    SystemMessageEvent,
    ToolCallItemEvent,
    ToolErrorEvent,
    ToolOutputItemEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessageEvent,
)

if TYPE_CHECKING:
    from ..types.items import InputMessageItem

PALETTE: dict[str, str] = {
    "border_tool_call": "#BEE4F7",
    "border_system": "#F7E1BE",
    "border_thinking": "#E1F8FA",
    "border_input": "#F7EEBE",
    "border_tool_result": "#EBFAAA",
    "separator": "#454141",
    "accent": "#AAACFA",
    "muted": "#696464",
    "tool_call": "#FFFFFF",
    "tool_result": "#969595",
    "thinking": "#969595",
    "success": "#3BBF69",
    "error": "#FCA9A9",
    "warn": "#BFB53B",
    "usage": "#9AA0AD",
    "arg_key": "#D0D0D8",
}

_TRUNC = 4000
_MAX_LINES = 60
_IMG_CELLS = (100, 50)  # max (cols, rows) for the rich-pixels half-block fallback
_IMG_COLS = 60  # chafa symbol-art target width (cells)
_IMG_ROWS_MAX = 40  # chafa symbol-art height cap (cells)


def render_event(
    event: Event[Any], *, inline_images: bool = True
) -> RenderableType | None:
    """
    Build a Rich renderable for an event, or ``None`` if it has no display.

    With ``inline_images=False`` the result omits images; a surface that renders
    images as its own widgets (the TUI) pairs this with :func:`event_images`.
    The default inlines them as symbol-art cells (the light console).
    """
    if isinstance(event, TurnStartEvent):
        agent = event.source or "agent"
        return render_turn_rule(agent, event.data.turn + 1)

    if isinstance(event, TurnEndEvent):
        reason = event.data.stop_reason
        val = str(getattr(reason, "value", reason)) if reason is not None else ""
        if val and val != "final_answer":
            return Text(f"stopped: {val}", style=f"italic {PALETTE['muted']}")
        return None

    if isinstance(event, OutputMessageItemEvent):
        text = event.data.text
        if not text:
            return None
        return _Markdown(text, code_theme=_code_theme, inline_code_theme=_code_theme)

    if isinstance(event, ReasoningItemEvent):
        parts = [
            p.text for p in (event.data.summary_parts or []) if getattr(p, "text", "")
        ]
        body = "\n".join(parts) or "thinking…"
        return panel(
            "thinking",
            Text(escape(body), style=PALETTE["thinking"]),
            PALETTE["border_thinking"],
        )

    if isinstance(event, ToolCallItemEvent):
        agent = event.source or "agent"
        tool = event.data.name or "tool"
        return panel(
            f"{escape(agent)} → {escape(tool)}",
            _build_args_renderable(event.data.arguments),
            PALETTE["border_tool_call"],
        )

    if isinstance(event, ToolOutputItemEvent):
        item = event.data
        agent = event.destination or "agent"
        tool = event.source or "tool"
        title = f"{escape(agent)} ← {escape(tool)}"
        text_out: Any = item.output if isinstance(item.output, str) else item.text
        images = item.images
        if not images or not inline_images:
            return panel(
                title,
                _build_result_renderable(
                    text_out, PALETTE["tool_result"], inline_images=inline_images
                ),
                PALETTE["border_tool_result"],
            )
        blocks: list[RenderableType] = []
        if isinstance(text_out, str) and text_out.strip():
            blocks.append(_build_result_renderable(text_out, PALETTE["tool_result"]))
        for img in images:
            if blocks:
                blocks.append(Text(""))
            blocks.append(render_input_image(img))
        return panel(title, Group(*blocks), PALETTE["border_tool_result"])

    if isinstance(event, ToolErrorEvent):
        info = event.data
        msg = f"✗ {info.error}" + (" (timed out)" if info.timed_out else "")
        return panel(
            info.tool_name,
            Text(msg, style=PALETTE["tool_result"]),
            PALETTE["error"],
        )

    if isinstance(event, (UserMessageEvent, SystemMessageEvent)):
        text = extract_input_text(event.data)
        if isinstance(event, SystemMessageEvent):
            label = f"System → {event.source or 'agent'}"
            border = PALETTE["border_system"]
        else:
            label = f"{event.source or 'User'} → {event.destination or 'agent'}"
            border = PALETTE["border_input"]
        return panel(label, Text(escape(truncate_lines(text, _MAX_LINES))), border)

    if isinstance(event, GenerationEndEvent):
        return _usage_line(event)

    if isinstance(event, BackgroundTaskLaunchedEvent):
        i = event.data
        return Text(
            f"⧗ {i.tool_name} launched in background (id: {i.task_id})",
            style=PALETTE["warn"],
        )

    if isinstance(event, BackgroundTaskCompletedEvent):
        i = event.data
        return Text(
            f"✓ {i.tool_name} completed (id: {i.task_id})",
            style=PALETTE["tool_result"],
        )

    if isinstance(event, (LLMStreamingErrorEvent, ProcStreamingErrorEvent)):
        return Text(f"Error: {event.data.error}", style=f"bold {PALETTE['error']}")

    return None


def render_turn_rule(agent: str, turn: int) -> RenderableType:
    """
    A turn-separator rule. ``turn`` is the displayed (1-based) number — a UI
    may pass its own monotonic per-agent count instead of the per-step value
    (which resets to 0 each run/step).
    """
    return Rule(f"[bold]{escape(agent)}[/] · turn {turn}", style=PALETTE["separator"])


def render_tool_stream(agent: str, tool: str, text: str) -> RenderableType:
    """Panel for in-progress (streaming) tool output, before the final result."""
    body = truncate_lines(truncate(text, _TRUNC), _MAX_LINES)
    return panel(
        f"{escape(agent)} ← {escape(tool)}",
        Text(body or "…", style=PALETTE["tool_result"]),
        PALETTE["border_tool_result"],
    )


def render_image(path: str) -> RenderableType:
    """Inline renderable for an image file (chafa symbol-art → rich-pixels → label)."""
    return _image_renderable(path)


def render_input_image(image: InputImage) -> RenderableType:
    """
    Symbol-art renderable for an ``InputImage`` (base64 data URI or file path).

    Tools that return inline images (e.g. a sandbox running matplotlib) surface
    them as ``InputImage`` parts; this decodes and renders them.
    """
    url = image.image_url or ""
    if url.startswith("data:"):
        try:
            payload = base64.b64decode(url.split(",", 1)[1])
        except Exception:
            return Text("[image: unreadable data uri]", style=PALETTE["muted"])
        return _image_renderable(payload)
    if url and not image.is_url and pathlib.Path(url).exists():
        return _image_renderable(url)
    return Text(f"[image: {url or 'attached'}]", style=PALETTE["muted"])


if TYPE_CHECKING:

    def _image_renderable(image: Any) -> RenderableType: ...

else:

    def _chafa_ansi(pil, max_cols, max_rows):
        # PIL RGB image → chafa terminal symbol-art (an ANSI string), or None if
        # chafa isn't installed / fails. chafa packs each cell with the best-fit
        # Unicode glyph + truecolor — far sharper than plain half-blocks, yet
        # still ordinary colored text, so it lives in the cell grid and scrolls
        # without the blanking that graphics protocols hit in a scroll pane.
        try:
            from chafa import Canvas, CanvasConfig, PixelType  # noqa: PLC0415
        except Exception:
            return None
        try:
            width, height = pil.size
            if not (width and height):
                return None
            cols = max_cols
            rows = max(1, round(cols * height / width / 2))  # cells ~twice as tall
            if rows > max_rows:
                rows = max_rows
                cols = max(1, round(rows * 2 * width / height))
            config = CanvasConfig()
            config.width = cols
            config.height = rows
            try:
                from chafa import CanvasMode  # noqa: PLC0415

                config.canvas_mode = CanvasMode.CHAFA_CANVAS_MODE_TRUECOLOR
            except Exception:
                pass
            canvas = Canvas(config)
            canvas.draw_all_pixels(
                PixelType.CHAFA_PIXEL_RGB8, pil.tobytes(), width, height, width * 3
            )
            return canvas.print().decode()
        except Exception:
            return None

    def _image_renderable(image):
        # image: a path (str), raw bytes, or a PIL image. Rendered as chafa
        # symbol-art (sharp, scroll-safe colored glyphs), falling back to
        # rich-pixels half-blocks, then a text label. Untyped optional libs stay
        # out of type-checking here.
        try:
            from PIL import Image as PILImage  # noqa: PLC0415
        except ImportError:
            return Text("[image]", style=PALETTE["muted"])
        try:
            if hasattr(image, "thumbnail"):
                pil = image
            elif isinstance(image, bytes):
                pil = PILImage.open(io.BytesIO(image))
            else:
                pil = PILImage.open(image)
            pil = pil.convert("RGB")
        except Exception:
            return Text("[image]", style=PALETTE["muted"])
        ansi = _chafa_ansi(pil, _IMG_COLS, _IMG_ROWS_MAX)
        if ansi is not None:
            return Text.from_ansi(ansi)
        try:
            from rich_pixels import Pixels  # noqa: PLC0415

            thumb = pil.copy()
            thumb.thumbnail(_IMG_CELLS)
            return Pixels.from_image(thumb)
        except Exception:
            return Text("[image]", style=PALETTE["muted"])


def image_path_of(event: Event[Any]) -> str | None:
    """Local image path from a tool result's ``image_path`` key, if present."""
    if not isinstance(event, ToolOutputItemEvent):
        return None
    data = _unwrap_json(event.data.output)
    if isinstance(data, dict):
        path = cast("dict[str, Any]", data).get("image_path")
        if isinstance(path, str) and pathlib.Path(path).exists():
            return path
    return None


def event_images(event: Event[Any]) -> list[InputImage | str]:
    """
    Image sources attached to an event, for a surface that renders images as its
    own widgets (the TUI) instead of inline half-blocks. Returns ``InputImage``
    parts (e.g. a sandbox chart from ``plt.show()``) and/or a local image path
    from a tool result's ``image_path`` key.
    """
    if not isinstance(event, ToolOutputItemEvent):
        return []
    out: list[InputImage | str] = list(event.data.images or [])
    path = image_path_of(event)
    if path is not None:
        out.append(path)
    return out


if TYPE_CHECKING:

    def image_to_pil(src: InputImage | str) -> Any: ...

else:

    def image_to_pil(src):
        # Decode an image source to a PIL image for high-fidelity widget
        # rendering (textual-image's Image widget), or None if it can't be read.
        # PIL is an untyped optional dep, so this lives behind a runtime branch.
        try:
            from PIL import Image as PILImage  # noqa: PLC0415
        except ImportError:
            return None
        try:
            if isinstance(src, str):
                return PILImage.open(src)
            url = src.image_url or ""
            if url.startswith("data:"):
                payload = base64.b64decode(url.split(",", 1)[1])
                return PILImage.open(io.BytesIO(payload))
            if url and not src.is_url and pathlib.Path(url).exists():
                return PILImage.open(url)
        except Exception:
            return None
        return None


# ── builders (ported from EventConsole; console.py keeps its own copy) ──


def panel(title: str, body: RenderableType, border: str) -> Panel:
    return Panel(
        body,
        title=f"[bold {border}]{title}[/]",
        title_align="left",
        border_style=border,
        box=ROUNDED,
        padding=(1, 2),
        expand=False,
    )


def _usage_line(event: GenerationEndEvent) -> RenderableType | None:
    usage = event.data.usage_with_cost
    if not usage:
        return None
    parts: list[str] = []
    if usage.input_tokens:
        parts.append(f"{usage.input_tokens:,} in")
        cached = usage.input_tokens_details.cached_tokens
        if cached:
            parts.append(f"{cached:,} cached")
    if usage.output_tokens:
        parts.append(f"{usage.output_tokens:,} out")
    if usage.cost:
        parts.append(f"${usage.cost:.4f}")
    model = event.data.model or ""
    line = f"{model} · {' · '.join(parts)}" if parts else model
    return Text(line, style=f"italic {PALETTE['usage']}")


# Code/markdown highlighting follows the active app theme so they stay
# consistent: the TUI calls set_markup_theme() with the current theme's colours;
# the plain console keeps Rich's defaults. Pygments is untyped, so its probe
# lives behind a runtime branch (kept out of type-checking).
_PYGMENTS_ALIASES = {"gruvbox": "gruvbox-dark"}

if TYPE_CHECKING:

    def _pygments_has(name: str) -> bool: ...

else:

    def _pygments_has(name):
        try:
            from pygments.styles import get_style_by_name  # noqa: PLC0415

            get_style_by_name(name)
        except Exception:
            return False
        return True


def _pygments_style_for(name: str) -> str:
    """A Pygments style matching the app theme name, or a sensible default."""
    for candidate in (name, _PYGMENTS_ALIASES.get(name)):
        if candidate and _pygments_has(candidate):
            return candidate
    if _pygments_has("catppuccin-macchiato"):
        return "catppuccin-macchiato"
    return "monokai"


_code_theme: str = _pygments_style_for("catppuccin-macchiato")
_md_styles: dict[str, str] = {}
# code background: "default" composites to near-black in Textual, so the TUI
# passes its theme background here to make code blocks blend into the pane
_code_bg: str = "default"


def set_markup_theme(
    theme_name: str, markdown_styles: dict[str, str], code_bg: str = "default"
) -> None:
    """
    Make code/markdown highlighting match the active app theme.

    The TUI calls this on launch and on every theme change, so syntax colours,
    Markdown element styles (headings, bullets, links) and the code-block
    background follow the selected theme. The plain console never calls it and
    keeps Rich's defaults.
    """
    global _code_theme, _md_styles, _code_bg
    _code_theme = _pygments_style_for(theme_name)
    _md_styles = markdown_styles
    _code_bg = code_bg


class _CodeBlock(CodeBlock):
    """A Markdown code fence with a transparent (app-background) syntax box."""

    def __rich_console__(  # noqa: PLW3201
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = str(self.text).rstrip()
        yield Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            background_color=_code_bg,
            word_wrap=True,
            padding=(1, 0),
        )


class _Markdown(Markdown):
    """Markdown with transparent code fences + app-theme element styling."""

    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "code_block": _CodeBlock,
        "fence": _CodeBlock,
    }

    def __rich_console__(  # noqa: PLW3201
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if not _md_styles:
            yield from super().__rich_console__(console, options)
            return
        console.push_theme(Theme(_md_styles, inherit=True))
        try:
            yield from super().__rich_console__(console, options)
        finally:
            console.pop_theme()


# Tool-arg keys whose string values are code → render with syntax highlighting
# (the lexer Rich/Pygments uses) instead of as a flat table cell.
_CODE_KEYS: dict[str, str] = {
    "code": "python",
    "source": "python",
    "python": "python",
    "command": "bash",
    "cmd": "bash",
    "bash": "bash",
    "shell": "bash",
    "script": "bash",
    "sql": "sql",
    "patch": "diff",
    "diff": "diff",
    "html": "html",
    "css": "css",
    "javascript": "javascript",
    "js": "javascript",
}


def _code_lexer(key: str, value: str) -> str | None:
    """Lexer name for a tool-arg value that is code, or ``None`` to leave as text."""
    lexer = _CODE_KEYS.get(key.lower())
    if lexer is None:
        return None
    # not worth a code block for a short one-liner (e.g. a trivial `command`)
    if "\n" not in value and len(value) < 40:
        return None
    return lexer


def _code_block(value: str, lexer: str) -> Syntax:
    return Syntax(
        value,
        lexer,
        theme=_code_theme,
        background_color=_code_bg,
        word_wrap=True,
    )


def _build_args_renderable(args_json: str) -> RenderableType:
    try:
        parsed = json.loads(args_json)
    except (json.JSONDecodeError, TypeError):
        return Text(escape(args_json))
    if not isinstance(parsed, dict):
        return Text(escape(str(parsed)))
    return _kv_table(cast("dict[str, Any]", parsed), PALETTE["tool_call"])


def _unwrap_json(value: Any) -> Any:
    """Peel up to two JSON-string layers (tools sometimes double-encode)."""
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            break
    return value


_MD_HEADING = re.compile(r"(?m)^#{1,6} \S")
_MD_BULLET = re.compile(r"(?m)^\s*[-*] \S")
_MD_NUMBERED = re.compile(r"(?m)^\s*\d+\. \S")


def _looks_like_markdown(text: str) -> bool:
    # render tool output as markdown only on a clear structural signal — a
    # heading, a fenced code block, or a list — so plain stdout/logs and diffs
    # (which lack these, or use "-foo" with no space) stay plain text
    return (
        bool(_MD_HEADING.search(text))
        or "```" in text
        or len(_MD_BULLET.findall(text)) >= 2
        or len(_MD_NUMBERED.findall(text)) >= 2
    )


def _build_result_renderable(
    output: Any, text_color: str, *, inline_images: bool = True
) -> RenderableType:
    raw = _unwrap_json(output)
    if isinstance(raw, dict):
        data = cast("dict[str, Any]", raw)
        img_path = data.get("image_path")
        if isinstance(img_path, str) and pathlib.Path(img_path).exists():
            # render the image itself; drop the (machine-specific) path row
            shown = {k: v for k, v in data.items() if k != "image_path"}
            table = _kv_table(shown, text_color)
            if not inline_images:
                return table
            return Group(table, Text(""), render_image(img_path))
        return _kv_table(data, text_color)
    content = _unescape_json_string(str(output))
    content = truncate(content, _TRUNC)
    if _looks_like_markdown(content):
        # agent tools often return markdown reports — render them formatted
        return _Markdown(content, code_theme=_code_theme, inline_code_theme=_code_theme)
    lines = [ln for ln in content.split("\n") if ln.strip()]
    return Text(truncate_lines("\n".join(lines), _MAX_LINES), style=text_color)


def _kv_table(data: dict[str, Any], text_color: str) -> Table:
    table = Table(
        show_header=False,
        show_edge=False,
        box=HORIZONTALS,
        show_lines=True,
        border_style=PALETTE["separator"],
        pad_edge=False,
        padding=(0, 2, 0, 0),
    )
    table.add_column("key", style=f"bold {PALETTE['arg_key']}", no_wrap=True)
    table.add_column("value", style=text_color, ratio=1)
    for k, v in data.items():
        table.add_row(str(k), _value_cell(str(k), v))
    return table


def _value_cell(key: str, v: Any) -> RenderableType:
    # code-bearing values (a tool's ``code``/``command``/…) render with syntax
    # highlighting in their value cell — the row layout (and column alignment)
    # is unchanged; everything else stays plain text.
    if isinstance(v, str):
        lexer = _code_lexer(key, v)
        if lexer is not None:
            return _code_block(v, lexer)
    return Text(_format_value(v))


def _format_value(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)) or v is None:
        return str(v)
    compact = json.dumps(v, ensure_ascii=False)
    if len(compact) > 60:
        return json.dumps(v, indent=2, ensure_ascii=False)
    return compact


def extract_input_text(msg: InputMessageItem) -> str:
    parts: list[str] = []
    for part in msg.content_parts:
        if isinstance(part, InputText):
            parts.append(part.text.strip())
        elif isinstance(part, InputImage):
            parts.append(part.image_url or "[image]" if part.is_url else "[image]")
    return "\n\n".join(parts)


def truncate(text: str, limit: int) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def truncate_lines(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return text
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    kept.append(f"… {len(lines) - max_lines} more lines")
    return "\n".join(kept)


def _unescape_json_string(text: str) -> str:
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    if isinstance(parsed, str):
        return parsed
    return json.dumps(parsed, indent=2)
