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

from rich.box import HEAVY, HORIZONTALS, ROUNDED
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.markdown import (
    CodeBlock,
    Heading,
    Markdown,
    MarkdownContext,
    MarkdownElement,
)
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text, TextType
from rich.theme import Theme

from grasp_agents.context.projection import fold_summary_text
from grasp_agents.context.untrusted_content import unwrap_untrusted
from grasp_agents.printer import sanitize_terminal_text
from grasp_agents.skills import match_invocation_wrapper
from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskLaunchedEvent,
    CompactionEvent,
    CompactionInfo,
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
    WebSearchCallItemEvent,
)
from grasp_agents.types.items import OpenPageAction, SearchAction, WebSearchCallItem
from grasp_agents.types.llm_events import ResponseFallback, ResponseRetrying

if TYPE_CHECKING:
    from grasp_agents.types.items import InputMessageItem

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
    # background-task progress: lighter than `muted` so the live log stays
    # legible, with a softer (lighter) border than the result panels
    "bg_tool": "#8E8888",
    "border_bg_tool": "#7A7474",
}

_TRUNC = 4000
_MAX_LINES = 60
_IMG_CELLS = (100, 50)  # max (cols, rows) for the rich-pixels half-block fallback
_IMG_COLS = 60  # chafa symbol-art target width (cells)
_IMG_ROWS_MAX = 40  # chafa symbol-art height cap (cells)


def render_event(
    event: Event[Any],
    *,
    inline_images: bool = True,
    hyperlinks: bool = True,
    markdown: bool = True,
) -> RenderableType | None:
    """
    Build a Rich renderable for an event, or ``None`` if it has no display.

    With ``inline_images=False`` the result omits images; a surface that renders
    images as its own widgets (the TUI) pairs this with :func:`event_images`.
    The default inlines them as symbol-art cells (the light console).

    ``hyperlinks`` controls whether links (markdown links, web-search sources)
    render as OSC-8 terminal hyperlinks — clean and clickable in a real terminal
    (Kitty, iTerm2, the TUI). Pass ``False`` for surfaces that can't render OSC-8
    (notebooks render ANSI→HTML, pipes), where it would leak as escape-code
    garbage; there links degrade to plain text the frontend can auto-linkify.

    ``markdown=False`` renders Markdown-bearing content (the assistant message,
    markdown tool reports) as plain text instead of formatted Markdown — for a
    linear surface that can't repaint and so streams raw tokens.
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
        if not markdown:
            return Text(text)  # raw text, no Markdown formatting or JSON tabling
        # A pure-JSON answer (e.g. a structured ``llm_output_schema`` output)
        # renders like tool-call arguments — structured, not as prose.
        structured = _structured_output(text)
        if structured is not None:
            return structured
        theme = _active_code_theme()
        # With hyperlinks=False, Rich renders links as "text (url)" plain text
        # instead of OSC-8 hyperlinks (which leak as escape-code garbage where
        # the surface can't render them); the plain URL is then auto-linkified.
        return _Markdown(
            text, code_theme=theme, inline_code_theme=theme, hyperlinks=hyperlinks
        )

    if isinstance(event, ReasoningItemEvent):
        parts = [p.text for p in (event.data.summary or []) if getattr(p, "text", "")]
        # A finalized reasoning item with no summary text (e.g. low effort, or
        # encrypted server-side reasoning) has nothing to show — skip it rather
        # than render an empty "thinking…" gutter. The live placeholder belongs
        # to the streaming path only.
        if not parts:
            return None
        return render_thinking_gutter("\n".join(parts))

    if isinstance(event, ToolCallItemEvent):
        agent = event.source or "agent"
        tool = event.data.name or "tool"
        return panel(
            f"{escape(agent)} → {escape(tool)}",
            _build_args_renderable(event.data.arguments),
            PALETTE["border_tool_call"],
        )

    if isinstance(event, WebSearchCallItemEvent):
        return render_web_search(
            event.data, event.source or "agent", hyperlinks=hyperlinks
        )

    if isinstance(event, ToolOutputItemEvent):
        item = event.data
        agent = event.destination or "agent"
        tool = event.source or "tool"
        text_out: Any = item.output if isinstance(item.output, str) else item.text
        # External tool output is fenced in <untrusted_content> for the model.
        # The UI peels the fence off and renders the inner result (so a JSON
        # result becomes a key/value panel rather than raw tags), flagging the
        # provenance in the title instead.
        untrusted = False
        if isinstance(text_out, str):
            inner, source = unwrap_untrusted(text_out)
            if source is not None:
                text_out, untrusted = inner, True
        title = f"{escape(agent)} ← {escape(tool)}"
        if untrusted:
            title += f" [{PALETTE['muted']}]{escape('[untrusted]')}[/]"
        # A failed tool result (a ToolErrorInfo the loop flagged) gets the red
        # error border; a normal result keeps the neutral one.
        border = PALETTE["error"] if item.is_error else PALETTE["border_tool_result"]
        images = item.images
        if not images or not inline_images:
            return panel(
                title,
                _build_result_renderable(
                    text_out,
                    PALETTE["tool_result"],
                    inline_images=inline_images,
                    hyperlinks=hyperlinks,
                    markdown=markdown,
                ),
                border,
            )
        blocks: list[RenderableType] = []
        if isinstance(text_out, str) and text_out.strip():
            blocks.append(
                _build_result_renderable(
                    text_out,
                    PALETTE["tool_result"],
                    hyperlinks=hyperlinks,
                    markdown=markdown,
                )
            )
        for img in images:
            if blocks:
                blocks.append(Text(""))
            blocks.append(render_input_image(img))
        return panel(title, Group(*blocks), border)

    if isinstance(event, ToolErrorEvent):
        # A tool failure is also committed as a ToolOutputItemEvent(is_error=True)
        # — the red-bordered result panel that the model actually sees. That is
        # the canonical record; this raw terminal event would only duplicate it,
        # so it has no display of its own. (Non-tool errors keep their rendering.)
        return None

    if isinstance(event, SystemMessageEvent):
        text = extract_input_text(event.data)
        return panel(
            f"System → {event.source or 'agent'}",
            _message_body(text),
            PALETTE["border_system"],
        )

    if isinstance(event, UserMessageEvent):
        text = extract_input_text(event.data)
        # Framework notices get a specialized render (shared by every surface).
        # The owning agent is the recipient: drain delivers with
        # destination=agent, resume injects with source=agent.
        if "<task_notification>" in text:
            return render_task_notification(
                text, agent=event.destination or event.source
            )
        if "<session_resumed>" in text:
            return render_resume_notice(text)
        # A user-invoked skill (slash-command) — show the message verbatim (the
        # ``<system-reminder subject="user invoked skill …">`` wrapper and the
        # full body, exactly as the agent receives it) under a distinct "skill" frame.
        skill = match_invocation_wrapper(text)
        if skill is not None:
            return panel(f"skill <{skill}>", _message_body(text), PALETTE["accent"])
        # A received input — drained mailbox mail names its sender ("user", or
        # a peer member); other paths may not know one, so fall back to showing
        # just the recipient.
        dst = event.destination or "agent"
        label = f"{event.source} → {dst}" if event.source else f"→ {dst}"
        return panel(label, _message_body(text), PALETTE["border_input"])

    if isinstance(event, GenerationEndEvent):
        return usage_line(event)

    if isinstance(event, CompactionEvent):
        return render_compaction_notice(event.data)

    if isinstance(event, BackgroundTaskLaunchedEvent):
        i = event.data
        text = f"⧗ {i.tool_name} launched in background (id: {i.task_id})"
        if i.output_name:
            text += f" · {i.output_name}"
        return Text(text, style=PALETTE["warn"])

    if isinstance(event, BackgroundTaskCompletedEvent):
        i = event.data
        return Text(
            f"✓ {i.tool_name} completed (id: {i.task_id})",
            style=PALETTE["tool_result"],
        )

    if isinstance(event, (LLMStreamingErrorEvent, ProcStreamingErrorEvent)):
        return Text(
            sanitize_terminal_text(f"Error: {event.data.error}"),
            style=f"bold {PALETTE['error']}",
        )

    return None


def render_retry_notice(event: ResponseRetrying | ResponseFallback) -> RenderableType:
    """
    Inline notice that an in-progress response attempt failed and a retry /
    model-fallback is starting. Its partial output is discarded, so this keeps
    the cleared text from vanishing without explanation.
    """
    if isinstance(event, ResponseFallback):
        head = (
            f"⇄ falling back: {escape(event.failed_model)} "
            f"→ {escape(event.fallback_model)}"
        )
        detail = event.error_type
    else:
        head = f"↻ retrying (attempt {event.attempt})"
        detail = event.error
    notice = Text(head, style=f"bold {PALETTE['warn']}")
    if detail:
        notice.append(f" — {truncate(detail, 200)}", style=PALETTE["muted"])
    return notice


def render_compaction_notice(info: CompactionInfo) -> RenderableType:
    """
    Marks where context-window compaction folded older turns into a summary.

    Shows the turns folded, the recent turns kept verbatim, and the resulting
    context size, plus the summary the folded span was replaced with (a panel
    when the summary is present, a thin separator rule otherwise).
    """
    folded = info.folded_turns
    kept = info.preserved_turns
    parts = [
        f"folded {folded} turn{'' if folded == 1 else 's'}",
        f"{kept} recent turn{'' if kept == 1 else 's'} kept",
    ]
    if info.context_window:
        parts.append(f"{info.context_tokens:,} / {info.context_window:,} tokens")
    elif info.context_tokens:
        parts.append(f"~{info.context_tokens:,} tokens")
    head = f"⊙ context compacted · {' · '.join(parts)}"
    if info.summary:
        # Show exactly what the agent receives — the wrapped <system-reminder>
        # message, rendered raw (XML-highlighted), like a skill invocation.
        return panel(
            head, _message_body(fold_summary_text(info.summary)), PALETTE["accent"]
        )
    return Rule(f"[bold]{head}[/]", style=PALETTE["accent"])


def render_turn_rule(agent: str, turn: int) -> RenderableType:
    """
    A turn-separator rule. ``turn`` is the displayed (1-based) number — a UI
    may pass its own monotonic per-agent count instead of the per-step value
    (which resets to 0 each run/step).
    """
    return Rule(f"[bold]{escape(agent)}[/] · turn {turn}", style=PALETTE["separator"])


def render_tool_stream(
    agent: str,
    tool: str,
    text: str,
    *,
    background: bool = False,
    log_name: str | None = None,
) -> RenderableType:
    """
    Panel for in-progress (streaming) tool output, before the final result.

    ``background=True`` marks a backgrounded task's live output: it is progress
    mirrored to the task's log, NOT a result in the agent's context (the agent
    receives only a summary when the task finishes). It is styled distinctly —
    muted, headed by the log file (``tool · <log_name>``) rather than an
    ``agent ← tool`` result arrow — so it is not mistaken for a tool result the
    agent actually saw. ``log_name`` is that log's basename, when one is written.
    """
    # Strip the trailing newline (tool output usually ends in one) so the box
    # hugs its content at the bottom instead of showing a blank line above the
    # panel's own padding — matching the finalised result panel.
    body = truncate_lines(truncate(text.rstrip("\n"), _TRUNC), _MAX_LINES)
    if background:
        title = f"{escape(tool)} · {escape(log_name)}" if log_name else escape(tool)
        return panel(
            title,
            Text(body or "…", style=PALETTE["bg_tool"]),
            PALETTE["border_bg_tool"],
        )
    return panel(
        f"{escape(agent)} ← {escape(tool)}",
        Text(body or "…", style=PALETTE["tool_result"]),
        PALETTE["border_tool_result"],
    )


def _notification_field(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.+?)\s*</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def render_task_notification(text: str, *, agent: str | None = None) -> RenderableType:
    """
    A background-task notification delivered to an agent.

    Shows the FULL message the model received, XML-highlighted (not stripped),
    under a ``<tool> → <agent>`` heading (the box keeps a status-cue border —
    red for interrupted/failed). An interrupted / failed task is prefixed with a
    gray ``✗ <tool> <status> (id)`` line — same gray as the ``✓ … completed``
    line a completed task gets from its :class:`BackgroundTaskCompletedEvent`
    (so a completed task adds none here).
    """
    tool = _notification_field(text, "tool_name") or "background task"
    status = _notification_field(text, "status")
    task_id = _notification_field(text, "task_id")
    failed = status in {"interrupted", "failed"}

    title = f"{escape(tool)} → {escape(agent)}" if agent else escape(tool)
    border = PALETTE["error"] if failed else PALETTE["border_tool_result"]
    box = panel(title, _message_body(text), border)
    if not failed:
        return box
    # Gray text (matching the "✓ … completed" line), even though the box keeps
    # its status-cue border.
    line = f"✗ {tool} {status}".rstrip()
    if task_id:
        line += f" (id: {task_id})"
    return Group(Text(line, style=PALETTE["tool_result"]), box)


def render_resume_notice(text: str) -> RenderableType:
    """Panel for the resume framing — the full message, XML-highlighted."""
    return panel("session resumed", _message_body(text), PALETTE["muted"])


class _ThinkingGutter:
    """
    Left-border "gutter" rendering of reasoning text, shared by the console and
    the Textual app so thinking looks the same on both surfaces.

    A ``┌ thinking`` header, each (wrapped) body line prefixed with ``│``, and a
    closing ``└``. Wrapping defers to the render-time width, so the gutter stays
    on every visual line at any pane width.
    """

    def __init__(self, body: str) -> None:
        self._body = body

    def __rich_console__(  # noqa: PLW3201
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        border = PALETTE["border_thinking"]
        indent, gutter = "  ", "│ "
        inner = max(options.max_width - len(indent) - len(gutter), 1)

        header = Text(f"{indent}┌ ", style=border)
        header.append("thinking", style=f"bold {border}")
        yield header
        yield Text(f"{indent}{gutter}", style=border)

        body = Text(self._body or "…", style=PALETTE["thinking"])
        for line in body.wrap(console, inner):
            row = Text(f"{indent}{gutter}", style=border)
            row.append_text(line)
            yield row

        yield Text(f"{indent}└", style=border)


def render_thinking_gutter(text: str) -> RenderableType:
    """Left-border gutter for reasoning text (streaming or finalized)."""
    return _ThinkingGutter(truncate_lines(truncate(text, _TRUNC), _MAX_LINES))


def render_thinking_stream(text: str) -> RenderableType:
    """Gutter for in-progress (streaming) reasoning, matching the finalized one."""
    return render_thinking_gutter(text)


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


# ── builders (shared by EventConsole and the Textual app) ──


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


def usage_line(
    event: GenerationEndEvent,
    *,
    total_cost: float = 0.0,
    generation_count: int = 0,
) -> RenderableType | None:
    usage = event.data.usage
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
        reasoning = usage.output_tokens_details.reasoning_tokens
        if reasoning:
            parts.append(f"{reasoning:,} thinking")
    if usage.cost:
        parts.append(f"${usage.cost:.4f}")
    model = event.data.model or ""
    line = f"{model} · {' · '.join(parts)}" if parts else model
    # Running session total, shown once more than one generation has cost — the
    # caller threads it through (the renderer itself is stateless).
    if generation_count > 1 and total_cost > 0:
        line += f" · Σ ${total_cost:.4f}"
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


# Resolved lazily, not at import: asking Pygments for the catppuccin style
# triggers its plugin entry point, which imports matplotlib. Importing this
# module must stay light (the plain console uses it too), so defer to first use.
_code_theme: str | None = None
_md_styles: dict[str, str] = {}
# code background: "default" composites to near-black in Textual, so the TUI
# passes its theme background here to make code blocks blend into the pane
_code_bg: str = "default"


def _active_code_theme() -> str:
    """The code-highlight theme, resolving the catppuccin default on first use."""
    global _code_theme
    if _code_theme is None:
        _code_theme = _pygments_style_for("catppuccin-macchiato")
    return _code_theme


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


class _Heading(Heading):
    """Headings left-aligned (Rich centers them by default)."""

    def __rich_console__(  # noqa: PLW3201
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # rich.markdown.Heading verbatim, except left-aligned: it hardcodes
        # justify="center" with no hook to override just the alignment, so the
        # method is reproduced (h1 box and the blank line above h2 are Rich's).
        self.text.justify = "left"
        if self.tag == "h1":
            yield Panel(self.text, box=HEAVY, style="markdown.h1.border")
        else:
            if self.tag == "h2":
                yield Text("")
            yield self.text


class _HtmlBlock(MarkdownElement):
    """Render raw HTML/XML blocks as highlighted XML (Rich drops them)."""

    def __init__(self) -> None:
        self._text = ""

    def on_text(
        self,
        context: MarkdownContext,  # noqa: ARG002 — MarkdownElement.on_text signature
        text: TextType,
    ) -> None:
        self._text += text if isinstance(text, str) else text.plain

    def __rich_console__(  # noqa: PLW3201
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = self._text.strip("\n")
        if not code.strip():
            return
        yield Syntax(
            code,
            "xml",
            theme=_active_code_theme(),
            background_color=_code_bg,
            word_wrap=True,
            padding=(1, 0),
        )


class _HtmlInline(MarkdownElement):
    """Keep inline HTML/XML tags as literal text (Rich strips them)."""

    new_line: ClassVar[bool] = False

    def on_text(self, context: MarkdownContext, text: TextType) -> None:
        # re-inject the raw tag into the surrounding paragraph buffer
        context.on_text(text if isinstance(text, str) else text.plain, "text")


class _Markdown(Markdown):
    """Markdown with transparent code fences + app-theme element styling."""

    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "code_block": _CodeBlock,
        "fence": _CodeBlock,
        "heading_open": _Heading,
        "html_block": _HtmlBlock,
        "html_inline": _HtmlInline,
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
    "new_source": "python",
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
        theme=_active_code_theme(),
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


def _link_style(url: str) -> str:
    """A Rich style rendering text as a clickable accent-colored OSC-8 hyperlink."""
    return f"{PALETTE['accent']} link {url}" if url else PALETTE["muted"]


def _source_url(url: str) -> str:
    """
    The URL to display for a search source as plain text, or ``""`` to omit it.

    Gemini grounding sources carry ``vertexaisearch.cloud.google.com/grounding-
    api-redirect/…`` URLs, not the real page — long, opaque, and identical bar a
    token, so listing them is pure clutter. They're dropped; the title is the real
    domain and identifies the source. Anthropic returns real URLs, which are kept.
    Used in every path — a grounding redirect is never shown nor used as the
    target of an OSC-8 link.
    """
    return "" if (not url or "grounding-api-redirect" in url) else url


def _web_search_body(
    item: WebSearchCallItem, *, hyperlinks: bool
) -> tuple[RenderableType, str]:
    """
    Key/value renderable + status for a server-side ``web_search_call`` item.

    Same kv-table style as tool-call args, but each query / source is its own
    bulleted entry. Opaque grounding-redirect URLs (Gemini) are never used as a
    link target nor shown (see ``_source_url``); such a source renders as just its
    domain title. For a real URL: with ``hyperlinks`` (a real terminal / the TUI)
    the title is a clickable OSC-8 link to it; without (notebooks/pipes, where
    OSC-8 leaks as escape-code garbage) the URL is shown as plain text on its own
    line for the frontend to auto-linkify.
    """
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
    table.add_column("value", ratio=1)

    action = item.action
    if isinstance(action, SearchAction):
        if action.queries:
            queries = Text()
            for i, query in enumerate(action.queries):
                if i:
                    queries.append("\n")
                queries.append("• ", PALETTE["accent"])
                queries.append(truncate(query, 300), PALETTE["tool_call"])
            table.add_row("queries", queries)
        if action.sources:
            sources = Text()
            for i, src in enumerate(action.sources[:12]):
                if i:
                    sources.append("\n")
                sources.append("• ", PALETTE["accent"])
                # Real URL only — grounding redirects are never linked or shown.
                url = _source_url(src.url)
                if hyperlinks and url:
                    # Clean label, clickable: OSC-8-link the title to the real URL
                    # (the raw URL stays behind the link).
                    sources.append(truncate(src.title or url, 200), _link_style(url))
                else:
                    # No OSC-8 (or no real URL): title, plus the real URL on its
                    # own line (auto-linkified) when there is one.
                    label = truncate(src.title or url or "(source)", 200)
                    sources.append(label, PALETTE["tool_call"])
                    if url and url != label:
                        sources.append(f"\n   {truncate(url, 300)}", PALETTE["muted"])
                if src.page_age:
                    sources.append(
                        f"\n   {truncate(src.page_age, 60)}", PALETTE["muted"]
                    )
            table.add_row("sources", sources)
    elif isinstance(action, OpenPageAction):
        url = action.url or ""
        style = _link_style(url) if (hyperlinks and url) else PALETTE["accent"]
        table.add_row("open page", Text(truncate(url, 400), style))
    else:  # FindInPageAction
        table.add_row(
            "find in page",
            Text(truncate(action.pattern or "", 200), PALETTE["tool_call"]),
        )
        if action.url:
            style = _link_style(action.url) if hyperlinks else PALETTE["accent"]
            table.add_row("page", Text(truncate(action.url, 400), style))

    if table.row_count == 0:
        table.add_row("web search", Text(""))
    return table, item.status or ""


def render_web_search(
    item: WebSearchCallItem, agent: str = "agent", *, hyperlinks: bool = True
) -> RenderableType:
    """Panel for a server-side ``web_search_call`` item (queries / sources / URL)."""
    body, status = _web_search_body(item, hyperlinks=hyperlinks)
    title = f"{escape(agent)} → web_search"
    if status and status != "completed":
        title += f" [{PALETTE['muted']}]{escape(f'[{status}]')}[/]"
    return panel(title, body, PALETTE["border_tool_call"])


def _structured_output(text: str) -> RenderableType | None:
    """
    Renderable for a pure-JSON output message, else ``None`` (prose).

    A JSON object renders as the same key/value table used for tool-call
    arguments; a JSON array as a highlighted JSON block. A payload that isn't a
    JSON object/array (ordinary prose, a bare string/number) returns ``None`` so
    the caller falls back to Markdown.
    """
    stripped = text.strip()
    if stripped[:1] not in {"{", "["}:
        return None
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, dict):
        return _kv_table(cast("dict[str, Any]", parsed), PALETTE["tool_call"])
    if isinstance(parsed, list):
        return _code_block(json.dumps(parsed, indent=2, ensure_ascii=False), "json")
    return None


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
# whole payload wrapped in a single root element (e.g. <task_notification>…</…>)
_XML_BLOCK = re.compile(r"(?s)\A\s*<([A-Za-z][\w.:-]*)(?:\s[^<>]*)?>.*</\1\s*>\s*\Z")


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


def _looks_like_xml(text: str) -> bool:
    # only when the whole payload is one root element, so ordinary logs/stdout
    # that merely contain a "<" stay plain text
    return bool(_XML_BLOCK.match(text))


def _message_body(text: str) -> RenderableType:
    # XML payloads (e.g. injected <task_notification>) render highlighted; typed
    # prose stays plain so normal user input isn't reinterpreted as markup
    if _looks_like_xml(text):
        return _code_block(truncate(text, _TRUNC), "xml")
    return Text(escape(truncate_lines(text, _MAX_LINES)))


def _build_result_renderable(
    output: Any,
    text_color: str,
    *,
    inline_images: bool = True,
    hyperlinks: bool = True,
    markdown: bool = True,
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
    if markdown and _looks_like_markdown(content):
        # agent tools often return markdown reports — render them formatted
        theme = _active_code_theme()
        return _Markdown(
            content, code_theme=theme, inline_code_theme=theme, hyperlinks=hyperlinks
        )
    if _looks_like_xml(content):
        # tagged payloads (e.g. background <task_notification>…) as highlighted XML
        return _code_block(content, "xml")
    # keep blank lines (output structure is meaningful); only drop the trailing
    # newline so the box hugs its content above the panel's own padding
    return Text(truncate_lines(content.rstrip("\n"), _MAX_LINES), style=text_color)


def _kv_table(data: dict[str, Any], text_color: str) -> RenderableType:
    if not data:
        # An empty table's flexible (ratio) value column measures to the full
        # available width, stretching the enclosing panel edge-to-edge. With no
        # rows to show, render nothing so the box hugs its title instead.
        return Text("")
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
        table.add_row(str(k), _value_cell(str(k), v, data))
    return table


def _value_cell(key: str, v: Any, row: dict[str, Any] | None = None) -> RenderableType:
    # code-bearing values (a tool's ``code``/``command``/…) render with syntax
    # highlighting in their value cell — the row layout (and column alignment)
    # is unchanged; everything else stays plain text.
    if isinstance(v, str):
        lexer = _code_lexer(key, v)
        if lexer is not None:
            # a notebook cell's source highlights by its declared type: markdown
            # for a markdown cell, python (the `source`/`new_source` default)
            # otherwise.
            if (
                key.lower() in {"source", "new_source"}
                and row is not None
                and row.get("cell_type") == "markdown"
            ):
                lexer = "markdown"
            return _code_block(v, lexer)
        # Parse ANSI in tool output (e.g. colorized `ls`) into Rich styles: the
        # colors render, dangerous control sequences (cursor / erase / OSC) are
        # dropped, and the cell width is measured from visible text — raw escape
        # bytes would inflate it and shove the panel border inward.
        return Text.from_ansi(v)
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
    for part in msg.content:
        if isinstance(part, InputText):
            parts.append(part.text.strip())
        elif isinstance(part, InputImage):
            parts.append(part.image_url or "[image]" if part.is_url else "[image]")
    return "\n\n".join(parts)


def truncate(text: str, limit: int) -> str:
    # Every untrusted body flows through here or truncate_lines — the
    # sanitization point against terminal-escape injection (CSI / OSC in
    # tool output clearing the screen or spoofing what an approver sees).
    text = sanitize_terminal_text(text)
    return text[:limit] + "…" if len(text) > limit else text


def truncate_lines(text: str, max_lines: int) -> str:
    text = sanitize_terminal_text(text)
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
