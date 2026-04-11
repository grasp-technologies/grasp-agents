"""
Console display for grasp-agents event streams.

Renders agent events with rich formatting. Works in terminals and Jupyter
notebooks.  In Jupyter the default Console uses ``force_jupyter=False`` so
that all output goes through stdout as a continuous ANSI stream instead of
creating a separate HTML block per ``console.print()`` call.
"""

import json
import textwrap
from collections.abc import AsyncIterator
from enum import StrEnum, auto
from typing import Any, TypedDict, cast

from rich.box import ROUNDED
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .types.content import InputImage, InputText
from .types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskLaunchedEvent,
    Event,
    GenerationEndEvent,
    LLMStreamEvent,
    LLMStreamingErrorEvent,
    OutputMessageItemEvent,
    ProcPacketOutEvent,
    ProcStreamingErrorEvent,
    ReasoningItemEvent,
    RunPacketOutEvent,
    SystemMessageEvent,
    ToolCallItemEvent,
    ToolErrorEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessageEvent,
)
from .types.items import (
    FunctionToolCallItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from .types.llm_events import (
    FunctionCallArgumentsDelta,
    OutputItemAdded,
    OutputItemDone,
    OutputMessageTextPartTextDelta,
    ReasoningContentPartTextDelta,
    ReasoningSummaryPartTextDelta,
)


class ColorPalette(TypedDict):
    border_tool_call: str  # tool-call panels
    border_input: str  # system msg panels
    border_tool_result: str  # tool-result panels
    border_system: str  # system msg panels
    border_thinking: str  # thinking panels
    separator: str  # turn separator rule
    accent: str  # arg keys, usage/cost
    muted: str  # tool output, stop, usage
    tool_call: str  # tool-call text
    tool_result: str  # tool-result text
    thinking: str  # thinking text
    success: str  # tool-result panels, user msg
    error: str  # errors
    warn: str  # background tasks


class ColorTheme(StrEnum):
    DARK = auto()
    LIGHT = auto()


COLOR_THEMES: dict[ColorTheme, ColorPalette] = {
    ColorTheme.DARK: {
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
    },
    ColorTheme.LIGHT: {
        "border_tool_call": "#BEE4F7",
        "border_system": "#F7E1BE",
        "border_input": "#F7EEBE",
        "border_tool_result": "#EBFAAA",
        "border_thinking": "#E1F8FA",
        "separator": "#454141",
        "accent": "#AAACFA",
        "muted": "#696464",
        "tool_call": "#FFFFFF",
        "tool_result": "#969595",
        "thinking": "#969595",
        "success": "#3BBF69",
        "error": "#FCA9A9",
        "warn": "#BFB53B",
    },
}

# ── Truncation defaults ──
_MAX_TOOL_OUTPUT_LINES = 12
_MAX_INPUT_MSG_LINES = 8


def _default_console() -> Console:
    """
    Console that renders cleanly in both terminals and Jupyter.

    Uses 256-color mode for maximum compatibility across terminal
    emulators and Jupyter environments. Truecolor (24-bit) causes
    background-fill artifacts in some terminals.
    """
    return Console(
        force_jupyter=False,
        force_terminal=True,
        highlight=False,
        color_system="256",
    )


class EventConsole:
    """
    Rich-based display for agent event streams.

    Handles both streaming mode (LLMStreamEvents with token deltas) and
    non-streaming mode (promoted item events with complete content).

    Usage::

        console = EventConsole()
        async for event in console.stream(agent.run_stream("hello", ctx=ctx)):
            if isinstance(event, ProcPacketOutEvent):
                result = event.data.payloads[0]
    """

    def __init__(
        self,
        *,
        console: Console | None = None,
        show_thinking: bool = False,
        show_tool_args: bool = True,
        show_usage: bool = True,
        show_input_messages: bool = False,
        show_packets: bool = False,
        trunc_len: int = 10000,
        max_tool_output_lines: int = _MAX_TOOL_OUTPUT_LINES,
        max_input_msg_lines: int = _MAX_INPUT_MSG_LINES,
        color_theme: ColorTheme = ColorTheme.DARK,
    ) -> None:
        self.console = console or _default_console()
        self.show_thinking = show_thinking
        self.show_tool_args = show_tool_args
        self.show_usage = show_usage
        self.show_input_messages = show_input_messages
        self.show_packets = show_packets
        self.trunc_len = trunc_len
        self.max_tool_output_lines = max_tool_output_lines
        self.max_input_msg_lines = max_input_msg_lines

        # Streaming state
        self._in_text = False
        self._in_thinking = False
        self._think_line_buf: str = ""
        self._streamed_message = False
        self._streamed_tool = False
        self._streamed_reasoning = False
        self._in_tool_block = False

        # Tool call_id → name mapping for result/error attribution
        self._tool_names: dict[str, str] = {}
        # Tool names launched as background tasks — their subagent
        # events (turns, thinking, text) are suppressed.
        self._bg_tool_names: set[str] = set()

        # Cumulative cost tracking
        self._total_cost: float = 0.0
        self._generation_count: int = 0

        # Colors
        self._color_theme = COLOR_THEMES[color_theme]

    async def stream(
        self,
        events: AsyncIterator[Event[Any]],
    ) -> AsyncIterator[Event[Any]]:
        """
        Wrap an event generator, displaying events while yielding
        them through.
        """
        async for event in events:
            self._handle(event)
            yield event

    # ── Dispatch ──

    def _is_bg_subagent(self, event: Event[Any]) -> bool:
        """True if event comes from a background tool's subagent."""
        src = event.source or ""
        return any(
            src == name or src.startswith(f"{name}:") for name in self._bg_tool_names
        )

    def _handle(self, event: Event[Any]) -> None:
        # Background task lifecycle events are always handled
        if isinstance(event, BackgroundTaskLaunchedEvent):
            self._on_bg_launched(event)
            return

        if isinstance(event, BackgroundTaskCompletedEvent):
            self._on_bg_completed(event)
            return

        # Suppress internal events from background subagents,
        # but let tool results through (placeholder "Task launched...")
        if self._is_bg_subagent(event) and not isinstance(
            event, (ToolResultEvent, ToolErrorEvent)
        ):
            return

        if isinstance(event, TurnStartEvent):
            self._on_turn_start(event)

        elif isinstance(event, TurnEndEvent):
            self._on_turn_end(event)

        elif isinstance(event, GenerationEndEvent):
            self._on_generation_end(event)

        elif isinstance(event, LLMStreamEvent):
            self._on_llm_stream(event)

        elif isinstance(event, OutputMessageItemEvent):
            self._on_output_message(event)

        elif isinstance(event, ToolCallItemEvent):
            self._on_tool_call_item(event)

        elif isinstance(event, ReasoningItemEvent):
            self._on_reasoning_item(event)

        elif isinstance(event, ToolResultEvent):
            self._on_tool_result(event)

        elif isinstance(event, ToolErrorEvent):
            self._on_tool_error(event)

        elif isinstance(event, UserMessageEvent):
            text = _extract_input_text(event.data)
            if "<task_notification>" in text:
                self._on_task_notification(text)
            elif self.show_input_messages:
                self._on_user_message(event)

        elif isinstance(event, SystemMessageEvent):
            if self.show_input_messages:
                self._on_system_message(event)

        elif isinstance(event, (ProcPacketOutEvent, RunPacketOutEvent)):
            if self.show_packets:
                self._on_packet(event)

        elif isinstance(event, (LLMStreamingErrorEvent, ProcStreamingErrorEvent)):
            self._on_error(event)

    # ── LLM stream events (streaming mode) ──

    def _on_llm_stream(self, event: LLMStreamEvent) -> None:
        se = event.data

        if isinstance(se, OutputItemAdded):
            item = se.item
            if isinstance(item, ReasoningItem):
                self._end_text()
                self._streamed_reasoning = True
                if self.show_thinking:
                    self._start_thinking()
            elif isinstance(item, FunctionToolCallItem):
                self._end_text()
                self._end_thinking()
                self._streamed_tool = True

        elif isinstance(se, OutputMessageTextPartTextDelta):
            self._streamed_message = True
            self._in_tool_block = False
            self._end_thinking()
            self._in_text = True
            self._write(se.delta)

        elif isinstance(
            se,
            ReasoningSummaryPartTextDelta | ReasoningContentPartTextDelta,
        ):
            if self.show_thinking:
                self._write_thinking(se.delta)

        elif isinstance(se, FunctionCallArgumentsDelta):
            pass  # Display complete args from ToolCallItemEvent

        elif isinstance(se, OutputItemDone):
            item = se.item
            if isinstance(item, OutputMessageItem):
                self._end_text()
            elif isinstance(item, ReasoningItem) and self.show_thinking:
                self._end_thinking()

    # ── Promoted item events (non-streaming fallback) ──

    def _on_output_message(self, event: OutputMessageItemEvent) -> None:
        if self._streamed_message:
            self._streamed_message = False
            return

        self._in_tool_block = False

        text = event.data.text
        if text:
            self._write(text + "\n")

    def _on_tool_call_item(self, event: ToolCallItemEvent) -> None:
        self._streamed_tool = False
        self._end_text()
        self._end_thinking()
        self._in_tool_block = True

        item = event.data
        name = item.name or "tool"

        # Track call_id → name for later result/error attribution
        if item.call_id:
            self._tool_names[item.call_id] = name

        title_color = self._color_theme["border_tool_call"]
        border_color = self._color_theme["border_tool_call"]

        if self.show_tool_args and item.arguments:
            renderable = self._build_args_renderable(item.arguments)
            panel = Panel(
                renderable,
                title=f"[bold {title_color}]→ {escape(name)}[/]",
                title_align="left",
                border_style=border_color,
                box=ROUNDED,
                padding=(0, 1),
                expand=True,
            )
            self.console.print(panel)
        else:
            self.console.print(f"[bold {title_color}]▶ {escape(name)}[/]")

    def _on_reasoning_item(self, event: ReasoningItemEvent) -> None:
        if self._streamed_reasoning:
            self._streamed_reasoning = False
            return

        if not self.show_thinking:
            return

        self._end_text()

        text_color = self._color_theme["border_thinking"]
        border_color = self._color_theme["border_thinking"]

        parts: list[str] = []
        for part in event.data.summary_parts or []:
            if hasattr(part, "text") and part.text:
                parts.append(part.text)
        if parts:
            content = "\n".join(parts)
            panel = Panel(
                Text(escape(content), style=f"{text_color}"),
                title=f"[bold {text_color}]thinking[/]",
                title_align="left",
                border_style=border_color,
                box=ROUNDED,
                padding=(0, 1),
                expand=True,
            )
            self.console.print(panel)
        else:
            self.console.print(f"[bold {text_color}]thinking…[/]")

    # ── Lifecycle events ──

    def _on_turn_start(self, event: TurnStartEvent) -> None:
        self._end_text()
        self._end_thinking()
        self._in_tool_block = False
        agent = event.source or "agent"
        turn = event.data.turn + 1
        self.console.print(
            Rule(
                f"[bold]{escape(agent)}[/bold] · turn {turn}",
                style=self._color_theme["separator"],
            )
        )
        self.console.print()

    def _on_turn_end(self, event: TurnEndEvent) -> None:
        self._end_text()
        self._end_thinking()
        self._in_tool_block = False
        reason = event.data.stop_reason
        if reason is None:
            return
        val = str(getattr(reason, "value", reason))
        color = self._color_theme["muted"]
        if val != "final_answer":
            self.console.print(f"[italic {color}]stopped: {escape(val)}[/]")

    def _on_generation_end(self, event: GenerationEndEvent) -> None:
        if not self.show_usage:
            return
        resp = event.data
        usage = resp.usage_with_cost
        if not usage:
            return

        color = self._color_theme["muted"]

        self._generation_count += 1

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
            self._total_cost += usage.cost
            parts.append(f"${usage.cost:.4f}")

        if parts:
            model = resp.model or ""
            line = f"{model} · {' · '.join(parts)}"
            if self._generation_count > 1 and self._total_cost > 0:
                line += f"  Σ ${self._total_cost:.4f}"

            # Right-aligned, muted — clearly metadata, not agent text
            self.console.print()
            self.console.print(Text(line, style=f"italic {color}"), justify="right")
            self.console.print()

    # ── Message events ──

    def _on_user_message(self, event: UserMessageEvent) -> None:
        self._end_text()
        self._in_tool_block = False
        text = _extract_input_text(event.data)
        display = _truncate_lines(text, self.max_input_msg_lines)
        agent = event.source or ""
        label = f"User → {agent}" if agent else "User"
        self.console.print(
            Panel(
                escape(display),
                title=f"[bold]{escape(label)}[/]",
                title_align="left",
                border_style=self._color_theme["border_input"],
                box=ROUNDED,
                padding=(0, 1),
            )
        )
        self.console.print()

    def _on_system_message(self, event: SystemMessageEvent) -> None:
        self._end_text()
        self._in_tool_block = False
        text = _extract_input_text(event.data)
        display = _truncate_lines(text, self.max_input_msg_lines)
        agent = event.source or ""
        label = f"System → {agent}" if agent else "System"
        self.console.print(
            Panel(
                escape(display),
                title=f"[bold]{escape(label)}[/]",
                title_align="left",
                border_style=self._color_theme["border_system"],
                box=ROUNDED,
                padding=(0, 1),
            )
        )
        self.console.print()

    # ── Tool events ──

    def _on_tool_result(self, event: ToolResultEvent) -> None:
        data = event.data

        title_color = self._color_theme["border_tool_result"]
        border_color = self._color_theme["border_tool_result"]
        text_color = self._color_theme["tool_result"]

        # Look up tool name from the call_id recorded in _on_tool_call_item
        tool_name = "← " + self._tool_names.get(data.call_id, "")
        title = f"[bold {title_color}]{escape(tool_name)}[/]" if tool_name else None

        # Try to render dicts/BaseModels as key-value tables
        renderable = self._build_result_renderable(data.output, text_color)

        panel = Panel(
            renderable,
            title=title,
            title_align="left",
            border_style=border_color,
            box=ROUNDED,
            padding=(0, 1),
            expand=True,
        )
        self.console.print(panel)
        self.console.print()

    def _on_tool_error(self, event: ToolErrorEvent) -> None:
        title_color = self._color_theme["error"]
        text_color = self._color_theme["tool_result"]
        border_color = self._color_theme["border_tool_result"]

        self.console.print()  # breathing room after usage line

        info = event.data
        msg = f"✗ {info.error}"
        if info.timed_out:
            msg += " (timed out)"
        panel = Panel(
            Text(msg, style=f"{text_color}"),
            title=f"[bold {title_color}]{escape(info.tool_name)}[/]",
            title_align="left",
            border_style=border_color,
            box=ROUNDED,
            padding=(0, 1),
            expand=True,
        )
        self.console.print(panel)

    # ── Packet events ──

    def _on_packet(self, event: ProcPacketOutEvent | RunPacketOutEvent) -> None:
        self._end_text()
        self._in_tool_block = False
        from pydantic import BaseModel as _BM  # noqa: PLC0415, N814

        color = self._color_theme["muted"]

        src = event.source or "processor"
        self.console.print(f"[bold]{escape(src)}[/] output:")
        for p in event.data.payloads:
            if isinstance(p, _BM):
                text = p.model_dump_json(indent=2)
            else:
                try:
                    text = json.dumps(p, indent=2)
                except TypeError:
                    text = str(p)
            self.console.print(f"[{color}]{escape(_truncate(text, self.trunc_len))}[/]")

    # ── Error events ──

    def _on_error(
        self,
        event: LLMStreamingErrorEvent | ProcStreamingErrorEvent,
    ) -> None:
        self._end_text()
        self._in_tool_block = False
        err = event.data.error
        color = self._color_theme["error"]
        self.console.print(f"[bold {color}]Error:[/] {escape(str(err))}")

    # ── Background task events ──

    def _on_bg_launched(self, event: BackgroundTaskLaunchedEvent) -> None:
        # Track name so we can suppress the subagent's internal events
        self._bg_tool_names.add(event.data.tool_name)

    def _on_bg_completed(self, event: BackgroundTaskCompletedEvent) -> None:
        color = self._color_theme["tool_result"]
        info = event.data
        self.console.print()
        self.console.print(
            Text(
                f"✓ {info.tool_name} completed (id: {info.task_id})",
                style=f"{color}",
            )
        )

    def _on_task_notification(self, text: str) -> None:
        """Display background task result from notification XML."""
        import re  # noqa: PLC0415

        tool = re.search(r"<tool_name>(.+?)</tool_name>", text)
        result = re.search(r"<result>\s*(.+?)\s*</result>", text, re.DOTALL)
        error = re.search(r"<error>\s*(.+?)\s*</error>", text, re.DOTALL)
        name = tool.group(1) if tool else "background task"

        text_error_color = self._color_theme["tool_result"]
        text_success_color = self._color_theme["tool_result"]
        title_success_color = self._color_theme["border_tool_result"]
        title_error_color = self._color_theme["error"]
        border_success_color = self._color_theme["border_tool_result"]
        border_error_color = self._color_theme["error"]

        if error:
            panel = Panel(
                Text(
                    f"✗ {error.group(1)}",
                    style=f"bold {text_error_color}",
                ),
                title=f"[bold {title_error_color}]{escape(name)}[/]",
                title_align="left",
                border_style=border_error_color,
                box=ROUNDED,
                padding=(0, 1),
                expand=True,
            )
            self.console.print(panel)

        elif result:
            content = _truncate(result.group(1), self.trunc_len)
            display = _truncate_lines(content, self.max_tool_output_lines)
            panel = Panel(
                Text(escape(display), style=text_success_color),
                title=f"[bold {title_success_color}]{escape(name)}[/]",
                title_align="left",
                border_style=border_success_color,
                box=ROUNDED,
                padding=(0, 1),
                expand=True,
            )
            self.console.print(panel)

        self.console.print()

    # ── Streaming text helpers ──

    def _write(self, text: str, *, muted: bool = False) -> None:
        """Write text for streaming output (no trailing newline)."""
        f = self.console.file
        if muted and self.console.color_system is not None:
            f.write(f"\033[90m{text}\033[39m")
        else:
            f.write(text)
        f.flush()

    def _end_text(self) -> None:
        if self._in_text:
            self._write("\n")
            self._in_text = False

    # ── Streaming thinking ──
    #
    # Thinking text is streamed line-by-line (buffered per logical line)
    # using console.print() with a "│" gutter.  textwrap.wrap() ensures
    # that long lines are broken BEFORE the terminal edge so the gutter
    # is always present — even on wrapped continuations.
    #
    # This works identically in terminals and Jupyter.

    def _start_thinking(self) -> None:
        """Open a streaming thinking block."""
        if self._in_thinking:
            return
        self._in_thinking = True
        self._think_line_buf = ""
        color = self._color_theme["border_thinking"]
        header = Text("  ┌ ", style=color)
        header.append("thinking", style=f"bold {color}")
        self.console.print(header)
        self.console.print(Text("  │ ", style=f"{color}"))

    def _write_thinking(self, text: str) -> None:
        """Buffer thinking text and flush at line boundaries."""
        if not self._in_thinking:
            self._start_thinking()
        for ch in text:
            if ch == "\n":
                self._flush_think_line()
            else:
                self._think_line_buf += ch

    def _flush_think_line(self) -> None:
        """Emit one buffered line of thinking text with gutter."""
        line = self._think_line_buf
        self._think_line_buf = ""
        gutter = "  │ "
        max_w = max((self.console.width or 80) - len(gutter), 20)

        text_color = self._color_theme["thinking"]
        border_color = self._color_theme["border_thinking"]

        if not line:
            # self.console.print(Text(gutter, style=f"{color}"))
            return
        wrapped = textwrap.wrap(line, width=max_w) or [line]

        for wl in wrapped:
            header = Text(gutter, style=border_color)
            header.append(Text(wl, style=f"{text_color}"))
            self.console.print(header)

    def _end_thinking(self) -> None:
        """Close the thinking block."""
        if not self._in_thinking:
            return
        self._in_thinking = False
        if self._think_line_buf:
            self._flush_think_line()
        color = self._color_theme["border_thinking"]
        self.console.print(Text("  └", style=f"{color}"))
        self.console.print()

    # ── Tool args rendering ──

    def _build_result_renderable(self, output: Any, text_color: str) -> Table | Text:
        """Build a renderable for tool output — table for dicts, text otherwise."""
        # If output is already a dict (not serialized), use it directly
        raw: Any = output
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass

        if isinstance(raw, dict):
            args: dict[str, Any] = cast("dict[str, Any]", raw)
            table = Table(
                show_header=False,
                show_edge=False,
                box=None,
                pad_edge=False,
                padding=(0, 2, 0, 0),
            )
            table.add_column("key", style=f"bold {text_color}", no_wrap=True)
            table.add_column("value", style=text_color, ratio=1)
            for k, v in args.items():
                table.add_row(k, Text(self._format_value(v)))
            return table

        # Plain text fallback
        content = str(output)
        content = _unescape_json_string(content)
        content = _truncate(content, self.trunc_len)
        lines = [ln for ln in content.split("\n") if ln.strip()]
        clean = "\n".join(lines)
        display = _truncate_lines(clean, self.max_tool_output_lines)
        return Text(escape(display), style=text_color)

    def _build_args_renderable(self, args_json: str) -> Table | Text:
        """Build a Rich Table or Text for tool arguments."""
        try:
            parsed = json.loads(args_json)
        except (json.JSONDecodeError, TypeError):
            return Text(escape(args_json))

        if not isinstance(parsed, dict):
            return Text(escape(str(parsed)))

        args: dict[str, Any] = cast("dict[str, Any]", parsed)
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            pad_edge=False,
            padding=(0, 2, 0, 0),
        )
        table.add_column("key", style="bold", no_wrap=True)
        table.add_column("value", ratio=1)

        for k, v in args.items():
            table.add_row(k, Text(self._format_value(v)))
        return table

    @staticmethod
    def _format_value(v: Any) -> str:
        """Format a JSON value for display."""
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float, bool)) or v is None:
            return str(v)
        compact = json.dumps(v, ensure_ascii=False)
        if len(compact) > 60:
            return json.dumps(v, indent=2, ensure_ascii=False)
        return compact


# ── Module-level helpers ──


def _extract_input_text(msg: InputMessageItem) -> str:
    parts: list[str] = []
    for part in msg.content_parts:
        if isinstance(part, InputText):
            parts.append(part.text.strip())
        elif isinstance(part, InputImage):
            parts.append(
                "[image]" if not part.is_url else (part.image_url or "[image]")
            )
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _truncate_lines(text: str, max_lines: int) -> str:
    """Truncate text to max_lines, showing a hidden-lines count."""
    if max_lines <= 0:
        return text
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    hidden = len(lines) - max_lines
    kept.append(f"… {hidden} more lines")
    return "\n".join(kept)


def _unescape_json_string(text: str) -> str:
    """If text is a JSON-encoded string, return its unescaped value."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, str):
            return parsed
        return json.dumps(parsed, indent=2)
    except Exception:
        return text


# ── Convenience wrapper ──


async def stream_events(
    events: AsyncIterator[Event[Any]],
    *,
    show_thinking: bool = False,
    show_tool_args: bool = True,
    show_usage: bool = True,
    show_input_messages: bool = False,
    show_packets: bool = False,
    trunc_len: int = 10000,
    console: Console | None = None,
) -> AsyncIterator[Event[Any]]:
    """
    Wrap an event stream with rich console display.

    Drop-in replacement for ``print_event_stream``::

        async for event in stream_events(
            agent.run_stream("hello", ctx=ctx)
        ):
            ...
    """
    ec = EventConsole(
        console=console,
        show_thinking=show_thinking,
        show_tool_args=show_tool_args,
        show_usage=show_usage,
        show_input_messages=show_input_messages,
        show_packets=show_packets,
        trunc_len=trunc_len,
    )
    async for event in ec.stream(events):
        yield event
