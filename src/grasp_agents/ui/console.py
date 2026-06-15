"""
Console display for grasp-agents event streams.

A light, linear ANSI stream that works in any terminal, a pipe, or a Jupyter
notebook (``force_jupyter=False`` keeps output a single continuous stream rather
than one HTML block per print). Needs only ``rich`` — a core dependency.

Finalized items (turns, messages, tool calls/results, errors, usage) are
rendered by the shared, Textual-free :mod:`._event_render` builders — the same
ones the ``tui`` app uses, so the two surfaces stay visually consistent.
:class:`EventConsole` owns only the stream-specific behavior on top: token-by-
token text + a line-buffered thinking gutter, suppression of a backgrounded
tool's subagent chatter, background-task notification panels, cumulative cost,
and the ``show_*`` toggles.
"""

import json
import textwrap
from collections import Counter
from collections.abc import AsyncIterator
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.text import Text

from grasp_agents.types.events import (
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
    ToolOutputItemEvent,
    ToolStreamEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_events import (
    FunctionCallArgumentsDelta,
    OutputItemAdded,
    OutputItemDone,
    OutputMessageTextPartTextDelta,
    ReasoningContentPartTextDelta,
    ReasoningSummaryPartTextDelta,
    ResponseFallback,
    ResponseRetrying,
)

from ._event_render import (
    PALETTE,
    extract_input_text,
    render_event,
    render_retry_notice,
    render_tool_stream,
    truncate,
)

# ── Truncation defaults (apply to the console-owned paths: packets + task
# notifications; delegated rendering uses _event_render's own limits) ──
_MAX_TOOL_OUTPUT_LINES = 12
_MAX_INPUT_MSG_LINES = 8


def _default_console() -> Console:
    """
    Console that renders cleanly in both terminals and Jupyter.

    256-color mode maximizes compatibility across emulators and Jupyter;
    truecolor causes background-fill artifacts in some terminals.
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

    Handles both streaming mode (``LLMStreamEvent`` token deltas) and
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
        show_input_messages: bool = True,
        show_system_messages: bool = False,
        show_packets: bool = False,
        show_background: bool = False,
        trunc_len: int = 10000,
        max_tool_output_lines: int = _MAX_TOOL_OUTPUT_LINES,
        max_input_msg_lines: int = _MAX_INPUT_MSG_LINES,
    ) -> None:
        self.console = console or _default_console()
        self.show_thinking = show_thinking
        self.show_tool_args = show_tool_args
        self.show_usage = show_usage
        self.show_input_messages = show_input_messages
        self.show_system_messages = show_system_messages
        self.show_packets = show_packets
        self.show_background = show_background
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
        self._pending_tool_spacing = False

        # Tool names with in-flight background tasks. By default their subagent
        # events (turns, thinking, text) are suppressed — interleaving several
        # concurrent streams into one linear column is unreadable; the launch /
        # completion notices and the result still show, and the full output is
        # in the task's `.grasp/tasks/<id>.log`. Set ``show_background=True`` to
        # print them inline anyway (each is tagged with its source). Refcounted
        # so a later *foreground* run of the same tool renders normally once
        # every background instance has completed.
        self._bg_tool_counts: Counter[str] = Counter()
        # tool name → its .grasp log basename, so a streamed chunk can be headed
        # by the log it's mirrored to (parity with the TUI).
        self._bg_tool_logs: dict[str, str | None] = {}
        # Per-agent deferred turn header: rendered right before that agent's
        # first content of the turn, not at turn start — so a concurrent agent's
        # output streaming in meanwhile can't wedge between the header and its
        # body. Keyed by source so interleaved agents stay correctly paired.
        self._pending_turns: dict[str, TurnStartEvent] = {}

        # Cumulative cost tracking
        self._total_cost: float = 0.0
        self._generation_count: int = 0

    async def stream(
        self,
        events: AsyncIterator[Event[Any]],
    ) -> AsyncIterator[Event[Any]]:
        """Wrap an event generator, displaying events while yielding them through."""
        async for event in events:
            self._handle(event)
            yield event

    # ── Dispatch ──

    def _print(self, renderable: Any) -> None:
        if renderable is not None:
            self.console.print(renderable)

    def _is_bg_subagent(self, event: Event[Any]) -> bool:
        """True if event comes from a background tool's subagent."""
        src = event.source or ""
        return any(
            src == name or src.startswith(f"{name}:") for name in self._bg_tool_counts
        )

    def _flush_tool_spacing(self) -> None:
        """Emit deferred blank line after a tool result group."""
        if self._pending_tool_spacing:
            self.console.print()
            self._pending_tool_spacing = False

    def _handle(self, event: Event[Any]) -> None:
        if isinstance(event, BackgroundTaskLaunchedEvent):
            self._on_bg_launched(event)
            return

        if isinstance(event, BackgroundTaskCompletedEvent):
            self._on_bg_completed(event)
            self._print(render_event(event, inline_images=False))
            return

        # Suppress internal events from background subagents (unless opted in),
        # but always let tool results, errors, and task notifications through.
        if (
            not self.show_background
            and self._is_bg_subagent(event)
            and not isinstance(
                event, (ToolOutputItemEvent, ToolErrorEvent, UserMessageEvent)
            )
        ):
            return

        # Flush deferred spacing from tool results before any
        # non-tool-result event — keeps consecutive result panels
        # tight but ensures a blank line after the last one.
        if not isinstance(event, (ToolOutputItemEvent, ToolErrorEvent)):
            self._flush_tool_spacing()

        # A turn header is deferred (see _on_turn_start) and flushed right before
        # its agent's first output of the turn — content or an error — so
        # concurrent agents' output can't wedge between a header and what it
        # labels. A turn that produces neither (e.g. a bare max-turns stop) drops
        # its header entirely.
        if isinstance(
            event,
            (
                LLMStreamEvent,
                OutputMessageItemEvent,
                ToolCallItemEvent,
                ReasoningItemEvent,
                LLMStreamingErrorEvent,
                ProcStreamingErrorEvent,
            ),
        ):
            self._flush_turn(event.source)

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

        elif isinstance(event, ToolOutputItemEvent):
            self._on_tool_result(event)

        elif isinstance(event, ToolStreamEvent):
            # Incremental tool output (e.g. a shell command's progressive
            # stdout). Off unless show_background: in the linear console it is
            # noise next to the final result, but it's the live progress you
            # want when watching background tasks.
            if self.show_background:
                self._on_tool_stream(event)

        elif isinstance(event, ToolErrorEvent):
            self._on_tool_error(event)

        elif isinstance(event, UserMessageEvent):
            text = extract_input_text(event.data)
            # Framework notices (task results, resume framing) always show and
            # are rendered specially by render_event; regular user input is on
            # by default but can be turned off.
            is_notice = "<task_notification>" in text or "<session_resumed>" in text
            if is_notice or self.show_input_messages:
                self._on_user_message(event)

        elif isinstance(event, SystemMessageEvent):
            if self.show_system_messages:
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

        elif isinstance(se, (ResponseRetrying, ResponseFallback)):
            # Linear stream — emitted tokens can't be retracted, so close the
            # open block and surface a notice that the partial output above
            # belongs to a superseded attempt. The retry streams fresh below.
            self._end_text()
            self._end_thinking()
            self._print(render_retry_notice(se))
            self._streamed_message = False
            self._streamed_reasoning = False

        elif isinstance(se, OutputItemDone):
            item = se.item
            if isinstance(item, OutputMessageItem):
                self._end_text()
            elif isinstance(item, ReasoningItem) and self.show_thinking:
                self._end_thinking()

    # ── Promoted item events → shared renderer ──

    def _on_output_message(self, event: OutputMessageItemEvent) -> None:
        if self._streamed_message:
            self._streamed_message = False
            return
        self._in_tool_block = False
        self._print(render_event(event, inline_images=False))

    def _on_tool_call_item(self, event: ToolCallItemEvent) -> None:
        self._streamed_tool = False
        self._end_text()
        self._end_thinking()
        self._in_tool_block = True
        if self.show_tool_args:
            self._print(render_event(event, inline_images=False))
        else:
            agent = event.source or "agent"
            tool = event.data.name or "tool"
            color = PALETTE["border_tool_call"]
            self.console.print(f"[bold {color}]{escape(agent)} → {escape(tool)}[/]")

    def _on_reasoning_item(self, event: ReasoningItemEvent) -> None:
        if self._streamed_reasoning:
            self._streamed_reasoning = False
            return
        if not self.show_thinking:
            return
        self._end_text()
        self._print(render_event(event, inline_images=False))

    # ── Lifecycle events ──

    def _on_turn_start(self, event: TurnStartEvent) -> None:
        # Defer: hold the header until this agent's first content of the turn
        # (see _flush_turn). Rendering it now would let a concurrent agent's
        # output stream in between the header and the body it labels.
        self._pending_turns[event.source or ""] = event

    def _flush_turn(self, source: str | None) -> None:
        """Render a deferred turn header, right before its agent's content."""
        event = self._pending_turns.pop(source or "", None)
        if event is None:
            return
        self._end_text()
        self._end_thinking()
        self._in_tool_block = False
        self._print(render_event(event, inline_images=False))
        self.console.print()

    def _on_turn_end(self, event: TurnEndEvent) -> None:
        self._end_text()
        self._end_thinking()
        self._in_tool_block = False
        self._print(render_event(event, inline_images=False))

    def _on_generation_end(self, event: GenerationEndEvent) -> None:
        if not self.show_usage:
            return
        usage = event.data.usage_with_cost
        if not usage:
            return

        color = PALETTE["muted"]
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
            model = event.data.model or ""
            line = f"{model} · {' · '.join(parts)}"
            if self._generation_count > 1 and self._total_cost > 0:
                line += f"  Σ ${self._total_cost:.4f}"
            self.console.print()
            self.console.print(Text(line, style=f"italic {color}"), justify="right")
            self.console.print()

    # ── Message events ──

    def _on_user_message(self, event: UserMessageEvent) -> None:
        self._end_text()
        self._in_tool_block = False
        self._print(render_event(event, inline_images=False))
        self.console.print()

    def _on_system_message(self, event: SystemMessageEvent) -> None:
        self._end_text()
        self._in_tool_block = False
        self._print(render_event(event, inline_images=False))
        self.console.print()

    # ── Tool events ──

    def _on_tool_result(self, event: ToolOutputItemEvent) -> None:
        self._print(render_event(event, inline_images=False))
        self._pending_tool_spacing = True

    def _on_tool_error(self, event: ToolErrorEvent) -> None:
        self.console.print()  # breathing room after usage line
        self._print(render_event(event, inline_images=False))

    # ── Packet events ──

    def _on_packet(self, event: ProcPacketOutEvent | RunPacketOutEvent) -> None:
        self._end_text()
        self._in_tool_block = False
        from pydantic import BaseModel as _BM  # noqa: PLC0415, N814

        color = PALETTE["muted"]
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
            self.console.print(f"[{color}]{escape(truncate(text, self.trunc_len))}[/]")

    # ── Error events ──

    def _on_error(
        self,
        event: LLMStreamingErrorEvent | ProcStreamingErrorEvent,
    ) -> None:
        self._end_text()
        self._in_tool_block = False
        self._print(render_event(event, inline_images=False))

    # ── Background task events ──

    def _on_bg_launched(self, event: BackgroundTaskLaunchedEvent) -> None:
        # Track name so we can suppress the subagent's internal events. The
        # launch itself stays silent — the eventual result panel covers it.
        self._bg_tool_counts[event.data.tool_name] += 1
        self._bg_tool_logs[event.data.tool_name] = event.data.output_name

    def _on_bg_completed(self, event: BackgroundTaskCompletedEvent) -> None:
        # Drop the suppression once the last in-flight task of this tool ends,
        # so a subsequent foreground run of the same tool renders normally.
        name = event.data.tool_name
        count = self._bg_tool_counts.get(name, 0)
        if count <= 1:
            self._bg_tool_counts.pop(name, None)
        else:
            self._bg_tool_counts[name] = count - 1

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

    def _on_tool_stream(self, event: ToolStreamEvent) -> None:
        """Render one incremental tool-output chunk (e.g. live shell stdout)."""
        text = str(event.data).rstrip("\n")
        if not text:
            return
        self._end_text()
        self._end_thinking()
        tool = event.source or "tool"
        # Same renderer as the TUI: a muted "background progress" panel headed by
        # the task's log, so live output isn't mistaken for a tool result.
        self._print(
            render_tool_stream(
                event.destination or "agent",
                tool,
                text,
                background=True,
                log_name=self._bg_tool_logs.get(tool),
            )
        )

    # ── Streaming thinking ──
    #
    # Thinking text is streamed line-by-line (buffered per logical line) using
    # console.print() with a "│" gutter. textwrap.wrap() breaks long lines
    # before the terminal edge so the gutter is always present — even on wrapped
    # continuations. Works identically in terminals and Jupyter.

    def _start_thinking(self) -> None:
        """Open a streaming thinking block."""
        if self._in_thinking:
            return
        self._in_thinking = True
        self._think_line_buf = ""
        color = PALETTE["border_thinking"]
        header = Text("  ┌ ", style=color)
        header.append("thinking", style=f"bold {color}")
        self.console.print(header)
        self.console.print(Text("  │ ", style=color))

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

        text_color = PALETTE["thinking"]
        border_color = PALETTE["border_thinking"]

        if not line:
            self.console.print(Text(gutter, style=border_color))
            return
        wrapped = textwrap.wrap(line, width=max_w) or [line]
        for wl in wrapped:
            header = Text(gutter, style=border_color)
            header.append(Text(wl, style=text_color))
            self.console.print(header)

    def _end_thinking(self) -> None:
        """Close the thinking block."""
        if not self._in_thinking:
            return
        self._in_thinking = False
        if self._think_line_buf:
            self._flush_think_line()
        color = PALETTE["border_thinking"]
        self.console.print(Text("  └", style=color))
        self.console.print()


# ── Convenience wrapper ──


async def stream_events(
    events: AsyncIterator[Event[Any]],
    *,
    show_thinking: bool = False,
    show_tool_args: bool = True,
    show_usage: bool = True,
    show_input_messages: bool = True,
    show_system_messages: bool = False,
    show_packets: bool = False,
    show_background: bool = False,
    trunc_len: int = 10000,
    max_tool_output_lines: int = _MAX_TOOL_OUTPUT_LINES,
    max_input_msg_lines: int = _MAX_INPUT_MSG_LINES,
    console: Console | None = None,
) -> AsyncIterator[Event[Any]]:
    """
    Wrap an event stream with rich console display, yielding events through::

        async for event in stream_events(agent.run_stream("hello", ctx=ctx)):
            ...

    User (input) messages show by default (``show_input_messages``); system
    messages are opt-in (``show_system_messages``). ``show_background=True``
    prints background subagents' internal events inline (off by default — only
    their launch/completion notices and results show).
    """
    ec = EventConsole(
        console=console,
        show_thinking=show_thinking,
        show_tool_args=show_tool_args,
        show_usage=show_usage,
        show_input_messages=show_input_messages,
        show_system_messages=show_system_messages,
        show_packets=show_packets,
        show_background=show_background,
        trunc_len=trunc_len,
        max_tool_output_lines=max_tool_output_lines,
        max_input_msg_lines=max_input_msg_lines,
    )
    async for event in ec.stream(events):
        yield event
