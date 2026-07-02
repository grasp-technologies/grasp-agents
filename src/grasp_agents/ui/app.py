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

import contextlib
import hashlib
import re
from typing import TYPE_CHECKING, Any, ClassVar

from rich.align import Align
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import (
    ContentSwitcher,
    Footer,
    Header,
    OptionList,
    Static,
    Tab,
    Tabs,
    TextArea,
)
from textual.worker import Worker, WorkerState

from grasp_agents.llm.model_info import get_context_window
from grasp_agents.skills import parse_slash_command
from grasp_agents.types.content import InputImage
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskLaunchedEvent,
    CompactionEvent,
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
    WebSearchCallItemEvent,
)
from grasp_agents.types.items import WebSearchCallItem
from grasp_agents.types.llm_events import (
    OutputItemDone,
    OutputMessageTextPartTextDelta,
    ReasoningContentPartTextDelta,
    ReasoningSummaryPartTextDelta,
    ResponseFallback,
    ResponseRetrying,
)

from ._approval import ApprovalScreen, TuiApprovalStore
from ._event_render import (
    PALETTE,
    event_images,
    render_event,
    render_image,
    render_input_image,
    render_retry_notice,
    render_thinking_stream,
    render_tool_stream,
    render_turn_rule,
    render_web_search,
    set_markup_theme,
    usage_line,
)
from ._images import ImageZoomScreen, prime_image_protocol
from ._theme import (
    DEFAULT_THEME,
    GRASP_DARK,
    GRASP_LIGHT,
    load_saved_theme,
    save_theme,
)
from ._widgets import (
    PromptArea,
    RollbackScreen,
    SelectableStatic,
    SkillPalette,
    ZoomableImage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from rich.console import RenderableType
    from textual.theme import Theme
    from textual.widget import Widget

    from grasp_agents.agent.llm_agent import LLMAgent
    from grasp_agents.session_context import SessionContext

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
    WebSearchCallItemEvent,
)

_GLYPH: dict[str, str] = {
    "working": "●",
    "done": "✓",
    "error": "✗",
    "idle": "○",
}

# Reserved prompt command (interactive): opens the rollback picker instead of
# being sent to the agent. Also bound to ctrl+r.
_ROLLBACK_COMMAND = "/rollback"


def _slug(source: str) -> str:
    base = re.sub(r"[^0-9A-Za-z_-]+", "-", source).strip("-").lower() or "x"
    # Distinct sources can collapse to the same base ("worker 1" vs "worker-1");
    # append a stable digest so the derived DOM ids stay unique (Textual rejects
    # duplicate ids).
    digest = hashlib.md5(source.encode()).hexdigest()[:6]  # noqa: S324
    return f"{base}-{digest}"


def _pane_id(source: str) -> str:
    return "pane-" + _slug(source)


def _tab_id(source: str) -> str:
    return "tab-" + _slug(source)


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
    #context-meter { width: 1fr; height: auto; padding: 0 3; margin-top: 1; }
    #skill-palette {
        height: auto; max-height: 10; margin: 0 3; padding: 0 1;
        border: round $accent; background: $surface;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Quit"),
        ("f", "toggle_follow", "Follow latest"),
        ("escape", "interrupt", "Interrupt run"),
    ]

    def __init__(
        self,
        events: AsyncIterator[Event[Any]] | None = None,
        *,
        on_submit: Callable[[str], AsyncIterator[Event[Any]]] | None = None,
        on_rollback: Callable[[int], Awaitable[None]] | None = None,
        main_agent: str | None = None,
        ctx: SessionContext[Any] | None = None,
    ) -> None:
        super().__init__()
        prime_image_protocol()
        self._ga_events = events
        self._ga_on_submit = on_submit
        # Rollback callback (interactive only): given the 0-based index of a prior
        # user message, the host rewinds the agent to that point (it owns the
        # step↔message mapping). Enables the /rollback command and ctrl+r.
        self._ga_on_rollback = on_rollback
        self._ga_main = main_agent
        # Skill palette (interactive mode only): typing "/" opens a filtered
        # picker; selecting one unwraps the skill into a user-message turn. The
        # skills come from the session ``ctx`` — the same registry the agent
        # uses — never a separate copy that could drift from it.
        skills = ctx.skills if ctx is not None else None
        self._ga_skills = skills
        self._ga_palette: SkillPalette | None = (
            SkillPalette(skills.all)
            if on_submit is not None and skills is not None
            else None
        )
        self._ga_last_agent: str | None = None
        self._ga_agents: set[str] = set()
        self._ga_turns: dict[str, int] = {}
        # owner → deferred turn number: the header is mounted right before that
        # agent's first content of the turn (see _flush_turn), not at turn start,
        # so a co-tenant of the pane (e.g. parallel replicas sharing a source
        # name) can't wedge output between the header and the body it labels.
        self._ga_pending_turn: dict[str, int] = {}
        self._ga_last_kind: dict[str, str] = {}
        self._ga_panes: dict[str, VerticalScroll] = {}
        self._ga_tab_source: dict[str, str] = {}
        self._ga_status: dict[str, str] = {}
        self._ga_parent: dict[str, str] = {}
        self._ga_follow = False
        self._ga_worker: Worker[None] | None = None
        # the worker driving the current interactive turn; cancelled by Esc
        self._ga_run_worker: Worker[None] | None = None
        self._ga_running = False
        # Approval gate (interactive mode): when the session ctx carries a
        # TuiApprovalStore, a worker drains its pending queue and pops a dialog
        # per gated tool call (see _consume_approvals). Any other ApprovalStore
        # still works — it just won't drive the dialog.
        store = ctx.approval_store if ctx is not None else None
        self._ga_approval_store = store if isinstance(store, TuiApprovalStore) else None
        # Session-wide cost roll-up: each generation shows its own cost,
        # plus a running Σ total once more than one generation has cost.
        self._ga_total_cost = 0.0
        self._ga_gen_count = 0
        # live-streaming widgets per owner (LLM tokens / reasoning / tool output),
        # finalised by the matching item event; empty when the agent isn't streaming
        self._ga_stream_msg: dict[str, SelectableStatic] = {}
        self._ga_stream_msg_text: dict[str, str] = {}
        self._ga_stream_think: dict[str, SelectableStatic] = {}
        self._ga_stream_think_text: dict[str, str] = {}
        # item id of each owner's in-flight reasoning, so the widget can be sealed
        # (and its promoted event suppressed) when a web search interleaves.
        self._ga_stream_think_id: dict[str, str] = {}
        # ids of reasoning / web-search items already rendered live during the
        # current generation (their widget was sealed when a search interleaved);
        # the promoted item event for the same id is then suppressed. Reset per
        # generation.
        self._ga_streamed_ids: set[str] = set()
        self._ga_stream_tool: dict[str, SelectableStatic] = {}
        self._ga_stream_tool_text: dict[str, str] = {}
        # name of the tool currently owning each pane's live tool-output widget,
        # so a different tool's stream starts its own widget instead of writing
        # into the previous one (which a backgrounded tool can leave un-finalised)
        self._ga_stream_tool_name: dict[str, str] = {}
        # tools whose work is currently running in the background (between their
        # launch and completion), mapped to their log basename (None if no log is
        # written). Their streamed output is live progress mirrored to that log,
        # not a result the agent sees — rendered distinctly, headed by the log.
        self._ga_bg_tools: dict[str, str | None] = {}
        # Context-token meter: the last reported input-token count + model per
        # agent (the running context size), shown for the active pane's agent
        # against its window. The window is inferred — learned from
        # CompactionEvents (the agent's managed budget), else the model's window;
        # ``run_tui_interactive`` seeds it from the agent up front.
        self._ga_input_tokens: dict[str, int] = {}
        self._ga_token_model: dict[str, str] = {}
        self._ga_token_window: dict[str, int] = {}
        self._ga_window_cache: dict[str, int | None] = {}
        self._ga_active_source: str | None = None
        # Rollback bookkeeping (interactive): one entry per user submission, in
        # order — its label and the main pane's child count at the time (the
        # rewind point, so a rollback can drop that turn's widgets and below).
        self._ga_user_turns: list[str] = []
        self._ga_turn_marks: list[int] = []
        # the displayed monotonic turn count at each submission, so a rollback
        # rewinds the counter to its value there (not the running maximum)
        self._ga_turn_counts: list[int] = []
        self._ga_rollback_open = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield ContentSwitcher(id="panes")
        if self._ga_on_submit is not None:
            if self._ga_palette is not None:
                yield self._ga_palette
            yield Static("", id="context-meter")
            with Horizontal(id="prompt-bar"):
                yield Static("❯", id="prompt-arrow")  # noqa: RUF001
                yield PromptArea(id="prompt")
        yield Tabs(id="agents")
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(GRASP_DARK)
        self.register_theme(GRASP_LIGHT)
        # restore the last-used theme (persisted across launches) instead of
        # resetting every time; default to Catppuccin Macchiato
        saved = load_saved_theme()
        self.theme = (
            saved if saved and self.get_theme(saved) is not None else DEFAULT_THEME
        )
        self._apply_markup_theme()
        self.theme_changed_signal.subscribe(self, self._on_theme_changed)
        if self._ga_events is not None:
            self._ga_worker = self._consume()
        if self._ga_on_submit is not None:
            # Interactive only — bound here (not in class BINDINGS) so monitor
            # mode shows no dead rollback key in the footer.
            self.bind("ctrl+r", "rollback", description="Rollback")
            self.query_one("#prompt", PromptArea).focus()
        if self._ga_approval_store is not None:
            self._consume_approvals()

    def _on_theme_changed(self, theme: Theme) -> None:
        save_theme(theme.name)  # persist the user's choice across launches
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

    @on(PromptArea.Submitted)
    def _on_prompt_submit(self, event: PromptArea.Submitted) -> None:
        text = event.value.strip()
        if not text or self._ga_on_submit is None:
            return
        self._hide_palette()
        if text == _ROLLBACK_COMMAND:
            prompt = self.query_one("#prompt", PromptArea)
            prompt.text = ""
            prompt.sync_height()
            self._open_rollback()
            return
        self._record_turn(text)
        self._dispatch(self._unwrap_slash_command(text))

    # ── skill palette (slash commands) ──

    @on(TextArea.Changed, "#prompt")
    def _on_prompt_changed(self, event: TextArea.Changed) -> None:
        self._update_palette(event.text_area.text)

    @on(PromptArea.SkillChosen)
    def _on_skill_chosen(self, event: PromptArea.SkillChosen) -> None:
        self._insert_skill(event.name)

    @on(OptionList.OptionSelected, "#skill-palette")
    def _on_skill_clicked(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            self._insert_skill(event.option.id)

    def _update_palette(self, text: str) -> None:
        """
        Show/filter the palette while the user is typing the command *name*.

        Once a space (or newline) follows — i.e. the user is typing arguments —
        the palette gets out of the way so the prompt is free to edit.
        """
        palette = self._ga_palette
        if palette is None:
            return
        if (
            text.startswith("/")
            and not any(c in text for c in " \t\n")
            and palette.filter_to(text[1:])
        ):
            palette.display = True
            return
        palette.display = False

    def _hide_palette(self) -> None:
        if self._ga_palette is not None:
            self._ga_palette.display = False

    def _insert_skill(self, name: str) -> None:
        """
        Insert ``/name `` into the prompt — selecting a skill does NOT submit.

        It fills in the command and leaves the cursor after it, ready for
        arguments; the trailing space also dismisses the palette. Submitting
        (Enter) then unwraps it (see :meth:`_unwrap_slash_command`).
        """
        if self._ga_skills is None or self._ga_skills.get_optional(name) is None:
            self._hide_palette()
            return
        prompt = self.query_one("#prompt", PromptArea)
        prompt.text = f"/{name} "
        prompt.sync_height()
        self._hide_palette()
        prompt.focus()
        prompt.move_cursor(prompt.document.end)

    def _unwrap_slash_command(self, text: str) -> str:
        """Expand a ``/name args`` line into the skill's wrapped body (else as-is)."""
        if self._ga_skills is None:
            return text
        parsed = parse_slash_command(text)
        if parsed is None or self._ga_skills.get_optional(parsed.name) is None:
            return text
        return self._ga_skills.render_invocation(
            parsed.name, args=parsed.args or None, wrap=True
        )

    def _dispatch(self, text: str) -> None:
        prompt = self.query_one("#prompt", PromptArea)
        prompt.text = ""
        prompt.sync_height()
        prompt.disabled = True
        self._ga_run_worker = self._run_turn(text)

    @work
    async def _run_turn(self, text: str) -> None:
        if self._ga_on_submit is None:
            return
        self._ga_running = True
        stream = self._ga_on_submit(text)
        try:
            async for event in stream:
                await self._feed(event)
        finally:
            self._ga_running = False
            # Re-enable the prompt (suppress NoMatches: the app may be tearing
            # down — e.g. quit pressed mid-run — and the widget already gone).
            with contextlib.suppress(NoMatches):
                prompt = self.query_one("#prompt", PromptArea)
                prompt.disabled = False
                prompt.focus()
            # Close the agent generator so its per-run cleanup runs even when
            # the turn was interrupted — Esc cancels this worker, raising
            # CancelledError out of the loop above. Re-enable the prompt first,
            # since aclose() may itself be cancelled during teardown.
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()

    # ── approval gate ──

    @work(exclusive=False)
    async def _consume_approvals(self) -> None:
        """Drain pending approvals, popping a dialog for each and resolving it."""
        store = self._ga_approval_store
        if store is None:
            return
        while True:
            pending = await store.pending_events.get()
            # Skip if it's no longer pending — already resolved by a prior
            # session decision, timed out, or cleared when the run was
            # interrupted (the gate hook denies the batch on cancellation).
            still = await store.list_pending(pending.session_key)
            if not any(p.call_id == pending.call_id for p in still):
                continue
            decision = await self.push_screen_wait(ApprovalScreen(pending))
            await store.resolve(pending.session_key, pending.call_id, decision)

    async def _feed(self, event: Event[Any]) -> None:
        self._record_edge(event)
        owner = self._owner(event)
        pane = await self._ensure(owner)
        # auto-scroll only when already at the bottom, so streaming content never
        # yanks the user down while they read scrolled-up history
        at_bottom = pane.scroll_offset.y >= pane.max_scroll_y - 1

        # Mount a deferred turn header right before this turn's first output —
        # content or an error (see _feed_item / _flush_turn) — so it isn't
        # separated from what it labels. A turn that produces neither drops it.
        if isinstance(
            event,
            (
                LLMStreamEvent,
                OutputMessageItemEvent,
                ToolCallItemEvent,
                ReasoningItemEvent,
                WebSearchCallItemEvent,
                LLMStreamingErrorEvent,
                ProcStreamingErrorEvent,
            ),
        ):
            await self._flush_turn(owner, pane, at_bottom)

        # live token / tool-output streaming when the agent streams; if it isn't
        # (or once an item completes) fall through to render the finished item
        if not await self._feed_streaming(event, owner, pane, at_bottom):
            await self._feed_item(event, owner, pane, at_bottom)

        self._update_status(event, owner)
        self._track_context(event, owner)
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
        if (
            isinstance(event, (ReasoningItemEvent, WebSearchCallItemEvent))
            and event.data.id in self._ga_streamed_ids
        ):
            # Already rendered live (sealed when a search interleaved) — drop the
            # redundant promoted event.
            return True
        if isinstance(event, LLMStreamEvent):
            data = event.data
            if isinstance(data, OutputMessageTextPartTextDelta) and data.delta:
                await self._stream_message(owner, pane, data.delta, at_bottom)
            elif (
                isinstance(
                    data,
                    (ReasoningContentPartTextDelta, ReasoningSummaryPartTextDelta),
                )
                and data.delta
            ):
                await self._stream_thinking(
                    owner, pane, data.delta, data.item_id, at_bottom
                )
            elif isinstance(data, OutputItemDone) and isinstance(
                data.item, WebSearchCallItem
            ):
                # Render each search as it completes, so several searches
                # interleave with the reasoning between them instead of batching
                # at the end (the promoted event fires only once the stream is
                # written).
                await self._render_web_search_live(owner, pane, data.item, at_bottom)
            elif isinstance(data, (ResponseRetrying, ResponseFallback)):
                # Failed attempt / model fallback — drop the partial widgets and
                # surface a notice so the cleared text isn't silently lost; the
                # retry streams a fresh widget below it.
                await self._discard_streamed_message(owner)
                self._ga_streamed_ids.clear()
                self._ga_stream_think_id.pop(owner, None)
                await pane.mount(
                    SelectableStatic(render_retry_notice(data), classes="ga-notice")
                )
                if at_bottom:
                    pane.scroll_end(animate=False)
            return True
        if isinstance(event, ToolStreamEvent):
            await self._stream_tool(
                owner, pane, event.source or "tool", str(event.data), at_bottom
            )
            return True
        if isinstance(event, ReasoningItemEvent) and owner in self._ga_stream_think:
            self._finalize_thinking(owner, event, pane, at_bottom)
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
            widget = SelectableStatic(Text(text), classes="ga-msg")
            self._ga_stream_msg[owner] = widget
            self._ga_last_kind[owner] = "text"
            await pane.mount(widget)
        else:
            widget.update(Text(text))
        if at_bottom:
            pane.scroll_end(animate=False)

    async def _stream_thinking(
        self,
        owner: str,
        pane: VerticalScroll,
        delta: str,
        item_id: str,
        at_bottom: bool,
    ) -> None:
        self._ga_stream_think_id[owner] = item_id
        text = self._ga_stream_think_text.get(owner, "") + delta
        self._ga_stream_think_text[owner] = text
        rend = render_thinking_stream(text)
        widget = self._ga_stream_think.get(owner)
        if widget is None:
            widget = SelectableStatic(rend, classes="ga-msg")
            self._ga_stream_think[owner] = widget
            self._ga_last_kind[owner] = "box"
            await pane.mount(widget)
        else:
            widget.update(rend)
        if at_bottom:
            pane.scroll_end(animate=False)

    async def _render_web_search_live(
        self,
        owner: str,
        pane: VerticalScroll,
        item: WebSearchCallItem,
        at_bottom: bool,
    ) -> None:
        # Seal any open thinking widget so the search panel sits below it (and the
        # next reasoning starts a fresh widget), and suppress that reasoning's now-
        # redundant promoted event by id. The sealed streamed gutter is byte-for-
        # byte what the promoted event would render, so nothing is lost.
        if owner in self._ga_stream_think:
            self._ga_stream_think.pop(owner, None)
            self._ga_stream_think_text.pop(owner, None)
            think_id = self._ga_stream_think_id.pop(owner, None)
            if think_id:
                self._ga_streamed_ids.add(think_id)
        self._ga_streamed_ids.add(item.id)
        self._ga_last_kind[owner] = "box"
        await pane.mount(
            SelectableStatic(render_web_search(item, owner), classes="ga-msg")
        )
        if at_bottom:
            pane.scroll_end(animate=False)

    async def _discard_streamed_message(self, owner: str) -> None:
        """
        Drop *owner*'s in-progress streamed message + reasoning on
        ``ResponseRetrying``: their partial content belongs to a failed attempt.
        The retry streams fresh widgets.
        """
        self._ga_stream_msg_text.pop(owner, None)
        widget = self._ga_stream_msg.pop(owner, None)
        if widget is not None:
            await widget.remove()
        self._ga_stream_think_text.pop(owner, None)
        think = self._ga_stream_think.pop(owner, None)
        if think is not None:
            await think.remove()

    async def _stream_tool(
        self, owner: str, pane: VerticalScroll, tool: str, delta: str, at_bottom: bool
    ) -> None:
        # A stream from a different tool than the one currently held means the
        # previous tool's output is finished (e.g. a backgrounded tool whose
        # drained output never got a finalising item event) — seal it so this
        # tool's output starts in its own widget rather than overwriting the old
        # one in place, above the new tool's call.
        if (
            owner in self._ga_stream_tool
            and self._ga_stream_tool_name.get(owner) != tool
        ):
            self._seal_stream_tool(owner)
        text = self._ga_stream_tool_text.get(owner, "") + delta
        self._ga_stream_tool_text[owner] = text
        rend = Align.right(
            render_tool_stream(
                owner,
                tool,
                text,
                background=tool in self._ga_bg_tools,
                log_name=self._ga_bg_tools.get(tool),
            )
        )
        widget = self._ga_stream_tool.get(owner)
        if widget is None:
            widget = SelectableStatic(rend, classes="ga-msg")
            self._ga_stream_tool[owner] = widget
            self._ga_stream_tool_name[owner] = tool
            self._ga_last_kind[owner] = "box"
            await pane.mount(widget)
        else:
            widget.update(rend)
        if at_bottom:
            pane.scroll_end(animate=False)

    def _seal_stream_tool(self, owner: str) -> None:
        """
        Stop tracking *owner*'s live tool-output widget, leaving it mounted as-is.

        Called when a streaming episode ends without a finalising
        ``ToolOutputItemEvent`` — e.g. a backgrounded tool whose drained output
        terminates with a background-completion notice. The next tool's stream
        then gets a fresh widget in its correct position instead of overwriting
        this one in place.
        """
        self._ga_stream_tool.pop(owner, None)
        self._ga_stream_tool_text.pop(owner, None)
        self._ga_stream_tool_name.pop(owner, None)

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

    def _finalize_thinking(
        self,
        owner: str,
        event: ReasoningItemEvent,
        pane: VerticalScroll,
        at_bottom: bool,
    ) -> None:
        widget = self._ga_stream_think.pop(owner)
        self._ga_stream_think_text.pop(owner, None)
        final = render_event(event, inline_images=False)
        # swap to the canonical panel unless the finalised item carries no summary
        # text (the "thinking…" placeholder), which would clobber streamed tokens
        has_text = any(getattr(p, "text", "") for p in (event.data.summary_parts or []))
        if final is not None and has_text:
            widget.update(final)
        if at_bottom:
            pane.scroll_end(animate=False)

    async def _finalize_tool(
        self, owner: str, event: Event[Any], pane: VerticalScroll, at_bottom: bool
    ) -> None:
        widget = self._ga_stream_tool.pop(owner)
        self._ga_stream_tool_text.pop(owner, None)
        self._ga_stream_tool_name.pop(owner, None)
        final = render_event(event, inline_images=False)
        if final is not None:
            widget.update(Align.right(final))
        for img in event_images(event):
            await pane.mount(self._image_widget(img))
        if at_bottom:
            pane.scroll_end(animate=False)

    def _usage_text(self, event: GenerationEndEvent) -> RenderableType | None:
        """Per-request usage line + a running session Σ total, like the console."""
        usage = event.data.usage_with_cost
        if usage:
            self._ga_gen_count += 1
            if usage.cost:
                self._ga_total_cost += usage.cost
        return usage_line(
            event,
            total_cost=self._ga_total_cost,
            generation_count=self._ga_gen_count,
        )

    # ── context-token meter ──

    def _track_context(self, event: Event[Any], owner: str) -> None:
        """Update an agent's running input-token count from its events."""
        if isinstance(event, GenerationEndEvent):
            usage = event.data.usage_with_cost
            if not usage or not usage.input_tokens:
                return
            self._ga_input_tokens[owner] = usage.input_tokens
            if event.data.model:
                self._ga_token_model[owner] = event.data.model
        elif isinstance(event, CompactionEvent):
            # the post-fold view size — reflect the reduced context at once
            self._ga_input_tokens[owner] = event.data.context_tokens
            if event.data.context_window:
                self._ga_token_window[owner] = event.data.context_window
        else:
            return
        if owner == self._ga_active_source:
            self._refresh_meter()

    def _window_for(self, source: str) -> int | None:
        """Window to meter ``source`` against: learned (budget/event) → model."""
        learned = self._ga_token_window.get(source)
        if learned is not None:
            return learned
        model = self._ga_token_model.get(source)
        if model is None:
            return None
        if model not in self._ga_window_cache:
            self._ga_window_cache[model] = get_context_window(model)
        return self._ga_window_cache[model]

    def seed_context_window(self, source: str, window: int) -> None:
        """
        Prime the token meter's window for ``source`` before its first event.

        :func:`run_tui_interactive` calls this with the agent's managed context
        window so the meter shows it from the start; afterwards the window is
        learned from CompactionEvents / the model (see :meth:`_window_for`).
        """
        self._ga_token_window[source] = window

    def _refresh_meter(self) -> None:
        try:
            meter = self.query_one("#context-meter", Static)
        except NoMatches:
            return
        source = self._ga_active_source
        tokens = self._ga_input_tokens.get(source, 0) if source else 0
        if not tokens or source is None:
            # Collapse the (margin-bearing) widget entirely so it leaves no gap
            # above the prompt before the first generation.
            meter.display = False
            meter.update("")
            return
        meter.display = True
        meter.update(self._meter_text(tokens, self._window_for(source)))

    def _meter_text(self, tokens: int, window: int | None) -> Text:
        if window and window > 0:
            pct = min(100, round(tokens / window * 100))
            if pct >= 85:
                color = f"bold {PALETTE['error']}"
            elif pct >= 65:
                color = PALETTE["warn"]
            else:
                color = PALETTE["usage"]
            body = f"context: {tokens:,} / {window:,} tokens ({pct}%)"
        else:
            color = PALETTE["usage"]
            body = f"context: {tokens:,} tokens"
        return Text(body, style=color, justify="right")

    async def _flush_turn(
        self, owner: str, pane: VerticalScroll, at_bottom: bool
    ) -> None:
        """Mount a deferred turn header, right before its agent's content."""
        turn = self._ga_pending_turn.pop(owner, None)
        if turn is None:
            return
        self._ga_last_kind[owner] = "turn"
        await pane.mount(
            SelectableStatic(render_turn_rule(owner, turn), classes="ga-turn")
        )
        if at_bottom:
            pane.scroll_end(animate=False)

    async def _feed_item(
        self, event: Event[Any], owner: str, pane: VerticalScroll, at_bottom: bool
    ) -> None:
        if isinstance(event, ToolCallItemEvent):
            # A new tool call ends any prior live tool-output stream for this
            # owner (a backgrounded tool's drained stream never gets a finalising
            # item event); seal it so this call and the next tool's output land
            # below it, not folded into the leaked widget above.
            self._seal_stream_tool(owner)
        if isinstance(event, TurnStartEvent):
            # monotonic per-agent turn count — the loop's own turn resets to 0
            # each step, so two consecutive turns can both read turn=0. Defer the
            # header: _flush_turn mounts it right before this turn's first
            # content (see _feed).
            self._ga_turns[owner] = self._ga_turns.get(owner, 0) + 1
            self._ga_pending_turn[owner] = self._ga_turns[owner]
            return
        if isinstance(event, GenerationEndEvent):
            # this generation's promoted item events have all been handled; reset
            # the streamed-id set so the next generation starts fresh.
            self._ga_streamed_ids.clear()
            text = self._usage_text(event)
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
                event,
                (
                    UserMessageEvent,
                    SystemMessageEvent,
                    ToolOutputItemEvent,
                    CompactionEvent,
                ),
            ):
                # incoming-message border to the right (the compaction summary is
                # injected into the agent's context); text inside the panel stays left
                text = Align.right(text)
            await pane.mount(SelectableStatic(text, classes=cls))
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
        return ZoomableImage(src, Align.right(rend))

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
            # its subsequent streamed output is background progress, not a result
            self._ga_bg_tools[event.data.tool_name] = event.data.output_name
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
        first = switcher.current is None
        await switcher.add_content(pane, set_current=first)
        self._ga_panes[source] = pane
        if first:  # the first pane is the one shown — track it for the meter
            self._ga_active_source = source
            self._refresh_meter()

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
        self._ga_active_source = source
        self._refresh_meter()

    @on(Tabs.TabActivated)
    def _on_tab_activated(self, event: Tabs.TabActivated) -> None:
        source = self._ga_tab_source.get(event.tab.id or "")
        if source is not None:
            self.query_one("#panes", ContentSwitcher).current = _pane_id(source)
            self._ga_active_source = source
            self._refresh_meter()

    @on(ZoomableImage.Zoom)
    def _on_image_zoom(self, event: ZoomableImage.Zoom) -> None:
        self.push_screen(ImageZoomScreen(event.src))

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
            self._ga_bg_tools.pop(event.data.tool_name, None)
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

    # ── rollback ──

    def _record_turn(self, label: str) -> None:
        """Remember a user submission + the main pane's rewind point for it."""
        main = self._ga_main
        pane = self._ga_panes.get(main) if main else None
        self._ga_user_turns.append(label)
        self._ga_turn_marks.append(len(pane.children) if pane is not None else 0)
        self._ga_turn_counts.append(self._ga_turns.get(main, 0) if main else 0)

    def action_rollback(self) -> None:
        self._open_rollback()

    @work
    async def _open_rollback(self) -> None:
        """Pop the rollback picker and rewind to the chosen message."""
        if self._ga_on_rollback is None:
            self.notify("Rollback isn't available in this session.")
            return
        if self._ga_running:
            self.notify("Can't roll back while the agent is running.")
            return
        if not self._ga_user_turns:
            self.notify("No messages to roll back to yet.")
            return
        if self._ga_rollback_open:
            return
        self._ga_rollback_open = True
        try:
            index = await self.push_screen_wait(
                RollbackScreen(list(self._ga_user_turns))
            )
        finally:
            self._ga_rollback_open = False
        if index is not None:
            await self._do_rollback(index)

    async def _do_rollback(self, index: int) -> None:
        callback = self._ga_on_rollback
        if callback is None or not (0 <= index < len(self._ga_user_turns)):
            return
        message = self._ga_user_turns[index]  # capture before truncation drops it
        try:
            await callback(index)
        except Exception as exc:
            self.notify(f"Rollback failed: {exc}", severity="error")
            return
        await self._truncate_to_turn(index)
        # Hand the rolled-back message back in the prompt, ready to edit + resend.
        self._prefill_prompt(message)
        self.notify(f"Rolled back to message #{index + 1}.")

    def _prefill_prompt(self, text: str) -> None:
        try:
            prompt = self.query_one("#prompt", PromptArea)
        except NoMatches:
            return
        prompt.text = text
        prompt.sync_height()
        prompt.move_cursor(prompt.document.end)
        prompt.focus()

    async def _truncate_to_turn(self, index: int) -> None:
        """Drop the chosen turn's widgets (and everything after) + reset the meter."""
        main = self._ga_main
        pane = self._ga_panes.get(main) if main else None
        if pane is not None and index < len(self._ga_turn_marks):
            for child in list(pane.children)[self._ga_turn_marks[index] :]:
                await child.remove()
        if main is not None and index < len(self._ga_turn_counts):
            # rewind the monotonic turn counter so the next run's turns continue
            # from the rolled-back point instead of climbing from the old total
            self._ga_turns[main] = self._ga_turn_counts[index]
        self._ga_user_turns = self._ga_user_turns[:index]
        self._ga_turn_marks = self._ga_turn_marks[:index]
        self._ga_turn_counts = self._ga_turn_counts[:index]
        if main is not None:
            # the rolled-back generations no longer describe the live context size
            self._ga_input_tokens.pop(main, None)
            self._refresh_meter()

    async def action_interrupt(self) -> None:
        """
        Cancel the in-flight interactive turn (``esc``).

        A no-op when nothing is running, or when a modal (e.g. the approval
        dialog) is open — that screen handles ``esc`` itself. Cancelling the
        turn worker raises ``CancelledError`` through the agent's stream, which
        closes it (see :meth:`_run_turn`).
        """
        worker = self._ga_run_worker
        if not self._ga_running or worker is None:
            return
        if worker.state in {WorkerState.PENDING, WorkerState.RUNNING}:
            worker.cancel()
        owner = self._ga_last_agent or self._ga_main
        if owner and owner in self._ga_panes:
            pane = self._ga_panes[owner]
            await pane.mount(
                SelectableStatic(
                    Text("⊘ interrupted", style="italic"), classes="ga-msg"
                )
            )
            pane.scroll_end(animate=False)


def run_tui(events: AsyncIterator[Event[Any]]) -> None:
    """Launch the TUI over an event stream (blocks until the user quits)."""
    GraspAgentsApp(events).run()


class _AgentTurns:
    """
    Drives an agent from the interactive callbacks.

    Each submitted message runs as its own step (``step=0, 1, 2, …``), recording
    a rewind boundary, so the rollback picker can return to a previous message
    via :meth:`LLMAgent.rollback_to_step`. ``on_rollback(i)`` rewinds to step
    ``i`` and re-parks the counter there, so the next message re-delivers it —
    matching the 0-based index the TUI assigns the i-th submission.
    """

    def __init__(self, agent: LLMAgent[Any, Any, Any]) -> None:
        self._agent = agent
        self._step = 0

    async def on_submit(self, text: str) -> AsyncIterator[Event[Any]]:
        step = self._step
        self._step += 1
        async for event in self._agent.run_stream(text, step=step):
            yield event

    async def on_rollback(self, index: int) -> None:
        await self._agent.rollback_to_step(index)
        self._step = index


def run_tui_interactive(
    agent: LLMAgent[Any, Any, Any] | None = None,
    *,
    on_submit: Callable[[str], AsyncIterator[Event[Any]]] | None = None,
    on_rollback: Callable[[int], Awaitable[None]] | None = None,
    main_agent: str | None = None,
    ctx: SessionContext[Any] | None = None,
    events: AsyncIterator[Event[Any]] | None = None,
) -> None:
    """
    Interactive TUI: type a message, the agent runs, events stream into panes.

    Pass an :class:`~grasp_agents.agent.LLMAgent` (recommended) and everything is
    inferred from it: each message runs as its own step so ``/rollback`` (or
    ``ctrl+r``) can rewind to a previous message, the slash-command palette picks
    up ``agent.ctx.skills``, and the token meter tracks the agent's context
    window (its compaction budget).

    For custom per-turn orchestration — a multi-agent ``Runner``, input
    pre/post-processing, calling several agents — pass ``on_submit`` instead (a
    coroutine ``(text) -> AsyncIterator[Event]``), with optional ``on_rollback``
    (rewind to a message's 0-based index), ``main_agent``, and ``ctx``. Either
    overrides the value inferred from ``agent`` when both are given.

    Pass ``events`` to also render a background event stream concurrently with
    human turns — e.g. a team member reacting to its mailbox while you type. Its
    events route to panes by ``source`` like any other.
    """
    window: int | None = None
    if agent is not None:
        turns = _AgentTurns(agent)
        if on_submit is None:
            on_submit = turns.on_submit
        if on_rollback is None and hasattr(agent, "rollback_to_step"):
            on_rollback = turns.on_rollback
        if main_agent is None:
            main_agent = agent.name
        if ctx is None:
            ctx = agent.ctx
        window = getattr(agent, "context_window", None)
    if on_submit is None:
        raise ValueError(
            "run_tui_interactive needs either an `agent` or an `on_submit` callback."
        )
    app = GraspAgentsApp(
        events=events,
        on_submit=on_submit,
        on_rollback=on_rollback,
        main_agent=main_agent,
        ctx=ctx,
    )
    if window and main_agent:
        # Seed the meter against the agent's managed context window; later updates
        # come from CompactionEvents / the model (see ``_window_for``).
        app.seed_context_window(main_agent, window)
    app.run()
