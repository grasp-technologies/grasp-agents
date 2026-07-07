"""
Relaunch-time session restore for the Textual app.

Mixed into :class:`~.app.GraspAgentsApp`: when its session ctx carries a
checkpoint store, :meth:`SessionRestoreMixin._restore_session` runs once at
mount — before the live event stream starts — and rebuilds the panes from the
persisted session. Reading is side-effect-free (``durability.session_history``);
rendering goes through the app's normal event path via :mod:`.replay`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from rich.rule import Rule
from rich.text import Text

from grasp_agents.durability.session_history import (
    read_agent_histories,
    read_pending_messages,
    read_task_records,
)
from grasp_agents.durability.task_record import TaskStatus
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskInfo,
    BackgroundTaskLaunchedEvent,
)
from grasp_agents.types.message import USER_SENDER

from ._event_render import PALETTE
from ._widgets import SelectableStatic
from .replay import history_events

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import Any

    from textual.containers import VerticalScroll
    from textual.notifications import SeverityLevel

    from grasp_agents.durability.task_record import TaskRecord
    from grasp_agents.processors.processor import Processor
    from grasp_agents.session_context import SessionContext
    from grasp_agents.types.events import Event
    from grasp_agents.types.items import InputItem

# Restore caps: only the tail of each transcript / task log is replayed at
# relaunch, so a long-lived session doesn't stall the launch.
_RESTORE_MAX_ITEMS = 200
_RESTORE_MAX_LOG_CHARS = 10_000


class SessionRestoreMixin:
    """Session-restore half of the app (see module docstring)."""

    if TYPE_CHECKING:
        # The host seam: state + methods GraspAgentsApp provides.
        _ga_ctx: SessionContext[Any] | None
        _ga_roster: list[str]
        _ga_members: dict[str, Processor[Any, Any, Any]]
        _ga_on_post: Callable[[str], Awaitable[None]] | None
        _ga_post_seq: int
        _ga_queued: list[tuple[int, str]]
        _ga_panes: dict[str, VerticalScroll]
        _ga_last_kind: dict[str, str]
        _ga_task_panes: dict[tuple[str, str], str]

        def notify(
            self,
            message: str,
            *,
            title: str = "",
            severity: SeverityLevel = "information",
            timeout: float | None = None,
            markup: bool = True,
        ) -> None: ...
        def _set_status(self, source: str, status: str) -> None: ...
        def _refresh_queue_strip(self) -> None: ...
        async def _ensure(
            self, source: str, *, status: str = "working"
        ) -> VerticalScroll: ...
        async def _feed(self, event: Event[Any]) -> None: ...
        async def _open_task_pane(self, event: BackgroundTaskLaunchedEvent) -> None: ...
        async def _close_task_pane(
            self,
            event: BackgroundTaskCompletedEvent,
            *,
            note: str = "✓ completed",
            failed: bool = False,
        ) -> None: ...
        async def _stream_task_log(
            self, owner: str, pane: VerticalScroll, delta: str, at_bottom: bool
        ) -> None: ...

    async def _restore_session(self) -> None:
        """
        Rebuild the panes from the session's persisted history.

        Reads — never mutates — the ctx checkpoint store: each agent's
        committed transcript replays through the normal event path (closed
        with a "restored session" rule), each background task's pane is
        rebuilt from its record + mirrored log, and still-pending human mail
        re-fills the queued strip. What was never persisted — per-generation
        usage, streaming partials, the ephemeral system prompt — does not
        come back.
        """
        ctx = self._ga_ctx
        store = ctx.checkpoint_store if ctx is not None else None
        if ctx is None or store is None:
            return
        try:
            histories = await read_agent_histories(store, ctx.session_key)
            tasks = await read_task_records(store, ctx.session_key)
            pending = await read_pending_messages(store, ctx.session_key)
        except Exception as exc:
            self.notify(f"Could not restore the session: {exc}", severity="warning")
            return
        # A roster scopes the restore to its members (and their nested
        # sub-agents): several member processes can share one store — each
        # window restores its own pane set, not every sibling's.
        scope = set(self._ga_roster)
        histories = [h for h in histories if not scope or h.root in scope]
        restored = [h.name for h in histories if h.messages]
        for hist in histories:
            if hist.messages:
                await self._restore_transcript(hist.name, hist.messages)
        launchers = scope | set(restored)
        for agent, record in tasks:
            if not scope or agent in launchers:
                await self._restore_task_pane(agent, record)
        for name in restored:
            # Replayed turns flipped the member "working"; it is parked now.
            self._set_status(name, "idle")
            await self._mount_restored_rule(name)
        if self._ga_on_post is not None:
            for message in pending:
                label = message.text.strip()
                if message.sender != USER_SENDER or not label:
                    continue
                self._ga_post_seq += 1
                self._ga_queued.append((self._ga_post_seq, label))
            self._refresh_queue_strip()

    async def _restore_transcript(
        self, name: str, messages: Sequence[InputItem]
    ) -> None:
        """Replay one agent's persisted transcript into its pane."""
        pane = await self._ensure(name, status="idle")
        # The ephemeral initial context (system prompt + leading messages) is
        # deliberately not in the transcript log, and a resumed run never
        # re-surfaces it — recompose it from the live member, when one was
        # passed (an object in ``agents``, not just a name).
        for event in history_events(await self._member_header(name), agent=name):
            await self._feed(event)
        omitted = len(messages) - _RESTORE_MAX_ITEMS
        if omitted > 0:
            await pane.mount(
                SelectableStatic(
                    Text(
                        f"… {omitted} earlier items not shown",
                        style=f"italic {PALETTE['usage']}",
                    ),
                    classes="ga-msg",
                )
            )
        for event in history_events(messages[-_RESTORE_MAX_ITEMS:], agent=name):
            await self._feed(event)

    async def _member_header(self, name: str) -> list[InputItem]:
        """A live member's recomposed initial context ([] when unavailable)."""
        member = self._ga_members.get(name)
        preview = getattr(member, "preview_initial_context", None)
        if preview is None:
            return []
        try:
            return list(await preview())
        except Exception:
            return []

    async def _mount_restored_rule(self, source: str) -> None:
        """Close a replayed pane: history above the rule, live output below."""
        pane = self._ga_panes.get(source)
        if pane is None:
            return
        self._ga_last_kind[source] = "turn"
        await pane.mount(
            SelectableStatic(
                Rule("[italic]↻ restored session[/]", style=PALETTE["separator"]),
                classes="ga-turn",
            )
        )
        # Land on the latest content, like a live session: during mount the
        # pane has no layout yet (an immediate scroll_end would no-op), so
        # anchor it — pinned to the bottom until the user scrolls away.
        pane.anchor()

    async def _restore_task_pane(self, agent: str, record: TaskRecord) -> None:
        """Rebuild one background task's log pane from its record + log file."""
        info = BackgroundTaskInfo(
            task_id=record.task_id,
            tool_name=record.tool_name,
            tool_call_id=record.tool_call_id,
            output_name=Path(record.output_path).name if record.output_path else None,
        )
        await self._open_task_pane(BackgroundTaskLaunchedEvent(source=agent, data=info))
        key = self._ga_task_panes[agent, record.task_id]
        log_text = await self._read_task_log(record.output_path)
        if log_text:
            await self._stream_task_log(
                key, self._ga_panes[key], log_text, at_bottom=True
            )
        # A RUNNING record stays open and routed: live resume either re-spawns
        # the task (its stream then lands here) or reports it interrupted in
        # the launcher's pane.
        if record.status is not TaskStatus.RUNNING:
            failed = record.status in {TaskStatus.FAILED, TaskStatus.CANCELLED} or bool(
                record.error
            )
            note = f"✗ {(record.error or record.status.value)[:200]}"
            await self._close_task_pane(
                BackgroundTaskCompletedEvent(source=agent, data=info),
                note="✓ completed" if not failed else note,
                failed=failed,
            )
        pane = self._ga_panes[key]
        pane.anchor()

    async def _read_task_log(self, path: str | None) -> str:
        """A task's mirrored log content, tail-capped; empty if unreadable."""
        if not path:
            return ""
        backend = self._ga_ctx.file_backend if self._ga_ctx is not None else None
        try:
            if backend is not None:
                data, _ = await backend.read_bytes(Path(path))
            else:
                data = await asyncio.to_thread(Path(path).read_bytes)
            text = data.decode("utf-8", errors="replace")
        except Exception:
            return ""
        if len(text) > _RESTORE_MAX_LOG_CHARS:
            text = "… earlier output truncated\n" + text[-_RESTORE_MAX_LOG_CHARS:]
        return text
