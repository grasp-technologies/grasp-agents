"""
Tool-call approval for the interactive TUI: a notifying store + the dialog.

:class:`TuiApprovalStore` is a
:class:`~grasp_agents.agent.approval_store.LocalApprovalStore` that also
announces each pending approval on a queue; the app drains it and pops an
:class:`ApprovalScreen` per gated call (see :meth:`.app.GraspAgentsApp.
_consume_approvals`).
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from grasp_agents.agent.approval_store import (
    ApprovalAllow,
    ApprovalDecision,
    ApprovalDeny,
    ApprovalScope,
    LocalApprovalStore,
    PendingApproval,
)

if TYPE_CHECKING:
    from pathlib import Path


class TuiApprovalStore(LocalApprovalStore):
    """
    A :class:`~grasp_agents.agent.approval_store.LocalApprovalStore` that
    announces each newly-pending approval on a queue, so the interactive TUI can
    pop a dialog the moment a gated tool call needs a decision (rather than
    polling :meth:`list_pending`).

    Wire it onto the run context and register the gate hook; the TUI drains the
    queue and resolves each request from the dialog::

        store = TuiApprovalStore()
        agent.add_before_tool_hook(build_store_approval(tool_names={"delete"}))
        ctx = SessionContext(approval_store=store, session_key="user-1")
        run_tui_interactive(on_submit=agent.run_stream, ctx=ctx)
    """

    def __init__(self, *, persist_path: Path | None = None) -> None:
        super().__init__(persist_path=persist_path)
        # Pre-approved (session/always) calls short-circuit before
        # submit_pending, so only calls that genuinely need a decision arrive
        # here — one dialog per queued item.
        self.pending_events: asyncio.Queue[PendingApproval] = asyncio.Queue()

    async def submit_pending(
        self, pending: PendingApproval
    ) -> asyncio.Future[ApprovalDecision]:
        fut = await super().submit_pending(pending)
        self.pending_events.put_nowait(pending)
        return fut


def _format_args(arguments: str) -> str:
    """Pretty-print a tool call's JSON arguments; fall back to the raw string."""
    if not arguments.strip():
        return "(no arguments)"
    try:
        return json.dumps(json.loads(arguments), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return arguments


class ApprovalScreen(ModalScreen[ApprovalDecision]):
    """
    Modal asking the user to approve or deny one gated tool call.

    The choice maps to an
    :class:`~grasp_agents.agent.approval_store.ApprovalDecision`:

    * **once** — allow just this call;
    * **session** — allow this tool for the rest of the session (skips
      re-prompting);
    * **always** — allow permanently; persisted to disk when the store has a
      ``persist_path``, so it survives restarts;
    * **deny** — block the call (``esc`` also denies).
    """

    CSS = """
    ApprovalScreen { align: center middle; background: $background 70%; }
    ApprovalScreen #approval-box {
        width: 60%; max-width: 80; min-width: 48; height: auto; max-height: 80%;
        padding: 1 2; border: round $accent; background: $surface;
    }
    ApprovalScreen #approval-title { text-style: bold; color: $accent; width: 1fr; }
    ApprovalScreen #approval-tool { margin-top: 1; width: 1fr; }
    ApprovalScreen #approval-args {
        margin-top: 1; height: auto; max-height: 14; width: 1fr;
        border: round $panel; padding: 0 1;
    }
    ApprovalScreen #approval-hint { margin-top: 1; width: 1fr; color: $text-muted; }
    ApprovalScreen #approval-buttons {
        margin-top: 1; height: auto; align-horizontal: left;
    }
    ApprovalScreen #approval-buttons Button {
        min-width: 13; height: 1; border: none; margin-right: 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("o", "once", "Allow once"),
        ("s", "session", "Allow session"),
        ("a", "always", "Allow always"),
        ("d", "deny", "Deny"),
        ("escape", "deny", "Deny"),
    ]

    def __init__(self, pending: PendingApproval) -> None:
        super().__init__()
        self._pending = pending

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-box"):
            yield Static("⚠ Tool approval required", id="approval-title")
            yield Static(
                Text.assemble(("Tool: ", "dim"), (self._pending.tool_name, "bold")),
                id="approval-tool",
            )
            with VerticalScroll(id="approval-args"):
                yield Static(Text(_format_args(self._pending.arguments)))
            yield Static(
                "session = skip for this tool · always = persist across restarts",
                id="approval-hint",
            )
            with Horizontal(id="approval-buttons"):
                yield Button("Once (o)", id="once", variant="success")
                yield Button("Session (s)", id="session", variant="primary")
                yield Button("Always (a)", id="always", variant="primary")
                yield Button("Deny (d)", id="deny", variant="error")

    @on(Button.Pressed)
    def _on_button(self, event: Button.Pressed) -> None:
        action = {
            "once": self.action_once,
            "session": self.action_session,
            "always": self.action_always,
            "deny": self.action_deny,
        }.get(event.button.id or "")
        if action is not None:
            action()

    def action_once(self) -> None:
        self.dismiss(ApprovalAllow(ApprovalScope.ONCE))

    def action_session(self) -> None:
        self.dismiss(ApprovalAllow(ApprovalScope.SESSION))

    def action_always(self) -> None:
        self.dismiss(ApprovalAllow(ApprovalScope.ALWAYS))

    def action_deny(self) -> None:
        self.dismiss(ApprovalDeny())
