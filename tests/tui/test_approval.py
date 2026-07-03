"""
Headless tests for the TUI approval dialog.

A :class:`TuiApprovalStore` on the run context makes the app drain pending
approvals and pop an :class:`ApprovalScreen` per gated tool call. Here a fake
agent plays the role of the before-tool gate hook — it submits a pending
approval and awaits the decision — so the dialog + resolve wiring is exercised
without an LLM.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("textual")

from collections.abc import AsyncIterator

if TYPE_CHECKING:
    from pathlib import Path

from grasp_agents.agent.approval_store import (
    ApprovalAllow,
    ApprovalDecision,
    ApprovalDeny,
    ApprovalScope,
    PendingApproval,
)
from grasp_agents.session_context import SessionContext
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import Event, OutputMessageItemEvent
from grasp_agents.types.items import OutputMessageItem
from grasp_agents.ui.app import (
    ApprovalScreen,
    GraspAgentsApp,
    PromptArea,
    TuiApprovalStore,
)

_SESSION = "t"


def _pending(call_id: str, key: str = "delete_record") -> PendingApproval:
    return PendingApproval(
        session_key=_SESSION,
        call_id=call_id,
        tool_name="delete_record",
        arguments='{"record_id": 3}',
        approval_key=key,
    )


def _gating_agent(store: TuiApprovalStore, sink: list[ApprovalDecision], *, calls: int):
    """
    A fake agent that mimics the before-tool gate: for each simulated call it
    skips pre-approved keys, else submits a pending approval and awaits it.
    """

    async def agent(text: str) -> AsyncIterator[Event[object]]:
        for i in range(calls):
            if await store.is_pre_approved("delete_record", session_key=_SESSION):
                continue
            fut = await store.submit_pending(_pending(f"c{i}"))
            sink.append(await fut)
        yield OutputMessageItemEvent(
            data=OutputMessageItem(
                content=[OutputMessageText(text="done")], status="completed"
            ),
            source="ops",
        )

    return agent


def _make_app(
    sink: list[ApprovalDecision],
    *,
    calls: int = 1,
    persist_path: Path | None = None,
) -> GraspAgentsApp:
    store = TuiApprovalStore(persist_path=persist_path)
    ctx = SessionContext(state=None, approval_store=store, session_key=_SESSION)
    return GraspAgentsApp(
        on_submit=_gating_agent(store, sink, calls=calls), main_agent="ops", ctx=ctx
    )


async def _wait(pilot, predicate, *, ticks: int = 40) -> None:
    for _ in range(ticks):
        await pilot.pause()
        if predicate():
            return


async def _submit(app: GraspAgentsApp, pilot) -> None:
    app.query_one("#prompt", PromptArea).text = "delete record 3"
    await pilot.press("enter")


@pytest.mark.asyncio
async def test_dialog_pops_and_allow_resolves() -> None:
    sink: list[ApprovalDecision] = []
    app = _make_app(sink)
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(app, pilot)
        await _wait(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        assert isinstance(app.screen, ApprovalScreen)
        await pilot.press("o")  # allow once
        await _wait(pilot, lambda: bool(sink))
        assert isinstance(sink[0], ApprovalAllow)
        assert sink[0].scope is ApprovalScope.ONCE


@pytest.mark.asyncio
async def test_dialog_deny_via_escape() -> None:
    sink: list[ApprovalDecision] = []
    app = _make_app(sink)
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(app, pilot)
        await _wait(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("escape")  # esc denies the call (does not interrupt)
        await _wait(pilot, lambda: bool(sink))
        assert isinstance(sink[0], ApprovalDeny)


@pytest.mark.asyncio
async def test_allow_always_persists_to_disk(tmp_path: Path) -> None:
    # "Allow always" records the key to the store's persist_path, so a fresh
    # store (a later process) pre-approves the same tool without prompting.
    sink: list[ApprovalDecision] = []
    persist = tmp_path / "approvals.json"
    app = _make_app(sink, persist_path=persist)
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(app, pilot)
        await _wait(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("a")  # allow always
        await _wait(pilot, lambda: bool(sink))
    assert isinstance(sink[0], ApprovalAllow)
    assert sink[0].scope is ApprovalScope.ALWAYS
    assert persist.is_file()
    assert "delete_record" in persist.read_text()
    # a brand-new store reading the same file already has the decision
    reloaded = TuiApprovalStore(persist_path=persist)
    assert await reloaded.is_pre_approved("delete_record", session_key=_SESSION)


@pytest.mark.asyncio
async def test_allow_session_skips_second_prompt() -> None:
    # Two gated calls; "allow session" on the first pre-approves the second, so
    # only ONE dialog is shown and both calls are allowed.
    sink: list[ApprovalDecision] = []
    app = _make_app(sink, calls=2)
    dialogs = 0
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(app, pilot)
        await _wait(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        dialogs += 1
        await pilot.press("s")  # allow for the session
        # the run finishes without a second dialog
        await _wait(pilot, lambda: app._ga_running is False and bool(sink))
        assert isinstance(app.screen, ApprovalScreen) is False
        assert dialogs == 1
        assert len(sink) == 1  # only the first call needed a decision
        assert isinstance(sink[0], ApprovalAllow)


@pytest.mark.asyncio
async def test_no_consumer_without_tui_store() -> None:
    # A plain (non-Tui) store must not start the approval consumer.
    app = GraspAgentsApp(
        on_submit=_gating_agent(TuiApprovalStore(), [], calls=0),
        main_agent="ops",
        ctx=SessionContext(state=None),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._ga_approval_store is None


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY to construct"
)
def test_example_build_copilot_constructs() -> None:
    from grasp_agents.examples.tui.approval_copilot import GATED_TOOLS, build_copilot

    agent, ctx = build_copilot()
    assert agent.name == "ops_assistant"
    assert isinstance(ctx.approval_store, TuiApprovalStore)
    assert set(agent.tools) >= GATED_TOOLS  # agent.tools is {name: BaseTool}
