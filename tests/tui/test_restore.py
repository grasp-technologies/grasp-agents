"""
Session restore on TUI relaunch: panes rebuilt from the persisted session.

Seeds a checkpoint store the way a previous run would have (agent heads +
message logs, task records + mirrored log files, pending mailbox records),
then launches the app over it and asserts the rebuilt state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("textual")

from typing import Any

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import (
    AgentCheckpoint,
    InMemoryCheckpointStore,
    StepWatermark,
    TaskRecord,
    TaskStatus,
)
from grasp_agents.session_context import SessionContext
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import InputMessageItem, OutputMessageItem
from grasp_agents.types.message import CONTROL_PRIORITY, TeamMessage
from grasp_agents.ui.app import GraspAgentsApp
from tests._helpers import MockLLM

if TYPE_CHECKING:
    from pathlib import Path


async def _noop_post(text: str) -> None:
    del text


def _make_ctx() -> tuple[SessionContext[None], InMemoryCheckpointStore]:
    store = InMemoryCheckpointStore()
    return SessionContext[None](state=None, checkpoint_store=store), store


async def _seed_transcript(
    store: InMemoryCheckpointStore, session_key: str, name: str
) -> None:
    key = f"{session_key}/agent/{name}"
    items = [
        InputMessageItem.from_text("why is the sky blue?"),
        OutputMessageItem(
            status="completed",
            content=[OutputMessageText(text="Rayleigh scattering.")],
        ),
    ]
    await store.append_messages(key, items)
    head = AgentCheckpoint(
        session_key=session_key,
        processor_name=name,
        current=StepWatermark(message_count=len(items)),
    )
    await store.save(key, head.model_dump_json(exclude={"messages"}).encode())


@pytest.mark.asyncio
async def test_restores_transcripts_into_panes() -> None:
    ctx, store = _make_ctx()
    await _seed_transcript(store, ctx.session_key, "lead")
    await _seed_transcript(store, ctx.session_key, "researcher")

    app = GraspAgentsApp(
        on_post=_noop_post,
        main_agent="lead",
        agents=["lead", "researcher"],
        ctx=ctx,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        # One replayed generation each; parked (idle) after the replay.
        assert app._ga_turns["lead"] == 1
        assert app._ga_status["lead"] == "idle"
        assert app._ga_status["researcher"] == "idle"
        # user message + turn rule + answer + "restored session" rule
        assert len(app._ga_panes["lead"].children) == 4


@pytest.mark.asyncio
async def test_member_objects_restore_the_initial_context() -> None:
    ctx, store = _make_ctx()
    await _seed_transcript(store, ctx.session_key, "lead")
    lead = LLMAgent[Any, Any, None](
        name="lead",
        ctx=ctx,
        llm=MockLLM(),
        sys_prompt="You are the lead.",
        env_info=False,
    )

    # A member passed as an object (not a name) also gets its ephemeral
    # initial context — never in the transcript log — recomposed at the top.
    app = GraspAgentsApp(on_post=_noop_post, agents=[lead], ctx=ctx)
    async with app.run_test() as pilot:
        await pilot.pause()
        # system prompt + user message + turn rule + answer + restored rule
        assert len(app._ga_panes["lead"].children) == 5


@pytest.mark.asyncio
async def test_roster_scopes_the_restore() -> None:
    ctx, store = _make_ctx()
    await _seed_transcript(store, ctx.session_key, "lead")
    await _seed_transcript(store, ctx.session_key, "researcher")

    # A member window (one shared store, one-member roster) restores only its
    # own pane — not every sibling that persisted to the same session.
    app = GraspAgentsApp(on_post=_noop_post, agents=["researcher"], ctx=ctx)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "lead" not in app._ga_panes
        assert app._ga_turns["researcher"] == 1


@pytest.mark.asyncio
async def test_rebuilds_completed_task_pane_with_log(tmp_path: Path) -> None:
    ctx, store = _make_ctx()
    await _seed_transcript(store, ctx.session_key, "lead")
    log = tmp_path / "call_1.log"
    log.write_text("indexing…\n[4/4] done\n")
    record = TaskRecord(
        session_key=ctx.session_key,
        task_id="bg1",
        tool_call_id="call_1",
        tool_name="index_sources",
        status=TaskStatus.COMPLETED,
        output_path=str(log),
    )
    await store.save(
        f"{ctx.session_key}/task/lead/tc_call_1", record.model_dump_json().encode()
    )

    app = GraspAgentsApp(on_post=_noop_post, agents=["lead"], ctx=ctx)
    async with app.run_test() as pilot:
        await pilot.pause()
        key = "index_sources bg1"
        assert key in app._ga_panes
        assert key in app._ga_task_keys  # tab keeps the task tint
        assert app._ga_parent[key] == "lead"  # nests under the launcher
        assert app._ga_status[key] == "done"  # terminal → ✓ completed
        assert ("lead", "bg1") not in app._ga_task_panes  # no live routing
        # header + replayed log + ✓ line
        assert len(app._ga_panes[key].children) == 3


@pytest.mark.asyncio
async def test_running_task_pane_stays_open_and_routed(tmp_path: Path) -> None:
    ctx, store = _make_ctx()
    await _seed_transcript(store, ctx.session_key, "lead")
    log = tmp_path / "call_2.log"
    log.write_text("halfway…\n")
    record = TaskRecord(
        session_key=ctx.session_key,
        task_id="bg2",
        tool_call_id="call_2",
        tool_name="index_sources",
        status=TaskStatus.RUNNING,
        output_path=str(log),
    )
    await store.save(
        f"{ctx.session_key}/task/lead/tc_call_2", record.model_dump_json().encode()
    )

    app = GraspAgentsApp(on_post=_noop_post, agents=["lead"], ctx=ctx)
    async with app.run_test() as pilot:
        await pilot.pause()
        key = "index_sources bg2"
        # Still routed: a live resume that re-spawns the task streams here.
        assert app._ga_task_panes["lead", "bg2"] == key
        assert app._ga_task_log_text[key] == "halfway…\n"


@pytest.mark.asyncio
async def test_prefills_queue_strip_from_pending_human_mail() -> None:
    ctx, _store = _make_ctx()
    human = TeamMessage.from_text(
        sender="user", to="lead", text="queued question", priority=CONTROL_PRIORITY
    )
    peer = TeamMessage.from_text(sender="writer", to="lead", text="peer note")
    await ctx.transport.post(human)
    await ctx.transport.post(peer)

    app = GraspAgentsApp(on_post=_noop_post, agents=["lead"], ctx=ctx)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Only the human's still-pending mail is listed as queued.
        assert [label for _, label in app._ga_queued] == ["queued question"]
