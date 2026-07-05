"""
A transcript rewind (failed-run settle / step rollback) must take the
background-task plane with it: tasks launched after the restored boundary are
orphans — their launching tool calls are no longer in the history the model
sees — so they are cancelled, their queued completions suppressed, and their
durable records flipped CANCELLED once the rewound transcript is durable.
Resume dead-letters records above the restored head's launch watermark: the
crash backstop for a deferred flip that never landed.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.store_keys import task_prefix
from grasp_agents.durability.task_record import TaskRecord, TaskStatus
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.events import BackgroundTaskCompletedEvent, UserMessageEvent
from grasp_agents.types.items import FunctionToolCallItem, InputMessageItem
from tests._helpers import MockLLM, _text_response, _tool_call_response
from tests.agent.test_background_tools import (
    EchoInput,
    FireAndForgetTool,
    SlowTool,
    _multi_tool_call_response,
)


def _make_manager() -> tuple[BackgroundTaskManager[None], LLMAgentTranscript]:
    transcript = LLMAgentTranscript()
    transcript.messages = [InputMessageItem.from_text("sys", role="system")]
    mgr = BackgroundTaskManager[None](
        agent_name="t", transcript=transcript, tools={}, path=[]
    )
    return mgr, transcript


async def _launch(
    mgr: BackgroundTaskManager[None],
    ctx: SessionContext[None],
    call_id: str,
    *,
    delay: float = 10.0,
) -> str:
    tool = SlowTool(delay=delay)
    call = FunctionToolCallItem(call_id=call_id, name="slow", arguments='{"text":"x"}')
    _note, event = await mgr.run_backgroundable(
        call, tool, EchoInput(text="x"), ctx=ctx, exec_id="t"
    )
    assert event is not None
    return event.data.task_id


async def _load_only_record(
    store: InMemoryCheckpointStore, session_key: str
) -> tuple[str, TaskRecord]:
    keys = await store.list_keys(task_prefix(session_key))
    assert len(keys) == 1
    raw = await store.load(keys[0])
    assert raw is not None
    return keys[0], TaskRecord.model_validate_json(raw)


class TestCancelLaunchedAfter:
    @pytest.mark.asyncio
    async def test_kills_only_tasks_above_the_watermark(self) -> None:
        ctx = SessionContext[None](state=None)
        mgr, _ = _make_manager()

        t1 = await _launch(mgr, ctx, "c1")
        watermark = mgr.last_launch_seq
        t2 = await _launch(mgr, ctx, "c2")
        killed_task = mgr.get(t2).task

        mgr.cancel_launched_after(watermark)

        # t1's launch predates the watermark and keeps running; t2 is gone.
        assert mgr.get(t1).task.done() is False
        with pytest.raises(ValueError, match=t2):
            mgr.get(t2)
        with contextlib.suppress(asyncio.CancelledError):
            await killed_task
        assert killed_task.cancelled()

        await mgr.cancel_all(ctx=ctx)

    @pytest.mark.asyncio
    async def test_queued_completion_is_suppressed_not_announced(self) -> None:
        # The task finished (completion queued for drain) but its launching
        # call was rolled back: drain must stay silent — no ghost note.
        ctx = SessionContext[None](state=None)
        mgr, transcript = _make_manager()

        await _launch(mgr, ctx, "c1", delay=0.01)
        await mgr.wait_idle()
        assert mgr.has_undelivered_completions

        mgr.cancel_launched_after(0)

        events = [e async for e in mgr.drain(exec_id="t", ctx=ctx)]
        assert not any(isinstance(e, UserMessageEvent) for e in events)
        assert not any(isinstance(e, BackgroundTaskCompletedEvent) for e in events)
        assert len(transcript.messages) == 1  # only the seeded system message
        assert not mgr.has_undelivered_completions

    @pytest.mark.asyncio
    async def test_record_flip_is_deferred_then_flushed(self) -> None:
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr, _ = _make_manager()

        await _launch(mgr, ctx, "c1")
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.PENDING
        assert record.launch_seq == 1

        mgr.cancel_launched_after(0)

        # Deferred: the record flips only once the rewound transcript is
        # durable (flush after the next checkpoint / the rollback persist).
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.PENDING

        await mgr.flush_delivered(ctx=ctx)
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.CANCELLED
        assert record.error is not None
        assert "rolled back" in record.error

    @pytest.mark.asyncio
    async def test_kill_flip_survives_watermark_restore(self) -> None:
        # A settle/rollback restore wholesale-replaces the deferred DELIVERED
        # map from the boundary snapshot; a kill recorded alongside must not
        # be lost with it — the killed launch is gone from the transcript no
        # matter which boundary is live.
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr, _ = _make_manager()

        await _launch(mgr, ctx, "c1")
        mgr.cancel_launched_after(0)
        mgr.restore_pending_delivered({})

        await mgr.flush_delivered(ctx=ctx)
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.CANCELLED


class TestLaunchSeqWatermark:
    @pytest.mark.asyncio
    async def test_snapshot_carries_high_water_and_restore_never_lowers(self) -> None:
        ctx = SessionContext[None](state=None)
        mgr, transcript = _make_manager()
        assert mgr.last_launch_seq == 0

        await _launch(mgr, ctx, "c1")
        assert mgr.last_launch_seq == 1

        agent_ctx = AgentContext.create(transcript=transcript, tools={}, bg_tasks=mgr)
        state = agent_ctx.snapshot()
        assert state.bg_launch_seq == 1

        # Restoring an older watermark must not resurrect burned seqs: a
        # re-run's fresh launches can't be mistaken for the cancelled ones.
        agent_ctx.restore(state.model_copy(update={"bg_launch_seq": 0}))
        assert mgr.last_launch_seq == 1
        # A fresh manager (cold resume) seeds forward from the head.
        agent_ctx.restore(state.model_copy(update={"bg_launch_seq": 5}))
        assert mgr.last_launch_seq == 5
        assert mgr._next_id() == ("bg_6", 6)  # pyright: ignore[reportPrivateUsage]

        await mgr.cancel_all(ctx=ctx)


class TestResumeOrphanGuard:
    @pytest.mark.asyncio
    async def test_records_above_head_watermark_are_dead_lettered(self) -> None:
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr, _ = _make_manager()

        for tid, cid, seq in [("bg_1", "c1", 1), ("bg_2", "c2", 2)]:
            key = mgr._task_store_key(ctx, cid)  # pyright: ignore[reportPrivateUsage]
            assert key is not None
            await store.save(
                key,
                TaskRecord(
                    session_key="s1",
                    task_id=tid,
                    launch_seq=seq,
                    tool_call_id=cid,
                    tool_name="slow",
                    status=TaskStatus.PENDING,
                )
                .model_dump_json()
                .encode(),
            )

        injected = await mgr.resume_durable(ctx=ctx, exec_id="t", bg_launch_seq=1)

        # bg_1's launch is inside the restored transcript → interrupted
        # notice as usual; bg_2's launching call was never persisted → dead-
        # lettered silently, terminal on disk, never re-spawned or reported.
        joined = "\n".join(str(m) for m in injected)
        assert "bg_1" in joined
        assert "bg_2" not in joined

        key2 = mgr._task_store_key(ctx, "c2")  # pyright: ignore[reportPrivateUsage]
        assert key2 is not None
        raw = await store.load(key2)
        assert raw is not None
        record = TaskRecord.model_validate_json(raw)
        assert record.status == TaskStatus.CANCELLED
        assert record.error is not None
        assert "never persisted" in record.error

        # Dead-lettered ids stay reserved — no bg_2 collision for new calls.
        assert mgr._next_id() == ("bg_3", 3)  # pyright: ignore[reportPrivateUsage]


class _BlockingTool(BaseTool[EchoInput, Any, Any]):
    """Parks until released, so the round stays in flight when cancelled."""

    def __init__(self, started: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__(name="block", description="Parks until released")
        self._started = started
        self._release = release

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        del inp, ctx, kwargs
        self._started.set()
        await self._release.wait()
        return "ok"


class TestAgentRewindsTaskPlane:
    @pytest.mark.asyncio
    async def test_settle_kills_tasks_launched_in_the_pruned_round(self) -> None:
        started, release = asyncio.Event(), asyncio.Event()
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        agent = LLMAgent[str, str, None](
            name="a",
            ctx=ctx,
            path=[],
            llm=MockLLM(
                model_name="mock",
                responses_queue=[
                    _multi_tool_call_response(
                        [
                            ("slow", '{"text":"x"}', "c1"),
                            ("block", '{"text":"x"}', "c2"),
                        ]
                    ),
                ],
            ),
            tools=[SlowTool(delay=30.0), _BlockingTool(started, release)],
            sys_prompt="x",
            env_info=False,
        )

        async def consume() -> None:
            async for _ in agent.run_stream("go"):
                pass

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(started.wait(), timeout=2.0)
            mgr = agent._loop.bg_tasks  # pyright: ignore[reportPrivateUsage]
            assert mgr.has_live_tasks
            bg_task = mgr.get("bg_1").task
            _, record = await _load_only_record(store, "s1")
            assert record.status == TaskStatus.PENDING
            assert record.launch_seq == 1

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

        # The settle pruned the round in flight — its launched task is an
        # orphan and was cancelled with it.
        agent.transcript.validate_tool_call_pairing()
        assert not agent._loop.bg_tasks.has_live_tasks  # pyright: ignore[reportPrivateUsage]
        with contextlib.suppress(asyncio.CancelledError):
            await bg_task
        assert bg_task.cancelled()

        # The flip is deferred (no checkpoint ran after the settle): a fresh
        # resume dead-letters the record off the head's watermark instead.
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.PENDING

        t2 = LLMAgentTranscript()
        t2.messages = [InputMessageItem.from_text("sys", role="system")]
        mgr2 = BackgroundTaskManager[None](
            agent_name="a", transcript=t2, tools={}, path=[]
        )
        injected = await mgr2.resume_durable(ctx=ctx, exec_id="t", bg_launch_seq=0)
        assert injected == []  # no "interrupted" ghost for the orphan
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_rollback_to_step_kills_tasks_and_flips_records(self) -> None:
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        agent = LLMAgent[str, str, None](
            name="a",
            ctx=ctx,
            path=[],
            llm=MockLLM(
                model_name="mock",
                responses_queue=[
                    _tool_call_response("fire_and_forget", '{"text":"x"}', "c1"),
                    _text_response("done"),
                ],
            ),
            tools=[FireAndForgetTool(delay=30.0)],
            sys_prompt="x",
            env_info=False,
        )

        out = await agent.run("go", step=1)
        assert out.payloads[0] == "done"
        mgr = agent._loop.bg_tasks  # pyright: ignore[reportPrivateUsage]
        assert mgr.has_live_tasks  # non-blocking task outlives the run
        bg_task = mgr.get("bg_1").task

        await agent.rollback_to_step(1)

        # The step's launches are gone with its transcript; rollback persists
        # the rewound head itself, so the CANCELLED flip lands immediately.
        assert not mgr.has_live_tasks
        with contextlib.suppress(asyncio.CancelledError):
            await bg_task
        assert bg_task.cancelled()
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.CANCELLED
