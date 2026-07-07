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
    async def test_queued_completion_is_suppressed_not_delivered(self) -> None:
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
        assert record.status == TaskStatus.RUNNING
        assert record.launch_seq == 1

        mgr.cancel_launched_after(0)

        # Deferred: the record flips only once the rewound transcript is
        # durable (flush after the next checkpoint / the rollback persist).
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.RUNNING

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
        mgr.restore_deferred_delivered({})

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
        assert state.task_launch_seq == 1

        # Restoring an older watermark must not resurrect burned seqs: a
        # re-run's fresh launches can't be mistaken for the cancelled ones.
        agent_ctx.restore(state.model_copy(update={"task_launch_seq": 0}))
        assert mgr.last_launch_seq == 1
        # A fresh manager (cold resume) seeds forward from the head.
        agent_ctx.restore(state.model_copy(update={"task_launch_seq": 5}))
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
                    status=TaskStatus.RUNNING,
                )
                .model_dump_json()
                .encode(),
            )

        injected = await mgr.resume_durable(ctx=ctx, exec_id="t", task_launch_seq=1)

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
            mgr = agent._loop.agent_ctx.bg_tasks  # pyright: ignore[reportPrivateUsage]
            assert mgr.has_live_tasks
            bg_task = mgr.get("bg_1").task
            _, record = await _load_only_record(store, "s1")
            assert record.status == TaskStatus.RUNNING
            assert record.launch_seq == 1

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

        # The settle pruned the round in flight — its launched task is an
        # orphan and was cancelled with it.
        agent.transcript.validate_tool_call_pairing()
        assert not agent._loop.agent_ctx.bg_tasks.has_live_tasks  # pyright: ignore[reportPrivateUsage]
        with contextlib.suppress(asyncio.CancelledError):
            await bg_task
        assert bg_task.cancelled()

        # The flip is deferred (no checkpoint ran after the settle): a fresh
        # resume dead-letters the record off the head's watermark instead.
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.RUNNING

        t2 = LLMAgentTranscript()
        t2.messages = [InputMessageItem.from_text("sys", role="system")]
        mgr2 = BackgroundTaskManager[None](
            agent_name="a", transcript=t2, tools={}, path=[]
        )
        injected = await mgr2.resume_durable(ctx=ctx, exec_id="t", task_launch_seq=0)
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
        mgr = agent._loop.agent_ctx.bg_tasks  # pyright: ignore[reportPrivateUsage]
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


class TestUndeliverAfter:
    @pytest.mark.asyncio
    async def test_rollback_reinjects_note_delivered_past_the_boundary(self) -> None:
        # A task launched BEFORE a step boundary whose completion note was
        # delivered (and its DELIVERED flip flushed) AFTER it: the rollback's
        # truncation removes the note while the launch survives — the note
        # must be re-injected, or the agent waits forever on an outcome
        # nothing would ever re-surface.
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        agent = LLMAgent[str, str, None](
            name="a",
            ctx=ctx,
            path=[],
            llm=MockLLM(responses_queue=[_text_response("a1"), _text_response("a2")]),
            sys_prompt="x",
            env_info=False,
        )
        await agent.run("q1", step=1)
        mgr = agent._loop.agent_ctx.bg_tasks  # pyright: ignore[reportPrivateUsage]

        # Launched in step 1's aftermath — before step 2's boundary.
        tool = FireAndForgetTool(delay=0.01)
        call = FunctionToolCallItem(
            call_id="c1", name="fire_and_forget", arguments='{"text":"x"}'
        )
        _note, event = await mgr.run_backgroundable(
            call, tool, EchoInput(text="x"), ctx=ctx, exec_id="t"
        )
        assert event is not None
        await agent.save_checkpoint(turn=0)  # the launch is durable pre-step-2
        await mgr.wait_idle()  # finished before step 2 starts

        # Step 2 drains the completion note into its own transcript span; its
        # run-end checkpoint flushes the DELIVERED flip with the position.
        await agent.run("q2", step=2)
        boundary = next(wm for wm in agent._step_watermarks if wm.step == 2)  # pyright: ignore[reportPrivateUsage]
        assert boundary.agent_ctx_state.task_launch_seq == 1
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.DELIVERED
        assert record.delivered_msg_count is not None
        assert record.delivered_msg_count > boundary.message_count
        assert str(agent.transcript.messages).count("<status>completed</status>") == 1

        await agent.rollback_to_step(2)

        # The note is back at the rewind point (once), queued for the next
        # run's event stream, and the record's position moved with it.
        assert str(agent.transcript.messages).count("<status>completed</status>") == 1
        assert agent._resume_notifications  # pyright: ignore[reportPrivateUsage]
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.DELIVERED
        assert record.delivered_msg_count == len(agent.transcript.messages)

        # A cold resume injects nothing extra (no duplicate note).
        t2 = LLMAgentTranscript()
        t2.messages = [InputMessageItem.from_text("sys", role="system")]
        mgr2 = BackgroundTaskManager[None](
            agent_name="a", transcript=t2, tools={}, path=[]
        )
        injected = await mgr2.resume_durable(ctx=ctx, exec_id="t", task_launch_seq=1)
        assert injected == []

    @pytest.mark.asyncio
    async def test_note_of_task_launched_after_the_boundary_stays_buried(self) -> None:
        # A DELIVERED record whose LAUNCH is also past the boundary is the
        # cancel path's business: its launching call is gone, so the note is
        # not re-injected.
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        agent = LLMAgent[str, str, None](
            name="a",
            ctx=ctx,
            path=[],
            llm=MockLLM(responses_queue=[_text_response("a1"), _text_response("a2")]),
            sys_prompt="x",
            env_info=False,
        )
        await agent.run("q1", step=1)
        await agent.run("q2", step=2)
        mgr = agent._loop.agent_ctx.bg_tasks  # pyright: ignore[reportPrivateUsage]

        # Launched, delivered, and flushed entirely inside step 2.
        tool = FireAndForgetTool(delay=0.01)
        call = FunctionToolCallItem(
            call_id="c1", name="fire_and_forget", arguments='{"text":"x"}'
        )
        _note, event = await mgr.run_backgroundable(
            call, tool, EchoInput(text="x"), ctx=ctx, exec_id="t"
        )
        assert event is not None
        await mgr.wait_idle()
        async for _ in mgr.drain(exec_id="t", ctx=ctx):
            pass
        await agent.save_checkpoint(turn=1)
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.DELIVERED

        await agent.rollback_to_step(2)

        assert "<status>completed</status>" not in str(agent.transcript.messages)
        assert not agent._resume_notifications  # pyright: ignore[reportPrivateUsage]


class TestUndeliverOverlay:
    @pytest.mark.asyncio
    async def test_drained_but_unflushed_note_is_reinjected(self) -> None:
        # A note drained in-process whose DELIVERED flip never flushed (its
        # record still COMPLETED): the rollback passes the pre-restore
        # deferred-flip map, and the overlay treats the note exactly like a
        # flushed one — re-injected, not silently lost until a cold resume.
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr, transcript = _make_manager()

        key = mgr._task_store_key(ctx, "c1")  # pyright: ignore[reportPrivateUsage]
        assert key is not None
        await store.save(
            key,
            TaskRecord(
                session_key="s1",
                task_id="bg_1",
                launch_seq=1,
                tool_call_id="c1",
                tool_name="slow",
                status=TaskStatus.COMPLETED,
                result="the outcome",
            )
            .model_dump_json()
            .encode(),
        )

        # Without the overlay the COMPLETED record stays buried.
        assert (
            await mgr.redeliver_after(message_count=4, task_launch_seq=1, ctx=ctx) == []
        )

        injected = await mgr.redeliver_after(
            message_count=4,
            task_launch_seq=1,
            ctx=ctx,
            deferred_delivered={
                key: {"status": TaskStatus.DELIVERED, "delivered_msg_count": 6}
            },
        )
        assert len(injected) == 1
        assert "the outcome" in str(injected[0])
        assert str(transcript.messages).count("<status>completed</status>") == 1


class TestKillsSurviveCrashBeforeFlush:
    @pytest.mark.asyncio
    async def test_restored_kill_is_not_resurrected_by_resume(self) -> None:
        # A settle cancels an orphan task and defers its CANCELLED flip; the
        # next head persists an inflated task_launch_seq BEFORE the flush. A
        # crash in that window must not resurrect the task on resume: the
        # head also carries the deferred kill, and resume honors it ahead of
        # the (defeated) orphan guard.
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr, transcript = _make_manager()

        await _launch(mgr, ctx, "c1")  # PENDING record, launch_seq=1
        mgr.cancel_launched_after(0)  # deferred kill, never flushed

        # The head snapshot carries both the raised counter and the kill.
        agent_ctx = AgentContext.create(transcript=transcript, tools={}, bg_tasks=mgr)
        state = agent_ctx.snapshot()
        assert state.task_launch_seq == 1
        assert state.deferred_killed

        # Cold resume: fresh manager, state restored from the head.
        t2 = LLMAgentTranscript()
        t2.messages = [InputMessageItem.from_text("sys", role="system")]
        mgr2 = BackgroundTaskManager[None](
            agent_name="t", transcript=t2, tools={}, path=[]
        )
        agent_ctx2 = AgentContext.create(transcript=t2, tools={}, bg_tasks=mgr2)
        agent_ctx2.restore(state)

        injected = await mgr2.resume_durable(
            ctx=ctx, exec_id="t", task_launch_seq=state.task_launch_seq
        )
        assert injected == []  # no phantom "interrupted" notice, no re-spawn

        # The re-armed kill lands at the next flush.
        await mgr2.flush_delivered(ctx=ctx)
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.CANCELLED


class TestResumeSkipsRestoredFlips:
    @pytest.mark.asyncio
    async def test_no_duplicate_note_when_flip_was_restored_with_the_head(
        self,
    ) -> None:
        # Crash between the checkpoint that persisted a completion note and
        # its record flush: the head's snapshot carries the deferred flip, so
        # the restored transcript already holds the note — resume must not
        # inject it again (the restored flip lands at the next save instead).
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr, _ = _make_manager()

        key = mgr._task_store_key(ctx, "c1")  # pyright: ignore[reportPrivateUsage]
        assert key is not None
        await store.save(
            key,
            TaskRecord(
                session_key="s1",
                task_id="bg_1",
                launch_seq=1,
                tool_call_id="c1",
                tool_name="slow",
                status=TaskStatus.COMPLETED,
                result="the outcome",
            )
            .model_dump_json()
            .encode(),
        )
        # The restored head carried this deferred flip.
        mgr.restore_deferred_delivered(
            {key: {"status": TaskStatus.DELIVERED, "delivered_msg_count": 2}}
        )

        injected = await mgr.resume_durable(ctx=ctx, exec_id="t", task_launch_seq=1)
        assert injected == []

        # Without the restored flip the same record IS re-injected.
        mgr.restore_deferred_delivered({})
        injected = await mgr.resume_durable(ctx=ctx, exec_id="t", task_launch_seq=1)
        assert len(injected) == 2  # framing + the note
        assert "the outcome" in str(injected[1])


class TestUndeliverOrder:
    @pytest.mark.asyncio
    async def test_reinjects_in_original_delivery_order(self) -> None:
        # Notes straddling the boundary are re-injected in the order the
        # model originally observed them (delivery order), which can differ
        # from launch order.
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr, transcript = _make_manager()

        for call_id, task_id, seq, delivered_at in [
            ("c1", "bg_1", 1, 6),  # launched first, delivered second
            ("c2", "bg_2", 2, 5),  # launched second, delivered first
        ]:
            key = mgr._task_store_key(ctx, call_id)  # pyright: ignore[reportPrivateUsage]
            assert key is not None
            await store.save(
                key,
                TaskRecord(
                    session_key="s1",
                    task_id=task_id,
                    launch_seq=seq,
                    tool_call_id=call_id,
                    tool_name=f"tool_{task_id}",
                    status=TaskStatus.DELIVERED,
                    result=f"result of {task_id}",
                    delivered_msg_count=delivered_at,
                )
                .model_dump_json()
                .encode(),
            )

        injected = await mgr.redeliver_after(
            message_count=4, task_launch_seq=2, ctx=ctx
        )

        assert len(injected) == 2
        assert "bg_2" in str(injected[0])  # delivered first → re-injected first
        assert "bg_1" in str(injected[1])
        assert len(transcript.messages) == 3  # seeded system message + 2 notes


class TestFlushDeliveredRetry:
    @pytest.mark.asyncio
    async def test_store_failure_keeps_unapplied_updates_deferred(self) -> None:
        # A store error mid-flush must not drop the un-applied flips: they
        # stay deferred, and the next flush applies them.
        class _FailOnceStore(InMemoryCheckpointStore):
            def __init__(self) -> None:
                super().__init__()
                self.fail_next_save = False

            async def save(self, key: str, data: bytes) -> None:
                if self.fail_next_save:
                    self.fail_next_save = False
                    raise RuntimeError("store down")
                await super().save(key, data)

        store = _FailOnceStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr, _ = _make_manager()

        await _launch(mgr, ctx, "c1")
        mgr.cancel_launched_after(0)

        store.fail_next_save = True
        with pytest.raises(RuntimeError, match="store down"):
            await mgr.flush_delivered(ctx=ctx)
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.RUNNING  # flip not applied yet

        await mgr.flush_delivered(ctx=ctx)  # retried at the next flush
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.CANCELLED


class TestSettleKeepsDeliveredFlips:
    @pytest.mark.asyncio
    async def test_flip_survives_settle_when_note_survives(self) -> None:
        # A completion note delivered BEFORE the round in flight survives the
        # settle (settling prunes only that round); its deferred DELIVERED
        # flip must survive with it — otherwise the record stays COMPLETED
        # and a later resume re-injects a note the transcript already holds.
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        agent = LLMAgent[str, str, None](
            name="a",
            ctx=ctx,
            path=[],
            llm=MockLLM(responses_queue=[_text_response("hi there")]),
            sys_prompt="x",
            env_info=False,
        )
        await agent.run("hi")  # a completed, checkpointed turn
        mgr = agent._loop.agent_ctx.bg_tasks  # pyright: ignore[reportPrivateUsage]

        # A task launched in a completed (checkpointed) round…
        tool = FireAndForgetTool(delay=0.01)
        call = FunctionToolCallItem(
            call_id="c1", name="fire_and_forget", arguments='{"text":"x"}'
        )
        _note, event = await mgr.run_backgroundable(
            call, tool, EchoInput(text="x"), ctx=ctx, exec_id="t"
        )
        assert event is not None
        await agent.save_checkpoint(turn=0)  # boundary covers the launch

        # …completes and its note is drained into the transcript.
        await mgr.wait_idle()
        async for _ in mgr.drain(exec_id="t", ctx=ctx):
            pass
        notes_before = str(agent.transcript.messages).count("task_notification")
        assert notes_before  # launch + completion notes
        assert mgr.export_deferred_delivered()  # the deferred DELIVERED flip

        # A later round is interrupted in flight → settle prunes only it.
        agent.transcript.update(
            [FunctionToolCallItem(call_id="c2", name="foo", arguments="{}")]
        )
        agent._settle_run(failed=True)  # pyright: ignore[reportPrivateUsage]

        # The notes are still in the transcript, and so is the flip.
        assert str(agent.transcript.messages).count("task_notification") == notes_before
        assert mgr.export_deferred_delivered()

        # The next save flushes the flip: the record lands DELIVERED…
        await agent.save_checkpoint(turn=1)
        _, record = await _load_only_record(store, "s1")
        assert record.status == TaskStatus.DELIVERED

        # …so a cold resume injects nothing (no duplicate note).
        t2 = LLMAgentTranscript()
        t2.messages = [InputMessageItem.from_text("sys", role="system")]
        mgr2 = BackgroundTaskManager[None](
            agent_name="a", transcript=t2, tools={}, path=[]
        )
        injected = await mgr2.resume_durable(ctx=ctx, exec_id="t", task_launch_seq=1)
        assert injected == []
