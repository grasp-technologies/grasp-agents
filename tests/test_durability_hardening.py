"""
B2.d hardening: extended store-failure coverage + task-record GC.

Covers:
- LLMAgent save failure propagates
- Runner save failure propagates
- Task-record save failure propagates
- LLMAgent and Runner tolerate corrupt checkpoints (log + treat as fresh)
- BackgroundTaskManager.prune_delivered deletes stale DELIVERED records
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import (
    AgentCheckpoint,
    InMemoryCheckpointStore,
    TaskRecord,
    TaskStatus,
)
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.runner.runner import END_PROC_NAME, Runner
from grasp_agents.types.errors import ProcRunError, RunnerError
from grasp_agents.types.events import (
    ProcPayloadOutEvent,
    RunPacketOutEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from grasp_agents.types.events import Event
    from grasp_agents.types.io import ProcName

from .test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
    SlowTool,
    _text_response,
    _tool_call_response,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


class _FailingStore(InMemoryCheckpointStore):
    """Raises ``OSError`` on save after ``fail_after`` successful writes."""

    def __init__(self, fail_after: int = 0) -> None:
        super().__init__()
        self._save_count = 0
        self._fail_after = fail_after

    async def save(self, key: str, data: bytes) -> None:
        self._save_count += 1
        if self._save_count > self._fail_after:
            raise OSError(f"Simulated I/O error on save #{self._save_count}")
        await super().save(key, data)


class _AppendProcessor(Processor[str, str, None]):
    """Minimal processor that forwards ``name->input`` to its recipients."""

    def __init__(self, name: str, *, recipients: list[ProcName] | None = None) -> None:
        super().__init__(name=name, recipients=recipients)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        del ctx, step
        base = in_args if in_args is not None else [str(chat_inputs)]
        for inp in base:
            yield ProcPayloadOutEvent(
                data=f"{inp}->{self.name}", source=self.name, exec_id=exec_id
            )


# ---------------------------------------------------------------------------
# FailingStore coverage — new surfaces
# ---------------------------------------------------------------------------


class TestFailingStoreCoverage:
    async def test_llm_agent_save_failure_kills_run(self) -> None:
        """LLMAgent's first AFTER_INPUT save failing must propagate."""
        store = _FailingStore(fail_after=0)  # fail on the first save
        agent = LLMAgent[str, str, None](
            name="agent",
            llm=MockLLM(responses_queue=[_text_response("ok")]),
            stream_llm=True,
        )
        ctx: RunContext[None] = RunContext(
            checkpoint_store=store, session_key="agent-fail"
        )

        with pytest.raises(ProcRunError) as exc:
            await agent.run("hi", ctx=ctx)
        assert isinstance(exc.value.__cause__, OSError)

    async def test_runner_save_failure_kills_run(self) -> None:
        """A Runner whose checkpoint save fails must propagate the error."""
        store = _FailingStore(fail_after=0)
        a = _AppendProcessor("A", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(
            checkpoint_store=store, session_key="run-fail", state=None
        )
        runner = Runner[str, None](entry_proc=a, procs=[a], ctx=ctx, name="r")

        with pytest.raises((RunnerError, ProcRunError, OSError)):
            async for _ in runner.run_stream(chat_inputs="x"):
                pass

    async def test_task_record_save_failure_kills_run(self) -> None:
        """
        Fail the TaskRecord save (2nd write — after the agent's AFTER_INPUT
        save succeeds) and assert the exception escapes the loop.
        """
        store = _FailingStore(fail_after=1)
        agent = LLMAgent[str, str, None](
            name="agent",
            llm=MockLLM(
                responses_queue=[
                    _tool_call_response("slow", '{"text":"x"}', "fc_1"),
                    _text_response("done"),
                ]
            ),
            tools=[SlowTool(delay=0.01)],
            stream_llm=True,
        )
        ctx: RunContext[None] = RunContext(
            checkpoint_store=store, session_key="task-fail"
        )

        with pytest.raises(ProcRunError) as exc:
            await agent.run("spawn", ctx=ctx)
        # The OSError is the root cause somewhere in the chain.
        root = exc.value
        while root.__cause__ is not None:
            root = root.__cause__  # type: ignore[assignment]
        assert isinstance(root, OSError)


# ---------------------------------------------------------------------------
# Corrupt-checkpoint tolerance (LLMAgent + Runner)
# ---------------------------------------------------------------------------


class TestCorruptCheckpointTolerance:
    async def test_llm_agent_corrupt_checkpoint_treated_as_fresh(self) -> None:
        store = InMemoryCheckpointStore()
        # Planted corrupt payload at the agent's store key.
        await store.save("agent/corrupt-a", b"not-valid-json{{{")

        agent = LLMAgent[str, str, None](
            name="agent",
            llm=MockLLM(responses_queue=[_text_response("fresh")]),
            stream_llm=True,
        )
        ctx: RunContext[None] = RunContext(
            checkpoint_store=store, session_key="corrupt-a"
        )

        result = await agent.run("hi", ctx=ctx)
        assert result.payloads[0] == "fresh"

        # A fresh checkpoint now sits at the key — the corrupt one was
        # treated as missing and overwritten by the normal run.
        data = await store.load("agent/corrupt-a")
        assert data is not None
        snap = AgentCheckpoint.model_validate_json(data)
        assert snap.session_key == "corrupt-a"

    async def test_runner_corrupt_checkpoint_treated_as_fresh(self) -> None:
        store = InMemoryCheckpointStore()
        await store.save("runner/corrupt-r", b"garbage")

        a = _AppendProcessor("A", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(
            checkpoint_store=store, session_key="corrupt-r", state=None
        )
        runner = Runner[str, None](entry_proc=a, procs=[a], ctx=ctx, name="r")

        payloads: list[str] = []
        async for event in runner.run_stream(chat_inputs="x"):
            if isinstance(event, RunPacketOutEvent):
                payloads = list(event.data.payloads)

        assert payloads == ["x->A"]


# ---------------------------------------------------------------------------
# prune_delivered
# ---------------------------------------------------------------------------


def _record(
    *,
    task_id: str,
    parent_session_key: str,
    status: TaskStatus,
    updated_at: datetime,
) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        parent_session_key=parent_session_key,
        tool_call_id=f"fc_{task_id}",
        tool_name="test",
        status=status,
        updated_at=updated_at,
    )


class TestPruneDelivered:
    async def test_deletes_delivered_older_than_cutoff(self) -> None:
        store = InMemoryCheckpointStore()
        now = datetime.now(UTC)
        old = _record(
            task_id="t-old",
            parent_session_key="s",
            status=TaskStatus.DELIVERED,
            updated_at=now - timedelta(days=2),
        )
        fresh = _record(
            task_id="t-fresh",
            parent_session_key="s",
            status=TaskStatus.DELIVERED,
            updated_at=now,
        )
        await store.save(old.store_key, old.model_dump_json().encode())
        await store.save(fresh.store_key, fresh.model_dump_json().encode())

        ctx: RunContext[None] = RunContext(
            checkpoint_store=store, session_key="s", state=None
        )
        pruned = await BackgroundTaskManager.prune_delivered(
            ctx, older_than=timedelta(hours=1)
        )
        assert pruned == 1
        assert await store.load(old.store_key) is None
        assert await store.load(fresh.store_key) is not None

    async def test_keeps_non_delivered(self) -> None:
        """Only DELIVERED records are swept; PENDING / FAILED stay."""
        store = InMemoryCheckpointStore()
        old = datetime.now(UTC) - timedelta(days=5)
        pending = _record(
            task_id="t-p",
            parent_session_key="s",
            status=TaskStatus.PENDING,
            updated_at=old,
        )
        failed = _record(
            task_id="t-f",
            parent_session_key="s",
            status=TaskStatus.FAILED,
            updated_at=old,
        )
        await store.save(pending.store_key, pending.model_dump_json().encode())
        await store.save(failed.store_key, failed.model_dump_json().encode())

        ctx: RunContext[None] = RunContext(
            checkpoint_store=store, session_key="s", state=None
        )
        pruned = await BackgroundTaskManager.prune_delivered(
            ctx, older_than=timedelta(seconds=1)
        )
        assert pruned == 0
        assert await store.load(pending.store_key) is not None
        assert await store.load(failed.store_key) is not None

    async def test_returns_zero_without_store(self) -> None:
        ctx: RunContext[None] = RunContext(session_key="s", state=None)
        assert (
            await BackgroundTaskManager.prune_delivered(
                ctx, older_than=timedelta(seconds=1)
            )
            == 0
        )

    async def test_scopes_to_session_key(self) -> None:
        """Records in other sessions are untouched."""
        store = InMemoryCheckpointStore()
        old = datetime.now(UTC) - timedelta(days=2)
        ours = _record(
            task_id="t-ours",
            parent_session_key="sA",
            status=TaskStatus.DELIVERED,
            updated_at=old,
        )
        theirs = _record(
            task_id="t-theirs",
            parent_session_key="sB",
            status=TaskStatus.DELIVERED,
            updated_at=old,
        )
        await store.save(ours.store_key, ours.model_dump_json().encode())
        await store.save(theirs.store_key, theirs.model_dump_json().encode())

        ctx: RunContext[None] = RunContext(
            checkpoint_store=store, session_key="sA", state=None
        )
        pruned = await BackgroundTaskManager.prune_delivered(
            ctx, older_than=timedelta(hours=1)
        )
        assert pruned == 1
        assert await store.load(ours.store_key) is None
        assert await store.load(theirs.store_key) is not None

    async def test_skips_corrupt_records(self) -> None:
        """A corrupt task record logs a warning; prune doesn't crash."""
        store = InMemoryCheckpointStore()
        await store.save("task/s/t-corrupt", b"not-json")

        ctx: RunContext[None] = RunContext(
            checkpoint_store=store, session_key="s", state=None
        )
        pruned = await BackgroundTaskManager.prune_delivered(
            ctx, older_than=timedelta(seconds=1)
        )
        assert pruned == 0
        # Record remains (we can't safely delete something we can't parse).
        assert await store.load("task/s/t-corrupt") is not None


# ---------------------------------------------------------------------------
# End-to-end: delivered records accumulate; prune_delivered clears them
# ---------------------------------------------------------------------------


class TestEndToEndGC:
    async def test_delivered_record_is_prunable_after_run(self) -> None:
        """
        After a background tool delivers, the record is DELIVERED in the
        store. A subsequent prune with an inclusive cutoff removes it.
        """
        store = InMemoryCheckpointStore()
        agent = LLMAgent[str, str, None](
            name="agent",
            llm=MockLLM(
                responses_queue=[
                    _tool_call_response("slow", '{"text":"data"}', "fc_1"),
                    _text_response("waiting"),
                    _text_response("done"),
                ]
            ),
            tools=[SlowTool(delay=0.01)],
            stream_llm=True,
        )
        ctx: RunContext[None] = RunContext(checkpoint_store=store, session_key="e2e")
        await agent.run("go", ctx=ctx)

        # Exactly one DELIVERED record.
        keys = await store.list_keys("task/e2e/")
        assert len(keys) == 1
        data = await store.load(keys[0])
        assert data is not None
        assert TaskRecord.model_validate_json(data).status == TaskStatus.DELIVERED

        # Wait past the cutoff, then prune. Using a very short window so
        # the DELIVERED record's ``updated_at`` is already older than it.
        await asyncio.sleep(0.05)
        pruned = await BackgroundTaskManager.prune_delivered(
            ctx, older_than=timedelta(milliseconds=1)
        )
        assert pruned == 1
        assert await store.list_keys("task/e2e/") == []
