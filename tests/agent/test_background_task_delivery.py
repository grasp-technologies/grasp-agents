"""
Turn-boundary delivery of background-task completions.

A task's outcome is its own terminal event; everything its completion needs
(log, spilled result, durable record) is settled when it finishes, so taking it
does no I/O, and a failure settling it is raised at the next turn boundary; its
log is written as its output flows; the whole batch of notes reaches the
transcript before any event fires; output produced after the last bubble pass
travels on the completion; and a note is excerpted only when its full text was
spilled to a file it names.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

import grasp_agents.agent.background_tasks as bt_mod
from grasp_agents.agent import task_progress
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import AgentCheckpointLocation, CheckpointKind
from grasp_agents.durability.store_keys import (
    make_store_key,
    make_tool_call_path,
    task_prefix,
)
from grasp_agents.durability.task_record import TaskRecord, TaskStatus
from grasp_agents.file_backend.base import FileBackend
from grasp_agents.sandbox import local_environment
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    Event,
    ToolErrorEvent,
    ToolErrorInfo,
    ToolOutputEvent,
    ToolStreamEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import FunctionToolCallItem, InputMessageItem
from tests._helpers import MockLLM, _text_response, _tool_call_response, _until
from tests.agent.test_background_tools import (
    BigOutputTool,
    EchoInput,
    SlowTool,
    _make_executor,
    _multi_tool_call_response,
)

# --- Tools ---


class _NestedErrorTool(BaseTool[EchoInput, Any, Any]):
    """
    A backgrounded sub-agent stand-in: one of its inner tools fails, then it
    answers. With ``finish=False`` it stops after the inner failure.
    """

    def __init__(self, *, finish: bool = True) -> None:
        super().__init__(name="nested", description="Nested", auto_background_at=0)
        self._finish = finish

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        del inp, ctx, kwargs
        return "child ok"

    async def _run_stream(
        self, inp: EchoInput, *, exec_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        del inp, kwargs
        # Claimed by the inner tool's own call, as a sub-agent's loop does.
        yield ToolErrorEvent(
            data=ToolErrorInfo(tool_name="inner", error="inner boom"),
            source="inner",
            exec_id=exec_id,
            tool_call_id="inner_call",
        )
        if not self._finish:
            await asyncio.Event().wait()
        yield ToolOutputEvent(data="child ok", source=self.name, exec_id=exec_id)


class _ChattyTool(BaseTool[EchoInput, Any, Any]):
    """Streams a large chunk to its log, then (unless blocked) a large result."""

    def __init__(self, *, block: bool = False, cap: int = 100) -> None:
        super().__init__(
            name="chatty",
            description="Streams a lot",
            auto_background_at=0,
            has_progress_log=True,
            max_inline_result_chars=cap,
        )
        self._block = block

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        del inp, ctx, kwargs
        return "R" * 1000

    async def _run_stream(
        self, inp: EchoInput, *, exec_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        del inp, kwargs
        yield ToolStreamEvent(data="Z" * 1000, source=self.name, exec_id=exec_id)
        if self._block:
            await asyncio.Event().wait()
        yield ToolOutputEvent(data="R" * 1000, source=self.name, exec_id=exec_id)


class _GatedTailTool(BaseTool[EchoInput, Any, Any]):
    """Streams ``a1``, waits on ``gate``, then streams ``A_TAIL`` and finishes."""

    def __init__(self, gate: asyncio.Event, *, auto_background_at: float = 0) -> None:
        super().__init__(
            name="tail",
            description="Streams a late tail",
            auto_background_at=auto_background_at,
            has_progress_log=True,
        )
        self._gate = gate

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        del inp, ctx, kwargs
        return "tail done"

    async def _run_stream(
        self, inp: EchoInput, *, exec_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        del inp, kwargs
        yield ToolStreamEvent(data="a1", source=self.name, exec_id=exec_id)
        await self._gate.wait()
        yield ToolStreamEvent(data="A_TAIL", source=self.name, exec_id=exec_id)
        yield ToolOutputEvent(data="tail done", source=self.name, exec_id=exec_id)


class _LongFailureTool(BaseTool[EchoInput, Any, Any]):
    """Backgrounded tool that fails with an error longer than its inline cap."""

    def __init__(self) -> None:
        super().__init__(
            name="long_fail",
            description="Fails verbosely",
            auto_background_at=0,
            max_inline_result_chars=100,
        )

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        del inp, ctx, kwargs
        raise RuntimeError("E" * 1000)


class _UnserializableTool(BaseTool[EchoInput, Any, Any]):
    """Backgrounded tool whose result has no JSON form."""

    def __init__(self) -> None:
        super().__init__(name="odd", description="Odd", auto_background_at=0)

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> Any:
        del inp, ctx, kwargs
        return object()


class _FlakyStore(InMemoryCheckpointStore):
    """Rejects every save while ``down``."""

    def __init__(self) -> None:
        super().__init__()
        self.down = False

    async def save(self, key: str, data: bytes) -> None:
        if self.down:
            raise RuntimeError("store down")
        await super().save(key, data)


# --- Helpers ---


def _call(call_id: str, name: str) -> FunctionToolCallItem:
    return FunctionToolCallItem(call_id=call_id, name=name, arguments='{"text": "x"}')


async def _launch(
    mgr: BackgroundTaskManager[None],
    ctx: SessionContext[None],
    tool: BaseTool[Any, Any, Any],
    call_id: str,
) -> str:
    _note, launched = await mgr.run_backgroundable(
        _call(call_id, tool.name), tool, EchoInput(text="x"), ctx=ctx, exec_id="t"
    )
    assert launched is not None
    return launched.task_id


def _tasks(mgr: BackgroundTaskManager[None]) -> dict[str, bt_mod.BackgroundTask]:
    return mgr._tasks  # pyright: ignore[reportPrivateUsage]


async def _all_finished(mgr: BackgroundTaskManager[None]) -> None:
    await _until(lambda: all(t.consumer.done() for t in _tasks(mgr).values()))


async def _only_record(store: InMemoryCheckpointStore, session_key: str) -> TaskRecord:
    keys = await store.list_keys(task_prefix(session_key))
    assert len(keys) == 1
    record = await store.load_json(keys[0], TaskRecord, subject="task record")
    assert record is not None
    return record


def _file_ctx(tmp_path: Path, **kwargs: Any) -> SessionContext[None]:
    return SessionContext[None](
        state=None, environment=local_environment(allowed_roots=[tmp_path]), **kwargs
    )


def _result_file(note: str) -> Path | None:
    tag = re.search(r"<result_file>\n(.+?)\n</result_file>", note)
    return Path(tag.group(1).strip()) if tag else None


def _completion_notes(agent: LLMAgent[Any, Any, Any]) -> list[str]:
    return [
        m.text
        for m in agent.transcript
        if isinstance(m, InputMessageItem) and "<task_notification>" in m.text
    ]


# --- Outcome: the task's own terminal event ---


class TestOutcomeIsTheTasksOwnTerminal:
    @pytest.mark.asyncio
    async def test_inner_error_does_not_fail_a_task_that_then_answers(self) -> None:
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr = BackgroundTaskManager[None](tools={}, path=[])

        await _launch(mgr, ctx, _NestedErrorTool(), "c1")
        await _all_finished(mgr)

        [done] = mgr.pop_completions(ctx=ctx)
        note = done.note.message.text
        assert "<status> completed </status>" in note
        assert "child ok" in note
        assert "inner boom" not in note

        record = await _only_record(store, "s1")
        assert record.status == TaskStatus.COMPLETED
        assert record.result is not None
        assert "child ok" in record.result
        assert record.error is None

    @pytest.mark.asyncio
    async def test_killing_an_unfinished_task_reports_cancelled(self) -> None:
        # An inner tool's error is buffered, but the task itself never finished:
        # that is a cancellation, not a failure.
        ctx = SessionContext[None](state=None)
        mgr = BackgroundTaskManager[None](tools={})

        task_id = await _launch(mgr, ctx, _NestedErrorTool(finish=False), "c1")
        await _until(lambda: bool(_tasks(mgr)[task_id].events))

        killed = await mgr.kill_task(task_id, ctx=ctx)
        assert killed.status == "cancelled"
        assert killed.result is None

    @pytest.mark.asyncio
    async def test_kill_output_points_at_the_streamed_log(self, tmp_path: Path) -> None:
        ctx = _file_ctx(tmp_path)
        mgr = BackgroundTaskManager[None](tools={})

        task_id = await _launch(mgr, ctx, _ChattyTool(block=True), "c1")
        await _until(lambda: bool(_tasks(mgr)[task_id].events))
        log_path = _tasks(mgr)[task_id].log_path
        assert log_path is not None

        killed = await mgr.kill_task(task_id, ctx=ctx)
        assert f"streamed output in {log_path}" in killed.output
        # Output still pending when the task was killed is written too.
        assert Path(log_path).read_text() == "Z" * 1000


# --- A completion is settled when its task finishes ---


class TestCompletionIsSettledAtFinish:
    @pytest.mark.asyncio
    async def test_spill_and_record_are_final_before_the_task_is_taken(
        self, tmp_path: Path
    ) -> None:
        store = InMemoryCheckpointStore()
        ctx = _file_ctx(tmp_path, checkpoint_store=store, session_key="s1")
        mgr = BackgroundTaskManager[None](tools={}, path=[])

        task_id = await _launch(mgr, ctx, BigOutputTool(size=1000, cap=100), "c1")
        await _all_finished(mgr)

        outcome = _tasks(mgr)[task_id].outcome
        assert outcome is not None
        assert outcome.result_path is not None
        assert Path(outcome.result_path).read_text() == "X" * 1000
        record = await _only_record(store, "s1")
        assert record.status == TaskStatus.COMPLETED

        [done] = mgr.pop_completions(ctx=ctx)
        assert _result_file(done.note.message.text) == Path(outcome.result_path)

    @pytest.mark.asyncio
    async def test_an_unserializable_result_fails_the_task(self) -> None:
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr = BackgroundTaskManager[None](tools={}, path=[])

        await _launch(mgr, ctx, _UnserializableTool(), "c1")
        await _all_finished(mgr)

        [done] = mgr.pop_completions(ctx=ctx)
        note = done.note.message.text
        assert "<status> failed </status>" in note
        assert "could not be serialized" in note

        record = await _only_record(store, "s1")
        assert record.status == TaskStatus.FAILED
        assert record.error is not None
        assert "could not be serialized" in record.error


class TestFinalizeFailure:
    @pytest.mark.asyncio
    async def test_a_failed_outcome_save_is_raised_at_the_next_boundary(
        self,
    ) -> None:
        store = _FlakyStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr = BackgroundTaskManager[None](tools={}, path=[])
        gate = asyncio.Event()

        task_id = await _launch(mgr, ctx, _GatedTailTool(gate), "c1")
        store.down = True
        gate.set()
        await _all_finished(mgr)

        with pytest.raises(RuntimeError, match="store down"):
            mgr.pop_completions(ctx=ctx)
        # Nothing was taken: the batch is delivered by the next boundary.
        assert task_id in _tasks(mgr)
        assert (await _only_record(store, "s1")).status == TaskStatus.RUNNING

        store.down = False
        [done] = mgr.pop_completions(ctx=ctx)
        assert "tail done" in done.note.message.text

        # Delivering the note persists the outcome the failed save lost.
        mgr.record_delivery(done.note, note_pos=1)
        await mgr.flush_flips(ctx=ctx)
        record = await _only_record(store, "s1")
        assert record.status == TaskStatus.DELIVERED
        assert record.result == "tail done"

    @pytest.mark.asyncio
    async def test_the_error_is_raised_once(self) -> None:
        store = _FlakyStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        mgr = BackgroundTaskManager[None](tools={}, path=[])
        gate = asyncio.Event()

        await _launch(mgr, ctx, _GatedTailTool(gate), "c1")
        store.down = True
        gate.set()
        await _all_finished(mgr)

        with pytest.raises(RuntimeError, match="store down"):
            mgr.pop_completions(ctx=ctx)
        # Still down: the boundary is not wedged on an error already raised.
        assert len(mgr.pop_completions(ctx=ctx)) == 1


# --- The log is written as the task's output flows ---


class TestTaskLog:
    @pytest.mark.asyncio
    async def test_a_quiet_tasks_output_is_written_within_the_interval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bt_mod, "_LOG_FLUSH_INTERVAL", 0.2)
        gate = asyncio.Event()
        ctx = _file_ctx(tmp_path)
        mgr = BackgroundTaskManager[None](tools={})

        task_id = await _launch(mgr, ctx, _GatedTailTool(gate), "c1")
        await _until(lambda: bool(_tasks(mgr)[task_id].events))
        log = Path(_tasks(mgr)[task_id].log_path or "")

        # Batched: not written at once, nor by a turn boundary...
        _ = [e async for e in mgr.bubble_events()]
        assert not log.exists()
        # ...but within the interval, though the task stays quiet.
        await _until(lambda: log.exists() and log.read_text() == "a1")

        gate.set()
        await _all_finished(mgr)
        assert log.read_text() == "a1A_TAIL"

    @pytest.mark.asyncio
    async def test_a_deadline_backgrounded_task_logs_its_earlier_output(
        self, tmp_path: Path
    ) -> None:
        gate = asyncio.Event()
        ctx = _file_ctx(tmp_path)
        mgr = BackgroundTaskManager[None](tools={})

        task_id = await _launch(
            mgr, ctx, _GatedTailTool(gate, auto_background_at=0.05), "c1"
        )
        log = Path(_tasks(mgr)[task_id].log_path or "")
        # Streamed before it had a log; written when it was backgrounded.
        assert log.read_text() == "a1"

        gate.set()
        await _all_finished(mgr)
        assert log.read_text() == "a1A_TAIL"


# --- The loop writes every note before yielding any event ---


class TestLoopDeliversTheWholeBatchFirst:
    @pytest.mark.asyncio
    async def test_stopping_after_the_first_completion_keeps_every_note(self) -> None:
        agent = LLMAgent[str, str, None](
            name="a",
            ctx=SessionContext[None](state=None),
            llm=MockLLM(
                responses_queue=[
                    _multi_tool_call_response(
                        [
                            ("slow_a", '{"text":"a"}', "c1"),
                            ("slow_b", '{"text":"b"}', "c2"),
                        ]
                    ),
                    _text_response("waiting"),
                    _text_response("done"),
                ]
            ),
            tools=[
                SlowTool(delay=0.05, name="slow_a"),
                SlowTool(delay=0.05, name="slow_b"),
            ],
            sys_prompt="x",
            env_info=False,
        )

        async def wait_for_both(*, exec_id: str, turn: int, **_: Any) -> None:
            # Let both tasks finish during the second generation, so they are
            # delivered together at the next turn boundary.
            del exec_id
            if turn == 1:
                await asyncio.sleep(0.2)

        agent.add_before_llm_hook(wait_for_both)

        completed = 0
        async with contextlib.aclosing(agent.run_stream("go")) as stream:
            async for event in stream:
                if isinstance(event, BackgroundTaskCompletedEvent):
                    completed += 1
                    break
        assert completed == 1

        notes = _completion_notes(agent)
        assert len(notes) == 2
        assert any("slow: a" in n for n in notes)
        assert any("slow: b" in n for n in notes)

        out = await agent.run("continue")
        assert out.payloads == ["done"]
        assert len(_completion_notes(agent)) == 2
        await agent.aclose()

    @pytest.mark.asyncio
    async def test_late_output_is_bubbled_ahead_of_its_completion(
        self, tmp_path: Path
    ) -> None:
        gate = asyncio.Event()
        tool = _GatedTailTool(gate)
        ctx = _file_ctx(tmp_path)
        executor, transcript, _ = _make_executor([], tools=[tool], ctx=ctx)
        mgr = executor.agent_ctx.bg_tasks

        _note, launched = await mgr.run_backgroundable(
            _call("c1", "tail"),
            tool,
            EchoInput(text="x"),
            ctx=ctx,
            exec_id="t",
            agent_ctx=executor.agent_ctx,
        )
        assert launched is not None
        task_id = launched.task_id
        await _until(lambda: bool(_tasks(mgr)[task_id].events))
        log_path = _tasks(mgr)[task_id].log_path
        assert log_path is not None

        bubbled = [e async for e in mgr.bubble_events()]
        assert [e.data for e in bubbled if isinstance(e, ToolStreamEvent)] == ["a1"]

        gate.set()
        await _all_finished(mgr)

        events: list[Event[Any]] = []
        note_written_first = False
        async for event in executor._drain_bg_completions(exec_id="t"):  # pyright: ignore[reportPrivateUsage]
            if not events:
                last = transcript[-1]
                note_written_first = (
                    isinstance(last, InputMessageItem) and "tail done" in last.text
                )
            events.append(event)

        assert note_written_first
        assert [type(e) for e in events] == [
            ToolStreamEvent,
            ToolOutputEvent,
            BackgroundTaskCompletedEvent,
            UserMessageEvent,
        ]
        tail = events[0]
        assert isinstance(tail, ToolStreamEvent)
        assert tail.data == "A_TAIL"
        assert tail.tool_call_id == "c1"

        # The tail was mirrored to the log before the note pointing at it.
        assert Path(log_path).read_text() == "a1A_TAIL"


# --- The excerpt names the file that holds the full text ---


class TestExcerptPointers:
    @pytest.mark.asyncio
    async def test_a_failed_spill_inlines_the_whole_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def unwritable(
            file_backend: FileBackend, *, name: str, text: str
        ) -> str | None:
            del file_backend, name, text
            return None

        monkeypatch.setattr(task_progress, "write_result_file", unwritable)
        ctx = _file_ctx(tmp_path)
        mgr = BackgroundTaskManager[None](tools={})

        await _launch(mgr, ctx, _ChattyTool(), "c1")
        await _all_finished(mgr)

        [done] = mgr.pop_completions(ctx=ctx)
        note = done.note.message.text
        assert "R" * 1000 in note
        assert "chars omitted" not in note
        assert "<result_file>" not in note

    @pytest.mark.asyncio
    async def test_failed_task_spill_names_the_full_error(self, tmp_path: Path) -> None:
        ctx = _file_ctx(tmp_path)
        mgr = BackgroundTaskManager[None](tools={})

        await _launch(mgr, ctx, _LongFailureTool(), "c1")
        await _all_finished(mgr)

        [done] = mgr.pop_completions(ctx=ctx)
        note = done.note.message.text
        assert "<status> failed </status>" in note
        assert "full error in" in note
        assert "full result in" not in note
        path = _result_file(note)
        assert path is not None
        assert path.read_text().count("E") >= 1000


class TestRebuiltNotesSpill:
    def _ctx(
        self, tmp_path: Path
    ) -> tuple[SessionContext[None], InMemoryCheckpointStore]:
        store = InMemoryCheckpointStore()
        return _file_ctx(tmp_path, checkpoint_store=store, session_key="s1"), store

    async def _save_record(
        self, store: InMemoryCheckpointStore, record: TaskRecord
    ) -> None:
        key = make_store_key(
            "s1", CheckpointKind.TASK, make_tool_call_path([], record.tool_call_id)
        )
        await store.save(key, record.model_dump_json().encode())

    @pytest.mark.asyncio
    async def test_resume_spills_an_over_cap_result(self, tmp_path: Path) -> None:
        ctx, store = self._ctx(tmp_path)
        await self._save_record(
            store,
            TaskRecord(
                session_key="s1",
                task_id="bg_1",
                launch_seq=1,
                tool_call_id="c1",
                tool_name="big",
                status=TaskStatus.COMPLETED,
                result="Y" * 9000,
            ),
        )
        mgr = BackgroundTaskManager[None](tools={}, path=[])

        notes = await mgr.resume_durable(ctx=ctx, exec_id="t")
        text = notes[-1].message.text
        assert "full result in" in text
        path = _result_file(text)
        assert path is not None
        assert path.name == "c1.result"
        assert path.read_text() == "Y" * 9000

    @pytest.mark.asyncio
    async def test_redelivery_spills_an_over_cap_result(self, tmp_path: Path) -> None:
        ctx, store = self._ctx(tmp_path)
        await self._save_record(
            store,
            TaskRecord(
                session_key="s1",
                task_id="bg_1",
                launch_seq=1,
                tool_call_id="c1",
                tool_name="big",
                status=TaskStatus.DELIVERED,
                result="Y" * 9000,
                note_transcript_pos=5,
            ),
        )
        mgr = BackgroundTaskManager[None](tools={}, path=[])

        notes = await mgr.redeliver_after(
            message_count=2,
            task_launch_seq=1,
            ctx=ctx,
            pre_restore_deferred_delivered={},
        )
        [note] = notes
        text = note.message.text
        assert "full result in" in text
        path = _result_file(text)
        assert path is not None
        assert path.name == "c1.result"
        assert path.read_text() == "Y" * 9000


# --- A delivered note is durable before the model responds to it ---


class TestPersistAtDelivery:
    @pytest.mark.asyncio
    async def test_delivered_note_is_checkpointed_before_the_next_generation(
        self,
    ) -> None:
        store = InMemoryCheckpointStore()
        ctx = SessionContext[None](state=None, checkpoint_store=store, session_key="s1")
        agent = LLMAgent[str, str, None](
            name="a",
            ctx=ctx,
            path=[],
            llm=MockLLM(
                responses_queue=[
                    _tool_call_response("slow", '{"text":"x"}', "c1"),
                    _text_response("waiting"),
                    _text_response("done"),
                ]
            ),
            tools=[SlowTool(delay=0.05)],
            sys_prompt="x",
            env_info=False,
        )

        locations: list[AgentCheckpointLocation] = []
        save = agent.save_checkpoint

        async def spy(
            *,
            turn: int = 0,
            location: AgentCheckpointLocation = AgentCheckpointLocation.AFTER_INPUT,
            output: str | None = None,
            stop_reason: Any = None,
        ) -> None:
            locations.append(location)
            await save(
                turn=turn, location=location, output=output, stop_reason=stop_reason
            )

        agent._loop.checkpoint_callback = spy  # pyright: ignore[reportPrivateUsage]

        probes: list[tuple[AgentCheckpointLocation | None, TaskStatus]] = []

        async def probe(*, exec_id: str, turn: int, **_: Any) -> None:
            del exec_id, turn
            if _completion_notes(agent):
                record = await _only_record(store, "s1")
                probes.append((locations[-1] if locations else None, record.status))

        agent.add_before_llm_hook(probe)

        await agent.run("go")

        # The first generation that sees the note runs after an AFTER_INPUT
        # checkpoint that persisted it and flipped the task's record.
        assert probes
        assert probes[0] == (AgentCheckpointLocation.AFTER_INPUT, TaskStatus.DELIVERED)
        await agent.aclose()
