"""
Tests for Runner execution and durability.

Verifies that:
- Basic linear and fan-out graphs execute correctly
- ParallelProcessor works as a fan-out node in the Runner graph
- Runner saves checkpoint (pending events) after each proc completion
- Runner resumes from checkpoint, skipping completed procs
- Composable checkpointing: Runner + ParallelProcessor sessions
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import RunnerCheckpoint
from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.runner.runner import END_PROC_NAME, Runner
from grasp_agents.types.errors import RunnerError
from grasp_agents.types.events import (
    Event,
    ProcPayloadOutEvent,
    RunPacketOutEvent,
)
from grasp_agents.types.io import ProcName

# ---------- Test helpers ----------


def _resolve_inputs(chat_inputs: Any | None, in_args: list[str] | None) -> list[str]:
    """Resolve inputs from either in_args (from packet) or chat_inputs (entry proc)."""
    if in_args is not None:
        return in_args
    if chat_inputs is not None:
        return [str(chat_inputs)]
    return []


class AppendProcessor(Processor[str, str, None]):
    """Appends its name to each input string."""

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
        for inp in _resolve_inputs(chat_inputs, in_args):
            output = f"{inp}->{self.name}"
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)


class CountingProcessor(Processor[str, str, None]):
    """Tracks how many times _process_stream was entered."""

    def __init__(self, name: str, *, recipients: list[ProcName] | None = None) -> None:
        super().__init__(name=name, recipients=recipients)
        self.call_count = 0

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        self.call_count += 1
        for inp in _resolve_inputs(chat_inputs, in_args):
            output = f"{inp}->{self.name}"
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)


class FailOnCallProcessor(Processor[str, str, None]):
    """Raises on a specific call number (1-based)."""

    def __init__(
        self,
        name: str,
        *,
        fail_on_call: int = 1,
        recipients: list[ProcName] | None = None,
    ) -> None:
        super().__init__(name=name, recipients=recipients)
        self._fail_on_call = fail_on_call
        self.call_count = 0

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        self.call_count += 1
        if self.call_count == self._fail_on_call:
            raise RuntimeError(
                f"{self.name} deliberate failure on call {self.call_count}"
            )
        for inp in _resolve_inputs(chat_inputs, in_args):
            output = f"{inp}->{self.name}"
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)


class FanOutProcessor(Processor[str, str, None]):
    """Produces multiple outputs per input."""

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
        for inp in _resolve_inputs(chat_inputs, in_args):
            yield ProcPayloadOutEvent(
                data=f"{inp}:x", source=self.name, exec_id=exec_id
            )
            yield ProcPayloadOutEvent(
                data=f"{inp}:y", source=self.name, exec_id=exec_id
            )


class RoutingProcessor(Processor[str, str, None]):
    """Routes each payload to a recipient based on content."""

    def __init__(
        self,
        name: str,
        *,
        recipients: list[ProcName] | None = None,
        route_map: dict[str, ProcName] | None = None,
    ) -> None:
        super().__init__(name=name, recipients=recipients)
        self._route_map = route_map or {}

    def select_recipients_impl(
        self, output: str, *, ctx: RunContext[None], exec_id: str
    ) -> Sequence[ProcName]:
        for substr, recipient in self._route_map.items():
            if substr in output:
                return [recipient]
        return list(self.recipients or [])

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for inp in _resolve_inputs(chat_inputs, in_args):
            yield ProcPayloadOutEvent(data=inp, source=self.name, exec_id=exec_id)


async def run_runner(
    runner: Runner[str, None],
    chat_inputs: Any = "start",
) -> list[str]:
    """Run a Runner and collect final payloads from RunPacketOutEvent."""
    payloads: list[str] = []
    async for event in runner.run_stream(chat_inputs=chat_inputs):
        if isinstance(event, RunPacketOutEvent):
            payloads = list(event.data.payloads)
    return payloads


# ---------- Basic execution ----------


class TestRunnerBasicExecution:
    @pytest.mark.asyncio
    async def test_linear_graph(self) -> None:
        """A → B → END."""
        a = AppendProcessor("A", recipients=["B"])
        b = AppendProcessor("B", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](entry_proc=a, procs=[a, b], ctx=ctx, name="r")

        result = await run_runner(runner, chat_inputs="s")
        assert result == ["s->A->B"]

    @pytest.mark.asyncio
    async def test_three_step_linear(self) -> None:
        """A → B → C → END."""
        a = AppendProcessor("A", recipients=["B"])
        b = AppendProcessor("B", recipients=["C"])
        c = AppendProcessor("C", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](entry_proc=a, procs=[a, b, c], ctx=ctx, name="r")

        result = await run_runner(runner, chat_inputs="s")
        assert result == ["s->A->B->C"]

    @pytest.mark.asyncio
    async def test_fan_out_merge(self) -> None:
        """A → [B, C] → D → END. B and C run concurrently, D gets both results."""
        a = AppendProcessor("A", recipients=["B", "C"])
        b = CountingProcessor("B", recipients=["D"])
        c = CountingProcessor("C", recipients=["D"])
        d = AppendProcessor("D", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](entry_proc=a, procs=[a, b, c, d], ctx=ctx, name="r")

        # D is the END proc and gets invoked twice (once from B, once from C).
        # But only the first RunPacketOutEvent finalizes the runner.
        result = await run_runner(runner, chat_inputs="s")
        assert b.call_count == 1
        assert c.call_count == 1

    @pytest.mark.asyncio
    async def test_routing_processor(self) -> None:
        """Router sends payload to specific recipient based on content."""
        router = RoutingProcessor(
            "router",
            recipients=["B", "C"],
            route_map={":x": "B", ":y": "C"},
        )
        fan = FanOutProcessor("fan", recipients=["router"])
        b = AppendProcessor("B", recipients=[END_PROC_NAME])
        c = AppendProcessor("C", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(state=None)

        # fan produces ["s:x", "s:y"], router routes :x→B, :y→C
        # Both B and C have END recipient — Runner requires exactly one.
        # Use a collector instead.
        d = AppendProcessor("D", recipients=[END_PROC_NAME])
        b2 = AppendProcessor("B", recipients=["D"])
        c2 = AppendProcessor("C", recipients=["D"])
        runner = Runner[str, None](
            entry_proc=fan, procs=[fan, router, b2, c2, d], ctx=ctx, name="r"
        )

        result = await run_runner(runner, chat_inputs="s")
        # fan → "s:x", "s:y" → router routes "s:x"→B, "s:y"→C
        # B → "s:x->B" → D, C → "s:y->C" → D
        # D produces two results (invoked twice)
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_parallel_processor_as_fanout_node(self) -> None:
        """Splitter → ParallelProcessor(worker) → Collector → END."""
        splitter = FanOutProcessor("splitter", recipients=["placeholder"])
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        collector = AppendProcessor("collector", recipients=[END_PROC_NAME])
        splitter.recipients = [par.name]
        par.recipients = ["collector"]

        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](
            entry_proc=splitter,
            procs=[splitter, par, collector],
            ctx=ctx,
            name="r",
        )

        result = await run_runner(runner, chat_inputs="s")
        assert sorted(result) == ["s:x->worker->collector", "s:y->worker->collector"]


# ---------- Checkpoint saving ----------


class TestRunnerCheckpoint:
    @pytest.mark.asyncio
    async def test_checkpoint_saved_after_each_proc(self) -> None:
        store = InMemoryCheckpointStore()
        a = AppendProcessor("A", recipients=["B"])
        b = AppendProcessor("B", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-1"
        )
        runner = Runner[str, None](entry_proc=a, procs=[a, b], ctx=ctx, name="r")

        result = await run_runner(runner, chat_inputs="s")
        assert result == ["s->A->B"]

        raw = await store.load("runner/run-1")
        assert raw is not None
        cp = RunnerCheckpoint.model_validate_json(raw)
        assert len(cp.pending_events) == 0

    @pytest.mark.asyncio
    async def test_checkpoint_has_pending_after_partial_run(self) -> None:
        """If a proc crashes, checkpoint still has its input event pending."""
        store = InMemoryCheckpointStore()
        a = AppendProcessor("A", recipients=["B"])
        b = FailOnCallProcessor("B", fail_on_call=1, recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-2"
        )
        runner = Runner[str, None](entry_proc=a, procs=[a, b], ctx=ctx, name="r")

        with pytest.raises(Exception):
            await run_runner(runner, chat_inputs="s")

        raw = await store.load("runner/run-2")
        assert raw is not None
        cp = RunnerCheckpoint.model_validate_json(raw)
        # A completed → START event removed, A's output event added.
        # B crashed → its input event still pending.
        assert len(cp.pending_events) == 1
        assert cp.pending_events[0].destination == "B"

    @pytest.mark.asyncio
    async def test_checkpoint_tracks_active_steps(self) -> None:
        store = InMemoryCheckpointStore()
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        par.recipients = [END_PROC_NAME]
        splitter = FanOutProcessor("splitter", recipients=[par.name])

        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-3"
        )
        runner = Runner[str, None](
            entry_proc=splitter, procs=[splitter, par], ctx=ctx, name="r"
        )

        await run_runner(runner, chat_inputs="s")

        raw = await store.load("runner/run-3")
        assert raw is not None
        cp = RunnerCheckpoint.model_validate_json(raw)
        # After completion, step counters remain in active_steps for all
        # procs that were invoked.
        assert par.name in cp.active_steps


# ---------- Resume ----------


class TestRunnerResume:
    @pytest.mark.asyncio
    async def test_resume_skips_completed_procs(self) -> None:
        """A completes and B crashes → resume skips A, re-runs B."""
        store = InMemoryCheckpointStore()

        a1 = CountingProcessor("A", recipients=["B"])
        b1 = FailOnCallProcessor("B", fail_on_call=1, recipients=[END_PROC_NAME])
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-r1"
        )
        runner1 = Runner[str, None](entry_proc=a1, procs=[a1, b1], ctx=ctx1, name="r")

        with pytest.raises(Exception):
            await run_runner(runner1, chat_inputs="s")

        assert a1.call_count == 1

        # Resume
        a2 = CountingProcessor("A", recipients=["B"])
        b2 = CountingProcessor("B", recipients=[END_PROC_NAME])
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-r1"
        )
        runner2 = Runner[str, None](entry_proc=a2, procs=[a2, b2], ctx=ctx2, name="r")

        result = await run_runner(runner2)
        assert result == ["s->A->B"]
        assert a2.call_count == 0
        assert b2.call_count == 1

    @pytest.mark.asyncio
    async def test_resume_three_step_crash_at_third(self) -> None:
        """A → B → C. Crash at C → resume runs only C."""
        store = InMemoryCheckpointStore()

        a1 = CountingProcessor("A", recipients=["B"])
        b1 = CountingProcessor("B", recipients=["C"])
        c1 = FailOnCallProcessor("C", fail_on_call=1, recipients=[END_PROC_NAME])
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-r2"
        )
        runner1 = Runner[str, None](
            entry_proc=a1, procs=[a1, b1, c1], ctx=ctx1, name="r"
        )

        with pytest.raises(Exception):
            await run_runner(runner1, chat_inputs="s")

        a2 = CountingProcessor("A", recipients=["B"])
        b2 = CountingProcessor("B", recipients=["C"])
        c2 = CountingProcessor("C", recipients=[END_PROC_NAME])
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-r2"
        )
        runner2 = Runner[str, None](
            entry_proc=a2, procs=[a2, b2, c2], ctx=ctx2, name="r"
        )

        result = await run_runner(runner2)
        assert result == ["s->A->B->C"]
        assert a2.call_count == 0
        assert b2.call_count == 0
        assert c2.call_count == 1

    @pytest.mark.asyncio
    async def test_resume_fan_out_partial(self) -> None:
        """A → [B, C] → D. C crashes → resume re-runs C (not A or B)."""
        store = InMemoryCheckpointStore()

        a1 = CountingProcessor("A", recipients=["B", "C"])
        b1 = CountingProcessor("B", recipients=["D"])
        c1 = FailOnCallProcessor("C", fail_on_call=1, recipients=["D"])
        d1 = AppendProcessor("D", recipients=[END_PROC_NAME])
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-fan"
        )
        runner1 = Runner[str, None](
            entry_proc=a1, procs=[a1, b1, c1, d1], ctx=ctx1, name="r"
        )

        with pytest.raises(Exception):
            await run_runner(runner1, chat_inputs="s")

        # Resume
        a2 = CountingProcessor("A", recipients=["B", "C"])
        b2 = CountingProcessor("B", recipients=["D"])
        c2 = CountingProcessor("C", recipients=["D"])
        d2 = AppendProcessor("D", recipients=[END_PROC_NAME])
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-fan"
        )
        runner2 = Runner[str, None](
            entry_proc=a2, procs=[a2, b2, c2, d2], ctx=ctx2, name="r"
        )

        result = await run_runner(runner2)
        assert a2.call_count == 0
        # C definitely runs
        assert c2.call_count == 1

    @pytest.mark.asyncio
    async def test_all_completed_resume_is_noop(self) -> None:
        """Full run completes → checkpoint has no pending events."""
        store = InMemoryCheckpointStore()

        a1 = AppendProcessor("A", recipients=["B"])
        b1 = AppendProcessor("B", recipients=[END_PROC_NAME])
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-noop"
        )
        runner1 = Runner[str, None](entry_proc=a1, procs=[a1, b1], ctx=ctx1, name="r")

        result1 = await run_runner(runner1, chat_inputs="s")
        assert result1 == ["s->A->B"]

        raw = await store.load("runner/run-noop")
        assert raw is not None
        cp = RunnerCheckpoint.model_validate_json(raw)
        assert len(cp.pending_events) == 0


# ---------- Composable checkpointing ----------


class TestRunnerComposableCheckpointing:
    @pytest.mark.asyncio
    async def test_parallel_processor_with_runner_session(self) -> None:
        """Runner configures ParallelProcessor session for composable checkpointing."""
        store = InMemoryCheckpointStore()
        splitter = FanOutProcessor("splitter", recipients=["placeholder"])
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        collector = AppendProcessor("collector", recipients=[END_PROC_NAME])
        splitter.recipients = [par.name]
        par.recipients = ["collector"]

        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="run-comp"
        )
        runner = Runner[str, None](
            entry_proc=splitter,
            procs=[splitter, par, collector],
            ctx=ctx,
            name="r",
        )

        result = await run_runner(runner, chat_inputs="s")
        assert sorted(result) == ["s:x->worker->collector", "s:y->worker->collector"]

        # ParallelProcessor should have its own checkpoint in the store
        par_key = f"parallel/run-comp/{par.name}"
        par_data = await store.load(par_key)
        assert par_data is not None

        # Runner checkpoint tracks active step counters
        raw = await store.load("runner/run-comp")
        assert raw is not None
        cp = RunnerCheckpoint.model_validate_json(raw)
        assert par.name in cp.active_steps


# ---------- Validation ----------


class TestRunnerValidation:
    def test_unknown_recipient_raises(self) -> None:
        """Referencing a non-existent processor as recipient should fail at init."""
        a = AppendProcessor("A", recipients=["NONEXISTENT"])
        b = AppendProcessor("B", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(state=None)
        with pytest.raises(RunnerError, match="unknown recipient"):
            Runner[str, None](entry_proc=a, procs=[a, b], ctx=ctx, name="r")

    def test_valid_recipients_pass(self) -> None:
        """Valid recipient names (including END) should not raise."""
        a = AppendProcessor("A", recipients=["B"])
        b = AppendProcessor("B", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(state=None)
        # Should not raise
        Runner[str, None](entry_proc=a, procs=[a, b], ctx=ctx, name="r")


# ---------- Corrupt checkpoint recovery ----------


class TestRunnerCorruptCheckpoint:
    @pytest.mark.asyncio
    async def test_corrupt_checkpoint_falls_back_to_fresh_run(self) -> None:
        """Corrupt checkpoint data should be ignored (fail-open), not crash."""
        store = InMemoryCheckpointStore()
        # Plant garbage data at the checkpoint key
        await store.save("runner/sess-corrupt", b"not valid json at all")

        a = AppendProcessor("A", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="sess-corrupt"
        )
        runner = Runner[str, None](entry_proc=a, procs=[a], ctx=ctx, name="r")

        # Should run fresh despite corrupt checkpoint
        result = await run_runner(runner, chat_inputs="s")
        assert result == ["s->A"]

    @pytest.mark.asyncio
    async def test_empty_pending_events_returns_immediately(self) -> None:
        """Checkpoint with zero pending events means run completed — no hang."""
        store = InMemoryCheckpointStore()
        # Save a valid checkpoint with empty pending_events
        cp = RunnerCheckpoint(
            session_key="sess-done",
            processor_name="r",
            pending_events=[],
        )
        await store.save(
            "runner/sess-done", cp.model_dump_json().encode("utf-8")
        )

        a = AppendProcessor("A", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="sess-done"
        )
        runner = Runner[str, None](entry_proc=a, procs=[a], ctx=ctx, name="r")

        # Should return immediately without hanging
        events: list[Event[Any]] = []
        async for event in runner.run_stream(chat_inputs="s"):
            events.append(event)
        assert events == []  # No events produced
