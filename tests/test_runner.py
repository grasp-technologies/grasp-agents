"""
Tests for Runner execution and durability.

Verifies that:
- Basic linear and fan-out graphs execute correctly
- ParallelProcessor works as a fan-out node in the Runner graph
- Runner saves checkpoint (pending events) after each proc completion
- Runner resumes from checkpoint, skipping completed procs
- Composable checkpointing: Runner + ParallelProcessor sessions
"""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import RunnerCheckpoint
from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.runner.event_bus import MAX_QUEUE_SIZE, EventBus
from grasp_agents.runner.runner import END_PROC_NAME, Runner
from grasp_agents.types.errors import ProcInputValidationError, RunnerError
from grasp_agents.types.events import (
    Event,
    ProcPacketOutEvent,
    ProcPayloadOutEvent,
    RoutedEvent,
    RunPacketOutEvent,
)
from grasp_agents.types.io import ProcName
from grasp_agents.types.packet import Packet
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow

from .test_sessions import MockLLM, _text_response

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
        self, output: str, *, exec_id: str
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
        await run_runner(runner, chat_inputs="s")
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
        assert sorted(result) == [
            "s:x->worker_0->collector",
            "s:y->worker_1->collector",
        ]


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

        raw = await store.load("run-1/runner")
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

        raw = await store.load("run-2/runner")
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

        raw = await store.load("run-3/runner")
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

        await run_runner(runner2)
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

        raw = await store.load("run-noop/runner")
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
        assert sorted(result) == [
            "s:x->worker_0->collector",
            "s:y->worker_1->collector",
        ]

        # ParallelProcessor should have its own checkpoint in the store
        par_key = f"run-comp/parallel/{par.name}"
        par_data = await store.load(par_key)
        assert par_data is not None

        # Runner checkpoint tracks active step counters
        raw = await store.load("run-comp/runner")
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
        await store.save("sess-corrupt/runner", b"not valid json at all")

        a = AppendProcessor("A", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="sess-corrupt"
        )
        runner = Runner[str, None](entry_proc=a, procs=[a], ctx=ctx, name="r")

        # Should run fresh despite corrupt checkpoint
        result = await run_runner(runner, chat_inputs="s")
        assert result == ["s->A"]

    @pytest.mark.asyncio
    async def test_empty_pending_events_raises_immediately(self) -> None:
        """
        Checkpoint with zero pending events means the run completed. Without a
        persisted final result (an older checkpoint) that is a clear error —
        not a hang, and not an AttributeError from ``run()``.
        """
        store = InMemoryCheckpointStore()
        # Save a valid checkpoint with empty pending_events and no final_event
        cp = RunnerCheckpoint(
            session_key="sess-done",
            processor_name="r",
            pending_events=[],
        )
        await store.save(
            "sess-done/runner", cp.model_dump_json().encode("utf-8")
        )

        a = AppendProcessor("A", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="sess-done"
        )
        runner = Runner[str, None](entry_proc=a, procs=[a], ctx=ctx, name="r")

        with pytest.raises(RunnerError, match="already completed"):
            async for _ in runner.run_stream(chat_inputs="s"):
                pass


# ---------- routing & resume edge cases ----------


class EmptyRoutingProcessor(Processor[str, str, None]):
    """Selects no recipients for any payload."""

    def __init__(self, name: str, *, recipients: list[ProcName] | None = None) -> None:
        super().__init__(name=name, recipients=recipients)

    def select_recipients_impl(
        self, output: str, *, exec_id: str
    ) -> Sequence[ProcName]:
        return []

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for inp in _resolve_inputs(chat_inputs, in_args):
            yield ProcPayloadOutEvent(data=inp, source=self.name, exec_id=exec_id)


class MixedRoutingProcessor(Processor[str, str, None]):
    """Yields two payloads, routing one to END and one to a real processor."""

    def __init__(self, name: str, *, recipients: list[ProcName] | None = None) -> None:
        super().__init__(name=name, recipients=recipients)

    def select_recipients_impl(
        self, output: str, *, exec_id: str
    ) -> Sequence[ProcName]:
        return [END_PROC_NAME] if output.endswith("x") else ["B"]

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        yield ProcPayloadOutEvent(data="px", source=self.name, exec_id=exec_id)
        yield ProcPayloadOutEvent(data="py", source=self.name, exec_id=exec_id)


class TestRoutingAndResume:
    @pytest.mark.asyncio
    async def test_runner_is_reusable_across_runs(self) -> None:
        """A second run() works — the bus must not stay stopped after a run."""
        a = AppendProcessor("A", recipients=[END_PROC_NAME])
        runner = Runner[str, None](entry_proc=a, procs=[a], name="r")

        out1 = await runner.run(chat_inputs="one")
        out2 = await runner.run(chat_inputs="two")

        assert list(out1.payloads) == ["one->A"]
        assert list(out2.payloads) == ["two->A"]

    @pytest.mark.asyncio
    async def test_resume_of_completed_run_returns_final_result(self) -> None:
        """run() on a completed session returns the persisted final packet."""
        store = InMemoryCheckpointStore()
        a1 = AppendProcessor("A", recipients=[END_PROC_NAME])
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="done-1"
        )
        runner1 = Runner[str, None](entry_proc=a1, procs=[a1], ctx=ctx1, name="r")
        out1 = await runner1.run(chat_inputs="s")
        assert list(out1.payloads) == ["s->A"]

        a2 = CountingProcessor("A", recipients=[END_PROC_NAME])
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="done-1"
        )
        runner2 = Runner[str, None](entry_proc=a2, procs=[a2], ctx=ctx2, name="r")
        out2 = await runner2.run()

        assert list(out2.payloads) == ["s->A"]
        assert a2.call_count == 0  # nothing re-ran

    @pytest.mark.asyncio
    async def test_empty_routing_raises_and_preserves_input(self) -> None:
        """No recipients selected → clear error; the input stays pending."""
        store = InMemoryCheckpointStore()
        a = EmptyRoutingProcessor("A", recipients=["B"])
        b = AppendProcessor("B", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="er-1"
        )
        runner = Runner[str, None](entry_proc=a, procs=[a, b], ctx=ctx, name="r")

        with pytest.raises(Exception) as excinfo:
            await run_runner(runner, chat_inputs="s")
        assert excinfo.group_contains(RunnerError, match="no routed recipients")

        # The consumed input was NOT checkpointed away — resume re-delivers it.
        raw = await store.load("er-1/runner")
        assert raw is not None
        cp = RunnerCheckpoint.model_validate_json(raw)
        assert len(cp.pending_events) == 1
        assert cp.pending_events[0].destination == "A"

    @pytest.mark.asyncio
    async def test_mixed_end_and_real_routing_raises_runner_error(self) -> None:
        """One packet routed both to END and to a proc → RunnerError, not KeyError."""
        a = MixedRoutingProcessor("A", recipients=[END_PROC_NAME, "B"])
        b = AppendProcessor("B")
        runner = Runner[str, None](entry_proc=a, procs=[a, b], name="r")

        with pytest.raises(Exception) as excinfo:
            await run_runner(runner, chat_inputs="s")
        assert excinfo.group_contains(RunnerError, match="both to END")


# ---------- parameterized-generic InT ----------


class ListProcessor(Processor[list[int], list[int], None]):
    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[list[int]] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for arg in in_args or []:
            yield ProcPayloadOutEvent(data=arg, source=self.name, exec_id=exec_id)


class TestParameterizedGenericInType:
    def test_single_list_arg_no_typeerror(self) -> None:
        proc = ListProcessor(name="p")
        out = proc.validate_inputs(exec_id="e", in_args=[1, 2, 3])
        # One argument of the declared list type — not three int args.
        assert out == [[1, 2, 3]]

    def test_packet_payloads_coerced(self) -> None:
        proc = ListProcessor(name="p")
        packet = Packet[Any](sender="x", payloads=[[1, 2], [3]])
        out = proc.validate_inputs(exec_id="e", in_packet=packet)
        assert out == [[1, 2], [3]]


# ---------- multi-payload fan-in to an LLMAgent ----------


class TestLLMAgentFanIn:
    def test_multi_payload_packet_aggregates_into_list_input(self) -> None:
        agent = LLMAgent[list[str], str, None](
            name="t",
            llm=MockLLM(responses_queue=[_text_response("ok")]),
            env_info=False,
        )
        packet = Packet[Any](sender="par", payloads=["a", "b", "c"])
        result = agent.validate_inputs(exec_id="e", in_packet=packet)
        assert result == [["a", "b", "c"]]

    def test_non_list_agent_raises_with_hint(self) -> None:
        agent = LLMAgent[str, str, None](
            name="t",
            llm=MockLLM(responses_queue=[]),
            env_info=False,
        )
        packet = Packet[Any](sender="par", payloads=["a", "b"])
        with pytest.raises(
            ProcInputValidationError, match=r"list\[\.\.\.\] to fan in"
        ):
            agent.validate_inputs(exec_id="e", in_packet=packet)


# ---------- bounded event-bus queues ----------


class TestEventBusBounds:
    def test_queues_are_bounded(self) -> None:
        bus = EventBus()
        assert bus._streamed_event_queue.maxsize == MAX_QUEUE_SIZE

        async def handler(event: Event[Any], **kwargs: Any) -> None:
            del event, kwargs

        bus.register_event_handler("x", handler)
        assert bus._routed_event_queues["x"].maxsize == MAX_QUEUE_SIZE

    @pytest.mark.asyncio
    async def test_shutdown_with_full_queue_does_not_hang(self) -> None:
        bus = EventBus()

        async def handler(event: Event[Any], **kwargs: Any) -> None:
            del event, kwargs

        bus.register_event_handler("x", handler)
        queue = bus._routed_event_queues["x"]
        dummy = ProcPayloadOutEvent(data="d", source="x", exec_id="e")
        while not queue.full():
            queue.put_nowait(dummy)

        # The consumer is gone (bus never entered) — shutdown must still
        # complete by making room for the sentinel.
        await asyncio.wait_for(bus.shutdown(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_crashing_handler_retrieves_future_exception(self) -> None:
        # A crashing handler delivers its error via the TaskGroup; the
        # final-result future's identical exception must be retrieved on exit
        # so a GC'd future can't trip asyncio's "Future exception was never
        # retrieved" warning in an unrelated later task.
        bus = EventBus()

        async def boom(event: Event[Any], **kwargs: Any) -> None:
            del event, kwargs
            raise RuntimeError("handler boom")

        async def drive() -> None:
            async with bus:
                bus.register_event_handler("p", boom)
                await bus.post(RoutedEvent(type="t", data="x", destination="p"))
                async for _ in bus.stream_events():
                    pass

        with pytest.raises(BaseExceptionGroup):
            await drive()

        fut = bus._final_result_fut
        assert fut is not None
        assert fut.done()
        assert fut._log_traceback is False  # retrieved during __aexit__
        assert isinstance(fut.exception(), RuntimeError)


# ---------- Runner[OutT] validates final payloads ----------


class IntProducer(Processor[str, int, None]):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, recipients=[END_PROC_NAME])

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        yield ProcPayloadOutEvent(data=42, source=self.name, exec_id=exec_id)


class TestRunnerOutTypeValidation:
    @pytest.mark.asyncio
    async def test_mismatched_final_payload_raises(self) -> None:
        p = IntProducer("P")
        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](entry_proc=p, procs=[p], ctx=ctx, name="r")

        with pytest.raises(Exception) as excinfo:
            await run_runner(runner, chat_inputs="s")
        assert excinfo.group_contains(
            RunnerError, match="does not match the runner's output type"
        )

    @pytest.mark.asyncio
    async def test_matching_final_payload_passes(self) -> None:
        a = AppendProcessor("A", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](entry_proc=a, procs=[a], ctx=ctx, name="r")

        result = await run_runner(runner, chat_inputs="s")
        assert result == ["s->A"]

    @pytest.mark.asyncio
    async def test_unsubscripted_runner_skips_validation(self) -> None:
        p = IntProducer("P")
        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner(entry_proc=p, procs=[p], ctx=ctx, name="r")

        payloads: list[Any] = []
        async for event in runner.run_stream(chat_inputs="s"):
            if type(event).__name__ == "RunPacketOutEvent":
                payloads = list(event.data.payloads)  # type: ignore[attr-defined]
        assert payloads == [42]


# ---------- sequential resume with re-delivered chat_inputs ----------


class TestSequentialResumeChatInputs:
    @pytest.mark.asyncio
    async def test_resume_with_chat_inputs_uses_checkpoint_packet(self) -> None:
        store = InMemoryCheckpointStore()

        async def drain(wf: SequentialWorkflow[str, str, None]) -> list[str]:
            payloads: list[str] = []
            async for event in wf.run_stream(chat_inputs="start", exec_id="t"):
                if (
                    isinstance(event, ProcPacketOutEvent)
                    and event.source == wf.name
                ):
                    payloads = list(event.data.payloads)
            return payloads

        # First run: A succeeds, B crashes — checkpoint has completed_step=0.
        a1 = AppendProcessor("A")
        b1 = FailOnCallProcessor("B", fail_on_call=1)
        c1 = AppendProcessor("C")
        wf1 = SequentialWorkflow[str, str, None](name="wf", subprocs=[a1, b1, c1])
        wf1.on_adopted(
            ctx=RunContext[None](
                state=None, checkpoint_store=store, session_key="seq-ci"
            )
        )
        with pytest.raises(Exception, match="Processor run failed"):
            await drain(wf1)

        # Resume re-delivering the same chat_inputs: the checkpointed packet
        # supersedes them — previously a dual-input validation error.
        a2 = AppendProcessor("A")
        b2 = FailOnCallProcessor("B", fail_on_call=99)
        c2 = AppendProcessor("C")
        wf2 = SequentialWorkflow[str, str, None](name="wf", subprocs=[a2, b2, c2])
        wf2.on_adopted(
            ctx=RunContext[None](
                state=None, checkpoint_store=store, session_key="seq-ci"
            )
        )
        result = await drain(wf2)
        assert result == ["start->A->B->C"]
