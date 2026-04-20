"""
Tests for workflow and parallel processor checkpointing.

Verifies that:
- SequentialWorkflow saves checkpoint after each step and resumes from last completed
- LoopedWorkflow saves checkpoint with iteration+step and resumes correctly
- ParallelProcessor saves checkpoint per-completion and resumes only pending copies
- All processors skip completed work on resume
- Checkpoint data round-trips correctly through the store
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import (
    ParallelCheckpoint,
    WorkflowCheckpoint,
)
from grasp_agents.packet import Packet
from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.events import (
    Event,
    ProcPacketOutEvent,
    ProcPayloadOutEvent,
)
from grasp_agents.types.io import ProcName
from grasp_agents.workflow.looped_workflow import LoopedWorkflow
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow

# ---------- Test helpers ----------


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
        for inp in in_args or []:
            output = f"{inp}->{self.name}"
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)


class CountingProcessor(Processor[str, str, None]):
    """Tracks how many times _process_stream was entered."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
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
        for inp in in_args or []:
            output = f"{inp}->{self.name}"
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)


class FailOnCallProcessor(Processor[str, str, None]):
    """Raises on a specific call number (1-based)."""

    def __init__(self, name: str, *, fail_on_call: int = 1) -> None:
        super().__init__(name=name)
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
        for inp in in_args or []:
            output = f"{inp}->{self.name}"
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)


class InputFailProcessor(Processor[str, str, None]):
    """Fails when input contains a specific substring. Survives deepcopy."""

    def __init__(self, name: str, *, fail_input: str = "FAIL") -> None:
        super().__init__(name=name)
        self._fail_input = fail_input

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for inp in in_args or []:
            if self._fail_input in inp:
                raise RuntimeError(f"{self.name} fails on input '{inp}'")
            output = f"{inp}->{self.name}"
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)


async def run_workflow(
    wf: SequentialWorkflow[str, str, None] | LoopedWorkflow[str, str, None],
    ctx: RunContext[None],
    in_args: str | None = None,
    step: int | None = None,
) -> list[str]:
    """Run a workflow via run_stream and collect final payloads."""
    payloads: list[str] = []
    async for event in wf.run_stream(
        in_args=in_args, ctx=ctx, exec_id="test", step=step
    ):
        if isinstance(event, ProcPacketOutEvent) and event.source == wf.name:
            payloads = list(event.data.payloads)
    return payloads


async def run_parallel(
    par: ParallelProcessor[str, str, None],
    ctx: RunContext[None],
    in_args: list[str] | None = None,
    step: int | None = None,
) -> list[str]:
    """Run a ParallelProcessor and collect final payloads."""
    payloads: list[str] = []
    async for event in par.run_stream(
        in_args=in_args, ctx=ctx, exec_id="test", step=step
    ):
        if isinstance(event, ProcPacketOutEvent) and event.source == par.name:
            payloads = list(event.data.payloads)
    return payloads


# ---------- SequentialWorkflow ----------


class TestSequentialWorkflowCheckpoint:
    @pytest.mark.asyncio
    async def test_basic_run_without_session(self) -> None:
        """Workflow works normally without agent/store."""
        a = AppendProcessor("A")
        b = AppendProcessor("B")
        c = AppendProcessor("C")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b, c])
        ctx: RunContext[None] = RunContext(state=None)

        result = await run_workflow(wf, ctx, in_args="start")
        assert result == ["start->A->B->C"]

    @pytest.mark.asyncio
    async def test_checkpoint_saved_after_each_step(self) -> None:
        """Each step saves a checkpoint to the store."""
        store = InMemoryCheckpointStore()
        a = AppendProcessor("A")
        b = AppendProcessor("B")
        c = AppendProcessor("C")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b, c])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="seq-1"
        )

        result = await run_workflow(wf, ctx, in_args="start")
        assert result == ["start->A->B->C"]

        raw = await store.load("workflow/seq-1")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.completed_step == 2  # 0-indexed, all 3 steps done
        assert cp.session_key == "seq-1"

    @pytest.mark.asyncio
    async def test_resume_skips_completed_steps(self) -> None:
        """After a crash, resume skips steps that already completed."""
        store = InMemoryCheckpointStore()

        # First run: step A succeeds, step B fails
        a1 = CountingProcessor("A")
        b1 = FailOnCallProcessor("B", fail_on_call=1)
        c1 = CountingProcessor("C")
        wf1 = SequentialWorkflow[str, str, None](name="wf", subprocs=[a1, b1, c1])
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="seq-2"
        )

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="start")

        # Checkpoint: step 0 completed
        raw = await store.load("workflow/seq-2")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.completed_step == 0

        # Resume
        a2 = CountingProcessor("A")
        b2 = CountingProcessor("B")
        c2 = CountingProcessor("C")
        wf2 = SequentialWorkflow[str, str, None](name="wf", subprocs=[a2, b2, c2])
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="seq-2"
        )

        result = await run_workflow(wf2, ctx2, step=0)
        assert result == ["start->A->B->C"]

        # A skipped, B and C ran once each
        assert a2.call_count == 0
        assert b2.call_count == 1
        assert c2.call_count == 1

    @pytest.mark.asyncio
    async def test_resume_with_multiple_payloads(self) -> None:
        """Resume works when packets have multiple payloads."""
        store = InMemoryCheckpointStore()

        class FanOutProcessor(Processor[str, str, None]):
            async def _process_stream(
                self,
                chat_inputs: Any | None = None,
                *,
                in_args: list[str] | None = None,
                exec_id: str,
                ctx: RunContext[None],
                step: int | None = None,
            ) -> AsyncIterator[Event[Any]]:
                for inp in in_args or []:
                    yield ProcPayloadOutEvent(
                        data=f"{inp}:x", source=self.name, exec_id=exec_id
                    )
                    yield ProcPayloadOutEvent(
                        data=f"{inp}:y", source=self.name, exec_id=exec_id
                    )

        fan = FanOutProcessor(name="fan")
        fail_b = FailOnCallProcessor("B", fail_on_call=1)
        wf1 = SequentialWorkflow[str, str, None](name="wf", subprocs=[fan, fail_b])
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="seq-multi"
        )

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="s")

        # Resume with working B
        fan2 = CountingProcessor("fan")
        b2 = AppendProcessor("B")
        wf2 = SequentialWorkflow[str, str, None](name="wf", subprocs=[fan2, b2])
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="seq-multi"
        )

        result = await run_workflow(wf2, ctx2, step=0)
        assert result == ["s:x->B", "s:y->B"]
        assert fan2.call_count == 0


# ---------- LoopedWorkflow ----------


class TestLoopedWorkflowCheckpoint:
    @pytest.mark.asyncio
    async def test_basic_looped_run_without_session(self) -> None:
        a = AppendProcessor("A")
        b = AppendProcessor("B")
        wf = LoopedWorkflow[str, str, None](
            name="loop", subprocs=[a, b], exit_proc=b, max_iterations=3
        )

        def _always_terminate(
            out_packet: Packet[str], *, ctx: RunContext[None], **kwargs: Any
        ) -> bool:
            return True

        wf.terminate_workflow_loop_impl = _always_terminate

        ctx: RunContext[None] = RunContext(state=None)
        result = await run_workflow(wf, ctx, in_args="start")
        assert result == ["start->A->B"]

    @pytest.mark.asyncio
    async def test_checkpoint_tracks_iteration(self) -> None:
        store = InMemoryCheckpointStore()
        a = AppendProcessor("A")
        b = AppendProcessor("B")
        wf = LoopedWorkflow[str, str, None](
            name="loop", subprocs=[a, b], exit_proc=b, max_iterations=3
        )

        iteration_count = 0

        def count_terminate(
            out_packet: Packet[str], *, ctx: RunContext[None], **kwargs: Any
        ) -> bool:
            nonlocal iteration_count
            iteration_count += 1
            return iteration_count >= 2

        wf.add_workflow_loop_terminator(count_terminate)

        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="loop-1"
        )

        result = await run_workflow(wf, ctx, in_args="s")
        assert result == ["s->A->B->A->B"]

        raw = await store.load("workflow/loop-1")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.completed_step == 3  # 2 iterations * 2 subprocs - 1

    @pytest.mark.asyncio
    async def test_resume_skips_completed_iteration_steps(self) -> None:
        """Resume in a loop skips completed steps within the current iteration."""
        store = InMemoryCheckpointStore()

        # First run: iteration 1, step A succeeds, step B fails
        a1 = CountingProcessor("A")
        b1 = FailOnCallProcessor("B", fail_on_call=1)
        wf1 = LoopedWorkflow[str, str, None](
            name="loop", subprocs=[a1, b1], exit_proc=b1, max_iterations=2
        )
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="loop-2"
        )

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="s")

        # Checkpoint: global step 0 completed (iteration 0, subproc A)
        raw = await store.load("workflow/loop-2")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.completed_step == 0

        # Resume — terminate after iteration 1
        a2 = CountingProcessor("A")
        b2 = CountingProcessor("B")
        wf2 = LoopedWorkflow[str, str, None](
            name="loop", subprocs=[a2, b2], exit_proc=b2, max_iterations=2
        )

        def _always_terminate2(
            out_packet: Packet[str], *, ctx: RunContext[None], **kwargs: Any
        ) -> bool:
            return True

        wf2.terminate_workflow_loop_impl = _always_terminate2
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="loop-2"
        )

        result = await run_workflow(wf2, ctx2, step=0)
        assert result == ["s->A->B"]

        # A was skipped (completed in iteration 1), B ran once
        assert a2.call_count == 0
        assert b2.call_count == 1


# ---------- ParallelProcessor ----------


class TestParallelProcessorCheckpoint:
    @pytest.mark.asyncio
    async def test_basic_parallel_without_session(self) -> None:
        subproc = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        ctx: RunContext[None] = RunContext(state=None)

        result = await run_parallel(par, ctx, in_args=["a", "b", "c"])
        assert sorted(result) == ["a->worker", "b->worker", "c->worker"]

    @pytest.mark.asyncio
    async def test_checkpoint_saved_per_completion(self) -> None:
        store = InMemoryCheckpointStore()
        subproc = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="par-1"
        )

        await run_parallel(par, ctx, in_args=["a", "b"])

        raw = await store.load("parallel/par-1")
        assert raw is not None
        cp = ParallelCheckpoint.model_validate_json(raw)
        assert len(cp.completed) == 2

    @pytest.mark.asyncio
    async def test_resume_skips_completed_copies(self) -> None:
        """
        After partial completion, resume only runs pending copies.

        stream_concurrent swallows per-copy errors, so the first run
        completes with None for the failed copy. On resume, only that
        copy is re-run.
        """
        store = InMemoryCheckpointStore()

        # "a" succeeds, "FAIL" crashes (InputFailProcessor survives deepcopy)
        subproc1 = InputFailProcessor("worker", fail_input="FAIL")
        par1 = ParallelProcessor[str, str, None](subproc=subproc1)
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="par-2"
        )

        # stream_concurrent catches the error — run completes
        await run_parallel(par1, ctx1, in_args=["a", "FAIL"])

        # Checkpoint: only index 0 completed
        raw = await store.load("parallel/par-2")
        assert raw is not None
        cp = ParallelCheckpoint.model_validate_json(raw)
        assert len(cp.completed) == 1
        assert 0 in cp.completed

        # Resume with a non-failing processor
        subproc2 = AppendProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=subproc2)
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="par-2"
        )

        result = await run_parallel(par2, ctx2, step=0)

        # "a" from checkpoint, "FAIL" re-run successfully
        assert sorted(result) == ["FAIL->worker", "a->worker"]

    @pytest.mark.asyncio
    async def test_resume_no_pending_is_noop(self) -> None:
        """If all copies completed, resume returns results from checkpoint."""
        store = InMemoryCheckpointStore()
        subproc = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="par-3"
        )

        await run_parallel(par, ctx, in_args=["a", "b"])

        # Resume
        subproc2 = CountingProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=subproc2)
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="par-3"
        )

        result = await run_parallel(par2, ctx2, step=0)
        assert sorted(result) == ["a->worker", "b->worker"]
        assert subproc2.call_count == 0

    @pytest.mark.asyncio
    async def test_checkpoint_stores_input_packet(self) -> None:
        """Checkpoint correctly round-trips the input packet."""
        store = InMemoryCheckpointStore()
        subproc = InputFailProcessor("worker", fail_input="FAIL")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="par-rt"
        )

        await run_parallel(par, ctx, in_args=["ok", "FAIL", "also_ok"])

        raw = await store.load("parallel/par-rt")
        assert raw is not None
        cp = ParallelCheckpoint.model_validate_json(raw)
        pkt = Packet[str].model_validate(cp.input_packet)
        assert list(pkt.payloads) == ["ok", "FAIL", "also_ok"]


# ---------- Crash-after-completion (re-delivery) ----------


class CrashAfterStepWorkflow(SequentialWorkflow[str, str, None]):
    """Crashes before saving the workflow checkpoint for a specific step."""

    def __init__(
        self,
        name: str,
        subprocs: list[Processor[Any, Any, None]],
        crash_after_step: int,
    ) -> None:
        super().__init__(name=name, subprocs=subprocs)
        self._crash_after_step = crash_after_step

    async def save_checkpoint(
        self,
        ctx: RunContext[None],
        *,
        completed_step: int,
        packet: Packet[Any],
    ) -> None:
        if completed_step == self._crash_after_step:
            raise RuntimeError(f"Simulated crash after step {completed_step}")
        await super().save_checkpoint(
            ctx, completed_step=completed_step, packet=packet
        )


class TestCrashAfterCompletion:
    """
    Verify the crash race window: child processor completes and saves its
    checkpoint, parent crashes before saving its own. On resume, the child
    must re-emit cached output without re-processing.
    """

    @pytest.mark.asyncio
    async def test_sequential_child_reemits_after_workflow_crash(self) -> None:
        """
        Workflow: [A, B].
        B completes (workflow checkpoint at step 0 only).
        Workflow crashes before saving step 1.
        Resume: B loads its checkpoint → all steps done → re-emits cached output.
        """
        store = InMemoryCheckpointStore()

        a1 = AppendProcessor("A")
        b1 = AppendProcessor("B")
        wf1 = CrashAfterStepWorkflow(
            name="wf", subprocs=[a1, b1], crash_after_step=1
        )
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="crash1"
        )

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="start")

        # Workflow checkpoint: step 0 done
        raw = await store.load("workflow/crash1")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.completed_step == 0

        # Resume: A skipped, B re-delivered (step=1 matches its checkpoint)
        a2 = CountingProcessor("A")
        b2 = CountingProcessor("B")
        wf2 = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[a2, b2]
        )
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="crash1"
        )

        result = await run_workflow(wf2, ctx2)
        assert result == ["start->A->B"]
        assert a2.call_count == 0  # skipped
        assert b2.call_count == 1  # re-ran (simple processor, no agent checkpoint)

    @pytest.mark.asyncio
    async def test_parallel_child_reemits_after_workflow_crash(self) -> None:
        """
        Workflow: [Fan, Parallel(worker)].
        Parallel completes all items. Workflow crashes before saving step 1.
        Resume: Parallel loads its checkpoint → all items done → emits from map.
        """
        store = InMemoryCheckpointStore()

        class FanOut(Processor[str, str, None]):
            async def _process_stream(
                self,
                chat_inputs: Any | None = None,
                *,
                in_args: list[str] | None = None,
                exec_id: str,
                ctx: RunContext[None],
                step: int | None = None,
            ) -> AsyncIterator[Event[Any]]:
                for inp in in_args or []:
                    yield ProcPayloadOutEvent(
                        data=f"{inp}:a", source=self.name, exec_id=exec_id
                    )
                    yield ProcPayloadOutEvent(
                        data=f"{inp}:b", source=self.name, exec_id=exec_id
                    )

        fan1 = FanOut(name="fan")
        worker1 = AppendProcessor("worker")
        par1 = ParallelProcessor[str, str, None](subproc=worker1)
        wf1 = CrashAfterStepWorkflow(
            name="wf", subprocs=[fan1, par1], crash_after_step=1
        )
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="crash2"
        )

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="start")

        # Parallel checkpoint: both items done
        par_key = f"parallel/crash2/{par1.name}"
        par_raw = await store.load(par_key)
        assert par_raw is not None
        par_cp = ParallelCheckpoint.model_validate_json(par_raw)
        assert len(par_cp.completed) == 2

        # Resume
        fan2 = CountingProcessor("fan")
        worker2 = CountingProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=worker2)
        wf2 = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[fan2, par2]
        )
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="crash2"
        )

        result = await run_workflow(wf2, ctx2)
        assert fan2.call_count == 0  # skipped
        assert worker2.call_count == 0  # all items from checkpoint
        assert sorted(result) == ["start:a->worker", "start:b->worker"]


# ---------- Store failure propagation ----------


class FailingStore(InMemoryCheckpointStore):
    """A store that raises on save after N successful saves."""

    def __init__(self, fail_after: int = 0) -> None:
        super().__init__()
        self._save_count = 0
        self._fail_after = fail_after

    async def save(self, key: str, data: bytes) -> None:
        self._save_count += 1
        if self._save_count > self._fail_after:
            raise OSError(f"Simulated I/O error on save #{self._save_count}")
        await super().save(key, data)


class TestStoreFailurePropagation:
    """Verify that store save failures propagate up and kill the run."""

    @pytest.mark.asyncio
    async def test_workflow_checkpoint_save_failure_kills_run(self) -> None:
        """If workflow.save_checkpoint fails, the run dies immediately."""
        store = FailingStore(fail_after=0)  # fail on first save
        a = AppendProcessor("A")
        b = AppendProcessor("B")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="fail-wf"
        )

        with pytest.raises(ProcRunError):
            await run_workflow(wf, ctx, in_args="start")

    @pytest.mark.asyncio
    async def test_parallel_checkpoint_save_failure_kills_run(self) -> None:
        """If parallel.save_checkpoint fails, the run dies immediately."""
        store = FailingStore(fail_after=0)
        subproc = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="fail-par"
        )

        with pytest.raises(ProcRunError):
            await run_parallel(par, ctx, in_args=["a", "b"])

    @pytest.mark.asyncio
    async def test_corrupt_checkpoint_treated_as_fresh(self) -> None:
        """A corrupt checkpoint is logged and treated as if no checkpoint exists."""
        store = InMemoryCheckpointStore()
        await store.save("workflow/corrupt-wf", b"not valid json{{{")

        a = AppendProcessor("A")
        b = AppendProcessor("B")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="corrupt-wf"
        )

        # Should proceed as fresh run (corrupt checkpoint ignored)
        result = await run_workflow(wf, ctx, in_args="start")
        assert result == ["start->A->B"]
