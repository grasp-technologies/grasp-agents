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
from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.processors.processor import Processor
from grasp_agents.session_context import SessionContext
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.events import (
    Event,
    ProcPacketOutEvent,
    ProcPayloadOutEvent,
)
from grasp_agents.types.io import ProcName
from grasp_agents.types.packet import Packet
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
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for inp in in_args or []:
            if self._fail_input in inp:
                raise RuntimeError(f"{self.name} fails on input '{inp}'")
            output = f"{inp}->{self.name}"
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)


async def run_workflow(
    wf: SequentialWorkflow[str, str, None] | LoopedWorkflow[str, str, None],
    ctx: SessionContext[None],
    in_args: str | None = None,
    step: int | None = None,
) -> list[str]:
    """Run a workflow via run_stream and collect final payloads."""
    wf.on_adopted(ctx=ctx)
    payloads: list[str] = []
    async for event in wf.run_stream(in_args=in_args, exec_id="test", step=step):
        if isinstance(event, ProcPacketOutEvent) and event.source == wf.name:
            payloads = list(event.data.payloads)
    return payloads


async def run_parallel(
    par: ParallelProcessor[str, str, None],
    ctx: SessionContext[None],
    in_args: list[str] | None = None,
    step: int | None = None,
) -> list[str]:
    """Run a ParallelProcessor and collect final payloads."""
    par.on_adopted(ctx=ctx)
    payloads: list[str] = []
    async for event in par.run_stream(in_args=in_args, exec_id="test", step=step):
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
        ctx: SessionContext[None] = SessionContext(state=None)
        wf.on_adopted(ctx=ctx)

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
        ctx: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="seq-1"
        )
        wf.on_adopted(ctx=ctx)

        result = await run_workflow(wf, ctx, in_args="start")
        assert result == ["start->A->B->C"]

        raw = await store.load("seq-1/workflow/wf")
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
        ctx1: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="seq-2"
        )
        wf1.on_adopted(ctx=ctx1)

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="start")

        # Checkpoint: step 0 completed
        raw = await store.load("seq-2/workflow/wf")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.completed_step == 0

        # Resume
        a2 = CountingProcessor("A")
        b2 = CountingProcessor("B")
        c2 = CountingProcessor("C")
        wf2 = SequentialWorkflow[str, str, None](name="wf", subprocs=[a2, b2, c2])
        ctx2: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="seq-2"
        )
        wf2.on_adopted(ctx=ctx2)

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
        ctx1: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="seq-multi"
        )
        wf1.on_adopted(ctx=ctx1)

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="s")

        # Resume with working B
        fan2 = CountingProcessor("fan")
        b2 = AppendProcessor("B")
        wf2 = SequentialWorkflow[str, str, None](name="wf", subprocs=[fan2, b2])
        ctx2: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="seq-multi"
        )
        wf2.on_adopted(ctx=ctx2)

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

        def _always_terminate(out_packet: Packet[str], **kwargs: Any) -> bool:
            return True

        wf.terminate_workflow_loop_impl = _always_terminate

        ctx: SessionContext[None] = SessionContext(state=None)
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

        def count_terminate(out_packet: Packet[str], **kwargs: Any) -> bool:
            nonlocal iteration_count
            iteration_count += 1
            return iteration_count >= 2

        wf.add_workflow_loop_terminator(count_terminate)

        ctx: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="loop-1"
        )

        result = await run_workflow(wf, ctx, in_args="s")
        assert result == ["s->A->B->A->B"]

        raw = await store.load("loop-1/workflow/loop")
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
        ctx1: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="loop-2"
        )
        wf1.on_adopted(ctx=ctx1)

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="s")

        # Checkpoint: global step 0 completed (iteration 0, subproc A)
        raw = await store.load("loop-2/workflow/loop")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.completed_step == 0

        # Resume — terminate after iteration 1
        a2 = CountingProcessor("A")
        b2 = CountingProcessor("B")
        wf2 = LoopedWorkflow[str, str, None](
            name="loop", subprocs=[a2, b2], exit_proc=b2, max_iterations=2
        )

        def _always_terminate2(out_packet: Packet[str], **kwargs: Any) -> bool:
            return True

        wf2.terminate_workflow_loop_impl = _always_terminate2
        ctx2: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="loop-2"
        )

        result = await run_workflow(wf2, ctx2, step=0)
        assert result == ["s->A->B"]

        # A was skipped (completed in iteration 1), B ran once
        assert a2.call_count == 0
        assert b2.call_count == 1

    @pytest.mark.asyncio
    async def test_resume_after_max_iterations_with_midloop_exit(self) -> None:
        """
        A max-iterations exit at an exit_proc that is NOT the last subproc
        resumes by re-emitting the saved output — without re-running the
        trailing subprocs.
        """
        store = InMemoryCheckpointStore()
        # exit_proc B sits in the MIDDLE, so the max-iterations exit leaves a
        # trailing C that must not re-run on resume.
        a, b, c = CountingProcessor("A"), CountingProcessor("B"), CountingProcessor("C")
        wf = LoopedWorkflow[str, str, None](
            name="loop", subprocs=[a, b, c], exit_proc=b, max_iterations=2
        )
        ctx: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="loop-midexit"
        )
        first = await run_workflow(wf, ctx, in_args="s")
        assert first == ["s->A->B->C->A->B"]  # exits at B of the last iteration

        a2, b2, c2 = (
            CountingProcessor("A"),
            CountingProcessor("B"),
            CountingProcessor("C"),
        )
        wf2 = LoopedWorkflow[str, str, None](
            name="loop", subprocs=[a2, b2, c2], exit_proc=b2, max_iterations=2
        )
        ctx2: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="loop-midexit"
        )
        resumed = await run_workflow(wf2, ctx2)
        # Output re-emitted (not empty) and no subproc re-run — least of all C.
        assert resumed == ["s->A->B->C->A->B"]
        assert (a2.call_count, b2.call_count, c2.call_count) == (0, 0, 0)


# ---------- ParallelProcessor ----------


class TestParallelProcessorCheckpoint:
    @pytest.mark.asyncio
    async def test_basic_parallel_without_session(self) -> None:
        subproc = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        ctx: SessionContext[None] = SessionContext(state=None)
        par.on_adopted(ctx=ctx)

        result = await run_parallel(par, ctx, in_args=["a", "b", "c"])
        assert sorted(result) == ["a->worker_0", "b->worker_1", "c->worker_2"]

    @pytest.mark.asyncio
    async def test_checkpoint_saved_per_completion(self) -> None:
        store = InMemoryCheckpointStore()
        subproc = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        ctx: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="par-1"
        )
        par.on_adopted(ctx=ctx)

        await run_parallel(par, ctx, in_args=["a", "b"])

        raw = await store.load("par-1/parallel/worker_par")
        assert raw is not None
        cp = ParallelCheckpoint.model_validate_json(raw)
        assert len(cp.completed) == 2

    @pytest.mark.asyncio
    async def test_resume_skips_completed_copies(self) -> None:
        """
        After a partial failure, resume only runs the pending copies.

        A failed copy fails the whole run (loudly), but the completed copies'
        checkpoint survives — resume re-runs only the failed one.
        """
        store = InMemoryCheckpointStore()

        # "a" succeeds, "FAIL" crashes (InputFailProcessor survives deepcopy)
        subproc1 = InputFailProcessor("worker", fail_input="FAIL")
        par1 = ParallelProcessor[str, str, None](subproc=subproc1)
        ctx1: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="par-2"
        )
        par1.on_adopted(ctx=ctx1)

        # The failed copy surfaces as an error (no silent None payloads) …
        with pytest.raises(ProcRunError):
            await run_parallel(par1, ctx1, in_args=["a", "FAIL"])

        # … but the checkpoint survives: only index 0 completed
        raw = await store.load("par-2/parallel/worker_par")
        assert raw is not None
        cp = ParallelCheckpoint.model_validate_json(raw)
        assert len(cp.completed) == 1
        assert 0 in cp.completed

        # Resume with a non-failing processor
        subproc2 = AppendProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=subproc2)
        ctx2: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="par-2"
        )
        par2.on_adopted(ctx=ctx2)

        result = await run_parallel(par2, ctx2, step=0)

        # "a" from checkpoint, "FAIL" re-run successfully
        assert sorted(result) == ["FAIL->worker_1", "a->worker_0"]

    @pytest.mark.asyncio
    async def test_resume_no_pending_is_noop(self) -> None:
        """If all copies completed, resume returns results from checkpoint."""
        store = InMemoryCheckpointStore()
        subproc = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        ctx: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="par-3"
        )
        par.on_adopted(ctx=ctx)

        await run_parallel(par, ctx, in_args=["a", "b"])

        # Resume
        subproc2 = CountingProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=subproc2)
        ctx2: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="par-3"
        )
        par2.on_adopted(ctx=ctx2)

        result = await run_parallel(par2, ctx2, step=0)
        assert sorted(result) == ["a->worker_0", "b->worker_1"]
        assert subproc2.call_count == 0

    @pytest.mark.asyncio
    async def test_drop_failed_drops_failed_copies(self) -> None:
        """drop_failed=True drops failures — no ``None`` payloads downstream."""
        subproc = InputFailProcessor("worker", fail_input="FAIL")
        par = ParallelProcessor[str, str, None](subproc=subproc, drop_failed=True)
        ctx: SessionContext[None] = SessionContext(state=None)

        result = await run_parallel(par, ctx, in_args=["ok", "FAIL"])

        assert result == ["ok->worker_0"]
        assert None not in result

    @pytest.mark.asyncio
    async def test_on_error_drop_matches_drop_failed(self) -> None:
        """on_error='drop' compacts failures, like the deprecated drop_failed=True."""
        subproc = InputFailProcessor("worker", fail_input="FAIL")
        par = ParallelProcessor[str, str, None](subproc=subproc, on_error="drop")
        ctx: SessionContext[None] = SessionContext(state=None)

        result = await run_parallel(par, ctx, in_args=["ok", "FAIL"])

        assert result == ["ok->worker_0"]

    @pytest.mark.asyncio
    async def test_on_error_keep_aligns_failed_slots(self) -> None:
        """on_error='keep' keeps a None at each failed slot and records why."""
        subproc = InputFailProcessor("worker", fail_input="FAIL")
        par = ParallelProcessor[str, str, None](subproc=subproc, on_error="keep")
        ctx: SessionContext[None] = SessionContext(state=None)
        par.on_adopted(ctx=ctx)

        out_packet: Packet[str] | None = None
        async for event in par.run_stream(
            in_args=["ok", "FAIL", "also_ok"], exec_id="test", step=0
        ):
            if isinstance(event, ProcPacketOutEvent) and event.source == par.name:
                out_packet = event.data

        assert out_packet is not None
        # Payloads stay aligned 1:1 with the inputs; the failed slot is None,
        # so zip(inputs, payloads, strict=True) still lines up.
        assert list(out_packet.payloads) == ["ok->worker_0", None, "also_ok->worker_2"]
        # The failure map travels on the packet.
        assert set(out_packet.failed) == {1}
        assert "FAIL" in out_packet.failed[1].error
        # error_type is the root cause (RuntimeError), not the retry wrapper.
        assert out_packet.failed[1].error_type == "RuntimeError"

    @pytest.mark.asyncio
    async def test_on_error_keep_no_failures_has_empty_map(self) -> None:
        """on_error='keep' with no failures returns aligned payloads, empty map."""
        subproc = InputFailProcessor("worker", fail_input="FAIL")
        par = ParallelProcessor[str, str, None](subproc=subproc, on_error="keep")
        ctx: SessionContext[None] = SessionContext(state=None)
        par.on_adopted(ctx=ctx)

        out_packet: Packet[str] | None = None
        async for event in par.run_stream(
            in_args=["ok", "also_ok"], exec_id="test", step=0
        ):
            if isinstance(event, ProcPacketOutEvent) and event.source == par.name:
                out_packet = event.data

        assert out_packet is not None
        assert list(out_packet.payloads) == ["ok->worker_0", "also_ok->worker_1"]
        assert out_packet.failed == {}

    def test_on_error_and_drop_failed_conflict(self) -> None:
        """Passing both on_error and drop_failed is rejected."""
        subproc = InputFailProcessor("worker")
        with pytest.raises(ValueError, match="not both"):
            ParallelProcessor[str, str, None](
                subproc=subproc, on_error="keep", drop_failed=True
            )

    @pytest.mark.asyncio
    async def test_checkpoint_stores_input_packet(self) -> None:
        """Checkpoint correctly round-trips the input packet."""
        store = InMemoryCheckpointStore()
        subproc = InputFailProcessor("worker", fail_input="FAIL")
        # drop_failed: the failed copy is dropped instead of failing the run.
        par = ParallelProcessor[str, str, None](subproc=subproc, drop_failed=True)
        ctx: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="par-rt"
        )
        par.on_adopted(ctx=ctx)

        await run_parallel(par, ctx, in_args=["ok", "FAIL", "also_ok"])

        raw = await store.load("par-rt/parallel/worker_par")
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
        *,
        completed_step: int,
        packet: Packet[Any],
    ) -> None:
        if completed_step == self._crash_after_step:
            raise RuntimeError(f"Simulated crash after step {completed_step}")
        await super().save_checkpoint(completed_step=completed_step, packet=packet)


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
        wf1 = CrashAfterStepWorkflow(name="wf", subprocs=[a1, b1], crash_after_step=1)
        ctx1: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="crash1"
        )

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="start")

        # Workflow checkpoint: step 0 done
        raw = await store.load("crash1/workflow/wf")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.completed_step == 0

        # Resume: A skipped, B re-delivered (step=1 matches its checkpoint)
        a2 = CountingProcessor("A")
        b2 = CountingProcessor("B")
        wf2 = SequentialWorkflow[str, str, None](name="wf", subprocs=[a2, b2])
        ctx2: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="crash1"
        )
        wf2.on_adopted(ctx=ctx2)

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
        ctx1: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="crash2"
        )
        par1.on_adopted(ctx=ctx1)

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="start")

        # Parallel checkpoint: both items done
        par_key = f"crash2/parallel/wf/{par1.name}"
        par_raw = await store.load(par_key)
        assert par_raw is not None
        par_cp = ParallelCheckpoint.model_validate_json(par_raw)
        assert len(par_cp.completed) == 2

        # Resume
        fan2 = CountingProcessor("fan")
        worker2 = CountingProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=worker2)
        wf2 = SequentialWorkflow[str, str, None](name="wf", subprocs=[fan2, par2])
        ctx2: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="crash2"
        )
        wf2.on_adopted(ctx=ctx2)

        result = await run_workflow(wf2, ctx2)
        assert fan2.call_count == 0  # skipped
        assert worker2.call_count == 0  # all items from checkpoint
        assert sorted(result) == ["start:a->worker_0", "start:b->worker_1"]


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
        ctx: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="fail-wf"
        )
        wf.on_adopted(ctx=ctx)

        with pytest.raises(ProcRunError):
            await run_workflow(wf, ctx, in_args="start")

    @pytest.mark.asyncio
    async def test_parallel_checkpoint_save_failure_kills_run(self) -> None:
        """If parallel.save_checkpoint fails, the run dies immediately."""
        store = FailingStore(fail_after=0)
        subproc = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        ctx: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="fail-par"
        )
        par.on_adopted(ctx=ctx)

        with pytest.raises(ProcRunError):
            await run_parallel(par, ctx, in_args=["a", "b"])

    @pytest.mark.asyncio
    async def test_corrupt_checkpoint_treated_as_fresh(self) -> None:
        """A corrupt checkpoint is logged and treated as if no checkpoint exists."""
        store = InMemoryCheckpointStore()
        await store.save("corrupt-wf/workflow/wf", b"not valid json{{{")

        a = AppendProcessor("A")
        b = AppendProcessor("B")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])
        ctx: SessionContext[None] = SessionContext(
            state=None, checkpoint_store=store, session_key="corrupt-wf"
        )
        wf.on_adopted(ctx=ctx)

        # Should proceed as fresh run (corrupt checkpoint ignored)
        result = await run_workflow(wf, ctx, in_args="start")
        assert result == ["start->A->B"]
