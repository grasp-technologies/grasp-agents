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
) -> list[str]:
    """Run a workflow via run_stream and collect final payloads."""
    payloads: list[str] = []
    async for event in wf.run_stream(in_args=in_args, ctx=ctx, exec_id="test"):
        if isinstance(event, ProcPacketOutEvent) and event.source == wf.name:
            payloads = list(event.data.payloads)
    return payloads


async def run_parallel(
    par: ParallelProcessor[str, str, None],
    ctx: RunContext[None],
    in_args: list[str] | None = None,
) -> list[str]:
    """Run a ParallelProcessor and collect final payloads."""
    payloads: list[str] = []
    async for event in par.run_stream(in_args=in_args, ctx=ctx, exec_id="test"):
        if isinstance(event, ProcPacketOutEvent) and event.source == par.name:
            payloads = list(event.data.payloads)
    return payloads


# ---------- SequentialWorkflow ----------


class TestSequentialWorkflowCheckpoint:
    @pytest.mark.asyncio
    async def test_basic_run_without_session(self) -> None:
        """Workflow works normally without session/store."""
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
        wf.reset_session("seq-1")
        ctx: RunContext[None] = RunContext(state=None, store=store)

        result = await run_workflow(wf, ctx, in_args="start")
        assert result == ["start->A->B->C"]

        raw = await store.load("workflow/seq-1")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.completed_step == 2  # 0-indexed, all 3 steps done
        assert cp.session_id == "seq-1"

    @pytest.mark.asyncio
    async def test_resume_skips_completed_steps(self) -> None:
        """After a crash, resume skips steps that already completed."""
        store = InMemoryCheckpointStore()

        # First run: step A succeeds, step B fails
        a1 = CountingProcessor("A")
        b1 = FailOnCallProcessor("B", fail_on_call=1)
        c1 = CountingProcessor("C")
        wf1 = SequentialWorkflow[str, str, None](name="wf", subprocs=[a1, b1, c1])
        wf1.reset_session("seq-2")
        ctx1: RunContext[None] = RunContext(state=None, store=store)

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
        wf2.reset_session("seq-2")
        ctx2: RunContext[None] = RunContext(state=None, store=store)

        result = await run_workflow(wf2, ctx2)
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
        wf1.reset_session("seq-multi")
        ctx1: RunContext[None] = RunContext(state=None, store=store)

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="s")

        # Resume with working B
        fan2 = CountingProcessor("fan")
        b2 = AppendProcessor("B")
        wf2 = SequentialWorkflow[str, str, None](name="wf", subprocs=[fan2, b2])
        wf2.reset_session("seq-multi")
        ctx2: RunContext[None] = RunContext(state=None, store=store)

        result = await run_workflow(wf2, ctx2)
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

        wf.reset_session("loop-1")
        ctx: RunContext[None] = RunContext(state=None, store=store)

        result = await run_workflow(wf, ctx, in_args="s")
        assert result == ["s->A->B->A->B"]

        raw = await store.load("workflow/loop-1")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.iteration == 2

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
        wf1.reset_session("loop-2")
        ctx1: RunContext[None] = RunContext(state=None, store=store)

        with pytest.raises(ProcRunError):
            await run_workflow(wf1, ctx1, in_args="s")

        # Checkpoint: iteration 1, step 0 completed
        raw = await store.load("workflow/loop-2")
        assert raw is not None
        cp = WorkflowCheckpoint.model_validate_json(raw)
        assert cp.iteration == 1
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
        wf2.reset_session("loop-2")
        ctx2: RunContext[None] = RunContext(state=None, store=store)

        result = await run_workflow(wf2, ctx2)
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
        par.reset_session("par-1")
        ctx: RunContext[None] = RunContext(state=None, store=store)

        await run_parallel(par, ctx, in_args=["a", "b"])

        raw = await store.load("parallel/par-1")
        assert raw is not None
        cp = ParallelCheckpoint.model_validate_json(raw)
        assert len(cp.completed) == 2

    @pytest.mark.asyncio
    async def test_resume_skips_completed_copies(self) -> None:
        """After partial completion, resume only runs pending copies.

        stream_concurrent swallows per-copy errors, so the first run
        completes with None for the failed copy. On resume, only that
        copy is re-run.
        """
        store = InMemoryCheckpointStore()

        # "a" succeeds, "FAIL" crashes (InputFailProcessor survives deepcopy)
        subproc1 = InputFailProcessor("worker", fail_input="FAIL")
        par1 = ParallelProcessor[str, str, None](subproc=subproc1)
        par1.reset_session("par-2")
        ctx1: RunContext[None] = RunContext(state=None, store=store)

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
        par2.reset_session("par-2")
        ctx2: RunContext[None] = RunContext(state=None, store=store)

        result = await run_parallel(par2, ctx2)

        # "a" from checkpoint, "FAIL" re-run successfully
        assert sorted(result) == ["FAIL->worker", "a->worker"]

    @pytest.mark.asyncio
    async def test_resume_no_pending_is_noop(self) -> None:
        """If all copies completed, resume returns results from checkpoint."""
        store = InMemoryCheckpointStore()
        subproc = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        par.reset_session("par-3")
        ctx: RunContext[None] = RunContext(state=None, store=store)

        await run_parallel(par, ctx, in_args=["a", "b"])

        # Resume
        subproc2 = CountingProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=subproc2)
        par2.reset_session("par-3")
        ctx2: RunContext[None] = RunContext(state=None, store=store)

        result = await run_parallel(par2, ctx2)
        assert sorted(result) == ["a->worker", "b->worker"]
        assert subproc2.call_count == 0

    @pytest.mark.asyncio
    async def test_checkpoint_stores_input_packet(self) -> None:
        """Checkpoint correctly round-trips the input packet."""
        store = InMemoryCheckpointStore()
        subproc = InputFailProcessor("worker", fail_input="FAIL")
        par = ParallelProcessor[str, str, None](subproc=subproc)
        par.reset_session("par-rt")
        ctx: RunContext[None] = RunContext(state=None, store=store)

        await run_parallel(par, ctx, in_args=["ok", "FAIL", "also_ok"])

        raw = await store.load("parallel/par-rt")
        assert raw is not None
        cp = ParallelCheckpoint.model_validate_json(raw)
        pkt = Packet[str].model_validate(cp.input_packet)
        assert list(pkt.payloads) == ["ok", "FAIL", "also_ok"]
