"""
Tests for recursive session propagation and checkpointing across nested
composite processors.

Verifies that:
- setup_session propagates session IDs through Workflow -> subprocs
- setup_session propagates through ParallelProcessor -> subproc
- Nested composites get correctly namespaced sessions at every level
- Checkpoints are saved at each level independently
- Resume works across multiple nesting levels: workflow skips completed steps,
  parallel skips completed items, inner processors resume from their own state
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

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
from grasp_agents.runner.runner import END_PROC_NAME, Runner
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.events import (
    Event,
    ProcPacketOutEvent,
    ProcPayloadOutEvent,
    RunPacketOutEvent,
)
from grasp_agents.types.io import ProcName
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
            yield ProcPayloadOutEvent(
                data=f"{inp}->{self.name}", source=self.name, exec_id=exec_id
            )


class CountingProcessor(Processor[str, str, None]):
    """Tracks call count and appends its name."""

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
        for inp in in_args or []:
            yield ProcPayloadOutEvent(
                data=f"{inp}->{self.name}", source=self.name, exec_id=exec_id
            )


class FanOutProcessor(Processor[str, str, None]):
    """Produces two outputs per input: {inp}:a and {inp}:b."""

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
            yield ProcPayloadOutEvent(
                data=f"{inp}:a", source=self.name, exec_id=exec_id
            )
            yield ProcPayloadOutEvent(
                data=f"{inp}:b", source=self.name, exec_id=exec_id
            )


class InputFailProcessor(Processor[str, str, None]):
    """Fails when any input contains a specific substring. Survives deepcopy."""

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
                raise RuntimeError(f"{self.name} fails on '{inp}'")
            yield ProcPayloadOutEvent(
                data=f"{inp}->{self.name}", source=self.name, exec_id=exec_id
            )


class CrashAfterStepWorkflow(SequentialWorkflow[str, str, None]):
    """
    SequentialWorkflow that raises before saving the checkpoint for a
    specific completed step, simulating a crash between a subproc finishing
    and the workflow persisting its checkpoint.
    """

    def __init__(
        self,
        name: str,
        subprocs: Sequence[Processor[Any, Any, None]],
        crash_after_step: int,
        session_id: str | None = None,
    ) -> None:
        super().__init__(name=name, subprocs=list(subprocs), session_id=session_id)
        self._crash_after_step = crash_after_step

    async def save_checkpoint(
        self,
        ctx: RunContext[None],
        *,
        completed_step: int,
        packet: Packet[Any],
    ) -> None:
        if completed_step == self._crash_after_step:
            raise RuntimeError(
                f"Simulated crash before saving step {completed_step}"
            )
        await super().save_checkpoint(
            ctx, completed_step=completed_step, packet=packet
        )


def _resolve_inputs(
    chat_inputs: Any | None, in_args: list[str] | None
) -> list[str]:
    if in_args is not None:
        return in_args
    if chat_inputs is not None:
        return [str(chat_inputs)]
    return []


class ChatAppendProcessor(Processor[str, str, None]):
    """Like AppendProcessor but also handles chat_inputs (for Runner entry)."""

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
                data=f"{inp}->{self.name}", source=self.name, exec_id=exec_id
            )


class ChatFanOutProcessor(Processor[str, str, None]):
    """FanOut that handles chat_inputs (for Runner entry)."""

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
                data=f"{inp}:a", source=self.name, exec_id=exec_id
            )
            yield ProcPayloadOutEvent(
                data=f"{inp}:b", source=self.name, exec_id=exec_id
            )


async def collect_payloads(
    proc: Processor[str, str, None]
    | SequentialWorkflow[str, str, None]
    | CrashAfterStepWorkflow,
    ctx: RunContext[None],
    in_args: str | list[str] | None = None,
    step: int | None = None,
) -> list[str]:
    """Run a processor via run_stream and collect final payloads."""
    payloads: list[str] = []
    async for event in proc.run_stream(
        in_args=in_args, ctx=ctx, exec_id="test", step=step
    ):
        if isinstance(event, ProcPacketOutEvent) and event.source == proc.name:
            payloads = list(event.data.payloads)
    return payloads


async def collect_runner_payloads(
    runner: Runner[str, None],
    chat_inputs: Any = "start",
) -> list[str]:
    """Run a Runner and collect final payloads."""
    payloads: list[str] = []
    async for event in runner.run_stream(chat_inputs=chat_inputs):
        if isinstance(event, RunPacketOutEvent):
            payloads = list(event.data.payloads)
    return payloads


# ---------- Session propagation ----------


class TestRecursiveSessionPropagation:
    def test_workflow_propagates_to_subprocs(self) -> None:
        """setup_session sets namespaced session_id on each subproc."""
        a = AppendProcessor("A")
        b = AppendProcessor("B")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])
        wf.setup_session("sess")

        assert wf._session_id == "sess"
        assert a._session_id == "sess/A"
        assert b._session_id == "sess/B"

    def test_parallel_propagates_to_subproc(self) -> None:
        """ParallelProcessor.setup_session propagates to inner subproc."""
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        par.setup_session("sess")

        assert par._session_id == "sess"
        assert worker._session_id == "sess/worker"

    def test_parallel_init_with_session_propagates(self) -> None:
        """session_id passed to ParallelProcessor.__init__ propagates to subproc."""
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](
            subproc=worker, session_id="sess"
        )

        assert par._session_id == "sess"
        assert worker._session_id == "sess/worker"

    def test_workflow_containing_parallel_three_levels(self) -> None:
        """Workflow -> ParallelProcessor -> worker gets 3-level namespacing."""
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        a = AppendProcessor("A")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, par])
        wf.setup_session("sess")

        assert wf._session_id == "sess"
        assert a._session_id == "sess/A"
        assert par._session_id == f"sess/{par.name}"
        assert worker._session_id == f"sess/{par.name}/worker"

    def test_nested_workflows_propagate_to_leaf_procs(self) -> None:
        """Outer -> Inner workflow propagates to all leaf processors."""
        x = AppendProcessor("X")
        y = AppendProcessor("Y")
        inner = SequentialWorkflow[str, str, None](name="inner", subprocs=[x, y])

        a = AppendProcessor("A")
        outer = SequentialWorkflow[str, str, None](name="outer", subprocs=[a, inner])
        outer.setup_session("sess")

        assert outer._session_id == "sess"
        assert a._session_id == "sess/A"
        assert inner._session_id == "sess/inner"
        assert x._session_id == "sess/inner/X"
        assert y._session_id == "sess/inner/Y"

    def test_all_levels_resumable_after_setup(self) -> None:
        """Resumable is True at every level after setup_session."""
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        a = AppendProcessor("A")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, par])

        # Before: nothing is resumable
        assert not wf.resumable
        assert not a.resumable
        assert not par.resumable
        assert not worker.resumable

        wf.setup_session("sess")

        # After: everything is resumable
        assert wf.resumable
        assert a.resumable
        assert par.resumable
        assert worker.resumable

    def test_runner_propagates_through_workflow_and_parallel(self) -> None:
        """Runner -> Workflow -> Parallel -> worker: 4-level propagation."""
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        fan = AppendProcessor("fan")
        wf = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[fan, par], recipients=[END_PROC_NAME]
        )

        entry = ChatAppendProcessor("entry", recipients=["wf"])
        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](
            entry_proc=entry, procs=[entry, wf], ctx=ctx, name="r"
        )
        runner.setup_session("sess")

        assert entry._session_id == "sess/entry"
        assert wf._session_id == "sess/wf"
        assert fan._session_id == "sess/wf/fan"
        assert par._session_id == f"sess/wf/{par.name}"
        assert worker._session_id == f"sess/wf/{par.name}/worker"


# ---------- Checkpoint storage ----------


class TestRecursiveCheckpointStorage:
    @pytest.mark.asyncio
    async def test_workflow_and_parallel_save_independent_checkpoints(self) -> None:
        """Both the workflow and its ParallelProcessor subproc save to the store."""
        store = InMemoryCheckpointStore()
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        fan = FanOutProcessor("fan")
        collect = AppendProcessor("collect")
        wf = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[fan, par, collect]
        )
        wf.setup_session("s1")
        ctx: RunContext[None] = RunContext(state=None, store=store)

        result = await collect_payloads(wf, ctx, in_args="start")
        assert sorted(result) == [
            "start:a->worker->collect",
            "start:b->worker->collect",
        ]

        # Workflow checkpoint exists
        wf_raw = await store.load("workflow/s1")
        assert wf_raw is not None
        wf_cp = WorkflowCheckpoint.model_validate_json(wf_raw)
        assert wf_cp.completed_step == 2  # all 3 steps done

        # Parallel checkpoint exists with correctly namespaced key
        par_key = f"parallel/s1/{par.name}"
        par_raw = await store.load(par_key)
        assert par_raw is not None
        par_cp = ParallelCheckpoint.model_validate_json(par_raw)
        assert len(par_cp.completed) == 2  # 2 items from FanOut

    @pytest.mark.asyncio
    async def test_nested_workflow_checkpoints_at_both_levels(self) -> None:
        """Outer and inner workflows save independent checkpoints."""
        store = InMemoryCheckpointStore()
        x = AppendProcessor("X")
        y = AppendProcessor("Y")
        inner = SequentialWorkflow[str, str, None](name="inner", subprocs=[x, y])

        a = AppendProcessor("A")
        outer = SequentialWorkflow[str, str, None](name="outer", subprocs=[a, inner])
        outer.setup_session("s2")
        ctx: RunContext[None] = RunContext(state=None, store=store)

        result = await collect_payloads(outer, ctx, in_args="start")
        assert result == ["start->A->X->Y"]

        # Both checkpoints exist with correct namespacing
        outer_raw = await store.load("workflow/s2")
        assert outer_raw is not None

        inner_raw = await store.load("workflow/s2/inner")
        assert inner_raw is not None

    @pytest.mark.asyncio
    async def test_checkpoint_keys_for_deep_nesting(self) -> None:
        """Workflow -> Parallel -> worker: all checkpoint keys are namespaced."""
        store = InMemoryCheckpointStore()
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        fan = FanOutProcessor("fan")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[fan, par])
        wf.setup_session("deep")
        ctx: RunContext[None] = RunContext(state=None, store=store)

        await collect_payloads(wf, ctx, in_args="start")

        # Workflow: "workflow/{session_id}"
        assert await store.load("workflow/deep") is not None
        # Parallel: "parallel/{session_id}/{par.name}"
        assert await store.load(f"parallel/deep/{par.name}") is not None


# ---------- Resume ----------


class TestRecursiveResume:
    @pytest.mark.asyncio
    async def test_parallel_checkpoint_survives_workflow_crash(self) -> None:
        """
        Workflow: fan -> parallel(worker) -> collect.
        Crash after parallel step (before workflow saves step 1 checkpoint).
        Resume: workflow skips fan, parallel loads its own checkpoint (all items
        done), collect runs.
        """
        store = InMemoryCheckpointStore()

        # First run: crash after step 1 (parallel)
        worker1 = AppendProcessor("worker")
        par1 = ParallelProcessor[str, str, None](subproc=worker1)
        fan1 = FanOutProcessor("fan")
        collect1 = AppendProcessor("collect")
        wf1 = CrashAfterStepWorkflow(
            name="wf",
            subprocs=[fan1, par1, collect1],
            crash_after_step=1,
            session_id="r1",
        )
        ctx1: RunContext[None] = RunContext(state=None, store=store)

        with pytest.raises(ProcRunError):
            await collect_payloads(wf1, ctx1, in_args="start")

        # Workflow checkpoint at step 0 (fan succeeded), step 1 save crashed
        wf_raw = await store.load("workflow/r1")
        assert wf_raw is not None
        wf_cp = WorkflowCheckpoint.model_validate_json(wf_raw)
        assert wf_cp.completed_step == 0

        # Parallel checkpoint exists (saved internally before the workflow crash)
        par_key = f"parallel/r1/{par1.name}"
        assert await store.load(par_key) is not None

        # Resume with fresh processors
        worker2 = AppendProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=worker2)
        fan2 = CountingProcessor("fan")
        collect2 = CountingProcessor("collect")
        wf2 = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[fan2, par2, collect2], session_id="r1"
        )
        ctx2: RunContext[None] = RunContext(state=None, store=store)

        result = await collect_payloads(wf2, ctx2, step=0)

        assert fan2.call_count == 0  # skipped (step 0 done)
        assert collect2.call_count == 1  # ran
        assert sorted(result) == [
            "start:a->worker->collect",
            "start:b->worker->collect",
        ]

    @pytest.mark.asyncio
    async def test_partial_parallel_survives_workflow_crash(self) -> None:
        """
        Workflow: fan -> parallel(failing_worker) -> collect.
        Parallel partially completes (1 of 2 items succeeds). Workflow crashes.
        Resume: parallel loads checkpoint, skips the completed item, re-runs only
        the failed one with a new (non-failing) worker.
        """
        store = InMemoryCheckpointStore()

        # First run: worker fails on ":b" inputs
        worker1 = InputFailProcessor("worker", fail_input=":b")
        par1 = ParallelProcessor[str, str, None](subproc=worker1)
        fan1 = FanOutProcessor("fan")
        collect1 = AppendProcessor("collect")
        wf1 = CrashAfterStepWorkflow(
            name="wf",
            subprocs=[fan1, par1, collect1],
            crash_after_step=1,
            session_id="r2",
        )
        ctx1: RunContext[None] = RunContext(state=None, store=store)

        with pytest.raises(ProcRunError):
            await collect_payloads(wf1, ctx1, in_args="start")

        # Parallel checkpoint: only item 0 completed (item 1 failed)
        par_key = f"parallel/r2/{par1.name}"
        par_raw = await store.load(par_key)
        assert par_raw is not None
        par_cp = ParallelCheckpoint.model_validate_json(par_raw)
        assert len(par_cp.completed) == 1
        assert 0 in par_cp.completed  # "start:a" succeeded

        # Resume with non-failing worker
        worker2 = AppendProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=worker2)
        fan2 = CountingProcessor("fan")
        collect2 = CountingProcessor("collect")
        wf2 = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[fan2, par2, collect2], session_id="r2"
        )
        ctx2: RunContext[None] = RunContext(state=None, store=store)

        result = await collect_payloads(wf2, ctx2, step=0)

        assert fan2.call_count == 0  # skipped
        assert collect2.call_count == 1
        # Both items present: item 0 from checkpoint, item 1 re-run
        assert sorted(result) == [
            "start:a->worker->collect",
            "start:b->worker->collect",
        ]

    @pytest.mark.asyncio
    async def test_inner_workflow_checkpoint_survives_outer_crash(self) -> None:
        """
        Outer: [A, Inner([X, Y])].
        Inner completes (saves its own checkpoint). Outer crashes before saving
        step 1. Resume: outer skips A, inner loads checkpoint (all steps done),
        emits cached output.
        """
        store = InMemoryCheckpointStore()

        # First run
        x1 = CountingProcessor("X")
        y1 = CountingProcessor("Y")
        inner1 = SequentialWorkflow[str, str, None](name="inner", subprocs=[x1, y1])
        a1 = AppendProcessor("A")
        outer1 = CrashAfterStepWorkflow(
            name="outer",
            subprocs=[a1, inner1],
            crash_after_step=1,
            session_id="r3",
        )
        ctx1: RunContext[None] = RunContext(state=None, store=store)

        with pytest.raises(ProcRunError):
            await collect_payloads(outer1, ctx1, in_args="start")

        # Inner checkpoint exists (both steps done)
        inner_raw = await store.load("workflow/r3/inner")
        assert inner_raw is not None
        inner_cp = WorkflowCheckpoint.model_validate_json(inner_raw)
        assert inner_cp.completed_step == 1  # both X and Y done

        # Resume
        x2 = CountingProcessor("X")
        y2 = CountingProcessor("Y")
        inner2 = SequentialWorkflow[str, str, None](name="inner", subprocs=[x2, y2])
        a2 = CountingProcessor("A")
        outer2 = SequentialWorkflow[str, str, None](
            name="outer", subprocs=[a2, inner2], session_id="r3"
        )
        ctx2: RunContext[None] = RunContext(state=None, store=store)

        result = await collect_payloads(outer2, ctx2, step=0)

        assert a2.call_count == 0  # skipped (outer checkpoint at step 0)
        assert x2.call_count == 0  # skipped (inner checkpoint)
        assert y2.call_count == 0  # skipped (inner checkpoint)
        assert result == ["start->A->X->Y"]

    @pytest.mark.asyncio
    async def test_workflow_parallel_nested_resume_end_to_end(self) -> None:
        """
        3-level recovery:
          Outer: [fan, Inner([par(failing_worker), collect])]
          Inner crashes after its parallel step saves. Parallel's internal
          checkpoint (item 0 done) survives but inner's workflow checkpoint
          does not.

        On resume all three levels contribute:
          - Outer checkpoint (step 0) -> skips fan
          - Inner has no checkpoint -> starts from step 0 (parallel)
          - Parallel checkpoint (item 0 done) -> only re-runs item 1
        """
        store = InMemoryCheckpointStore()

        # First run: worker fails on ":b", inner crashes after parallel step
        worker1 = InputFailProcessor("worker", fail_input=":b")
        par1 = ParallelProcessor[str, str, None](subproc=worker1)
        collect1 = AppendProcessor("collect")
        inner1 = CrashAfterStepWorkflow(
            name="inner",
            subprocs=[par1, collect1],
            crash_after_step=0,  # crash after parallel step, before inner saves
        )
        fan1 = FanOutProcessor("fan")
        outer1 = SequentialWorkflow[str, str, None](
            name="outer", subprocs=[fan1, inner1], session_id="r4"
        )
        ctx1: RunContext[None] = RunContext(state=None, store=store)

        with pytest.raises(ProcRunError):
            await collect_payloads(outer1, ctx1, in_args="start")

        # Outer checkpoint at step 0 (fan succeeded, inner crashed)
        outer_raw = await store.load("workflow/r4")
        assert outer_raw is not None
        outer_cp = WorkflowCheckpoint.model_validate_json(outer_raw)
        assert outer_cp.completed_step == 0

        # Inner has NO checkpoint (crash prevented saving)
        assert await store.load("workflow/r4/inner") is None

        # But parallel's checkpoint exists with item 0 completed
        par_key = f"parallel/r4/inner/{par1.name}"
        par_raw = await store.load(par_key)
        assert par_raw is not None
        par_cp = ParallelCheckpoint.model_validate_json(par_raw)
        assert len(par_cp.completed) == 1
        assert 0 in par_cp.completed  # "start:a" succeeded

        # Resume: non-failing worker, normal inner workflow
        worker2 = AppendProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=worker2)
        collect2 = CountingProcessor("collect")
        inner2 = SequentialWorkflow[str, str, None](
            name="inner", subprocs=[par2, collect2]
        )
        fan2 = CountingProcessor("fan")
        outer2 = SequentialWorkflow[str, str, None](
            name="outer", subprocs=[fan2, inner2], session_id="r4"
        )
        ctx2: RunContext[None] = RunContext(state=None, store=store)

        result = await collect_payloads(outer2, ctx2, step=0)

        assert fan2.call_count == 0  # skipped (outer step 0 done)
        assert collect2.call_count == 1  # ran
        # Both items: item 0 from parallel checkpoint, item 1 re-run
        assert sorted(result) == [
            "start:a->worker->collect",
            "start:b->worker->collect",
        ]

    @pytest.mark.asyncio
    async def test_runner_with_workflow_parallel_checkpoints(self) -> None:
        """
        Runner: entry -> wf(fan, par(worker), collect) -> END.
        All levels save checkpoints with correctly namespaced keys.
        """
        store = InMemoryCheckpointStore()
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        fan = FanOutProcessor("fan")
        collect = AppendProcessor("collect")
        wf = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[fan, par, collect], recipients=[END_PROC_NAME]
        )

        entry = ChatAppendProcessor("entry", recipients=["wf"])
        ctx: RunContext[None] = RunContext(state=None, store=store)
        runner = Runner[str, None](
            entry_proc=entry, procs=[entry, wf], ctx=ctx, name="r"
        )
        runner.setup_session("rs")

        result = await collect_runner_payloads(runner, chat_inputs="start")
        assert sorted(result) == [
            "start->entry:a->worker->collect",
            "start->entry:b->worker->collect",
        ]

        # Runner checkpoint
        assert await store.load("runner/rs") is not None
        # Workflow checkpoint (namespaced under runner session)
        assert await store.load("workflow/rs/wf") is not None
        # Parallel checkpoint (namespaced under workflow session)
        assert await store.load(f"parallel/rs/wf/{par.name}") is not None
