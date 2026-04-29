"""
Tests for recursive session propagation and checkpointing across nested
composite processors.

Verifies that:
- Adoption propagates session paths through Workflow -> subprocs
- Adoption propagates through ParallelProcessor -> subproc
- Nested composites get correctly namespaced sessions at every level
- Checkpoints are saved at each level independently
- Resume works across multiple nesting levels: workflow skips completed steps,
  parallel skips completed items, inner processors resume from their own state
"""

from collections.abc import AsyncIterator, Sequence
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
    ) -> None:
        super().__init__(name=name, subprocs=list(subprocs))
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
        """Workflow adoption sets namespaced path on each subproc."""
        a = AppendProcessor("A")
        b = AppendProcessor("B")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])

        # Root workflow path includes its own name (Option D)
        assert wf.path == ["wf"]
        assert a.path == ["wf", "A"]
        assert b.path == ["wf", "B"]

    def test_parallel_propagates_to_subproc(self) -> None:
        """ParallelProcessor propagates its path to inner subproc."""
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)

        assert par.path == [par.name]
        assert worker.path == [par.name, "worker"]

    def test_parallel_init_with_session_propagates(self) -> None:
        """ParallelProcessor construction propagates path to subproc."""
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)

        assert par.path == [par.name]
        assert worker.path == [par.name, "worker"]

    def test_workflow_containing_parallel_three_levels(self) -> None:
        """Workflow -> ParallelProcessor -> worker gets 3-level namespacing."""
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        a = AppendProcessor("A")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, par])

        assert wf.path == ["wf"]
        assert a.path == ["wf", "A"]
        assert par.path == ["wf", par.name]
        assert worker.path == ["wf", par.name, "worker"]

    def test_nested_workflows_propagate_to_leaf_procs(self) -> None:
        """Outer -> Inner workflow propagates to all leaf processors."""
        x = AppendProcessor("X")
        y = AppendProcessor("Y")
        inner = SequentialWorkflow[str, str, None](name="inner", subprocs=[x, y])

        a = AppendProcessor("A")
        outer = SequentialWorkflow[str, str, None](name="outer", subprocs=[a, inner])

        assert outer.path == ["outer"]
        assert a.path == ["outer", "A"]
        assert inner.path == ["outer", "inner"]
        assert x.path == ["outer", "inner", "X"]
        assert y.path == ["outer", "inner", "Y"]

    def test_is_resumable_depends_on_ctx(self) -> None:
        """Processor.is_resumable() is inferred from ctx.checkpoint_store."""
        worker = AppendProcessor("worker")

        # No ctx -> not resumable
        assert not Processor.is_resumable(None)

        # Ctx with no checkpoint_store -> not resumable
        ctx_empty: RunContext[None] = RunContext(state=None)
        assert not Processor.is_resumable(ctx_empty)

        # Ctx with checkpoint_store -> resumable
        store = InMemoryCheckpointStore()
        ctx_with_store: RunContext[None] = RunContext(
            state=None, checkpoint_store=store
        )
        assert Processor.is_resumable(ctx_with_store)

        # worker not used for runtime; only docs its path
        _ = worker

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
        Runner[str, None](
            entry_proc=entry, procs=[entry, wf], ctx=ctx, name="r"
        )

        # Runner adopts each top-level proc at empty parent path, so
        # direct children get single-entry paths and descendants get
        # the accumulated path.
        assert entry.path == ["entry"]
        assert wf.path == ["wf"]
        assert fan.path == ["wf", "fan"]
        assert par.path == ["wf", par.name]
        assert worker.path == ["wf", par.name, "worker"]


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
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="s1"
        )

        result = await collect_payloads(wf, ctx, in_args="start")
        assert sorted(result) == [
            "start:a->worker->collect",
            "start:b->worker->collect",
        ]

        # Workflow checkpoint exists
        wf_raw = await store.load("s1/workflow/wf")
        assert wf_raw is not None
        wf_cp = WorkflowCheckpoint.model_validate_json(wf_raw)
        assert wf_cp.completed_step == 2  # all 3 steps done

        # Parallel checkpoint exists with correctly namespaced key
        par_key = f"s1/parallel/wf/{par.name}"
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
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="s2"
        )

        result = await collect_payloads(outer, ctx, in_args="start")
        assert result == ["start->A->X->Y"]

        # Both checkpoints exist with correct namespacing
        outer_raw = await store.load("s2/workflow/outer")
        assert outer_raw is not None

        inner_raw = await store.load("s2/workflow/outer/inner")
        assert inner_raw is not None

    @pytest.mark.asyncio
    async def test_checkpoint_keys_for_deep_nesting(self) -> None:
        """Workflow -> Parallel -> worker: all checkpoint keys are namespaced."""
        store = InMemoryCheckpointStore()
        worker = AppendProcessor("worker")
        par = ParallelProcessor[str, str, None](subproc=worker)
        fan = FanOutProcessor("fan")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[fan, par])
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="deep"
        )

        await collect_payloads(wf, ctx, in_args="start")

        # Workflow: "<session>/workflow/<wf_name>"
        assert await store.load("deep/workflow/wf") is not None
        # Parallel: "<session>/parallel/<wf_name>/<par.name>"
        assert await store.load(f"deep/parallel/wf/{par.name}") is not None


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
        )
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="r1"
        )

        with pytest.raises(ProcRunError):
            await collect_payloads(wf1, ctx1, in_args="start")

        # Workflow checkpoint at step 0 (fan succeeded), step 1 save crashed
        wf_raw = await store.load("r1/workflow/wf")
        assert wf_raw is not None
        wf_cp = WorkflowCheckpoint.model_validate_json(wf_raw)
        assert wf_cp.completed_step == 0

        # Parallel checkpoint exists (saved internally before the workflow crash)
        par_key = f"r1/parallel/wf/{par1.name}"
        assert await store.load(par_key) is not None

        # Resume with fresh processors
        worker2 = AppendProcessor("worker")
        par2 = ParallelProcessor[str, str, None](subproc=worker2)
        fan2 = CountingProcessor("fan")
        collect2 = CountingProcessor("collect")
        wf2 = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[fan2, par2, collect2]
        )
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="r1"
        )

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
        )
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="r2"
        )

        with pytest.raises(ProcRunError):
            await collect_payloads(wf1, ctx1, in_args="start")

        # Parallel checkpoint: only item 0 completed (item 1 failed)
        par_key = f"r2/parallel/wf/{par1.name}"
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
            name="wf", subprocs=[fan2, par2, collect2]
        )
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="r2"
        )

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
        )
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="r3"
        )

        with pytest.raises(ProcRunError):
            await collect_payloads(outer1, ctx1, in_args="start")

        # Inner checkpoint exists (both steps done)
        inner_raw = await store.load("r3/workflow/outer/inner")
        assert inner_raw is not None
        inner_cp = WorkflowCheckpoint.model_validate_json(inner_raw)
        assert inner_cp.completed_step == 1  # both X and Y done

        # Resume
        x2 = CountingProcessor("X")
        y2 = CountingProcessor("Y")
        inner2 = SequentialWorkflow[str, str, None](name="inner", subprocs=[x2, y2])
        a2 = CountingProcessor("A")
        outer2 = SequentialWorkflow[str, str, None](
            name="outer", subprocs=[a2, inner2]
        )
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="r3"
        )

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
            name="outer", subprocs=[fan1, inner1]
        )
        ctx1: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="r4"
        )

        with pytest.raises(ProcRunError):
            await collect_payloads(outer1, ctx1, in_args="start")

        # Outer checkpoint at step 0 (fan succeeded, inner crashed)
        outer_raw = await store.load("r4/workflow/outer")
        assert outer_raw is not None
        outer_cp = WorkflowCheckpoint.model_validate_json(outer_raw)
        assert outer_cp.completed_step == 0

        # Inner has NO checkpoint (crash prevented saving)
        assert await store.load("r4/workflow/outer/inner") is None

        # But parallel's checkpoint exists with item 0 completed
        par_key = f"r4/parallel/outer/inner/{par1.name}"
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
            name="outer", subprocs=[fan2, inner2]
        )
        ctx2: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="r4"
        )

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
        ctx: RunContext[None] = RunContext(
            state=None, checkpoint_store=store, session_key="rs"
        )
        runner = Runner[str, None](
            entry_proc=entry, procs=[entry, wf], ctx=ctx, name="r"
        )

        result = await collect_runner_payloads(runner, chat_inputs="start")
        assert sorted(result) == [
            "start->entry:a->worker->collect",
            "start->entry:b->worker->collect",
        ]

        # Runner checkpoint
        assert await store.load("rs/runner") is not None
        # Workflow checkpoint (namespaced under runner session)
        assert await store.load("rs/workflow/wf") is not None
        # Parallel checkpoint (namespaced under workflow session)
        assert await store.load(f"rs/parallel/wf/{par.name}") is not None
