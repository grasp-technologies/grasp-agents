"""
Container ctx resolution (PR #30) + session cascade (PR #15).

Containers no longer borrow a ctx from an arbitrary child (``start_proc`` /
``subproc``). Instead ``shared_child_ctx`` inherits the single ctx the
children were *built* with (if any) and otherwise falls back to a fresh one;
either way every subprocessor and the container end up sharing one
``RunContext`` instance. Conflicting explicit child ctxs are surfaced, not
silently resolved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext, shared_child_ctx
from grasp_agents.runner.runner import END_PROC_NAME, Runner
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow

if TYPE_CHECKING:
    from grasp_agents.types.io import ProcName


class _Pass(Processor[str, str, None]):
    """Minimal str->str passthrough processor."""


def _proc(
    name: str,
    ctx: RunContext[None] | None = None,
    recipients: list[ProcName] | None = None,
) -> _Pass:
    return _Pass(name=name, ctx=ctx, recipients=recipients)


class TestSharedChildCtx:
    def test_none_when_all_children_built_bare(self) -> None:
        assert shared_child_ctx([_proc("a"), _proc("b")]) is None

    def test_returns_the_single_explicit_child_ctx(self) -> None:
        ctx: RunContext[None] = RunContext(state=None)
        # ``b`` is bare (placeholder) and must be ignored.
        assert shared_child_ctx([_proc("a", ctx), _proc("b")]) is ctx

    def test_same_instance_on_two_children_is_not_a_conflict(self) -> None:
        ctx: RunContext[None] = RunContext(state=None)
        assert shared_child_ctx([_proc("a", ctx), _proc("b", ctx)]) is ctx

    def test_raises_on_conflicting_explicit_child_ctxs(self) -> None:
        a = _proc("a", RunContext(state=None))
        b = _proc("b", RunContext(state=None))
        with pytest.raises(ValueError, match="different RunContext"):
            shared_child_ctx([a, b])

    def test_explicit_ctx_via_on_adopted_counts_as_deliberate(self) -> None:
        # A processor built bare but later given a ctx via on_adopted should
        # count as deliberate (supports "build bare -> on_adopted -> wrap").
        ctx: RunContext[None] = RunContext(state=None)
        a = _proc("a")
        a.on_adopted(ctx=ctx)
        assert shared_child_ctx([a, _proc("b")]) is ctx


class TestContainersShareOneCtx:
    def test_workflow_bare_subprocs_all_share_fresh_ctx(self) -> None:
        a, b = _proc("a"), _proc("b")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])
        assert a.ctx is wf.ctx
        assert b.ctx is wf.ctx

    def test_workflow_inherits_shared_subproc_ctx(self) -> None:
        ctx: RunContext[None] = RunContext(state=None)
        a, b = _proc("a", ctx), _proc("b", ctx)
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])
        assert wf.ctx is ctx
        assert a.ctx is ctx
        assert b.ctx is ctx

    def test_workflow_raises_on_conflicting_subproc_ctxs(self) -> None:
        a = _proc("a", RunContext(state=None))
        b = _proc("b", RunContext(state=None))
        with pytest.raises(ValueError, match="different RunContext"):
            SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])

    def test_parallel_processor_shares_ctx_with_subproc(self) -> None:
        sub = _proc("sub")
        par = ParallelProcessor[str, str, None](subproc=sub)
        assert sub.ctx is par.ctx

    def test_runner_bare_procs_share_fresh_ctx(self) -> None:
        a = _proc("a", recipients=[END_PROC_NAME])
        runner: Runner[str, None] = Runner(entry_proc=a, procs=[a])
        assert a.ctx is runner.ctx

    def test_runner_inherits_shared_proc_ctx(self) -> None:
        ctx: RunContext[None] = RunContext(state=None)
        a = _proc("a", ctx, recipients=[END_PROC_NAME])
        runner: Runner[str, None] = Runner(entry_proc=a, procs=[a])
        assert runner.ctx is ctx


class TestSessionCascade:
    """PR #15: on_adopted sets one or both axes and cascades to children."""

    def test_adopt_ctx_cascades_ctx_and_keeps_path_lineage(self) -> None:
        a, b = _proc("a"), _proc("b")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])
        ctx: RunContext[None] = RunContext(state=None)
        wf.on_adopted(ctx=ctx)
        # ctx axis cascaded to every child...
        assert wf.ctx is ctx
        assert a.ctx is ctx
        assert b.ctx is ctx
        # ...without disturbing the path axis (lineage under the container).
        assert a.path == ["wf", "a"]
        assert b.path == ["wf", "b"]

    def test_adopt_path_cascades_path_and_keeps_ctx(self) -> None:
        ctx: RunContext[None] = RunContext(state=None)
        a, b = _proc("a", ctx), _proc("b", ctx)
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b], ctx=ctx)
        wf.on_adopted(path=["root", "wf"])
        assert a.path == ["root", "wf", "a"]
        # ctx axis untouched by a path-only change.
        assert a.ctx is ctx
        assert wf.ctx is ctx
