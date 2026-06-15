"""
ctx is a top-down session concern: set it at the top (a container, a
standalone processor, or a ``with RunContext(...)`` block) and it cascades
onto every subprocessor via ``on_adopted``. A subprocessor never carries its
own session — a container that adopts it overrides whatever ctx it was built
with (one session per composition tree). A bare container resolves the
ambient / process-default ctx like any bare processor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import (
    RunContext,
    current_run_context,
    reset_default_run_context,
)
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


class TestContainersShareOneCtx:
    def test_workflow_bare_subprocs_all_share_ctx(self) -> None:
        a, b = _proc("a"), _proc("b")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])
        assert a.ctx is wf.ctx
        assert b.ctx is wf.ctx

    def test_explicit_container_ctx_cascades_to_subprocs(self) -> None:
        ctx: RunContext[None] = RunContext(state=None)
        a, b = _proc("a"), _proc("b")
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b], ctx=ctx)
        assert wf.ctx is ctx
        assert a.ctx is ctx
        assert b.ctx is ctx

    def test_container_ctx_overrides_subproc_explicit_ctx(self) -> None:
        # ctx flows top-down: a subprocessor's own ctx is just its standalone
        # default; the container's wins (same as ProcessorTool's per-call
        # rebind). No bottom-up inheritance, no conflict error.
        child_ctx: RunContext[None] = RunContext(state=None)
        container_ctx: RunContext[None] = RunContext(state=None)
        a, b = _proc("a", child_ctx), _proc("b", child_ctx)
        wf = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[a, b], ctx=container_ctx
        )
        assert wf.ctx is container_ctx
        assert a.ctx is container_ctx
        assert b.ctx is container_ctx

    def test_divergent_subproc_ctxs_unify_to_container(self) -> None:
        # Two subprocs built with different ctxs no longer raise — the
        # container's ctx silently wins for both.
        a = _proc("a", RunContext(state=None))
        b = _proc("b", RunContext(state=None))
        wf = SequentialWorkflow[str, str, None](name="wf", subprocs=[a, b])
        assert a.ctx is wf.ctx
        assert b.ctx is wf.ctx

    def test_parallel_processor_shares_ctx_with_subproc(self) -> None:
        sub = _proc("sub")
        par = ParallelProcessor[str, str, None](subproc=sub)
        assert sub.ctx is par.ctx

    def test_runner_bare_procs_share_ctx(self) -> None:
        a = _proc("a", recipients=[END_PROC_NAME])
        runner: Runner[str, None] = Runner(entry_proc=a, procs=[a])
        assert a.ctx is runner.ctx

    def test_runner_explicit_ctx_cascades(self) -> None:
        ctx: RunContext[None] = RunContext(state=None)
        a = _proc("a", recipients=[END_PROC_NAME])
        runner: Runner[str, None] = Runner(entry_proc=a, procs=[a], ctx=ctx)
        assert runner.ctx is ctx
        assert a.ctx is ctx


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


class TestDefaultAndAmbientCtx:
    """Bare construction binds to one shared ctx, never a per-agent throwaway."""

    def test_uncomposed_bare_agents_share_the_process_default(self) -> None:
        # The leak this design closes: two agents that are never composed
        # still belong to one session.
        a, b = _proc("a"), _proc("b")
        assert a.ctx is b.ctx
        assert a.ctx is current_run_context()

    def test_reset_default_gives_a_fresh_one(self) -> None:
        first = _proc("a").ctx
        reset_default_run_context()
        assert _proc("b").ctx is not first

    def test_with_block_binds_bare_agents_to_that_ctx(self) -> None:
        with RunContext[None](state=None) as ctx:
            a, b = _proc("a"), _proc("b")
            assert a.ctx is ctx
            assert b.ctx is ctx
        # Outside the block, bare construction falls back to the default.
        assert _proc("c").ctx is not ctx

    def test_with_block_restores_outer_ambient(self) -> None:
        with RunContext[None](state=None) as outer:
            assert _proc("a").ctx is outer
            with RunContext[None](state=None) as inner:
                assert _proc("b").ctx is inner
            # Inner exit restores the outer ambient ctx.
            assert _proc("c").ctx is outer

    def test_explicit_ctx_wins_over_ambient(self) -> None:
        explicit: RunContext[None] = RunContext(state=None)
        with RunContext[None](state=None) as ambient:
            p = _proc("a", explicit)
            assert p.ctx is explicit
            assert p.ctx is not ambient
