import asyncio
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from grasp_agents.evals import (
    Dataset,
    EvalContext,
    Example,
    FunctionTask,
    LocalRunStore,
    Mean,
    PassRate,
    ResumeError,
    RunStatus,
    Score,
    TrialContext,
    TrialProgress,
    Usage,
    evaluate,
    evaluator,
    rescore,
)

type Ctx = EvalContext[int, int, int]


def _numbers(n: int = 4) -> Dataset[int, int]:
    return Dataset(
        [Example(id=f"n{i}", input=i, reference=i * 2) for i in range(n)],
        name="numbers",
    )


async def double(x: int) -> int:
    return x * 2


@evaluator
def correct(ctx: Ctx) -> bool:
    return ctx.output == ctx.reference


@pytest.fixture
def store(tmp_path: Path) -> LocalRunStore:
    return LocalRunStore(tmp_path / "evals")


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_scores_persist_and_reload(self, store: LocalRunStore) -> None:
        run = await evaluate(double, _numbers(), [correct], store=store)
        assert run.status == RunStatus.COMPLETED
        assert run.counts.trials_done == 4
        assert run.metric("pass_rate(correct)") is not None
        assert run.metric("pass_rate(correct)").value == pytest.approx(1.0)  # type: ignore[union-attr]
        assert run.task.name == "double"
        assert run.evaluators[0].name == "correct"
        assert run.provenance.python

        loaded = store.load(run.id)
        assert [t.key for t in loaded.trials] == [t.key for t in run.trials]
        assert loaded.trials[0].scores[0].evaluator == "correct"
        assert [e.id for e in loaded.examples] == ["n0", "n1", "n2", "n3"]
        assert (store.run_dir(run.id) / "report.md").read_text().startswith("# double")

    @pytest.mark.asyncio
    async def test_outcome_taxonomy(self, store: LocalRunStore) -> None:
        async def flaky(x: int) -> int:
            if x == 1:
                raise ValueError("cannot handle 1")
            return x * 2

        @evaluator
        def judge(ctx: Ctx) -> Score | None:
            if ctx.input == 0:
                return None  # not applicable
            if ctx.input == 2:
                return Score.unscored("judge", reason="invalid_response_format")
            return Score(name="judge", value=0.5, explanation="half")

        @evaluator
        def broken(ctx: Ctx) -> bool:
            raise RuntimeError("judge crashed")

        run = await evaluate(flaky, _numbers(), [judge, broken], store=store)
        by_id = {t.example_id: t for t in run.trials}
        assert by_id["n1"].error is not None
        assert by_id["n1"].scores == []
        assert by_id["n0"].scores == []  # not applicable: nothing recorded
        assert "judge" in by_id["n0"].evaluated
        assert by_id["n2"].scores[0].value is None
        assert by_id["n3"].scores[0].value == pytest.approx(0.5)
        assert all(
            [f.evaluator for f in t.evaluator_failures] == ["broken"]
            for t in run.trials
            if t.ok
        )
        assert run.counts.task_errors == 1
        assert run.counts.unscored == 1
        assert run.counts.evaluator_failures == 3
        mean = run.metric("mean(judge)")
        assert mean is not None
        assert mean.value == pytest.approx(0.5)
        assert mean.n == 1
        assert mean.n_missing == 3

    @pytest.mark.asyncio
    async def test_duplicate_score_names_are_a_failure(self) -> None:
        @evaluator(name="a")
        def first(ctx: Ctx) -> dict[str, float]:
            return {"shared": 1.0}

        @evaluator(name="b")
        def second(ctx: Ctx) -> dict[str, float]:
            return {"shared": 0.0}

        run = await evaluate(double, _numbers(1), [first, second], persist=False)
        trial = run.trials[0]
        assert [s.name for s in trial.scores] == ["shared"]
        assert trial.evaluator_failures[0].error.type == "DuplicateScoreName"

    @pytest.mark.asyncio
    async def test_duplicate_evaluator_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate evaluator"):
            await evaluate(double, _numbers(1), [correct, correct], persist=False)

    @pytest.mark.asyncio
    async def test_repetitions_and_concurrency(self) -> None:
        active = 0
        peak = 0

        async def slow(x: int) -> int:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return x * 2

        run = await evaluate(
            slow, _numbers(6), [correct], repetitions=3, concurrency=2, persist=False
        )
        assert run.counts.trials_done == 18
        assert peak == 2
        assert {t.repetition for t in run.trials} == {0, 1, 2}
        result = run.metric("pass_rate(correct)")
        assert result is not None
        assert result.n == 6

    @pytest.mark.asyncio
    async def test_timeout_is_a_task_error(self) -> None:
        async def hang(x: int) -> int:
            await asyncio.sleep(10)
            return x

        run = await evaluate(
            hang, _numbers(1), [correct], timeout_s=0.05, persist=False
        )
        error = run.trials[0].error
        assert error is not None
        assert error.type == "TimeoutError"
        assert "timeout" in error.message

    @pytest.mark.asyncio
    async def test_max_error_rate_invalidates(self) -> None:
        async def bad(x: int) -> int:
            raise ValueError("no")

        run = await evaluate(
            bad, _numbers(2), [correct], max_error_rate=0.1, persist=False
        )
        assert run.status == RunStatus.COMPLETED
        assert run.invalid_reason is not None
        assert "error rate" in run.invalid_reason

    @pytest.mark.asyncio
    async def test_budget_stops_scheduling(self) -> None:
        async def costly(x: int, trial: TrialContext) -> int:
            trial.usage_by_agent["agent"] = Usage(input_tokens=10, cost_usd=1.0)
            return x * 2

        run = await evaluate(
            FunctionTask(costly),
            _numbers(5),
            [correct],
            concurrency=1,
            max_cost_usd=2.0,
            persist=False,
        )
        assert run.status == RunStatus.PARTIAL
        assert run.counts.trials_done == 2
        assert run.usage.cost_usd == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_measurements_and_progress(self) -> None:
        seen: list[TrialProgress] = []

        async def measured(x: int, trial: TrialContext) -> int:
            trial.record("steps", x + 1)
            return x * 2

        run = await evaluate(
            FunctionTask(measured),
            _numbers(3),
            [correct],
            [Mean("steps")],
            progress=seen.append,
            persist=False,
        )
        assert [p.done for p in seen] == [1, 2, 3]
        assert run.metric("mean(steps)").value == pytest.approx(2.0)  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_sealed_splits_flag_trials(self) -> None:
        dataset = Dataset(
            [
                Example(id="dev", input=1, reference=2, splits=["dev"]),
                Example(id="test", input=2, reference=4, splits=["test"]),
            ]
        )
        run = await evaluate(
            double, dataset, [correct], sealed_splits=["test"], persist=False
        )
        assert {t.example_id: t.sealed for t in run.trials} == {
            "dev": False,
            "test": True,
        }

    @pytest.mark.asyncio
    async def test_trace_ids_link_trials(self) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        from grasp_agents.evals import _execution

        original = _execution._tracer
        _execution._tracer = provider.get_tracer("test")
        try:
            run = await evaluate(double, _numbers(2), [correct], persist=False)
        finally:
            _execution._tracer = original
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        ids = {format(s.context.trace_id, "032x") for s in spans}
        assert {t.trace_id for t in run.trials} == ids
        attributes = spans[0].attributes or {}
        assert attributes["grasp.eval.run_id"] == run.id


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_reruns_only_failed_trials(self, store: LocalRunStore) -> None:
        calls: list[int] = []
        fail = {1}

        async def sometimes(x: int) -> int:
            calls.append(x)
            if x in fail:
                raise ValueError("transient")
            return x * 2

        task = FunctionTask(sometimes, name="sometimes")
        first = await evaluate(task, _numbers(3), [correct], store=store)
        assert first.counts.task_errors == 1
        fail.clear()
        calls.clear()
        resumed = await evaluate(
            task, _numbers(3), [correct], store=store, resume=first.id
        )
        assert calls == [1]
        assert resumed.id == first.id
        assert resumed.counts.task_errors == 0
        assert store.load(first.id).counts.task_errors == 0

    @pytest.mark.asyncio
    async def test_resume_scores_missing_evaluators(self, store: LocalRunStore) -> None:
        unscored = await evaluate(
            double, _numbers(2), [correct], store=store, score=False
        )
        assert all(t.scores == [] for t in unscored.trials)
        resumed = await evaluate(
            double, _numbers(2), [correct], store=store, resume=unscored.id
        )
        assert all(t.score("correct") is not None for t in resumed.trials)

    @pytest.mark.asyncio
    async def test_resume_refuses_changed_configuration(
        self, store: LocalRunStore
    ) -> None:
        run = await evaluate(double, _numbers(2), [correct], store=store)
        with pytest.raises(ResumeError, match="differ"):
            await evaluate(double, _numbers(3), [correct], store=store, resume=run.id)


class TestRescore:
    @pytest.mark.asyncio
    async def test_child_run_reuses_outputs(self, store: LocalRunStore) -> None:
        calls = 0

        async def counted(x: int) -> int:
            nonlocal calls
            calls += 1
            return x * 2

        @evaluator(name="strict", version="1")
        def strict_v1(ctx: Ctx) -> bool:
            return False

        @evaluator(name="strict", version="2")
        def strict_v2(ctx: Ctx) -> bool:
            return ctx.output == ctx.reference

        parent = await evaluate(counted, _numbers(3), [correct, strict_v1], store=store)
        assert calls == 3
        child = await rescore(parent.id, [strict_v2], store=store, output_type=int)
        assert calls == 3  # the task did not run again
        assert child.parent_run_id == parent.id
        assert child.kind == "rescore"
        assert {e.name: e.version for e in child.evaluators} == {
            "correct": "1",
            "strict": "2",
        }
        assert child.metric("pass_rate(strict)").value == pytest.approx(1.0)  # type: ignore[union-attr]
        assert child.metric("pass_rate(correct)").value == pytest.approx(1.0)  # type: ignore[union-attr]
        reloaded_parent = store.load(parent.id)
        assert reloaded_parent.metric("pass_rate(strict)").value == pytest.approx(0.0)  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_rescore_sees_transcripts_and_typed_outputs(
        self, store: LocalRunStore
    ) -> None:
        seen: list[Any] = []

        @evaluator
        def inspect_output(ctx: Ctx) -> None:
            seen.append(ctx.output)

        parent = await evaluate(double, _numbers(2), [correct], store=store)
        await rescore(parent, [inspect_output], store=store, output_type=int)
        assert sorted(seen) == [0, 2]
        assert PassRate("correct").name in {m.name for m in parent.metrics}
