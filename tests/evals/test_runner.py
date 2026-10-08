import asyncio
import importlib
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from grasp_agents.evals import (
    Dataset,
    Example,
    FunctionTask,
    LocalRunStore,
    Mean,
    PassRate,
    ResumeError,
    RunStatus,
    Score,
    ScoreContext,
    Scorer,
    SealedSelectionError,
    TrialContext,
    TrialProgress,
    Usage,
    evaluate,
    rescore,
    scorer,
)

type Ctx = ScoreContext[int, int, int]


def _numbers(n: int = 4) -> Dataset[int, int]:
    return Dataset(
        [Example(id=f"n{i}", input=i, reference=i * 2) for i in range(n)],
        name="numbers",
    )


async def double(x: int) -> int:
    return x * 2


@scorer
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
        assert run.scorers[0].name == "correct"
        assert run.provenance.python

        loaded = store.load(run.id)
        assert [t.key for t in loaded.trials] == [t.key for t in run.trials]
        assert loaded.trials[0].scores[0].scorer == "correct"
        assert [e.id for e in loaded.examples] == ["n0", "n1", "n2", "n3"]
        assert (store.run_dir(run.id) / "report.md").read_text().startswith("# double")

    @pytest.mark.asyncio
    async def test_outcome_taxonomy(self, store: LocalRunStore) -> None:
        async def flaky(x: int) -> int:
            if x == 1:
                raise ValueError("cannot handle 1")
            return x * 2

        @scorer
        def judge(ctx: Ctx) -> Score | None:
            if ctx.input == 0:
                return None  # not applicable
            if ctx.input == 2:
                return Score.unscored("judge", reason="invalid_response_format")
            return Score(name="judge", value=0.5, explanation="half")

        @scorer
        def broken(ctx: Ctx) -> bool:
            raise RuntimeError("judge crashed")

        run = await evaluate(flaky, _numbers(), [judge, broken], store=store)
        by_id = {t.example_id: t for t in run.trials}
        assert by_id["n1"].error is not None
        assert by_id["n1"].scores == []
        assert by_id["n0"].scores == []  # not applicable: nothing recorded
        assert "judge" in by_id["n0"].scorers_run
        assert by_id["n2"].scores[0].value is None
        assert by_id["n3"].scores[0].value == pytest.approx(0.5)
        assert all(
            [f.scorer for f in t.scorer_failures] == ["broken"]
            for t in run.trials
            if t.ok
        )
        assert run.counts.task_errors == 1
        assert run.counts.unscored == 1
        assert run.counts.scorer_failures == 3
        mean = run.metric("mean(judge)")
        assert mean is not None
        assert mean.value == pytest.approx(0.5)
        assert mean.n == 1
        assert mean.n_missing == 3

    @pytest.mark.asyncio
    async def test_duplicate_score_names_are_a_failure(self) -> None:
        @scorer(name="a")
        def first(ctx: Ctx) -> dict[str, float]:
            return {"shared": 1.0}

        @scorer(name="b")
        def second(ctx: Ctx) -> dict[str, float]:
            return {"shared": 0.0}

        run = await evaluate(double, _numbers(1), [first, second], persist=False)
        trial = run.trials[0]
        assert [s.name for s in trial.scores] == ["shared"]
        assert trial.scorer_failures[0].error.type == "DuplicateScoreName"

    @pytest.mark.asyncio
    async def test_duplicate_scorer_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate scorer"):
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
        async def flaky(x: int) -> int:
            if x == 0:
                raise ValueError("no")
            return x * 2

        run = await evaluate(
            flaky, _numbers(4), [correct], max_error_rate=0.1, persist=False
        )
        assert run.status == RunStatus.COMPLETED
        assert run.invalid_reason is not None
        assert "error rate" in run.invalid_reason

    @pytest.mark.asyncio
    async def test_a_run_where_every_trial_failed_is_invalid(self) -> None:
        async def bad(x: int) -> int:
            raise ValueError("no")

        run = await evaluate(bad, _numbers(2), [correct], persist=False)
        assert run.invalid_reason == "every trial failed"

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
        retried = await evaluate(
            task, _numbers(3), [correct], store=store, resume=first.id
        )
        assert calls == [1]
        # A completed run is never modified: the retry is a child run.
        assert retried.kind == "retry"
        assert retried.parent_run_id == first.id
        assert retried.counts.task_errors == 0
        assert store.load(first.id).counts.task_errors == 1
        reloaded = store.load(retried.id)
        assert len(reloaded.trials) == 3
        assert all(t.ok for t in reloaded.trials)

    @pytest.mark.asyncio
    async def test_resume_scores_missing_scorers(self, store: LocalRunStore) -> None:
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

        @scorer(name="strict", version="1")
        def strict_v1(ctx: Ctx) -> bool:
            return False

        @scorer(name="strict", version="2")
        def strict_v2(ctx: Ctx) -> bool:
            return ctx.output == ctx.reference

        parent = await evaluate(counted, _numbers(3), [correct, strict_v1], store=store)
        assert calls == 3
        child = await rescore(parent.id, [strict_v2], store=store, output_type=int)
        assert calls == 3  # the task did not run again
        assert child.parent_run_id == parent.id
        assert child.kind == "rescore"
        assert {e.name: e.version for e in child.scorers} == {
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

        @scorer
        def inspect_output(ctx: Ctx) -> None:
            seen.append(ctx.output)

        parent = await evaluate(double, _numbers(2), [correct], store=store)
        await rescore(parent, [inspect_output], store=store, output_type=int)
        assert sorted(seen) == [0, 2]
        assert PassRate("correct").name in {m.name for m in parent.metrics}


def _mark_interrupted(store: LocalRunStore, run_id: str) -> None:
    run = store.load(run_id)
    run.status = RunStatus.CANCELLED
    store.save(run)


def _costly(cost: float | None, tokens: int = 10) -> FunctionTask[int, int]:
    async def costly(x: int, trial: TrialContext) -> int:
        trial.usage_by_agent["agent"] = Usage(input_tokens=tokens, cost_usd=cost)
        return x * 2

    return FunctionTask(costly)


class TestResumeRules:
    @pytest.mark.asyncio
    async def test_unfinished_runs_continue_in_place(
        self, store: LocalRunStore
    ) -> None:
        fail = {1}

        async def flaky(x: int) -> int:
            if x in fail:
                raise ValueError("transient")
            return x * 2

        task = FunctionTask(flaky, name="flaky")
        first = await evaluate(task, _numbers(3), [correct], store=store)
        _mark_interrupted(store, first.id)
        fail.clear()
        resumed = await evaluate(
            task, _numbers(3), [correct], store=store, resume=first.id, concurrency=2
        )
        assert resumed.id == first.id
        reloaded = store.load(first.id)
        assert len(reloaded.trials) == 3
        assert all(t.ok for t in reloaded.trials)  # the later record wins
        (record,) = reloaded.metadata["resumed"]
        assert record["changed_settings"] == {"concurrency": [4, 2]}

    @pytest.mark.asyncio
    async def test_code_changes_refuse_resume_unless_forced(
        self, store: LocalRunStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from grasp_agents.evals import runner as runner_module

        first = await evaluate(double, _numbers(2), [correct], store=store)
        _mark_interrupted(store, first.id)
        original = runner_module.capture_provenance

        def edited(*args: Any, **kwargs: Any) -> Any:
            return original(*args, **kwargs).model_copy(
                update={"source_hash": "edited"}
            )

        monkeypatch.setattr(runner_module, "capture_provenance", edited)
        with pytest.raises(ResumeError, match="source files"):
            await evaluate(double, _numbers(2), [correct], store=store, resume=first.id)
        forced = await evaluate(
            double, _numbers(2), [correct], store=store, resume=first.id, force=True
        )
        assert forced.metadata["resumed"][-1]["forced_past"]

    @pytest.mark.asyncio
    async def test_changed_scorers_refuse_resume(self, store: LocalRunStore) -> None:
        @scorer
        def positive(ctx: Ctx) -> bool:
            return ctx.output >= 0

        first = await evaluate(double, _numbers(2), [correct], store=store)
        _mark_interrupted(store, first.id)
        with pytest.raises(ResumeError, match="scorers"):
            await evaluate(
                double, _numbers(2), [correct, positive], store=store, resume=first.id
            )

    @pytest.mark.asyncio
    async def test_a_model_change_stops_a_resume(self, store: LocalRunStore) -> None:
        model = {"name": "m1"}
        fail = {1}

        async def answer(x: int, trial: TrialContext) -> int:
            trial.models["agent"] = [model["name"]]
            if x in fail:
                raise ValueError("transient")
            return x * 2

        task = FunctionTask(answer, name="answer")
        first = await evaluate(task, _numbers(2), [correct], store=store)
        _mark_interrupted(store, first.id)
        model["name"], fail = "m2", set()
        with pytest.raises(ResumeError, match="m2"):
            await evaluate(task, _numbers(2), [correct], store=store, resume=first.id)

    @pytest.mark.asyncio
    async def test_a_model_change_stops_a_resume_after_a_crash(
        self, store: LocalRunStore
    ) -> None:
        model = {"name": "m1"}
        fail = {1, 2}

        async def answer(x: int, trial: TrialContext) -> int:
            trial.models["agent"] = [model["name"]]
            if x in fail:
                raise ValueError("transient")
            return x * 2

        task = FunctionTask(answer, name="answer")
        first = await evaluate(task, _numbers(4), [correct], store=store)
        # A killed process never writes its totals into the header.
        crashed = store.load(first.id)
        crashed.status = RunStatus.RUNNING
        crashed.provenance.observed_models = {}
        store.save(crashed)
        model["name"], fail = "m2", set()
        with pytest.raises(ResumeError, match="m2"):
            await evaluate(
                task,
                _numbers(4),
                [correct],
                store=store,
                resume=first.id,
                concurrency=2,
            )

    @pytest.mark.asyncio
    async def test_nothing_left_to_retry(self, store: LocalRunStore) -> None:
        first = await evaluate(double, _numbers(2), [correct], store=store)
        again = await evaluate(
            double, _numbers(2), [correct], store=store, resume=first.id
        )
        assert again.id == first.id
        assert len(store.list_runs()) == 1


class TestSelectionGuards:
    @pytest.mark.asyncio
    async def test_empty_selections_are_refused(self) -> None:
        with pytest.raises(ValueError, match="no examples"):
            await evaluate(double, Dataset([], name="empty"), [correct], persist=False)

    @pytest.mark.asyncio
    async def test_sealed_splits_are_evaluated_whole(self) -> None:
        data = Dataset(
            [
                Example(
                    id=f"n{i}",
                    input=i,
                    reference=i * 2,
                    splits=["test"] if i >= 2 else [],
                )
                for i in range(4)
            ],
            name="numbers",
        )
        with pytest.raises(ValueError, match="not splits"):
            await evaluate(
                double, data, [correct], sealed_splits=["Test"], persist=False
            )
        with pytest.raises(SealedSelectionError, match="1 of the 2"):
            await evaluate(
                double,
                data.select(["n0", "n3"]),
                [correct],
                sealed_splits=["test"],
                persist=False,
            )
        whole = await evaluate(
            double, data.split("test"), [correct], sealed_splits=["test"], persist=False
        )
        assert all(t.sealed for t in whole.trials)


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_an_unserializable_output_is_a_task_error(self) -> None:
        async def tangled(x: int) -> Any:
            if x == 0:
                loop: list[Any] = []
                loop.append(loop)
                return loop
            return x * 2

        run = await evaluate(tangled, _numbers(3), [correct], persist=False)
        assert run.status == RunStatus.COMPLETED
        error = run.trial("n0").error  # type: ignore[union-attr]
        assert error is not None
        assert error.type == "OutputNotSerializable"
        assert run.trial("n1").ok  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_non_finite_measurements_are_refused(self) -> None:
        async def ratio(x: int, trial: TrialContext) -> int:
            trial.record("ratio", float("inf"))
            return x

        run = await evaluate(ratio, _numbers(1), [correct], persist=False)
        assert "finite" in run.trials[0].error.message  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_sync_scorers_run_off_the_event_loop(self) -> None:
        import time

        @scorer
        def slow(ctx: Ctx) -> bool:
            time.sleep(0.3)
            return True

        started = time.perf_counter()
        run = await evaluate(double, _numbers(4), [slow], concurrency=4, persist=False)
        assert time.perf_counter() - started < 0.9
        assert all(t.duration_s < 0.25 for t in run.trials)

    def test_a_hanging_sync_scorer_does_not_delay_the_exit(self) -> None:
        import time

        @scorer
        def hangs(ctx: Ctx) -> bool:
            time.sleep(3)
            return True

        started = time.perf_counter()
        run = asyncio.run(
            evaluate(double, _numbers(1), [hangs], scorer_timeout_s=0.1, persist=False)
        )
        assert time.perf_counter() - started < 2
        (failure,) = run.trials[0].scorer_failures
        assert failure.error.type == "TimeoutError"

    @pytest.mark.asyncio
    async def test_bytes_outputs_survive_storage(self, store: LocalRunStore) -> None:
        seen: list[Any] = []

        @scorer
        def is_payload(ctx: ScoreContext[int, bytes, None]) -> bool:
            seen.append(ctx.output)
            return ctx.output == b"\x00\xff"

        async def emit(x: int) -> bytes:
            return b"\x00\xff"

        data = Dataset[int, None]([Example(id="a", input=1)], name="blobs")
        first = await evaluate(emit, data, [is_payload], store=store, score=False)
        later = await evaluate(emit, data, [is_payload], store=store, resume=first.id)
        assert seen == [b"\x00\xff"]
        assert later.trials[0].scores[0].value is True

    @pytest.mark.asyncio
    async def test_hanging_scorers_time_out(self) -> None:
        @scorer
        async def hangs(ctx: Ctx) -> bool:
            await asyncio.sleep(5)
            return True

        run = await evaluate(
            double, _numbers(1), [hangs], scorer_timeout_s=0.05, persist=False
        )
        (failure,) = run.trials[0].scorer_failures
        assert failure.error.type == "TimeoutError"


class TestBudget:
    @pytest.mark.asyncio
    async def test_concurrency_does_not_overshoot_the_budget(self) -> None:
        run = await evaluate(
            _costly(1.0),
            _numbers(8),
            [correct],
            concurrency=8,
            max_cost_usd=2.0,
            persist=False,
        )
        assert run.status == RunStatus.PARTIAL
        assert run.usage.cost_usd == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_a_free_first_trial_does_not_open_the_batch(self) -> None:
        async def priced(x: int, trial: TrialContext) -> int:
            if x == 0:
                raise ValueError("fails before calling a model")
            await asyncio.sleep(0.02)  # trials overlap, as model calls do
            trial.usage_by_agent["agent"] = Usage(input_tokens=10, cost_usd=1.0)
            return x * 2

        run = await evaluate(
            FunctionTask(priced),
            _numbers(12),
            [correct],
            concurrency=8,
            max_cost_usd=2.5,
            persist=False,
        )
        assert run.status == RunStatus.PARTIAL
        assert run.usage.cost_usd is not None
        assert run.usage.cost_usd <= 3.0

    @pytest.mark.asyncio
    async def test_unpriced_models_are_reported(self) -> None:
        run = await evaluate(
            _costly(None), _numbers(2), [correct], max_cost_usd=1.0, persist=False
        )
        assert run.metadata["unpriced_agents"] == ["agent"]

    @pytest.mark.asyncio
    async def test_a_spent_budget_runs_nothing(self) -> None:
        run = await evaluate(
            _costly(1.0), _numbers(2), [correct], max_cost_usd=0.0, persist=False
        )
        assert run.status == RunStatus.PARTIAL
        assert run.invalid_reason == "no trial ran"


class TestRescoreRules:
    @pytest.mark.asyncio
    async def test_unchanged_scorers_are_reused(self, store: LocalRunStore) -> None:
        calls: list[str] = []

        def judge(version: str) -> Any:
            @scorer(name="judged", version=version)
            def judged(ctx: Ctx) -> bool:
                calls.append(ctx.example.id)
                return ctx.output == ctx.reference

            return judged

        parent = await evaluate(double, _numbers(3), [judge("1")], store=store)
        calls.clear()
        await rescore(parent.id, [judge("1")], store=store)
        assert calls == []
        await rescore(parent.id, [judge("1")], store=store, rerun=True)
        assert sorted(calls) == ["n0", "n1", "n2"]
        calls.clear()
        await rescore(parent.id, [judge("2")], store=store)
        assert sorted(calls) == ["n0", "n1", "n2"]

    @pytest.mark.asyncio
    async def test_reuse_survives_configs_that_storage_reshapes(
        self, store: LocalRunStore
    ) -> None:
        calls: list[str] = []

        class Labelled(Scorer[int, int, int]):
            name = "labelled"

            def config(self) -> dict[str, Any]:
                return {"labels": ("good", "bad"), "tags": {"b", "a"}}

            def score(self, ctx: Ctx) -> bool:
                calls.append(ctx.example.id)
                return ctx.output == ctx.reference

        run = await evaluate(double, _numbers(3), [Labelled()], store=store)
        await rescore(run.id, [Labelled()], store=store, output_type=int)
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_edited_scorer_code_is_rerun(
        self, store: LocalRunStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        @scorer(name="judged")
        def judged(ctx: Ctx) -> bool:
            calls.append(ctx.example.id)
            return ctx.output == ctx.reference

        run = await evaluate(double, _numbers(3), [judged], store=store)
        # (The package exports a function named like the module.)
        module = importlib.import_module("grasp_agents.evals.scorer")
        monkeypatch.setattr(module, "code_hash", lambda _: "edited")
        await rescore(run.id, [judged], store=store, output_type=int)
        assert len(calls) == 6

    @pytest.mark.asyncio
    async def test_failed_scorer_outputs_are_filled(self, store: LocalRunStore) -> None:
        broken = {"n0"}
        calls: list[str] = []

        @scorer
        def fragile(ctx: Ctx) -> bool:
            calls.append(ctx.example.id)
            if ctx.example.id in broken:
                raise RuntimeError("judge down")
            return True

        parent = await evaluate(double, _numbers(3), [fragile], store=store)
        assert parent.counts.scorer_failures == 1
        broken.clear()
        calls.clear()
        child = await rescore(parent.id, [fragile], store=store)
        assert calls == ["n0"]
        assert child.counts.scorer_failures == 0

    @pytest.mark.asyncio
    async def test_rescoring_a_partial_run_stays_partial(
        self, store: LocalRunStore
    ) -> None:
        parent = await evaluate(
            _costly(1.0),
            _numbers(4),
            [correct],
            concurrency=1,
            max_cost_usd=2.0,
            store=store,
        )
        assert parent.status == RunStatus.PARTIAL
        child = await rescore(parent.id, [correct], store=store)
        assert child.status == RunStatus.PARTIAL
        assert child.counts.trials_expected == 4

    @pytest.mark.asyncio
    async def test_a_replaced_judge_is_not_counted_twice(
        self, store: LocalRunStore
    ) -> None:
        def paid(version: str) -> Any:
            @scorer(name="paid", version=version)
            def judge(ctx: Ctx) -> bool:
                ctx.record_usage(Usage(input_tokens=1, cost_usd=1.0))
                return True

            return judge

        parent = await evaluate(double, _numbers(2), [paid("1")], store=store)
        assert parent.usage.cost_usd == pytest.approx(2.0)
        child = await rescore(parent.id, [paid("2")], store=store)
        assert child.usage.cost_usd == pytest.approx(2.0)
        assert all(t.total_usage.cost_usd == pytest.approx(1.0) for t in child.trials)

    @pytest.mark.asyncio
    async def test_running_parents_are_refused(self, store: LocalRunStore) -> None:
        parent = await evaluate(double, _numbers(2), [correct], store=store)
        loaded = store.load(parent.id)
        loaded.status = RunStatus.RUNNING
        store.save(loaded)
        with pytest.raises(ValueError, match="not finished"):
            await rescore(parent.id, [correct], store=store)
