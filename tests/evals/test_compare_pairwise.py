import asyncio
import importlib
import threading
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.evals import (
    Dataset,
    Example,
    FunctionPairwiseJudge,
    FunctionTask,
    LocalRunStore,
    Mean,
    Measure,
    PairwiseContext,
    PairwiseVerdict,
    Pooled,
    Score,
    ScoreContext,
    Trial,
    TrialContext,
    WinRate,
    compare,
    evaluate,
    pairwise,
    render_comparison_markdown,
    scorer,
)
from grasp_agents.evals.compare import regressions

type Ctx = ScoreContext[int, int, int]


def _dataset(n: int = 20) -> Dataset[int, int]:
    return Dataset([Example(id=f"x{i}", input=i, reference=i * 2) for i in range(n)])


@scorer
def correct(ctx: Ctx) -> bool:
    return ctx.output == ctx.reference


async def good(x: int) -> int:
    return x * 2


async def flawed(x: int) -> int:
    return x * 2 if x % 2 == 0 else x * 2 + 1


@pytest.fixture
def store(tmp_path: Path) -> LocalRunStore:
    return LocalRunStore(tmp_path)


class TestCompare:
    @pytest.mark.asyncio
    async def test_paired_binary_comparison(self, store: LocalRunStore) -> None:
        base = await evaluate(
            FunctionTask(flawed, name="solver"), _dataset(), [correct], store=store
        )
        candidate = await evaluate(
            FunctionTask(good, name="solver"), _dataset(), [correct], store=store
        )
        result = compare(base, candidate)
        target = result.target("correct")
        assert target is not None
        assert result.n_paired_examples == 20
        assert target.test == "mcnemar"
        assert target.diff == pytest.approx(0.5)
        assert target.improved == 10
        assert target.regressed == 0
        assert target.p_value is not None
        assert target.p_value < 0.01
        assert target.significant
        assert not target.top_regressions
        assert len(target.top_improvements) == 5
        assert result.target(str(Measure.DURATION)) is not None
        text = render_comparison_markdown(result)
        assert "correct" in text
        assert "✱" in text

    @pytest.mark.asyncio
    async def test_changed_examples_are_not_paired(self, store: LocalRunStore) -> None:
        base = await evaluate(good, _dataset(4), [correct], store=store)
        changed = Dataset(
            [*_dataset(3).examples, Example(id="x3", input=3, reference=7)]
        )
        candidate = await evaluate(good, changed, [correct], store=store)
        result = compare(base, candidate)
        assert result.n_paired_examples == 3
        assert result.n_changed_examples == 1
        assert any("different dataset content" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_identical_configuration_warns_about_noise(
        self, store: LocalRunStore
    ) -> None:
        a = await evaluate(good, _dataset(3), [correct], store=store)
        b = await evaluate(good, _dataset(3), [correct], store=store)
        assert any("noise" in w for w in compare(a, b).warnings)

    @pytest.mark.asyncio
    async def test_sealed_examples_never_listed(self, store: LocalRunStore) -> None:
        dataset = Dataset(
            [
                Example(id=f"x{i}", input=i, reference=i * 2, splits=["test"])
                for i in range(6)
            ]
        )
        base = await evaluate(
            flawed,
            dataset.split("test"),
            [correct],
            sealed_splits=["test"],
            store=store,
        )
        candidate = await evaluate(
            good, dataset.split("test"), [correct], sealed_splits=["test"], store=store
        )
        target = compare(base, candidate).target("correct")
        assert target is not None
        assert target.improved == 3
        assert target.top_improvements == []


def _prefer_longer(ctx: PairwiseContext[int, str, Any]) -> PairwiseVerdict:
    if len(ctx.first) == len(ctx.second):
        return PairwiseVerdict(winner="tie")
    return PairwiseVerdict(
        winner="first" if len(ctx.first) > len(ctx.second) else "second",
        explanation="longer is better",
    )


def _always_first(ctx: PairwiseContext[int, str, Any]) -> PairwiseVerdict:
    return PairwiseVerdict(winner="first")


class TestPairwise:
    @pytest.mark.asyncio
    async def test_order_swapped_win_rate(self, store: LocalRunStore) -> None:
        async def short(x: int) -> str:
            return "a"

        async def long(x: int) -> str:
            return "a" * (2 if x < 8 else 1)

        base = await evaluate(FunctionTask(short), _dataset(10), store=store)
        candidate = await evaluate(FunctionTask(long), _dataset(10), store=store)
        run = await pairwise(
            base.id,
            candidate.id,
            FunctionPairwiseJudge(_prefer_longer, name="length"),
            store=store,
        )
        assert run.kind == "pairwise"
        assert run.parent_run_id is None
        assert run.counts.trials_done == 10
        win = run.metric("win_rate(length)")
        assert win is not None
        assert win.value == pytest.approx((8 + 0.5 * 2) / 10)
        assert win.details["candidate"] == 8
        assert win.details["tie"] == 2
        consistency = run.metric("pass_rate(length.position_consistent)")
        assert consistency is not None
        assert consistency.value == pytest.approx(1.0)
        assert store.load(run.id).trials[0].output == {"base": "a", "candidate": "aa"}

    @pytest.mark.asyncio
    async def test_position_bias_is_exposed(self, store: LocalRunStore) -> None:
        async def same(x: int) -> str:
            return "same"

        base = await evaluate(FunctionTask(same), _dataset(5), store=store)
        candidate = await evaluate(FunctionTask(same), _dataset(5), store=store)
        run = await pairwise(
            base,
            candidate,
            FunctionPairwiseJudge(_always_first, name="biased"),
            store=store,
        )
        consistency = run.metric("pass_rate(biased.position_consistent)")
        assert consistency is not None
        assert consistency.value == pytest.approx(0.0)
        win = run.metric("win_rate(biased)")
        assert win is not None
        assert win.value == pytest.approx(0.5)
        assert win.details["inconsistent"] == 5

    @pytest.mark.asyncio
    async def test_sync_judges_run_off_the_event_loop(
        self, store: LocalRunStore
    ) -> None:
        async def same(x: int) -> str:
            return "same"

        loop_alive = threading.Event()

        async def tick() -> None:
            await asyncio.sleep(0.01)
            loop_alive.set()

        def judge(ctx: PairwiseContext[int, str, int]) -> PairwiseVerdict:
            # Blocks; only the event loop (the tick task) can release it.
            released = loop_alive.wait(timeout=2)
            return PairwiseVerdict(winner="tie" if released else "first")

        base = await evaluate(FunctionTask(same), _dataset(1), store=store)
        candidate = await evaluate(FunctionTask(same), _dataset(1), store=store)
        ticker = asyncio.create_task(tick())
        run = await pairwise(
            base, candidate, FunctionPairwiseJudge(judge, name="j"), store=store
        )
        await ticker
        trial = run.trial("x0")
        assert trial is not None
        winner = trial.score("j.winner")
        assert winner is not None
        assert winner.value == "tie"

    @pytest.mark.asyncio
    async def test_a_failed_arm_loses_its_pair(self, store: LocalRunStore) -> None:
        async def broken(x: int) -> str:
            if x == 0:
                raise ValueError("no")
            return "ok"

        async def fine(x: int) -> str:
            return "ok"

        asked: list[str] = []

        def judge(ctx: PairwiseContext[int, str, int]) -> PairwiseVerdict:
            asked.append(ctx.example.id)
            return PairwiseVerdict(winner="tie")

        base = await evaluate(FunctionTask(broken), _dataset(3), store=store)
        candidate = await evaluate(FunctionTask(fine), _dataset(3), store=store)
        run = await pairwise(
            base, candidate, FunctionPairwiseJudge(judge, name="j"), store=store
        )
        assert run.counts.trials_done == 3
        assert "x0" not in asked
        decided = run.trial("x0")
        assert decided is not None
        winner = decided.score("j.winner")
        assert winner is not None
        assert winner.value == "candidate"

    @pytest.mark.asyncio
    async def test_nothing_to_judge_is_refused(self, store: LocalRunStore) -> None:
        async def broken(x: int) -> str:
            raise ValueError("no")

        base = await evaluate(FunctionTask(broken), _dataset(2), store=store)
        with pytest.raises(ValueError, match="nothing to judge"):
            await pairwise(
                base, base, FunctionPairwiseJudge(_prefer_longer, name="j"), store=store
            )


@pytest.mark.asyncio
async def test_pairwise_validates_stored_types(store: LocalRunStore) -> None:
    from pydantic import BaseModel

    class Item(BaseModel):
        word: str

    def by_length(ctx: PairwiseContext[Item, str, Any]) -> PairwiseVerdict:
        assert isinstance(ctx.input, Item)  # stored JSON re-validated
        longer = len(ctx.first) - len(ctx.second)
        return PairwiseVerdict(
            winner="tie" if not longer else ("first" if longer > 0 else "second")
        )

    dataset = Dataset([Example(id=w, input=Item(word=w)) for w in ["a", "bb", "ccc"]])

    async def echo(item: Item) -> str:
        return item.word

    async def double(item: Item) -> str:
        return item.word * 2

    base = await evaluate(FunctionTask(echo), dataset, store=store)
    candidate = await evaluate(FunctionTask(double), dataset, store=store)
    run = await pairwise(
        base.id,
        candidate.id,
        FunctionPairwiseJudge(by_length, name="len"),
        input_type=Item,
        store=store,
    )
    assert run.counts.scorer_failures == 0
    assert run.metric("win_rate(len)").value == pytest.approx(1.0)  # type: ignore[union-attr]


def _timed(delay: float) -> Any:
    async def solve(x: int, trial: TrialContext) -> int:
        trial.record("latency", delay + x / 100)
        return x * 2

    return FunctionTask(solve, name="solver")


class TestGate:
    @pytest.mark.asyncio
    async def test_lower_is_better_targets_count_slower_as_worse(
        self, store: LocalRunStore
    ) -> None:
        base = await evaluate(good, _dataset(6), [correct], store=store)
        slower = base.model_copy(deep=True)
        for trial in slower.trials:
            trial.duration_s += 1.0
        result = compare(base, slower, targets=[Measure.DURATION])
        target = result.target("duration_s")
        assert target is not None
        assert target.regressed == 6
        assert target.improved == 0
        assert target.worse

    @pytest.mark.asyncio
    async def test_the_gate_reads_scores_not_timings(
        self, store: LocalRunStore
    ) -> None:
        base = await evaluate(good, _dataset(12), [correct], store=store)
        slower = base.model_copy(deep=True)
        for trial in slower.trials:
            trial.duration_s += 1.0
        result = compare(base, slower)
        duration = result.target("duration_s")
        assert duration is not None
        assert duration.significant
        assert regressions(result) == []
        assert regressions(result, targets=["duration_s"])
        assert regressions(result, targets=["duration_s"], min_effect=5.0) == []

    @pytest.mark.asyncio
    async def test_the_gate_fails_when_nothing_pairs(
        self, store: LocalRunStore
    ) -> None:
        base = await evaluate(good, _dataset(3), [correct], store=store)
        other = await evaluate(
            good,
            Dataset([Example(id=f"y{i}", input=i, reference=i * 2) for i in range(3)]),
            [correct],
            store=store,
        )
        result = compare(base, other)
        assert any("No example could be paired" in w for w in result.warnings)
        assert "no example could be paired with the baseline" in regressions(result)

    @pytest.mark.asyncio
    async def test_significance_follows_the_reported_test(
        self, store: LocalRunStore
    ) -> None:
        # 4 discordant pairs, all regressions: the exact McNemar p is 0.125.
        async def mostly(x: int) -> int:
            return -1 if x < 4 else x * 2

        base = await evaluate(good, _dataset(100), [correct], store=store)
        candidate = await evaluate(mostly, _dataset(100), [correct], store=store)
        target = compare(base, candidate).target("correct")
        assert target is not None
        assert target.p_value == pytest.approx(0.125)
        assert not target.significant

    @pytest.mark.asyncio
    async def test_task_errors_count_as_failures(self, store: LocalRunStore) -> None:
        async def crashes(x: int) -> int:
            if x < 10:
                raise RuntimeError("crash")
            return x * 2

        base = await evaluate(good, _dataset(20), [correct], store=store)
        candidate = await evaluate(crashes, _dataset(20), [correct], store=store)
        result = compare(base, candidate)
        target = result.target("correct")
        assert target is not None
        assert target.n_pairs == 20
        assert target.regressed == 10
        assert any("task errors in only one run" in w for w in result.warnings)
        assert any("correct" in failure for failure in regressions(result))

    @pytest.mark.asyncio
    async def test_the_gate_catches_crashes_behind_numeric_scores(
        self, store: LocalRunStore
    ) -> None:
        @scorer
        def closeness(ctx: Ctx) -> float:
            return 1.0 if ctx.output == ctx.reference else 0.5

        async def crashes(x: int) -> int:
            if x < 8:
                raise RuntimeError("crash")
            return x * 2

        base = await evaluate(good, _dataset(20), [closeness], store=store)
        candidate = await evaluate(crashes, _dataset(20), [closeness], store=store)
        result = compare(base, candidate)
        score = result.target("closeness")
        assert score is not None
        assert score.n_pairs == 12  # pairs with a crash have no score to compare
        assert not score.worse
        assert any(
            f.startswith("significant regression in error") for f in regressions(result)
        )

    @pytest.mark.asyncio
    async def test_resources_are_adjusted_apart_from_outcomes(
        self, store: LocalRunStore
    ) -> None:
        # 9 regressions and 1 improvement: exact McNemar p = 0.0215.
        async def base_task(x: int) -> int:
            return -1 if x == 19 else x * 2

        async def candidate_task(x: int) -> int:
            return -1 if x < 9 else x * 2

        base = await evaluate(base_task, _dataset(20), [correct], store=store)
        candidate = await evaluate(candidate_task, _dataset(20), [correct], store=store)
        result = compare(base, candidate)
        target = result.target("correct")
        assert target is not None
        assert target.p_value == pytest.approx(0.021484375)
        # The unchanged error rate and the duration do not dilute the score.
        assert target.p_adjusted == pytest.approx(target.p_value)
        assert target.significant
        assert regressions(result)

    @pytest.mark.asyncio
    async def test_a_target_that_cannot_be_tested_fails_the_gate(
        self, store: LocalRunStore
    ) -> None:
        data = Dataset(
            [
                Example(id=f"x{i}", input=i, reference=i * 2, metadata={"course": "c"})
                for i in range(10)
            ]
        )
        base = await evaluate(good, data, [correct], store=store, cluster_by="course")
        candidate = await evaluate(
            flawed, data, [correct], store=store, cluster_by="course"
        )
        result = compare(base, candidate)
        assert any("one 'course' cluster" in w for w in result.warnings)
        target = result.target("correct")
        assert target is not None
        assert target.p_value is None
        assert any("correct cannot be tested" in f for f in regressions(result))

    @pytest.mark.asyncio
    async def test_a_score_missing_from_the_candidate_fails_the_gate(
        self, store: LocalRunStore
    ) -> None:
        @scorer(name="correct")
        def broken(ctx: Ctx) -> bool:
            raise RuntimeError("judge down")

        base = await evaluate(good, _dataset(4), [correct], store=store)
        candidate = await evaluate(good, _dataset(4), [broken], store=store)
        result = compare(base, candidate)
        assert result.missing_scores == ["correct"]
        assert any("no values for ['correct']" in w for w in result.warnings)
        assert "the candidate has no values for correct" in regressions(result)
        # Naming other targets leaves it out of the gate.
        assert not any("correct" in f for f in regressions(result, targets=["error"]))

    @pytest.mark.asyncio
    async def test_changed_scorer_code_is_flagged(
        self, store: LocalRunStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = await evaluate(good, _dataset(4), [correct], store=store)
        # (The package exports a function named like the module.)
        module = importlib.import_module("grasp_agents.evals.scorer")
        monkeypatch.setattr(module, "code_hash", lambda _: "v2:edited")
        candidate = await evaluate(good, _dataset(4), [correct], store=store)
        assert any("changed its code" in w for w in compare(base, candidate).warnings)

    @pytest.mark.asyncio
    async def test_code_hashes_of_an_older_scheme_are_not_called_changes(
        self, store: LocalRunStore
    ) -> None:
        base = await evaluate(good, _dataset(4), [correct], store=store)
        candidate = await evaluate(good, _dataset(4), [correct], store=store)
        base.scorers[0].source = "0123456789ab"  # recorded before schemes
        warnings = compare(base, candidate).warnings
        assert any("different versions of grasp-evals" in w for w in warnings)
        assert not any("changed its code" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_changed_declared_identity_is_flagged(
        self, store: LocalRunStore
    ) -> None:
        def judged(model: str) -> Any:
            return scorer(name="correct", identity=[model])(correct.fn)

        base = await evaluate(good, _dataset(4), [judged("a")], store=store)
        candidate = await evaluate(good, _dataset(4), [judged("b")], store=store)
        assert any("declared identity" in w for w in compare(base, candidate).warnings)

    @pytest.mark.asyncio
    async def test_clustered_runs_compare_on_clustered_errors(
        self, store: LocalRunStore
    ) -> None:
        data = Dataset(
            [
                Example(
                    id=f"x{i}", input=i, reference=i * 2, metadata={"course": i % 3}
                )
                for i in range(9)
            ]
        )
        base = await evaluate(
            _timed(0.0), data, [correct], cluster_by="course", store=store
        )
        candidate = await evaluate(
            _timed(0.5), data, [correct], cluster_by="course", store=store
        )
        result = compare(base, candidate, targets=["latency"])
        target = result.target("latency")
        assert target is not None
        assert result.cluster_by == "course"
        assert target.test in {"clustered_paired_t", "sign"}

    @pytest.mark.asyncio
    async def test_regressions_explain_the_failing_repetition(
        self, store: LocalRunStore
    ) -> None:
        @scorer
        def explained(ctx: Ctx) -> Score:
            ok = ctx.output == ctx.reference
            return Score(
                name="explained", value=ok, explanation="ok" if ok else "WRONG"
            )

        attempts: dict[int, int] = {}

        async def wobbly(x: int) -> int:
            attempts[x] = attempts.get(x, 0) + 1
            return -1 if x == 0 and attempts[x] == 2 else x * 2

        base = await evaluate(
            good, _dataset(2), [explained], repetitions=3, store=store
        )
        candidate = await evaluate(
            wobbly, _dataset(2), [explained], repetitions=3, concurrency=1, store=store
        )
        target = compare(base, candidate).target("explained")
        assert target is not None
        (regression,) = target.top_regressions
        assert regression.candidate_explanation == "WRONG"

    @pytest.mark.asyncio
    async def test_different_subsets_are_flagged(self, store: LocalRunStore) -> None:
        base = await evaluate(good, _dataset(6).head(4), [correct], store=store)
        candidate = await evaluate(good, _dataset(6).head(3), [correct], store=store)
        assert any("different subsets" in w for w in compare(base, candidate).warnings)


class TestDirections:
    @staticmethod
    def _defects(lower_is_better: bool) -> Any:
        @scorer(name="defects", lower_is_better=lower_is_better)
        def defects(ctx: Ctx) -> float:
            return float(ctx.output != ctx.reference)

        return defects

    @pytest.mark.asyncio
    async def test_a_lower_is_better_score_improves_downwards(
        self, store: LocalRunStore
    ) -> None:
        defects = self._defects(lower_is_better=True)
        base = await evaluate(flawed, _dataset(), [defects], store=store)
        candidate = await evaluate(good, _dataset(), [defects], store=store)
        reloaded = store.load(candidate.id)
        assert reloaded.directions == {"defects": "lower"}
        result = compare(base, reloaded)
        target = result.target("defects")
        assert target is not None
        assert not target.higher_is_better
        assert target.diff == pytest.approx(-0.5)
        assert (target.improved, target.regressed) == (10, 0)
        assert target.significant
        assert not target.top_regressions
        assert regressions(result) == []
        backwards = compare(reloaded, base)
        assert any("defects" in f for f in regressions(backwards))
        assert "↓ defects" in render_comparison_markdown(result)

    @pytest.mark.asyncio
    async def test_runs_that_disagree_leave_the_score_out(
        self, store: LocalRunStore
    ) -> None:
        base = await evaluate(
            flawed, _dataset(), [self._defects(lower_is_better=False)], store=store
        )
        candidate = await evaluate(
            good, _dataset(), [self._defects(lower_is_better=True)], store=store
        )
        result = compare(base, candidate)
        assert result.target("defects") is None
        assert any("disagree" in w and "defects" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_a_run_without_directions_takes_the_other_runs(
        self, store: LocalRunStore
    ) -> None:
        defects = self._defects(lower_is_better=True)
        base = await evaluate(flawed, _dataset(), [defects], store=store)
        candidate = await evaluate(good, _dataset(), [defects], store=store)
        base.directions = {}  # recorded before directions existed
        target = compare(base, candidate).target("defects")
        assert target is not None
        assert not target.higher_is_better

    @pytest.mark.asyncio
    async def test_metrics_declare_directions_of_measurements(
        self, store: LocalRunStore
    ) -> None:
        async def solve(x: int, trial: TrialContext) -> int:
            trial.record("retries", float(x % 3))
            return x * 2

        run = await evaluate(
            solve,
            _dataset(4),
            [correct],
            [Mean("retries", higher_is_better=False)],
            store=store,
        )
        assert run.directions == {"correct": "higher", "retries": "lower"}
        with pytest.raises(ValueError, match="declares the opposite"):
            await evaluate(
                solve,
                _dataset(4),
                [self._defects(lower_is_better=True)],
                [Mean("defects", higher_is_better=True)],
                store=store,
            )
        with pytest.raises(ValueError, match="built-in direction"):
            Mean(Measure.DURATION, higher_is_better=True)


class TestWinRateUnits:
    def test_the_sign_test_counts_examples_not_judgments(self) -> None:
        from datetime import UTC, datetime

        def trial(example: str, rep: int, winner: str) -> Trial:
            return Trial(
                example_id=example,
                repetition=rep,
                example_hash="h",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                duration_s=0.0,
                scores=[Score(name="j.winner", value=winner)],
            )

        trials = [
            trial(f"e{e}", r, "candidate" if e < 4 else "base")
            for e in range(5)
            for r in range(10)
        ]
        result = WinRate("j").compute(trials)
        assert result.details["examples_won"] == 4
        assert result.details["sign_test_p"] == pytest.approx(0.375)
        assert result.n == 5


@pytest.mark.asyncio
async def test_differences_of_shares_stay_in_range(store: LocalRunStore) -> None:
    async def sometimes(x: int) -> int:
        return x * 2 if x % 3 else -1

    base = await evaluate(sometimes, _dataset(6), [correct], repetitions=3, store=store)
    candidate = await evaluate(good, _dataset(6), [correct], repetitions=3, store=store)
    target = compare(base, candidate).target("correct")
    assert target is not None
    assert target.ci_high is not None
    assert target.ci_high <= 1.0


class TestPooled:
    @staticmethod
    def _pages() -> Dataset[int, int]:
        # A page with 2 headings and pages with 40; arXiv papers and others.
        return Dataset(
            [
                Example(
                    id=f"p{i}",
                    input=i,
                    reference=2 if i == 0 else 40,
                    metadata={"source": "arxiv" if i % 2 else "other"},
                )
                for i in range(12)
            ],
            name="pages",
        )

    @staticmethod
    def _found(share_of_large: float) -> Any:
        @scorer(name="found")
        def found(ctx: ScoreContext[int, int, int]) -> Score:
            items = ctx.reference or 0
            share = 0.0 if items == 2 else share_of_large + 0.01 * (ctx.input % 3)
            return Score(name="found", value=share, weight=items)

        return found

    @pytest.mark.asyncio
    async def test_weighted_scores_are_pooled_in_metrics_and_comparisons(
        self, store: LocalRunStore
    ) -> None:
        base = await evaluate(good, self._pages(), [self._found(0.5)], store=store)
        pooled = base.metric("pooled(found)")
        assert pooled is not None
        assert pooled.value == pytest.approx(
            sum(40 * (0.5 + 0.01 * (i % 3)) for i in range(1, 12)) / (2 + 40 * 11)
        )
        assert pooled.details["weight"] == pytest.approx(442)
        candidate = await evaluate(good, self._pages(), [self._found(0.7)], store=store)
        result = compare(base, candidate, group_by="source")
        target = result.target("found")
        assert target is not None
        assert target.pooled
        assert target.test == "pooled_randomization"
        assert target.diff == pytest.approx(0.2 * 440 / 442)
        assert target.significant
        assert sorted(result.groups) == ["source=arxiv", "source=other"]
        arxiv = result.groups["source=arxiv"]
        assert arxiv.n_paired_examples == 6
        arxiv_found = arxiv.target("found")
        assert arxiv_found is not None
        assert arxiv_found.p_adjusted is not None
        assert regressions(compare(candidate, base))
        only = compare(base, candidate, where={"source": "other"})
        assert only.n_paired_examples == 6
        text = render_comparison_markdown(result)
        assert "found (pooled)" in text
        assert "## By `source`" in text

    @pytest.mark.asyncio
    async def test_weights_in_one_run_only_are_compared_on_means(
        self, store: LocalRunStore
    ) -> None:
        @scorer(name="found")
        def unweighted(ctx: ScoreContext[int, int, int]) -> float:
            return 0.5

        base = await evaluate(good, self._pages(), [unweighted], store=store)
        candidate = await evaluate(good, self._pages(), [self._found(0.5)], store=store)
        result = compare(base, candidate)
        target = result.target("found")
        assert target is not None
        assert not target.pooled
        assert any("weights in only one run" in w for w in result.warnings)

    def test_failed_repetitions_do_not_change_an_example_weight(self) -> None:
        from datetime import UTC, datetime

        def trial(example: str, rep: int, value: float | None, weight: float) -> Trial:
            scores = (
                [] if value is None else [Score(name="f", value=value, weight=weight)]
            )
            return Trial(
                example_id=example,
                repetition=rep,
                example_hash="h",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                duration_s=0.0,
                scores=scores,
            )

        # "big" has one scored repetition of two; it still weighs 40, not 20.
        trials = [
            trial("small", 0, 1.0, 10.0),
            trial("small", 1, 1.0, 10.0),
            trial("big", 0, 0.0, 40.0),
            trial("big", 1, None, 40.0),
        ]
        result = Pooled("f").compute(trials)
        assert result.value == pytest.approx(10 / 50)
        assert result.details["weight"] == pytest.approx(50)


class TestCompareFilters:
    @staticmethod
    def _data() -> Dataset[int, int]:
        flags: list[Any] = [True, 1, True, 1, "2024", 2024]
        return Dataset(
            [
                Example(
                    id=f"x{i}",
                    input=i,
                    reference=i * 2,
                    metadata={"flag": flags[i % 6], "year": 2024 if i % 2 else "2024"},
                    splits=["test" if i >= 10 else "dev"],
                )
                for i in range(12)
            ],
            name="filtered",
        )

    @pytest.mark.asyncio
    async def test_true_and_one_are_different_groups(
        self, store: LocalRunStore
    ) -> None:
        base = await evaluate(
            flawed, self._data(), [correct], include_sealed=True, store=store
        )
        candidate = await evaluate(
            good, self._data(), [correct], include_sealed=True, store=store
        )
        result = compare(base, candidate, group_by="flag")
        assert result.groups["flag=True"].n_paired_examples == 4
        assert result.groups["flag=1"].n_paired_examples == 4
        # "2024" and 2024 have the same text, so they form one group.
        assert result.groups["flag=2024"].n_paired_examples == 4
        assert len(result.groups) == 3

    @pytest.mark.asyncio
    async def test_filters_match_text_and_leave_out_sealed_examples(
        self, store: LocalRunStore
    ) -> None:
        data = self._data()
        base = await evaluate(flawed, data, [correct], include_sealed=True, store=store)
        candidate = await evaluate(
            good, data, [correct], include_sealed=True, store=store
        )
        sealed_ids = {e.id for e in data if "test" in e.splits}
        base.config.sealed_splits = ["test"]
        for run in (base, candidate):
            for trial in run.trials:
                trial.sealed = trial.example_id in sealed_ids
        result = compare(base, candidate, where={"year": "2024"})
        assert result.n_paired_examples == 10
        assert any("sealed splits are left out" in w for w in result.warnings)
        assert compare(base, candidate).n_paired_examples == 12
        nothing = compare(base, candidate, where={"year": 1999})
        assert nothing.n_paired_examples == 0
        assert any("No paired example matches" in w for w in nothing.warnings)


class TestDescriptiveMetadata:
    @staticmethod
    def _data(**metadata: str) -> Dataset[int, int]:
        return Dataset(
            [
                Example(id=f"x{i}", input=i, reference=i * 2, metadata=dict(metadata))
                for i in range(4)
            ],
            name="numbers",
        )

    @pytest.mark.asyncio
    async def test_reworded_descriptive_keys_keep_runs_pairing(
        self, store: LocalRunStore
    ) -> None:
        # The base run predates the provenance note; the candidate declares it
        # descriptive after rewording it.
        base = await evaluate(good, self._data(), [correct], store=store)
        candidate = await evaluate(
            flawed,
            self._data(note="checked twice").with_descriptive_metadata(["note"]),
            [correct],
            store=store,
        )
        assert candidate.dataset.descriptive_metadata == ["note"]
        result = compare(base, candidate)
        assert result.n_paired_examples == 4
        assert result.n_changed_examples == 0
        judged = await pairwise(
            base, candidate, FunctionPairwiseJudge(_always_first), store=store
        )
        assert judged.counts.trials_done == 4
        undeclared = await evaluate(
            flawed, self._data(note="checked twice"), [correct], store=store
        )
        assert compare(base, undeclared).n_paired_examples == 0
