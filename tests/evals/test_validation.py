import importlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.evals import (
    ClassRecall,
    CohenKappa,
    ConfusionMatrix,
    Consistency,
    CorrectedPassRate,
    EvalContext,
    Evaluation,
    JudgeErrorRates,
    LocalRunStore,
    PassRate,
    Perturbation,
    Score,
    Trial,
    UnvalidatedJudgeError,
    ValidationGate,
    compare,
    evaluator,
    judge_probes,
    judge_validation,
)
from grasp_agents.evals.stats import cohens_kappa, corrected_prevalence
from grasp_agents.evals.validation import find_validation, summarize_validation

type Judged = EvalContext[str, str, Any]


@evaluator(name="good", annotator="LLM")
def good_judge(ctx: Judged) -> Score:
    return Score(name="good", value="good" in ctx.output, explanation="said good")


@evaluator(name="good", version="2", annotator="LLM")
def good_judge_v2(ctx: Judged) -> bool:
    return "good" in ctx.output and not ctx.output.endswith(".")


# (output, label, split): the judge says pass for "good"; people want "!".
_ROWS = [
    ("good!", True, "dev"),
    ("good.", False, "dev"),
    ("bad!", True, "dev"),
    ("bad.", False, "dev"),
    ("very good!", True, "test"),
    ("so bad.", False, "test"),
    ("good enough.", False, "test"),
    ("not bad!", True, "test"),
]


def _labels(path: Path, rows: list[tuple[str, Any, str]] = _ROWS) -> Path:
    records = [
        {
            "id": f"o{i}",
            "input": {"input": "q", "output": output},
            "reference": label,
            "metadata": {"example_id": f"e{i}"},
            "splits": [split],
        }
        for i, (output, label, split) in enumerate(rows)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.fixture
def store(tmp_path: Path) -> LocalRunStore:
    return LocalRunStore(tmp_path / "evals")


def _trial(example_id: str, repetition: int = 0, **scores: Any) -> Trial:
    return Trial(
        example_id=example_id,
        repetition=repetition,
        example_hash="h",
        started_at=datetime.now(UTC),
        duration_s=0.0,
        scores=[Score(name=k.replace("__", "."), value=v) for k, v in scores.items()],
        evaluated=["e"],
    )


class TestAgreementMetrics:
    def test_kappa_matches_the_pairs(self) -> None:
        pairs = [("true", "true"), ("true", "false"), ("false", "false")] * 3
        trials = [
            _trial(f"e{i}", judge="true" if j == "true" else "false", label=t)
            for i, (t, j) in enumerate(pairs)
        ]
        result = CohenKappa("judge", "label").compute(trials)
        assert result.value == pytest.approx(cohens_kappa(pairs))
        assert result.ci_low is not None
        assert result.ci_high is not None
        assert result.ci_low <= result.value <= result.ci_high
        assert (result.n, result.details["judgments"]) == (9, 9)

    def test_class_recall_averages_repetitions_per_example(self) -> None:
        trials = [
            _trial("a", 0, judge="true", label="true"),
            _trial("a", 1, judge="false", label="true"),
            _trial("b", 0, judge="true", label="true"),
            _trial("b", 1, judge="true", label="true"),
            _trial("c", 0, judge="true", label="false"),
        ]
        tpr = ClassRecall("judge", "label", True).compute(trials)
        assert tpr.value == pytest.approx(0.75)
        assert tpr.n == 2
        tnr = ClassRecall("judge", "label", False).compute(trials)
        assert tnr.value == 0.0
        assert tnr.n == 1

    def test_confusion_counts_every_judgment(self) -> None:
        trials = [
            _trial("a", judge="true", label="true"),
            _trial("b", judge="true", label="false"),
            _trial("c", judge="false", label="false"),
            _trial("d", judge="false", label="false"),
        ]
        result = ConfusionMatrix("judge", "label").compute(trials)
        assert result.details["counts"] == {
            "false": {"false": 2, "true": 1},
            "true": {"false": 0, "true": 1},
        }

    def test_consistency_needs_two_repetitions(self) -> None:
        trials = [
            _trial("same", 0, v=True),
            _trial("same", 1, v=True),
            _trial("mixed", 0, v=True),
            _trial("mixed", 1, v=True),
            _trial("mixed", 2, v=False),
            _trial("single", 0, v=True),
        ]
        result = Consistency("v").compute(trials)
        assert result.value == 0.5
        assert result.n == 2
        assert result.n_na == 1
        assert result.details["pair_agreement"] == pytest.approx((1 + 1 / 3) / 2)


class TestJudgeValidation:
    @pytest.mark.asyncio
    async def test_reports_agreement_with_labels(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        validation = judge_validation(
            good_judge, _labels(tmp_path / "labels.jsonl"), input_type=str
        )
        run = await validation.run(store=store)
        values = {m.name: m.value for m in run.metrics}
        # TP: good!, very good!; FP: good., good enough.; FN: bad!, not bad!
        assert values["accuracy(good)"] == pytest.approx(4 / 8)
        assert values["tpr(good)"] == pytest.approx(2 / 4)
        assert values["tnr(good)"] == pytest.approx(2 / 4)
        assert values["kappa(good)"] == pytest.approx(0.0)
        assert run.task.kind == "evaluator"
        assert run.config.sealed_splits == ["test"]
        assert {t.sealed for t in run.trials} == {True, False}
        explanation = run.trials[1].score("good.agrees")
        assert explanation is not None
        assert explanation.explanation == "judge: true, label: false — said good"

    @pytest.mark.asyncio
    async def test_judge_versions_compare_on_the_same_labels(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        labels = _labels(tmp_path / "labels.jsonl")
        first = await judge_validation(good_judge, labels).run(split="dev", store=store)
        second = await judge_validation(good_judge_v2, labels).run(
            split="dev", store=store
        )
        comparison = compare(first, second)
        target = next(t for t in comparison.targets if t.target == "good.agrees")
        assert target.n_pairs == 4
        assert target.improved == 1
        assert target.regressed == 0

    @pytest.mark.asyncio
    async def test_several_scores_numeric_verdicts_and_missing_ones(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        @evaluator(name="rubric", annotator="LLM")
        def rubric(ctx: Judged) -> dict[str, Any]:
            return {"clarity": len(ctx.output) / 10, "tone": "kind"}

        @evaluator(name="unsure", annotator="LLM")
        def unsure(ctx: Judged) -> Score:
            return Score.unscored("unsure", reason="refusal")

        rows: list[tuple[str, Any, str]] = [
            ("short", {"clarity": False, "tone": "kind"}, "dev"),
            ("a long answer", {"clarity": True, "tone": "rude"}, "dev"),
            ("mid sized", {"clarity": True}, "dev"),
        ]
        labels = _labels(tmp_path / "rubric.jsonl", rows)
        run = await judge_validation(
            rubric, labels, scores=["clarity", "tone"], threshold=0.7, sealed_splits=()
        ).run(store=store)
        values = {m.name: (m.value, m.n) for m in run.metrics}
        # clarity: 0.5 → false (agrees), 1.3 → true (agrees), 0.9 → true (agrees).
        assert values["accuracy(clarity)"] == (1.0, 3)
        # tone: the third output has no tone label (not applicable).
        assert values["accuracy(tone)"] == (0.5, 2)

        missing = await judge_validation(
            unsure, _labels(tmp_path / "u.jsonl"), sealed_splits=()
        ).run(store=store)
        accuracy = missing.metric("accuracy(unsure)")
        assert accuracy is not None
        assert accuracy.n == 0
        assert accuracy.n_missing == 8

    @pytest.mark.asyncio
    async def test_a_single_label_cannot_validate_several_scores(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        run = await judge_validation(
            good_judge, _labels(tmp_path / "l.jsonl"), scores=["good", "other"]
        ).run(split="dev", store=store)
        assert run.counts.evaluator_failures == 4

    @pytest.mark.asyncio
    async def test_a_failing_judge_is_a_task_error_and_usage_is_the_trials(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        @evaluator(name="good", annotator="LLM")
        async def costly(ctx: Judged) -> bool:
            from grasp_agents.evals import Usage  # noqa: PLC0415

            ctx.record_usage(Usage(input_tokens=7, cost_usd=0.01))
            if ctx.output.startswith("bad"):
                raise RuntimeError("judge timed out")
            return "good" in ctx.output

        run = await judge_validation(costly, _labels(tmp_path / "l.jsonl")).run(
            split="dev", store=store
        )
        assert run.counts.task_errors == 2
        assert all(t.usage.input_tokens == 7 for t in run.trials)
        assert run.usage.cost_usd == pytest.approx(0.04)

    @pytest.mark.asyncio
    async def test_repetitions_measure_self_consistency(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        rng = random.Random(0)

        @evaluator(name="good", annotator="LLM")
        def coin(ctx: Judged) -> bool:
            return rng.random() < 0.5

        run = await judge_validation(
            coin, _labels(tmp_path / "l.jsonl"), repetitions=4
        ).run(split="dev", store=store)
        consistency = run.metric("consistency(good)")
        assert consistency is not None
        assert consistency.value is not None
        assert consistency.value < 1.0
        assert consistency.n == 4


class TestValidationGates:
    @pytest.mark.asyncio
    async def test_finds_the_current_judges_sealed_run(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        labels = _labels(tmp_path / "labels.jsonl")
        validation = judge_validation(good_judge, labels)
        assert find_validation(store, [good_judge], "good") is None
        await validation.run(split="dev", store=store)
        # Dev only: nothing held out yet.
        assert find_validation(store, [good_judge], "good") is None
        run = await validation.run(split="test", store=store)
        found = find_validation(store, [good_judge], "good")
        assert found is not None
        assert found.run_id == run.id
        assert found.sealed
        # Over the sealed split: very good! (TP), so bad. (TN), good enough. (FP),
        # not bad! (FN).
        assert found.accuracy.value == 0.5
        assert find_validation(store, [good_judge_v2], "good") is None
        assert find_validation(store, [good_judge], "other") is None

    @pytest.mark.asyncio
    async def test_a_code_change_invalidates_the_validation(
        self, tmp_path: Path, store: LocalRunStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await judge_validation(good_judge, _labels(tmp_path / "l.jsonl")).run(
            split="test", store=store
        )
        assert find_validation(store, [good_judge], "good") is not None
        # Same name, version and configuration; the function's code changed.
        module = importlib.import_module("grasp_agents.evals.evaluator")
        monkeypatch.setattr(module, "code_hash", lambda _: "edited")
        assert find_validation(store, [good_judge], "good") is None

    def test_gate_reads_the_lower_bound_by_default(self) -> None:
        from grasp_agents.evals import ComponentInfo, JudgeValidation, MetricResult  # noqa: PLC0415

        validation = JudgeValidation(
            run_id="r",
            score="good",
            evaluator=ComponentInfo(name="good", kind="k"),
            labels="labels",
            labels_fingerprint="f",
            sealed=True,
            judgments=10,
            missing=0,
            accuracy=MetricResult(name="a", value=0.9, n=10, ci_low=0.6, ci_high=1.0),
            kappa=MetricResult(name="k", value=0.7, n=10, ci_low=0.3, ci_high=0.9),
        )
        assert ValidationGate(min_kappa=0.5).failures(validation)
        assert not ValidationGate(min_kappa=0.5, bound="value").failures(validation)
        assert not ValidationGate(min_accuracy=0.6).failures(validation)
        missing = ValidationGate(min_tpr=0.5).failures(validation)
        assert missing == ["good: tpr ci_low unknown < 0.5 (validation run r)"]

    @pytest.mark.asyncio
    async def test_gated_evaluation_needs_a_passing_validation(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        from grasp_agents.evals import Dataset, Example, FunctionTask  # noqa: PLC0415

        answers = Dataset(
            [Example(id=str(i), input=w) for i, w in enumerate(["good", "bad", "good"])]
        )

        async def shout(x: str) -> str:
            return x + "!"

        def gated(metrics: Any = None, minimum: float = 0.0) -> Evaluation:
            return Evaluation(
                name="answers",
                task=FunctionTask(shout),
                dataset=answers,
                evaluators=[good_judge],
                metrics=metrics,
                validation_gates={"good": ValidationGate(min_accuracy=minimum)},
            )

        with pytest.raises(UnvalidatedJudgeError, match="no finished validation run"):
            await gated().run(store=store)
        allowed = await gated().run(store=store, allow_unvalidated=True)
        assert allowed.metadata["unvalidated_judges"]
        assert allowed.metric("corrected_pass_rate(good)") is None

        labels = _labels(tmp_path / "labels.jsonl")
        await judge_validation(good_judge, labels).run(split="test", store=store)
        with pytest.raises(UnvalidatedJudgeError, match="accuracy ci_low"):
            await gated(minimum=0.9).run(store=store)

        run = await gated().run(store=store)
        assert run.metadata["unvalidated_judges"] == []
        assert run.metadata["judge_validations"]["good"]["accuracy"]["value"] == 0.5
        # A judge no better than chance (TPR + TNR = 1) cannot be corrected for.
        corrected = run.metric("corrected_pass_rate(good)")
        assert corrected is not None
        assert corrected.value is None
        assert run.metric("pass_rate(good)") is not None

        explicit = await gated(metrics=[PassRate("good")]).run(store=store)
        assert [m.name for m in explicit.metrics] == [
            "pass_rate(good)",
            "corrected_pass_rate(good)",
        ]
        rescored = await gated().rescore(run, store=store)
        assert rescored.metadata["judge_validations"]["good"]["run_id"]

    @pytest.mark.asyncio
    async def test_summaries_can_cover_every_trial(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        run = await judge_validation(good_judge, _labels(tmp_path / "l.jsonl")).run(
            store=store
        )
        everything = summarize_validation(run, "good", sealed_only=False)
        assert everything.accuracy.n == 8
        rates = everything.rates()
        assert rates is not None
        assert (rates.tpr, rates.tnr) == (0.5, 0.5)
        assert rates.source == run.id


class TestCorrectedPassRate:
    def test_corrects_for_the_judges_errors(self) -> None:
        rates = JudgeErrorRates(tpr=0.9, tnr=0.8, n_positive=50, n_negative=50)
        trials = [_trial(f"e{i}", ok=i < 70) for i in range(100)]
        result = CorrectedPassRate("ok", rates).compute(trials)
        expected = corrected_prevalence(0.7, 0.9, 0.8)
        assert result.value == pytest.approx(expected)
        assert result.details["observed"] == pytest.approx(0.7)
        assert result.ci_low is not None
        assert result.ci_high is not None
        assert result.ci_low < result.value < result.ci_high

    def test_no_value_for_a_judge_no_better_than_chance(self) -> None:
        rates = JudgeErrorRates(tpr=0.5, tnr=0.4, n_positive=10, n_negative=10)
        trials = [_trial(f"e{i}", ok=i % 2 == 0) for i in range(10)]
        assert CorrectedPassRate("ok", rates).compute(trials).value is None

    def test_interval_covers_the_true_rate(self) -> None:
        rng = random.Random(1)
        prevalence, tpr, tnr = 0.6, 0.85, 0.8
        covered = 0
        simulations = 150
        for _ in range(simulations):
            measured_tpr = rng.binomialvariate(60, tpr) / 60
            measured_tnr = rng.binomialvariate(60, tnr) / 60
            trials = []
            for i in range(150):
                truth = rng.random() < prevalence
                verdict = rng.random() < (tpr if truth else 1 - tnr)
                trials.append(_trial(f"e{i}", ok=verdict))
            rates = JudgeErrorRates(
                tpr=measured_tpr, tnr=measured_tnr, n_positive=60, n_negative=60
            )
            result = CorrectedPassRate(
                "ok", rates, n_resamples=400, seed=rng.randrange(1000)
            ).compute(trials)
            assert result.ci_low is not None
            assert result.ci_high is not None
            covered += result.ci_low <= prevalence <= result.ci_high
        assert covered / simulations >= 0.9


def _flip(item: Any) -> str:
    return item.output.replace("good", "bad")


class TestProbes:
    @pytest.mark.asyncio
    async def test_sensitivity_and_invariance(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        probes = [
            Perturbation("degraded", _flip, expect="lower"),
            Perturbation("shouted", lambda item: item.output.upper(), expect="same"),
            Perturbation(
                "only_bad",
                lambda item: None if "good" in item.output else item.output + "?",
                expect="changed",
            ),
        ]
        run = await judge_probes(
            good_judge, _labels(tmp_path / "items.jsonl"), probes, input_type=str
        ).run(store=store)
        values = {m.name: (m.value, m.n, m.n_na) for m in run.metrics}
        # Degrading a "good" output always lowers the verdict; the rest were
        # already fails (not applicable).
        assert values["sensitivity(good.degraded)"] == (1.0, 4, 4)
        # Upper-casing hides "good" from this judge: never invariant on them.
        assert values["invariance(good.shouted)"] == (0.5, 8, 0)
        # Adding "?" to a bad output does not change a fail.
        assert values["sensitivity(good.only_bad)"] == (0.0, 4, 4)

    def test_perturbation_names_must_be_unique(self) -> None:
        probe = Perturbation("x", _flip, expect="lower")
        with pytest.raises(ValueError, match="unique"):
            judge_probes(good_judge, "items.jsonl", [probe, probe])
        with pytest.raises(ValueError, match="unique"):
            judge_probes(
                good_judge, "items.jsonl", [Perturbation("original", _flip, "same")]
            )

    @pytest.mark.asyncio
    async def test_numeric_verdicts_with_a_threshold(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        @evaluator(name="length", annotator="LLM")
        def length(ctx: Judged) -> float:
            return float(len(ctx.output))

        probes = [
            Perturbation("halved", lambda i: i.output[: len(i.output) // 2], "lower"),
            Perturbation("trimmed", lambda i: i.output[:-1], "same"),
        ]
        rows: list[tuple[str, Any, str]] = [
            ("abcdefgh", None, "dev"),
            ("abc", None, "dev"),
        ]
        run = await judge_probes(
            length, _labels(tmp_path / "n.jsonl", rows), probes, threshold=5
        ).run(store=store)
        values = {m.name: (m.value, m.n, m.n_na) for m in run.metrics}
        # 8 → 4 crosses the threshold; 3 is already below it.
        assert values["sensitivity(length.halved)"] == (1.0, 1, 1)
        # 8 → 7 and 3 → 2 stay on their side.
        assert values["invariance(length.trimmed)"] == (1.0, 2, 0)


class TestValidationCoverage:
    @pytest.mark.asyncio
    async def test_abstentions_and_failures_count_as_missing(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        @evaluator(name="good", annotator="LLM")
        def picky(ctx: Judged) -> Score | bool:
            if "bad" in ctx.output:
                return Score.unscored("good", reason="refusal")
            if "enough" in ctx.output:
                raise RuntimeError("could not parse the verdict")
            return "good" in ctx.output

        await judge_validation(picky, _labels(tmp_path / "l.jsonl")).run(
            split="test", store=store
        )
        found = find_validation(store, [picky], "good")
        assert found is not None
        # Sealed: "very good!" judged; "so bad." and "not bad!" abstained;
        # "good enough." failed.
        assert (found.judgments, found.missing) == (4, 3)
        assert found.kappa.n_missing == 3
        assert found.tpr is not None
        assert (found.tpr.n, found.tpr.n_missing) == (1, 1)
        failures = ValidationGate(min_accuracy=0.0).failures(found)
        assert failures
        assert "3 of 4 labeled judgments have no verdict" in failures[0]
        assert not ValidationGate(max_missing=1.0).failures(found)

    def test_kappa_interval_stays_wide_on_a_few_labels(self) -> None:
        trials = [
            _trial("a", judge="true", label="true"),
            _trial("b", judge="false", label="false"),
        ]
        result = CohenKappa("judge", "label").compute(trials)
        assert result.value == 1.0
        assert result.ci_low is not None
        assert result.ci_low < 0.0
        many = [
            _trial(f"e{i}", judge=v, label=v)
            for i, v in enumerate(["true", "false"] * 30)
        ]
        tight = CohenKappa("judge", "label").compute(many)
        assert tight.ci_low is not None
        assert tight.ci_low > 0.8

    @pytest.mark.asyncio
    async def test_labels_named_for_the_gate(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        @evaluator(name="good", annotator="LLM")
        def quality(ctx: Judged) -> str:
            return "good" if "good" in ctx.output else "bad"

        rows: list[tuple[str, Any, str]] = [
            (out, "good" if label else "bad", split) for out, label, split in _ROWS
        ]
        validation = judge_validation(
            quality,
            _labels(tmp_path / "words.jsonl", rows),
            positive="good",
            negative="bad",
        )
        await validation.run(split="test", store=store)
        found = find_validation(store, [quality], "good")
        assert found is not None
        assert (found.positive, found.negative) == ("good", "bad")
        assert found.tpr is not None
        assert found.tpr.value == 0.5
        assert found.rates() is not None

    @pytest.mark.asyncio
    async def test_gate_reads_the_named_labels_and_the_newest_measurement(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        easy = _labels(
            tmp_path / "easy.jsonl", [(o, "good" in o, s) for o, _, s in _ROWS]
        )
        real = _labels(tmp_path / "real.jsonl")
        old = await judge_validation(good_judge, easy).run(split="test", store=store)
        new = await judge_validation(good_judge, real).run(split="test", store=store)
        assert find_validation(store, [good_judge], "good").run_id == new.id  # type: ignore[union-attr]
        named = find_validation(store, [good_judge], "good", labels="easy")
        assert named is not None
        assert named.run_id == old.id
        assert named.accuracy.value == 1.0
        # Re-judging the old run does not make its labels the newest.
        from grasp_agents.evals import rescore  # noqa: PLC0415
        from grasp_agents.evals.validation import LabelAgreement  # noqa: PLC0415

        await rescore(old, [LabelAgreement(["good"])], rerun=True, store=store)
        assert find_validation(store, [good_judge], "good").run_id == new.id  # type: ignore[union-attr]
        # Probe runs are not validations, even with sealed splits.
        probes = judge_probes(
            good_judge,
            real,
            [Perturbation("x", lambda i: i.output + "?", "same")],
            sealed_splits=["test"],
        )
        await probes.run(store=store)
        assert find_validation(store, [good_judge], "good").run_id == new.id  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_outputs_of_one_example_are_clustered(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        rows = [
            {
                "id": f"o{e}-{k}",
                "input": {"input": "q", "output": f"good {e}.{k}!"},
                "reference": True,
                "metadata": {"example_id": f"e{e}"},
                "splits": ["dev"],
            }
            for e in range(6)
            for k in range(4)
        ]
        path = tmp_path / "many.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        clustered = await judge_validation(good_judge, path, sealed_splits=()).run(
            store=store
        )
        flat = await judge_validation(
            good_judge, path, sealed_splits=(), cluster_by=None
        ).run(store=store)
        low = clustered.metric("accuracy(good)").ci_low  # type: ignore[union-attr]
        naive = flat.metric("accuracy(good)").ci_low  # type: ignore[union-attr]
        assert low is not None
        assert naive is not None
        assert low < 0.7 < naive

    @pytest.mark.asyncio
    async def test_numbers_bools_and_mismatched_labels(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        @evaluator(name="good", annotator="LLM")
        def binary(ctx: Judged) -> int:
            return 1 if "good" in ctx.output else 0

        run = await judge_validation(binary, _labels(tmp_path / "l.jsonl")).run(
            split="dev", store=store
        )
        assert run.metric("accuracy(good)").value == 0.5  # type: ignore[union-attr]

        rows: list[tuple[str, Any, str]] = [
            (o, "correct" if label else "incorrect", s) for o, label, s in _ROWS
        ]
        mismatched = await judge_validation(
            good_judge, _labels(tmp_path / "words.jsonl", rows)
        ).run(split="dev", store=store)
        assert mismatched.counts.evaluator_failures == 4
        failure = mismatched.trials[0].evaluator_failures[0].error.message
        assert "pass/fail verdict needs a pass/fail label" in failure

    @pytest.mark.asyncio
    async def test_validation_reports_each_split(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        run = await judge_validation(good_judge, _labels(tmp_path / "l.jsonl")).run(
            store=store
        )
        accuracy = run.metric("accuracy(good)")
        assert accuracy is not None
        assert set(accuracy.groups) == {"split=dev", "split=test"}


class TestCorrectedPassRateErrors:
    def test_task_errors_stay_failures(self) -> None:
        rates = JudgeErrorRates(tpr=0.9, tnr=0.9, n_positive=50, n_negative=50)
        judged = [_trial(f"e{i}", ok=i < 4) for i in range(5)]
        failed = [
            Trial(
                example_id=f"x{i}",
                example_hash="h",
                started_at=datetime.now(UTC),
                duration_s=0.0,
                error={"type": "RuntimeError", "message": "boom"},  # type: ignore[arg-type]
            )
            for i in range(5)
        ]
        result = CorrectedPassRate("ok", rates).compute([*judged, *failed])
        expected = corrected_prevalence(0.8, 0.9, 0.9)
        assert expected is not None
        assert result.value == pytest.approx(0.5 * expected)
        assert result.n == 10
        kept_out = CorrectedPassRate("ok", rates, errors_as_failures=False).compute(
            [*judged, *failed]
        )
        assert kept_out.value == pytest.approx(expected)

    def test_rates_measured_at_one_still_vary(self) -> None:
        rates = JudgeErrorRates(tpr=1.0, tnr=1.0, n_positive=10, n_negative=10)
        trials = [_trial(f"e{i}", ok=i < 120) for i in range(200)]
        result = CorrectedPassRate("ok", rates).compute(trials)
        assert result.value == pytest.approx(0.6)
        assert result.ci_low is not None
        assert result.ci_high is not None
        # Wider than the observed rate's own interval alone would be.
        assert result.ci_high - result.ci_low > 0.15


class TestProbeEdges:
    @pytest.mark.asyncio
    async def test_lower_needs_ordered_verdicts(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        @evaluator(name="tone", annotator="LLM")
        def tone(ctx: Judged) -> str:
            return "kind" if "!" in ctx.output else "flat"

        run = await judge_probes(
            tone,
            _labels(tmp_path / "l.jsonl"),
            [Perturbation("flat", lambda i: i.output.rstrip("!."), "lower")],
        ).run(store=store)
        assert run.counts.evaluator_failures == 8
        assert (
            "use expect='changed'" in run.trials[0].evaluator_failures[0].error.message
        )

    def test_perturbation_code_is_part_of_the_identity(self) -> None:
        def first(item: Any) -> str:
            return item.output + "!"

        def second(item: Any) -> str:
            return item.output + "?"

        a = judge_probes(good_judge, "x.jsonl", [Perturbation("p", first, "same")])
        b = judge_probes(good_judge, "x.jsonl", [Perturbation("p", second, "same")])
        assert a.build_task().describe() != b.build_task().describe()
