import math
from datetime import UTC, datetime

import pytest

from grasp_agents.evals import (
    Distribution,
    ErrorRate,
    Example,
    Mean,
    Measure,
    PassAtK,
    PassHatK,
    PassRate,
    Percentile,
    Proportion,
    Score,
    Total,
    Trial,
    Usage,
    compute_metrics,
    default_metrics,
)
from grasp_agents.evals.metrics import all_pass, max_
from grasp_agents.evals.stats import bounded_mean_estimate
from grasp_agents.evals.types import ErrorInfo

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _trial(
    example_id: str,
    repetition: int = 0,
    *,
    scores: dict[str, bool | float | str | None] | None = None,
    error: bool = False,
    duration: float = 1.0,
    cost: float | None = None,
) -> Trial:
    return Trial(
        example_id=example_id,
        repetition=repetition,
        example_hash="h",
        started_at=_T0,
        duration_s=duration,
        error=ErrorInfo(type="ValueError", message="boom") if error else None,
        usage=Usage(cost_usd=cost),
        scores=[Score(name=k, value=v) for k, v in (scores or {}).items()],
    )


class TestMean:
    def test_reduces_repetitions_per_example(self) -> None:
        trials = [
            _trial("a", 0, scores={"s": 1.0}),
            _trial("a", 1, scores={"s": 0.0}),
            _trial("b", 0, scores={"s": 1.0}),
            _trial("b", 1, scores={"s": 1.0}),
        ]
        result = Mean("s").compute(trials)
        assert result.n == 2  # examples, not trials
        assert result.value == pytest.approx(0.75)

    def test_custom_reducer_in_name(self) -> None:
        trials = [_trial("a", 0, scores={"s": 0.2}), _trial("a", 1, scores={"s": 0.9})]
        result = Mean("s", reduce=max_).compute(trials)
        assert result.name == "mean(s)[max]"
        assert result.value == pytest.approx(0.9)

    def test_missing_scores_are_counted_not_zeroed(self) -> None:
        trials = [
            _trial("a", scores={"s": 1.0}),
            _trial("b", error=True),
            _trial("c", scores={"s": None}),
        ]
        result = Mean("s").compute(trials)
        assert result.value == pytest.approx(1.0)
        assert result.n == 1
        assert result.n_missing == 2

    def test_bools_count_as_one_and_zero(self) -> None:
        result = Mean("ok").compute(
            [_trial("a", scores={"ok": True}), _trial("b", scores={"ok": False})]
        )
        assert result.value == pytest.approx(0.5)

    def test_measures(self) -> None:
        trials = [_trial("a", duration=2.0), _trial("b", duration=4.0)]
        assert Mean(Measure.DURATION).compute(trials).value == pytest.approx(3.0)

    def test_trial_unit_clusters_repetitions(self) -> None:
        trials = [
            _trial("a", r, scores={"s": float(v)}) for r, v in enumerate([1, 1, 1, 1])
        ] + [_trial("b", r, scores={"s": float(v)}) for r, v in enumerate([0, 0, 0, 0])]
        trial_level = Mean("s", unit="trial").compute(trials)
        naive_se = math.sqrt(0.25 * 8 / 7 / 8)
        assert trial_level.n == 8
        assert trial_level.stderr is not None
        assert trial_level.stderr > naive_se


class TestPassRate:
    def test_bool_scores_use_wilson(self) -> None:
        trials = [_trial(str(i), scores={"ok": i < 8}) for i in range(10)]
        result = PassRate("ok").compute(trials)
        assert result.value == pytest.approx(0.8)
        assert result.ci_low == pytest.approx(0.4902, abs=1e-3)

    def test_threshold(self) -> None:
        trials = [_trial("a", scores={"s": 0.9}), _trial("b", scores={"s": 0.4})]
        result = PassRate("s", threshold=0.5).compute(trials)
        assert result.name == "pass_rate(s>=0.5)"
        assert result.value == pytest.approx(0.5)

    def test_errors_count_as_failures(self) -> None:
        trials = [_trial("a", scores={"ok": True}), _trial("b", error=True)]
        assert PassRate("ok").compute(trials).value == pytest.approx(0.5)
        excluded = PassRate("ok", errors_as_failures=False).compute(trials)
        assert excluded.value == pytest.approx(1.0)
        assert excluded.n_missing == 1

    def test_all_pass_reducer(self) -> None:
        trials = [
            _trial("a", 0, scores={"ok": True}),
            _trial("a", 1, scores={"ok": False}),
            _trial("b", 0, scores={"ok": True}),
            _trial("b", 1, scores={"ok": True}),
        ]
        assert PassRate("ok", reduce=all_pass).compute(trials).value == pytest.approx(
            0.5
        )


class TestPassAtK:
    def _trials(self) -> list[Trial]:
        # Example a: 1 of 4 pass; example b: 4 of 4 pass.
        return [_trial("a", r, scores={"ok": r == 0}) for r in range(4)] + [
            _trial("b", r, scores={"ok": True}) for r in range(4)
        ]

    def test_pass_at_k_unbiased(self) -> None:
        result = PassAtK("ok", 2).compute(self._trials())
        # a: 1 - C(3,2)/C(4,2) = 0.5; b: 1
        assert result.value == pytest.approx(0.75)

    def test_pass_hat_k(self) -> None:
        result = PassHatK("ok", 2).compute(self._trials())
        # a: C(1,2)/C(4,2) = 0; b: 1
        assert result.value == pytest.approx(0.5)
        assert result.name == "pass^2(ok)"

    def test_too_few_repetitions_is_not_applicable(self) -> None:
        result = PassAtK("ok", 5).compute(self._trials())
        assert result.value is None
        assert result.n_na == 2
        assert result.n_missing == 0

    def test_errored_repetitions_count_as_failures(self) -> None:
        trials = [_trial("a", r, scores={"ok": True}) for r in range(2)]
        trials.append(_trial("a", 2, error=True))
        result = PassHatK("ok", 3).compute(trials)
        assert result.value == pytest.approx(0.0)
        assert result.n_missing == 0
        excluded = PassHatK("ok", 3, errors_as_failures=False).compute(trials)
        assert excluded.n_missing == 1
        assert excluded.n_na == 0


class TestOtherMetrics:
    def test_percentile_trial_level(self) -> None:
        trials = [_trial(str(i), duration=float(i)) for i in range(1, 101)]
        result = Percentile(Measure.DURATION, 95).compute(trials)
        assert result.value == pytest.approx(95.05)
        assert result.ci_low is not None

    def test_error_rate(self) -> None:
        trials = [_trial("a"), _trial("b", error=True), _trial("c"), _trial("d")]
        result = ErrorRate().compute(trials)
        assert result.value == pytest.approx(0.25)
        assert result.details["by_type"] == {"ValueError": 1}

    def test_total(self) -> None:
        trials = [_trial("a", cost=0.5), _trial("b", cost=0.25), _trial("c")]
        result = Total(Measure.COST).compute(trials)
        assert result.value == pytest.approx(0.75)
        assert result.n_missing == 1

    def test_distribution_and_proportion(self) -> None:
        trials = [
            _trial("a", scores={"verdict": "good"}),
            _trial("b", scores={"verdict": "bad"}),
            _trial("c", scores={"verdict": "good"}),
        ]
        dist = Distribution("verdict").compute(trials)
        assert dist.details["counts"] == {"good": 2, "bad": 1}
        share = Proportion("verdict", "good").compute(trials)
        assert share.value == pytest.approx(2 / 3)


class TestComputeMetrics:
    def test_group_by_and_cluster_by(self) -> None:
        examples = [
            Example[str, None](
                id="a", input="x", metadata={"topic": "math", "course": "c1"}
            ),
            Example[str, None](
                id="b", input="y", metadata={"topic": "math", "course": "c1"}
            ),
            Example[str, None](
                id="c", input="z", metadata={"topic": "geo", "course": "c2"}
            ),
        ]
        trials = [
            _trial("a", scores={"ok": True}),
            _trial("b", scores={"ok": False}),
            _trial("c", scores={"ok": True}),
        ]
        [result] = compute_metrics(
            trials,
            [PassRate("ok")],
            examples=examples,
            group_by=["topic"],
            cluster_by="course",
        )
        assert result.value == pytest.approx(2 / 3)
        assert set(result.groups) == {"topic=geo", "topic=math"}
        assert result.groups["topic=math"].value == pytest.approx(0.5)

    def test_defaults_follow_score_types(self) -> None:
        trials = [
            _trial("a", scores={"ok": True, "quality": 0.7, "label": "x"}, cost=0.1),
        ]
        names = [m.name for m in default_metrics(trials)]
        assert names == [
            "pass_rate(ok)",
            "mean(quality)",
            "dist(label)",
            "error_rate",
            "p50(duration_s)",
            "p95(duration_s)",
            "total(cost_usd)",
        ]


class TestPassRules:
    def test_pass_hat_k_counts_partial_passes(self) -> None:
        trials = [_trial("a", r, scores={"ok": r != 3}) for r in range(4)]
        # 3 of 4 pass: C(3,2)/C(4,2) = 0.5
        assert PassHatK("ok", 2).compute(trials).value == pytest.approx(0.5)

    def test_numeric_scores_need_a_threshold(self) -> None:
        trials = [_trial("a", scores={"s": 3.0}), _trial("b", scores={"s": 5.0})]
        result = PassRate("s").compute(trials)
        assert result.value is None
        assert "threshold" in result.details["error"]
        assert PassHatK("s", 1).compute(trials).value is None
        assert PassRate("s", threshold=4.0).compute(trials).value == pytest.approx(0.5)

    def test_rates_and_k_metrics_agree_on_passes(self) -> None:
        trials = [
            _trial(str(i), scores={"q": q}) for i, q in enumerate([0.95, 0.4, 0.9])
        ]
        rate = PassRate("q", threshold=0.9).compute(trials).value
        at_1 = PassAtK("q", 1, threshold=0.9).compute(trials).value
        assert rate == pytest.approx(at_1)  # type: ignore[arg-type]

    def test_measure_names_are_measures(self) -> None:
        trials = [_trial(str(i), duration=float(i)) for i in range(1, 5)]
        assert Mean("duration_s").compute(trials).value == pytest.approx(2.5)


class TestIntervals:
    def test_percentile_needs_enough_units_beyond_the_quantile(self) -> None:
        few = [_trial(str(i), duration=float(i)) for i in range(20)]
        result = Percentile(Measure.DURATION, 95).compute(few)
        assert result.value is not None
        assert result.ci_low is None
        assert "too few" in result.details["note"]
        many = [_trial(str(i), duration=float(i)) for i in range(200)]
        assert Percentile(Measure.DURATION, 95).compute(many).ci_low is not None

    def test_error_rate_clusters_repetitions_by_example(self) -> None:
        trials = [
            _trial(f"e{e}", r, error=e == 0) for e in range(10) for r in range(10)
        ]
        clustered = ErrorRate().compute(trials)
        independent = ErrorRate().compute(
            [_trial(f"t{i}", error=i < 10) for i in range(100)]
        )
        assert clustered.value == pytest.approx(independent.value)  # type: ignore[arg-type]
        width = clustered.ci_high - clustered.ci_low  # type: ignore[operator]
        naive = independent.ci_high - independent.ci_low  # type: ignore[operator]
        assert width > naive

    def test_one_cluster_per_group_gives_no_fake_certainty(self) -> None:
        examples = [
            Example(id=str(i), input=i, metadata={"course": "x" if i < 4 else "y"})
            for i in range(8)
        ]
        trials = [_trial(str(i), scores={"ok": i % 2 == 0}) for i in range(8)]
        (result,) = compute_metrics(
            trials,
            [PassRate("ok")],
            examples=examples,
            group_by=["course"],
            cluster_by="course",
        )
        group = result.groups["course=x"]
        assert group.value == pytest.approx(0.5)
        assert group.ci_low is not None
        assert group.ci_high - group.ci_low > 0.5  # type: ignore[operator]


class TestAccounting:
    def test_not_applicable_is_not_missing(self) -> None:
        trials = [
            _trial("a", scores={"ok": True}),
            _trial("b"),  # the evaluator returned nothing for b
        ]
        result = PassRate("ok").compute(trials)
        assert result.n == 1
        assert result.n_na == 1
        assert result.n_missing == 0

    def test_trials_that_never_ran_are_missing(self) -> None:
        trials = [_trial("a", scores={"ok": True})]
        (rate, errors) = compute_metrics(
            trials,
            [PassRate("ok"), ErrorRate()],
            expected=[("a", 0), ("b", 0), ("b", 1)],
        )
        assert rate.n_missing == 1  # example b
        assert errors.n_missing == 2  # two trials of b

    def test_partly_failed_examples_are_reported(self) -> None:
        trials = [
            _trial("a", 0, scores={"ok": True}),
            _trial("a", 1, scores={"ok": None}),
        ]
        result = PassRate("ok").compute(trials)
        assert result.details["partial_examples"] == 1


def test_repetitions_count_towards_share_intervals() -> None:
    # 20 examples that behave alike, 3 repetitions each: the interval rests on
    # 60 outcomes, not 20.
    values = [2 / 3] * 10 + [1.0] * 10
    by_examples = bounded_mean_estimate(values)
    by_trials = bounded_mean_estimate(values, observations=60)
    assert by_trials.ci_low is not None
    assert by_trials.ci_high is not None
    assert by_examples.ci_low is not None
    assert by_examples.ci_high is not None
    assert (
        by_trials.ci_high - by_trials.ci_low < by_examples.ci_high - by_examples.ci_low
    )
    # When every unit passes, each example still counts once.
    saturated = bounded_mean_estimate([1.0] * 20, observations=60)
    assert saturated.ci_low == pytest.approx(bounded_mean_estimate([1.0] * 20).ci_low)
