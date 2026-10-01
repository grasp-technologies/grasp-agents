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

    def test_errors_as_failures(self) -> None:
        trials = [_trial("a", scores={"ok": True}), _trial("b", error=True)]
        assert PassRate("ok").compute(trials).value == pytest.approx(1.0)
        assert PassRate("ok", errors_as_failures=True).compute(
            trials
        ).value == pytest.approx(0.5)

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

    def test_errored_repetitions_are_missing(self) -> None:
        trials = [_trial("a", r, scores={"ok": True}) for r in range(2)]
        trials.append(_trial("a", 2, error=True))
        result = PassHatK("ok", 3).compute(trials)
        assert result.n_missing == 1
        assert result.n_na == 0


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
