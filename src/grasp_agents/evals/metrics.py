"""
Metrics: aggregates over the trials of a run.

A metric reads one *target* per trial — a score name, a custom measurement
recorded by the task, or a built-in :class:`Measure` — and aggregates it.
Example-level metrics (the default) first reduce each example's repetitions
to one value, so the statistical unit is the example and repetitions never
inflate ``n``. Trial-level metrics (latency percentiles, error rates) use
every trial.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from enum import StrEnum
from statistics import median as _median
from typing import Any, Literal, override

from .stats import (
    Estimate,
    bootstrap_estimate,
    mean_estimate,
    percentile,
    proportion_estimate,
)
from .types import Example, MetricResult, Trial

type Unit = Literal["example", "trial"]


class Measure(StrEnum):
    """Built-in per-trial measurements, usable as a metric target."""

    DURATION = "duration_s"
    COST = "cost_usd"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    TOTAL_TOKENS = "total_tokens"
    ERROR = "error"


type Target = str | Measure


def trial_value(trial: Trial, of: Target) -> float | None:
    """
    Numeric value of ``of`` for one trial, or ``None`` when it has none.

    A plain string is a score name (bools count as 1/0; labels have no numeric
    value), falling back to a measurement the task recorded under that name.
    """
    if isinstance(of, Measure):
        match of:
            case Measure.DURATION:
                return trial.duration_s
            case Measure.COST:
                return trial.total_usage.cost_usd
            case Measure.INPUT_TOKENS:
                return float(trial.usage.input_tokens)
            case Measure.OUTPUT_TOKENS:
                return float(trial.usage.output_tokens)
            case Measure.TOTAL_TOKENS:
                return float(trial.usage.total_tokens)
            case Measure.ERROR:
                return 0.0 if trial.ok else 1.0
    score = trial.score(of)
    if score is not None:
        return score.as_float()
    return trial.measurements.get(of)


# --- Reducers (per-example aggregation of repetitions) ---

type Reducer = Callable[[Sequence[float]], float]


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def median(values: Sequence[float]) -> float:
    return float(_median(values))


def max_(values: Sequence[float]) -> float:
    return max(values)


def min_(values: Sequence[float]) -> float:
    return min(values)


def all_pass(values: Sequence[float]) -> float:
    """1 when every repetition passed (pass^k over the observed repetitions)."""
    return 1.0 if all(v >= 1.0 for v in values) else 0.0


def any_pass(values: Sequence[float]) -> float:
    """1 when at least one repetition passed (empirical pass@k)."""
    return 1.0 if any(v >= 1.0 for v in values) else 0.0


def at_least(k: int) -> Reducer:
    def reduce(values: Sequence[float]) -> float:
        return 1.0 if sum(1 for v in values if v >= 1.0) >= k else 0.0

    reduce.__name__ = f"at_least_{k}"
    return reduce


max_.__name__ = "max"
min_.__name__ = "min"


def _reducer_suffix(reduce: Reducer) -> str:
    name = getattr(reduce, "__name__", "custom")
    return "" if reduce is mean else f"[{name}]"


def _group_by_example(trials: Iterable[Trial]) -> dict[str, list[Trial]]:
    groups: dict[str, list[Trial]] = {}
    for trial in trials:
        groups.setdefault(trial.example_id, []).append(trial)
    return groups


def _not_applicable(trial: Trial, target: Target | None) -> bool:
    if target is None or isinstance(target, Measure) or not trial.ok:
        return False
    if trial.evaluator_failures or target in trial.measurements:
        return False
    return trial.score(target) is None


def _collect(
    trials: Sequence[Trial],
    value_of: Callable[[Trial], float | None],
    *,
    unit: Unit,
    reduce: Reducer,
    target: Target | None = None,
) -> tuple[list[float], list[str], int, int]:
    """``(values, example_ids, n_missing, n_na)`` at the requested unit."""
    values: list[float] = []
    keys: list[str] = []
    missing = na = 0
    if unit == "trial":
        for trial in trials:
            value = value_of(trial)
            if value is not None:
                values.append(value)
                keys.append(trial.example_id)
            elif _not_applicable(trial, target):
                na += 1
            else:
                missing += 1
        return values, keys, missing, na
    for example_id, group in _group_by_example(trials).items():
        found = [v for t in group if (v := value_of(t)) is not None]
        if found:
            values.append(reduce(found))
            keys.append(example_id)
        elif all(_not_applicable(t, target) for t in group):
            na += 1
        else:
            missing += 1
    return values, keys, missing, na


def _clusters_for(
    keys: Sequence[str],
    clusters: Mapping[str, Hashable] | None,
    *,
    unit: Unit,
) -> list[Hashable] | None:
    if clusters is not None:
        return [clusters.get(k, k) for k in keys]
    if unit == "trial" and len(set(keys)) < len(keys):
        # Repetitions of one example are correlated: cluster on the example.
        return list(keys)
    return None


def _result(
    name: str,
    estimate: Estimate,
    *,
    missing: int,
    na: int = 0,
    bounded: bool = False,
    **details: Any,
) -> MetricResult:
    value = None if math.isnan(estimate.value) else estimate.value
    low, high = estimate.ci_low, estimate.ci_high
    if bounded:
        # Shares live in [0, 1]; a small-sample t-interval can overshoot.
        low = None if low is None else max(0.0, low)
        high = None if high is None else min(1.0, high)
    return MetricResult(
        name=name,
        value=value,
        n=estimate.n,
        n_missing=missing,
        n_na=na,
        stderr=estimate.stderr,
        ci_low=low,
        ci_high=high,
        details={k: v for k, v in details.items() if v is not None},
    )


class Metric(ABC):
    name: str

    @abstractmethod
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        """
        Aggregate ``trials``. ``clusters`` maps example ids to a cluster key
        (from ``cluster_by``) for clustered standard errors.
        """


class Mean(Metric):
    def __init__(
        self,
        of: Target,
        *,
        reduce: Reducer = mean,
        unit: Unit = "example",
        name: str | None = None,
        confidence: float = 0.95,
    ) -> None:
        self.of = of
        self.reduce = reduce
        self.unit: Unit = unit
        self.confidence = confidence
        self.name = name or f"mean({of}){_reducer_suffix(reduce)}"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        values, keys, missing, na = _collect(
            trials,
            lambda t: trial_value(t, self.of),
            unit=self.unit,
            reduce=self.reduce,
            target=self.of,
        )
        estimate = mean_estimate(
            values,
            clusters=_clusters_for(keys, clusters, unit=self.unit),
            confidence=self.confidence,
        )
        return _result(self.name, estimate, missing=missing, na=na)


class PassRate(Metric):
    """
    Share of passing units. Bool scores pass when ``True``; numeric scores
    pass at or above ``threshold``. Trials whose task failed have no score and
    are excluded unless ``errors_as_failures`` counts them as failures.
    """

    def __init__(
        self,
        of: Target,
        *,
        threshold: float | None = None,
        errors_as_failures: bool = False,
        reduce: Reducer = mean,
        name: str | None = None,
        confidence: float = 0.95,
    ) -> None:
        self.of = of
        self.threshold = threshold
        self.errors_as_failures = errors_as_failures
        self.reduce = reduce
        self.confidence = confidence
        bar = "" if threshold is None else f">={threshold:g}"
        self.name = name or f"pass_rate({of}{bar}){_reducer_suffix(reduce)}"

    def _passed(self, trial: Trial) -> float | None:
        if not trial.ok and self.errors_as_failures:
            return 0.0
        value = trial_value(trial, self.of)
        if value is None:
            return None
        if self.threshold is not None:
            return 1.0 if value >= self.threshold else 0.0
        return value

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        values, keys, missing, na = _collect(
            trials, self._passed, unit="example", reduce=self.reduce, target=self.of
        )
        non_binary = any(v not in {0.0, 1.0} for v in values)
        if non_binary or clusters is not None:
            estimate = mean_estimate(
                values,
                clusters=_clusters_for(keys, clusters, unit="example"),
                confidence=self.confidence,
            )
        else:
            passes = sum(1 for v in values if v > 0.5)
            estimate = proportion_estimate(
                passes, len(values), confidence=self.confidence
            )
        return _result(self.name, estimate, missing=missing, na=na, bounded=True)


class _KMetric(Metric):
    def __init__(
        self,
        of: Target,
        k: int,
        *,
        threshold: float | None,
        name: str,
        confidence: float,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.of = of
        self.k = k
        self.threshold = threshold
        self.confidence = confidence
        self.name = name

    def _counts(self, group: Sequence[Trial]) -> tuple[int, int]:
        n = c = 0
        for trial in group:
            value = trial_value(trial, self.of)
            if value is None:
                continue
            n += 1
            passed = (
                value >= self.threshold if self.threshold is not None else value >= 1.0
            )
            c += int(passed)
        return n, c

    @abstractmethod
    def _per_example(self, n: int, c: int) -> float: ...

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        values: list[float] = []
        keys: list[str] = []
        missing = na = 0
        for example_id, group in _group_by_example(trials).items():
            n, c = self._counts(group)
            if n < self.k:
                # Too few repetitions for k is a configuration matter, not
                # missing data; too few *usable* ones (errors) is missing.
                if len(group) < self.k or all(
                    _not_applicable(t, self.of) for t in group
                ):
                    na += 1
                else:
                    missing += 1
                continue
            values.append(self._per_example(n, c))
            keys.append(example_id)
        estimate = mean_estimate(
            values,
            clusters=_clusters_for(keys, clusters, unit="example"),
            confidence=self.confidence,
        )
        return _result(
            self.name, estimate, missing=missing, na=na, bounded=True, k=self.k
        )


class PassAtK(_KMetric):
    """Unbiased pass@k: chance that at least one of k repetitions passes."""

    def __init__(
        self,
        of: Target,
        k: int,
        *,
        threshold: float | None = None,
        name: str | None = None,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(
            of,
            k,
            threshold=threshold,
            name=name or f"pass@{k}({of})",
            confidence=confidence,
        )

    def _per_example(self, n: int, c: int) -> float:
        return 1.0 - math.comb(n - c, self.k) / math.comb(n, self.k)


class PassHatK(_KMetric):
    """
    Unbiased pass^k: chance that all of k repetitions pass — the reliability
    a user experiences when the same request must work every time.
    """

    def __init__(
        self,
        of: Target,
        k: int,
        *,
        threshold: float | None = None,
        name: str | None = None,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(
            of,
            k,
            threshold=threshold,
            name=name or f"pass^{k}({of})",
            confidence=confidence,
        )

    def _per_example(self, n: int, c: int) -> float:
        return math.comb(c, self.k) / math.comb(n, self.k)


class Percentile(Metric):
    """A percentile (0-100) with a bootstrap interval; trial-level by default."""

    def __init__(
        self,
        of: Target,
        q: float,
        *,
        unit: Unit = "trial",
        reduce: Reducer = mean,
        name: str | None = None,
        n_resamples: int = 2000,
        seed: int = 0,
        confidence: float = 0.95,
    ) -> None:
        self.of = of
        self.q = q
        self.unit: Unit = unit
        self.reduce = reduce
        self.n_resamples = n_resamples
        self.seed = seed
        self.confidence = confidence
        self.name = name or f"p{q:g}({of})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        values, _, missing, na = _collect(
            trials,
            lambda t: trial_value(t, self.of),
            unit=self.unit,
            reduce=self.reduce,
            target=self.of,
        )
        estimate = bootstrap_estimate(
            values,
            lambda v: percentile(v, self.q),
            n_resamples=self.n_resamples,
            seed=self.seed,
            confidence=self.confidence,
        )
        return _result(self.name, estimate, missing=missing, na=na)


class ErrorRate(Metric):
    """Share of trials whose task raised or timed out (Wilson interval)."""

    def __init__(self, *, name: str = "error_rate", confidence: float = 0.95) -> None:
        self.name = name
        self.confidence = confidence

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        errors = sum(1 for t in trials if not t.ok)
        estimate = proportion_estimate(errors, len(trials), confidence=self.confidence)
        types: dict[str, int] = {}
        for trial in trials:
            if trial.error is not None:
                types[trial.error.type] = types.get(trial.error.type, 0) + 1
        return _result(self.name, estimate, missing=0, by_type=types or None)


class Total(Metric):
    def __init__(self, of: Target, *, name: str | None = None) -> None:
        self.of = of
        self.name = name or f"total({of})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        values = [v for t in trials if (v := trial_value(t, self.of)) is not None]
        return MetricResult(
            name=self.name,
            value=math.fsum(values) if values else None,
            n=len(values),
            n_missing=len(trials) - len(values),
        )


def _label(trial: Trial, of: str) -> str | None:
    score = trial.score(of)
    if score is None or score.value is None:
        return None
    if isinstance(score.value, bool):
        return "true" if score.value else "false"
    if isinstance(score.value, str):
        return score.value
    return f"{score.value:g}"


class Distribution(Metric):
    """Counts and shares of a categorical score's labels across trials."""

    def __init__(self, of: str, *, name: str | None = None) -> None:
        self.of = of
        self.name = name or f"dist({of})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        counts: dict[str, int] = {}
        missing = 0
        for trial in trials:
            label = _label(trial, self.of)
            if label is None:
                missing += 1
            else:
                counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        return MetricResult(
            name=self.name,
            value=None,
            n=total,
            n_missing=missing,
            details={
                "counts": ordered,
                "shares": {k: v / total for k, v in ordered.items()} if total else {},
            },
        )


class Proportion(Metric):
    """Share of scored units carrying one label (per-example mean, Wilson CI)."""

    def __init__(
        self,
        of: str,
        label: str,
        *,
        name: str | None = None,
        confidence: float = 0.95,
    ) -> None:
        self.of = of
        self.label = label
        self.confidence = confidence
        self.name = name or f"share({of}={label})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        def indicator(trial: Trial) -> float | None:
            label = _label(trial, self.of)
            return None if label is None else float(label == self.label)

        values, keys, missing, na = _collect(
            trials, indicator, unit="example", reduce=mean, target=self.of
        )
        if clusters is None and all(v in {0.0, 1.0} for v in values):
            estimate = proportion_estimate(
                sum(1 for v in values if v > 0.5),
                len(values),
                confidence=self.confidence,
            )
        else:
            estimate = mean_estimate(
                values,
                clusters=_clusters_for(keys, clusters, unit="example"),
                confidence=self.confidence,
            )
        return _result(self.name, estimate, missing=missing, na=na, bounded=True)


# --- Computing a run's metrics ---


def default_metrics(trials: Sequence[Trial]) -> list[Metric]:
    """A metric per score (pass rate, mean or label shares), plus errors and cost."""
    kinds: dict[str, set[type]] = {}
    for trial in trials:
        for score in trial.scores:
            if score.value is not None:
                kinds.setdefault(score.name, set()).add(type(score.value))
            else:
                kinds.setdefault(score.name, set())
    metrics: list[Metric] = []
    for name, value_types in kinds.items():
        if value_types == {bool}:
            metrics.append(PassRate(name))
        elif value_types and value_types <= {float, int, bool}:
            metrics.append(Mean(name))
        elif str in value_types:
            metrics.append(Distribution(name))
        else:
            metrics.append(Mean(name))
    metrics.extend(
        [
            ErrorRate(),
            Percentile(Measure.DURATION, 50),
            Percentile(Measure.DURATION, 95),
        ]
    )
    if any(t.total_usage.cost_usd is not None for t in trials):
        metrics.append(Total(Measure.COST))
    return metrics


def compute_metrics(
    trials: Sequence[Trial],
    metrics: Sequence[Metric] | None,
    *,
    examples: Iterable[Example[Any, Any]] = (),
    group_by: Sequence[str] = (),
    cluster_by: str | None = None,
) -> list[MetricResult]:
    selected = list(metrics) if metrics is not None else default_metrics(trials)
    by_id = {e.id: e for e in examples}
    clusters: dict[str, Hashable] | None = None
    if cluster_by is not None:
        clusters = {
            example_id: _hashable(example.metadata.get(cluster_by, example_id))
            for example_id, example in by_id.items()
        }
    results: list[MetricResult] = []
    for metric in selected:
        result = metric.compute(trials, clusters=clusters)
        for key in group_by:
            buckets: dict[str, list[Trial]] = {}
            for trial in trials:
                example = by_id.get(trial.example_id)
                value = None if example is None else example.metadata.get(key)
                buckets.setdefault(f"{key}={value}", []).append(trial)
            for label, bucket in sorted(buckets.items()):
                result.groups[label] = metric.compute(bucket, clusters=clusters)
        results.append(result)
    return results


def _hashable(value: Any) -> Hashable:
    if isinstance(value, Hashable):
        return value
    return repr(value)
