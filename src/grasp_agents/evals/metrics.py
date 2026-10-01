"""
Metrics: aggregates over the trials of a run.

A metric reads one *target* per trial — a score name, a custom measurement
recorded by the task, or a built-in :class:`Measure` — and aggregates it.
Example-level metrics (the default) first reduce each example's repetitions
to one value, so the statistical unit is the example and repetitions never
inflate ``n``. Trial-level metrics (latency percentiles, error rates) use
every trial, with uncertainty clustered on the example.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from statistics import median as _median
from typing import Any, Literal, override

from .stats import (
    Estimate,
    bootstrap_estimate,
    bounded_mean_estimate,
    cohens_kappa,
    mean_estimate,
    percentile,
    proportion_estimate,
)
from .types import Example, MetricResult, Trial

type Unit = Literal["example", "trial"]

# A percentile's interval needs this many units beyond the quantile on each
# side; below it the bootstrap cannot reach past the observed extremes.
_MIN_TAIL_UNITS = 5


class Measure(StrEnum):
    """Built-in per-trial measurements, usable as a metric target."""

    DURATION = "duration_s"
    COST = "cost_usd"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    TOTAL_TOKENS = "total_tokens"
    ERROR = "error"


type Target = str | Measure

_MEASURES = {m.value: m for m in Measure}


def as_target(of: Target) -> Target:
    """``of`` with built-in measure names (``"duration_s"``) as :class:`Measure`."""
    if isinstance(of, Measure):
        return of
    return _MEASURES.get(of, of)


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
    value = trial.measurements.get(of)
    return value if value is None or math.isfinite(value) else None


class ThresholdRequiredError(ValueError):
    pass


def pass_value(value: float, threshold: float | None) -> float:
    """
    1.0 / 0.0 for one value: at or above ``threshold``, or for pass/fail
    values (bools read as 1/0) themselves. Any other value needs a threshold.
    """
    if threshold is not None:
        return 1.0 if value >= threshold else 0.0
    if value in {0.0, 1.0}:
        return value
    raise ThresholdRequiredError(value)


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


@dataclass
class _Collected:
    values: list[float]
    # Example id of each value (the cluster key for trial-level values).
    keys: list[str]
    missing: int = 0
    na: int = 0
    # Examples with a value whose repetitions were only partly usable.
    partial: int = 0
    # Trial values behind ``values``.
    observations: int = 0


def _collect(
    trials: Sequence[Trial],
    value_of: Callable[[Trial], float | None],
    *,
    unit: Unit,
    reduce: Reducer,
    target: Target | None = None,
) -> _Collected:
    collected = _Collected(values=[], keys=[])
    if unit == "trial":
        for trial in trials:
            value = value_of(trial)
            if value is not None:
                collected.values.append(value)
                collected.keys.append(trial.example_id)
                collected.observations += 1
            elif _not_applicable(trial, target):
                collected.na += 1
            else:
                collected.missing += 1
        return collected
    for example_id, group in _group_by_example(trials).items():
        per_trial = [(t, value_of(t)) for t in group]
        found = [v for _, v in per_trial if v is not None]
        if found:
            collected.values.append(reduce(found))
            collected.keys.append(example_id)
            collected.observations += len(found)
            unusable = [t for t, v in per_trial if v is None]
            if unusable and not all(_not_applicable(t, target) for t in unusable):
                collected.partial += 1
        elif all(_not_applicable(t, target) for t in group):
            collected.na += 1
        else:
            collected.missing += 1
    return collected


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


def _share_estimate(
    values: Sequence[float],
    keys: Sequence[str],
    clusters: Mapping[str, Hashable] | None,
    *,
    unit: Unit,
    confidence: float,
    observations: int | None = None,
) -> Estimate:
    # Shares of independent pass/fail units get the exact Wilson interval;
    # fractional or clustered ones the effective-sample-size Wilson.
    cluster_keys = _clusters_for(keys, clusters, unit=unit)
    if cluster_keys is None and all(v in {0.0, 1.0} for v in values):
        passes = sum(1 for v in values if v > 0.5)
        return proportion_estimate(passes, len(values), confidence=confidence)
    return bounded_mean_estimate(
        values,
        clusters=cluster_keys,
        observations=observations,
        confidence=confidence,
    )


def _result(
    name: str,
    estimate: Estimate,
    *,
    missing: int,
    na: int = 0,
    confidence: float = 0.95,
    **details: Any,
) -> MetricResult:
    value = None if math.isnan(estimate.value) else estimate.value
    return MetricResult(
        name=name,
        value=value,
        n=estimate.n,
        n_missing=missing,
        n_na=na,
        stderr=estimate.stderr,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        confidence=confidence,
        details={k: v for k, v in details.items() if v},
    )


def _threshold_error(name: str, of: Target) -> MetricResult:
    return MetricResult(
        name=name,
        value=None,
        n=0,
        details={
            "error": (
                f"{of} has values other than pass/fail; set threshold= to "
                "say what counts as a pass"
            )
        },
    )


class Metric(ABC):
    name: str
    # Whether ``n`` counts examples (repetitions reduced first) or trials.
    unit: Unit = "example"

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
        self.of = as_target(of)
        self.reduce = reduce
        self.unit = unit
        self.confidence = confidence
        self.name = name or f"mean({self.of}){_reducer_suffix(reduce)}"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        c = _collect(
            trials,
            lambda t: trial_value(t, self.of),
            unit=self.unit,
            reduce=self.reduce,
            target=self.of,
        )
        estimate = mean_estimate(
            c.values,
            clusters=_clusters_for(c.keys, clusters, unit=self.unit),
            confidence=self.confidence,
        )
        return _result(
            self.name,
            estimate,
            missing=c.missing,
            na=c.na,
            confidence=self.confidence,
            partial_examples=c.partial,
        )


class PassRate(Metric):
    """
    Share of passing examples. Bool scores pass when ``True``; numeric scores
    pass at or above ``threshold`` (required for them). Repetitions are
    reduced per example (``reduce``, the mean by default).

    Trials whose task failed have no score; they count as failures unless
    ``errors_as_failures=False`` excludes them (then a system that crashes on
    hard examples would look better). The error rate is reported separately
    either way.
    """

    def __init__(
        self,
        of: Target,
        *,
        threshold: float | None = None,
        errors_as_failures: bool = True,
        reduce: Reducer = mean,
        name: str | None = None,
        confidence: float = 0.95,
    ) -> None:
        self.of = as_target(of)
        self.threshold = threshold
        self.errors_as_failures = errors_as_failures
        self.reduce = reduce
        self.confidence = confidence
        bar = "" if threshold is None else f">={threshold:g}"
        self.name = name or f"pass_rate({self.of}{bar}){_reducer_suffix(reduce)}"

    def _passed(self, trial: Trial) -> float | None:
        value = trial_value(trial, self.of)
        if value is None:
            return 0.0 if not trial.ok and self.errors_as_failures else None
        return pass_value(value, self.threshold)

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        try:
            c = _collect(
                trials, self._passed, unit="example", reduce=self.reduce, target=self.of
            )
        except ThresholdRequiredError:
            return _threshold_error(self.name, self.of)
        estimate = _share_estimate(
            c.values,
            c.keys,
            clusters,
            unit="example",
            confidence=self.confidence,
            # A mean over repetitions rests on every repetition's outcome.
            observations=c.observations if self.reduce is mean else None,
        )
        return _result(
            self.name,
            estimate,
            missing=c.missing,
            na=c.na,
            confidence=self.confidence,
            partial_examples=c.partial,
        )


class _KMetric(Metric):
    def __init__(
        self,
        of: Target,
        k: int,
        *,
        threshold: float | None,
        errors_as_failures: bool,
        name: str,
        confidence: float,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.of = as_target(of)
        self.k = k
        self.threshold = threshold
        self.errors_as_failures = errors_as_failures
        self.confidence = confidence
        self.name = name

    def _counts(self, group: Sequence[Trial]) -> tuple[int, int]:
        n = c = 0
        for trial in group:
            value = trial_value(trial, self.of)
            if value is None:
                if not trial.ok and self.errors_as_failures:
                    n += 1
                continue
            n += 1
            c += int(pass_value(value, self.threshold) > 0.5)
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
        observations = 0
        missing = na = 0
        for example_id, group in _group_by_example(trials).items():
            try:
                n, c = self._counts(group)
            except ThresholdRequiredError:
                return _threshold_error(self.name, self.of)
            if n < self.k:
                # Too few repetitions for k is a configuration matter, not
                # missing data; too few *usable* ones (failed scoring) is missing.
                if len(group) < self.k or all(
                    _not_applicable(t, self.of) for t in group
                ):
                    na += 1
                else:
                    missing += 1
                continue
            values.append(self._per_example(n, c))
            keys.append(example_id)
            observations += n
        estimate = bounded_mean_estimate(
            values,
            clusters=_clusters_for(keys, clusters, unit="example"),
            observations=observations,
            confidence=self.confidence,
        )
        return _result(
            self.name,
            estimate,
            missing=missing,
            na=na,
            confidence=self.confidence,
            k=self.k,
        )


class PassAtK(_KMetric):
    """
    Unbiased pass@k: chance that at least one of k repetitions passes. Failed
    repetitions count as failures unless ``errors_as_failures=False``.
    """

    def __init__(
        self,
        of: Target,
        k: int,
        *,
        threshold: float | None = None,
        errors_as_failures: bool = True,
        name: str | None = None,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(
            of,
            k,
            threshold=threshold,
            errors_as_failures=errors_as_failures,
            name=name or f"pass@{k}({as_target(of)})",
            confidence=confidence,
        )

    def _per_example(self, n: int, c: int) -> float:
        return 1.0 - math.comb(n - c, self.k) / math.comb(n, self.k)


class PassHatK(_KMetric):
    """
    Unbiased pass^k: chance that all of k repetitions pass — the reliability
    a user experiences when the same request must work every time. Failed
    repetitions count as failures unless ``errors_as_failures=False``.
    """

    def __init__(
        self,
        of: Target,
        k: int,
        *,
        threshold: float | None = None,
        errors_as_failures: bool = True,
        name: str | None = None,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(
            of,
            k,
            threshold=threshold,
            errors_as_failures=errors_as_failures,
            name=name or f"pass^{k}({as_target(of)})",
            confidence=confidence,
        )

    def _per_example(self, n: int, c: int) -> float:
        return math.comb(c, self.k) / math.comb(n, self.k)


class Percentile(Metric):
    """
    A percentile (0-100) with a bootstrap interval that resamples examples
    (keeping their repetitions together); trial-level by default. With too
    few units beyond the quantile to bound it, no interval is reported.
    """

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
        if not 0.0 <= q <= 100.0:
            raise ValueError("q must be in [0, 100]")
        if n_resamples < 2:
            raise ValueError("n_resamples must be >= 2")
        self.of = as_target(of)
        self.q = q
        self.unit = unit
        self.reduce = reduce
        self.n_resamples = n_resamples
        self.seed = seed
        self.confidence = confidence
        self.name = name or f"p{q:g}({self.of})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        c = _collect(
            trials,
            lambda t: trial_value(t, self.of),
            unit=self.unit,
            reduce=self.reduce,
            target=self.of,
        )
        cluster_keys = _clusters_for(c.keys, clusters, unit=self.unit)
        units = len(set(cluster_keys)) if cluster_keys is not None else len(c.values)
        share = self.q / 100.0
        tail = units * min(share, 1.0 - share)
        note: str | None = None
        if c.values and tail < _MIN_TAIL_UNITS:
            estimate = Estimate(
                value=percentile(c.values, self.q), n=len(c.values), units=units
            )
            note = f"too few units ({units}) for a p{self.q:g} interval"
        else:
            estimate = bootstrap_estimate(
                c.values,
                lambda v: percentile(v, self.q),
                clusters=cluster_keys,
                n_resamples=self.n_resamples,
                seed=self.seed,
                confidence=self.confidence,
            )
        return _result(
            self.name,
            estimate,
            missing=c.missing,
            na=c.na,
            confidence=self.confidence,
            note=note,
        )


class ErrorRate(Metric):
    """Share of trials whose task raised or timed out (clustered on the example)."""

    unit: Unit = "trial"

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
        values = [0.0 if t.ok else 1.0 for t in trials]
        keys = [t.example_id for t in trials]
        estimate = _share_estimate(
            values, keys, clusters, unit="trial", confidence=self.confidence
        )
        types: dict[str, int] = {}
        for trial in trials:
            if trial.error is not None:
                types[trial.error.type] = types.get(trial.error.type, 0) + 1
        return _result(
            self.name,
            estimate,
            missing=0,
            confidence=self.confidence,
            by_type=types,
        )


class Total(Metric):
    unit: Unit = "trial"

    def __init__(self, of: Target, *, name: str | None = None) -> None:
        self.of = as_target(of)
        self.name = name or f"total({self.of})"

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


def label_text(value: bool | float | str) -> str:
    """A score value as a label: ``true``/``false``, the label, or the number."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return f"{value:g}"


def _label(trial: Trial, of: str) -> str | None:
    score = trial.score(of)
    if score is None or score.value is None:
        return None
    return label_text(score.value)


class Distribution(Metric):
    """Counts and shares of a categorical score's labels across trials."""

    unit: Unit = "trial"

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
    """Share of scored examples carrying one label (per-example mean, Wilson CI)."""

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

        c = _collect(trials, indicator, unit="example", reduce=mean, target=self.of)
        estimate = _share_estimate(
            c.values,
            c.keys,
            clusters,
            unit="example",
            confidence=self.confidence,
            observations=c.observations,
        )
        return _result(
            self.name, estimate, missing=c.missing, na=c.na, confidence=self.confidence
        )


class Consistency(Metric):
    """
    Share of examples whose repetitions all give the same value of ``of``
    (a judge's or a task's self-consistency), with the mean agreement of
    repetition pairs in ``details``. Examples with fewer than two scored
    repetitions do not count.
    """

    def __init__(
        self, of: str, *, name: str | None = None, confidence: float = 0.95
    ) -> None:
        self.of = of
        self.confidence = confidence
        self.name = name or f"consistency({of})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        values: list[float] = []
        keys: list[str] = []
        pair_agreement: list[float] = []
        missing = na = 0
        for example_id, group in _group_by_example(trials).items():
            labels = [label for t in group if (label := _label(t, self.of)) is not None]
            if len(labels) < 2:
                if len(group) < 2 or all(_not_applicable(t, self.of) for t in group):
                    na += 1
                else:
                    missing += 1
                continue
            values.append(1.0 if len(set(labels)) == 1 else 0.0)
            keys.append(example_id)
            pairs = list(combinations(labels, 2))
            pair_agreement.append(sum(a == b for a, b in pairs) / len(pairs))
        estimate = _share_estimate(
            values, keys, clusters, unit="example", confidence=self.confidence
        )
        result = _result(
            self.name, estimate, missing=missing, na=na, confidence=self.confidence
        )
        if pair_agreement:
            result.details["pair_agreement"] = math.fsum(pair_agreement) / len(
                pair_agreement
            )
        return result


# --- Agreement between two labels (judge validation) ---


def _label_pairs(
    trials: Sequence[Trial], of: str, truth: str
) -> tuple[list[tuple[str, str]], list[str], int, int]:
    """``(truth, judged)`` label pairs, their example ids, missing and n/a."""
    pairs: list[tuple[str, str]] = []
    keys: list[str] = []
    missing = na = 0
    for trial in trials:
        expected, judged = _label(trial, truth), _label(trial, of)
        if expected is not None and judged is not None:
            pairs.append((expected, judged))
            keys.append(trial.example_id)
        elif _not_applicable(trial, truth) or _not_applicable(trial, of):
            na += 1
        else:
            missing += 1
    return pairs, keys, missing, na


class CohenKappa(Metric):
    """
    Cohen's κ between two labels of each trial — a judge's verdict ``of``
    and the ``truth`` it is validated against (e.g. a human label). κ is
    agreement beyond what the two label distributions produce by chance: 0 is
    chance, 1 perfect. Its interval is the observed agreement's (Wilson, on
    examples, clustered like other example-level metrics) mapped through the
    chance agreement, so it stays wide on a few labels even when they all
    agree.
    """

    def __init__(
        self,
        of: str,
        truth: str,
        *,
        name: str | None = None,
        confidence: float = 0.95,
    ) -> None:
        self.of = of
        self.truth = truth
        self.confidence = confidence
        self.name = name or f"kappa({of}, {truth})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        pairs, keys, missing, na = _label_pairs(trials, self.of, self.truth)
        per_example: dict[str, list[float]] = {}
        for (expected, judged), key in zip(pairs, keys, strict=True):
            per_example.setdefault(key, []).append(float(expected == judged))
        examples = list(per_example)
        agreement = _share_estimate(
            [mean(v) for v in per_example.values()],
            examples,
            clusters,
            unit="example",
            confidence=self.confidence,
            observations=len(pairs),
        )
        kappa = cohens_kappa(pairs)
        result = MetricResult(
            name=self.name,
            value=kappa,
            n=len(examples),
            n_missing=missing,
            n_na=na,
            confidence=self.confidence,
            details={"judgments": len(pairs)},
        )
        if kappa is None or not pairs:
            return result
        chance = _chance_agreement(pairs)
        scale = 1.0 - chance
        result.details["observed_agreement"] = sum(1 for a, b in pairs if a == b) / len(
            pairs
        )
        result.details["chance_agreement"] = chance
        if agreement.ci_low is not None and agreement.ci_high is not None:
            result.ci_low = max(-1.0, (agreement.ci_low - chance) / scale)
            result.ci_high = min(1.0, (agreement.ci_high - chance) / scale)
        if agreement.stderr is not None:
            result.stderr = agreement.stderr / scale
        return result


def _chance_agreement(pairs: Sequence[tuple[str, str]]) -> float:
    n = len(pairs)
    left: dict[str, int] = {}
    right: dict[str, int] = {}
    for a, b in pairs:
        left[a] = left.get(a, 0) + 1
        right[b] = right.get(b, 0) + 1
    return sum(left[k] * right.get(k, 0) for k in left) / (n * n)


class ClassRecall(Metric):
    """
    Among examples whose ``truth`` is ``label``, the share of trials where
    ``of`` is ``label`` too: a judge's true positive rate (``label`` the
    passing label) or true negative rate (the failing one). Each example's
    repetitions are averaged first.
    """

    def __init__(
        self,
        of: str,
        truth: str,
        label: str | bool,
        *,
        name: str | None = None,
        confidence: float = 0.95,
    ) -> None:
        self.of = of
        self.truth = truth
        self.label = label_text(label)
        self.confidence = confidence
        self.name = name or f"recall({of}={self.label} | {truth})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        in_class = [t for t in trials if _label(t, self.truth) == self.label]

        def hit(trial: Trial) -> float | None:
            judged = _label(trial, self.of)
            return None if judged is None else float(judged == self.label)

        c = _collect(in_class, hit, unit="example", reduce=mean, target=self.of)
        estimate = _share_estimate(
            c.values,
            c.keys,
            clusters,
            unit="example",
            confidence=self.confidence,
            observations=c.observations,
        )
        return _result(
            self.name,
            estimate,
            missing=c.missing,
            na=c.na,
            confidence=self.confidence,
            label=self.label,
        )


class ConfusionMatrix(Metric):
    """Counts of ``truth`` → ``of`` label pairs across trials (no single value)."""

    unit: Unit = "trial"

    def __init__(self, of: str, truth: str, *, name: str | None = None) -> None:
        self.of = of
        self.truth = truth
        self.name = name or f"confusion({of}, {truth})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        pairs, _, missing, na = _label_pairs(trials, self.of, self.truth)
        labels = sorted({label for pair in pairs for label in pair})
        counts = {expected: dict.fromkeys(labels, 0) for expected in labels}
        for expected, judged in pairs:
            counts[expected][judged] += 1
        return MetricResult(
            name=self.name,
            value=None,
            n=len(pairs),
            n_missing=missing,
            n_na=na,
            details={
                "labels": labels,
                "counts": counts,
                "truth": self.truth,
                "judged": self.of,
            },
        )


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


type MetricsSpec = (
    Sequence[Metric] | Callable[[Sequence[Trial]], Sequence[Metric]] | None
)
"""
Metrics for a run: a list, a function of the finished trials returning one
(e.g. ``lambda trials: [*default_metrics(trials), extra]``), or ``None`` for
:func:`default_metrics`.
"""


def compute_metrics(
    trials: Sequence[Trial],
    metrics: MetricsSpec,
    *,
    examples: Iterable[Example[Any, Any]] = (),
    group_by: Sequence[str] = (),
    cluster_by: str | None = None,
    expected: Sequence[tuple[str, int]] | None = None,
) -> list[MetricResult]:
    """
    Compute ``metrics`` (see :data:`MetricsSpec`) over ``trials``.
    ``expected`` lists the trials that should exist: those that never ran
    (e.g. the budget ran out) count as missing.
    """
    if metrics is None:
        selected = default_metrics(trials)
    elif callable(metrics):
        selected = list(metrics(trials))
    else:
        selected = list(metrics)
    by_id = {e.id: e for e in examples}
    clusters: dict[str, Hashable] | None = None
    if cluster_by is not None:
        clusters = {
            example_id: _hashable(example.metadata.get(cluster_by, example_id))
            for example_id, example in by_id.items()
        }
    present = {t.key for t in trials}
    present_examples = {key[0] for key in present}
    never_ran = [key for key in expected or () if key not in present]
    never_ran_examples = {key[0] for key in never_ran} - present_examples
    results: list[MetricResult] = []
    for metric in selected:
        result = metric.compute(trials, clusters=clusters)
        result.n_missing += (
            len(never_ran) if metric.unit == "trial" else len(never_ran_examples)
        )
        for key in group_by:
            buckets: dict[str, list[Trial]] = {}
            for trial in trials:
                example = by_id.get(trial.example_id)
                value = None if example is None else _group_value(example, key)
                buckets.setdefault(f"{key}={value}", []).append(trial)
            for label, bucket in sorted(buckets.items()):
                result.groups[label] = metric.compute(bucket, clusters=clusters)
        results.append(result)
    return results


def _group_value(example: Example[Any, Any], key: str) -> Any:
    # ``split`` groups by the example's splits unless its metadata has one.
    if key == "split" and key not in example.metadata:
        return "+".join(example.splits) or None
    return example.metadata.get(key)


def _hashable(value: Any) -> Hashable:
    if isinstance(value, Hashable):
        return value
    return repr(value)
