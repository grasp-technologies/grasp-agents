from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from .metrics import Measure, Reducer, Target, mean, trial_value
from .stats import paired_difference
from .types import EvaluationRun, RunStatus, Trial

_LOWER_IS_BETTER = frozenset(
    {
        Measure.DURATION,
        Measure.COST,
        Measure.INPUT_TOKENS,
        Measure.OUTPUT_TOKENS,
        Measure.TOTAL_TOKENS,
        Measure.ERROR,
    }
)
_EPS = 1e-9


class ExampleDelta(BaseModel):
    example_id: str
    base: float
    candidate: float
    delta: float
    base_explanation: str | None = None
    candidate_explanation: str | None = None


class TargetComparison(BaseModel):
    """Paired comparison of one score or measure (candidate - base)."""

    target: str
    kind: Literal["score", "measure"] = "score"
    higher_is_better: bool = True
    n_pairs: int
    base_mean: float | None = None
    candidate_mean: float | None = None
    diff: float | None = None
    stderr: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    # Smallest true difference this comparison could detect (80% power).
    mde: float | None = None
    test: str = "none"
    improved: int = 0
    regressed: int = 0
    unchanged: int = 0
    top_regressions: list[ExampleDelta] = Field(default_factory=list[ExampleDelta])
    top_improvements: list[ExampleDelta] = Field(default_factory=list[ExampleDelta])

    @property
    def significant(self) -> bool:
        return (
            self.ci_low is not None
            and self.ci_high is not None
            and (self.ci_low > 0 or self.ci_high < 0)
        )


class Comparison(BaseModel):
    base_run: str
    candidate_run: str
    base_name: str
    candidate_name: str
    n_paired_examples: int
    # Shared ids whose content changed between the runs (not compared).
    n_changed_examples: int
    n_base_only: int
    n_candidate_only: int
    warnings: list[str] = Field(default_factory=list[str])
    targets: list[TargetComparison] = Field(default_factory=list[TargetComparison])

    def target(self, name: str) -> TargetComparison | None:
        for target in self.targets:
            if target.target == name:
                return target
        return None


class _ExampleView:
    def __init__(self, trials: Sequence[Trial]) -> None:
        self.trials = list(trials)
        self.hash = trials[0].example_hash
        self.sealed = any(t.sealed for t in trials)

    def value(self, target: Target, reduce: Reducer) -> float | None:
        values = [v for t in self.trials if (v := trial_value(t, target)) is not None]
        return reduce(values) if values else None

    def explanation(self, target: Target) -> str | None:
        if isinstance(target, Measure):
            return None
        for trial in self.trials:
            score = trial.score(target)
            if score is not None and score.explanation:
                return score.explanation
        return None


def _views(run: EvaluationRun) -> dict[str, _ExampleView]:
    groups: dict[str, list[Trial]] = {}
    for trial in run.trials:
        groups.setdefault(trial.example_id, []).append(trial)
    return {example_id: _ExampleView(trials) for example_id, trials in groups.items()}


def _numeric_scores(run: EvaluationRun) -> list[str]:
    names: dict[str, None] = {}
    for trial in run.trials:
        for score in trial.scores:
            if score.as_float() is not None:
                names.setdefault(score.name, None)
    return list(names)


def _default_targets(base: EvaluationRun, candidate: EvaluationRun) -> list[Target]:
    shared = [n for n in _numeric_scores(base) if n in set(_numeric_scores(candidate))]
    targets: list[Target] = [*shared, Measure.ERROR, Measure.DURATION]
    if any(
        t.total_usage.cost_usd is not None for t in (*base.trials, *candidate.trials)
    ):
        targets.append(Measure.COST)
    return targets


def _warnings(
    base: EvaluationRun, candidate: EvaluationRun, n_changed: int
) -> list[str]:
    warnings: list[str] = []
    if base.config_hash == candidate.config_hash:
        warnings.append(
            "Identical configuration (same task, evaluators and data): differences "
            "measure run-to-run noise."
        )
    if base.dataset.fingerprint != candidate.dataset.fingerprint or n_changed:
        warnings.append(
            "The runs evaluated different dataset content; only shared examples with "
            f"unchanged content are compared ({n_changed} changed examples skipped)."
        )
    base_versions = {e.name: e.version for e in base.evaluators}
    for evaluator in candidate.evaluators:
        old = base_versions.get(evaluator.name)
        if evaluator.name in base_versions and old != evaluator.version:
            warnings.append(
                f"Evaluator {evaluator.name!r} changed version ({old} → "
                f"{evaluator.version}): score differences may come from the instrument."
            )
    if base.config.repetitions != candidate.config.repetitions:
        warnings.append(
            f"Repetitions differ ({base.config.repetitions} vs "
            f"{candidate.config.repetitions})."
        )
    for run, label in ((base, "base"), (candidate, "candidate")):
        if run.invalid_reason:
            warnings.append(f"The {label} run is invalid: {run.invalid_reason}.")
        if run.status != RunStatus.COMPLETED:
            warnings.append(f"The {label} run did not complete (status {run.status}).")
    return warnings


def compare(
    base: EvaluationRun,
    candidate: EvaluationRun,
    *,
    targets: Sequence[Target] | None = None,
    top: int = 5,
    reduce: Reducer = mean,
    confidence: float = 0.95,
) -> Comparison:
    """
    Paired comparison of two runs on the examples they share.

    Examples are paired by id and only when their content is unchanged;
    repetitions are reduced per example first. Each target reports the mean
    paired difference with a confidence interval, a p-value (McNemar for
    binary scores, paired t otherwise), the minimum detectable effect, and the
    examples that moved most. Examples in sealed splits contribute to the
    statistics but are never listed.
    """
    base_views = _views(base)
    candidate_views = _views(candidate)
    shared = [i for i in base_views if i in candidate_views]
    paired = [i for i in shared if base_views[i].hash == candidate_views[i].hash]
    n_changed = len(shared) - len(paired)
    comparisons: list[TargetComparison] = []
    for target in targets if targets is not None else _default_targets(base, candidate):
        higher_is_better = not (
            isinstance(target, Measure) and target in _LOWER_IS_BETTER
        )
        rows: list[tuple[str, float, float]] = []
        for example_id in paired:
            b = base_views[example_id].value(target, reduce)
            c = candidate_views[example_id].value(target, reduce)
            if b is not None and c is not None:
                rows.append((example_id, b, c))
        estimate = paired_difference(
            [b for _, b, _ in rows], [c for _, _, c in rows], confidence=confidence
        )
        bounded = all(0.0 <= v <= 1.0 for _, b, c in rows for v in (b, c))
        ci_low, ci_high = estimate.ci_low, estimate.ci_high
        if bounded:
            # A difference of [0, 1] values lives in [-1, 1].
            ci_low = None if ci_low is None else max(-1.0, ci_low)
            ci_high = None if ci_high is None else min(1.0, ci_high)
        sign = 1.0 if higher_is_better else -1.0
        deltas: list[ExampleDelta] = []
        improved = regressed = 0
        for example_id, b, c in rows:
            gain = sign * (c - b)
            if gain > _EPS:
                improved += 1
            elif gain < -_EPS:
                regressed += 1
            if abs(c - b) > _EPS and not (
                base_views[example_id].sealed or candidate_views[example_id].sealed
            ):
                deltas.append(
                    ExampleDelta(
                        example_id=example_id,
                        base=b,
                        candidate=c,
                        delta=c - b,
                        base_explanation=base_views[example_id].explanation(target),
                        candidate_explanation=candidate_views[example_id].explanation(
                            target
                        ),
                    )
                )
        deltas.sort(key=lambda d: sign * d.delta)
        has_rows = estimate.n > 0
        comparisons.append(
            TargetComparison(
                target=str(target),
                kind="measure" if isinstance(target, Measure) else "score",
                higher_is_better=higher_is_better,
                n_pairs=estimate.n,
                base_mean=estimate.base_mean if has_rows else None,
                candidate_mean=estimate.candidate_mean if has_rows else None,
                diff=estimate.diff if has_rows else None,
                stderr=estimate.stderr,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=estimate.p_value,
                mde=estimate.mde,
                test=estimate.test,
                improved=improved,
                regressed=regressed,
                unchanged=len(rows) - improved - regressed,
                top_regressions=[d for d in deltas if sign * d.delta < 0][:top],
                top_improvements=[d for d in reversed(deltas) if sign * d.delta > 0][
                    :top
                ],
            )
        )
    return Comparison(
        base_run=base.id,
        candidate_run=candidate.id,
        base_name=base.name,
        candidate_name=candidate.name,
        n_paired_examples=len(paired),
        n_changed_examples=n_changed,
        n_base_only=sum(1 for i in base_views if i not in candidate_views),
        n_candidate_only=sum(1 for i in candidate_views if i not in base_views),
        warnings=_warnings(base, candidate, n_changed),
        targets=comparisons,
    )
