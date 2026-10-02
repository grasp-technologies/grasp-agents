from collections.abc import Hashable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from .metrics import Measure, Reducer, Target, as_target, mean, trial_value
from .stats import holm_adjust, paired_difference
from .types import EvaluationRun, RunStatus, Trial

_RESOURCES = frozenset(
    {
        Measure.DURATION,
        Measure.COST,
        Measure.INPUT_TOKENS,
        Measure.OUTPUT_TOKENS,
        Measure.TOTAL_TOKENS,
    }
)
_RESOURCE_NAMES = frozenset(str(m) for m in _RESOURCES)
_LOWER_IS_BETTER = _RESOURCES | {Measure.ERROR}
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
    # Holm-adjusted within the target's family: outcomes (scores, the error
    # rate, recorded measures) or resources (duration, cost, tokens).
    p_adjusted: float | None = None
    significant: bool = False
    # Smallest true difference this comparison could detect (80% power).
    mde: float | None = None
    test: str = "none"
    improved: int = 0
    regressed: int = 0
    unchanged: int = 0
    top_regressions: list[ExampleDelta] = Field(default_factory=list[ExampleDelta])
    top_improvements: list[ExampleDelta] = Field(default_factory=list[ExampleDelta])

    @property
    def resource(self) -> bool:
        """Duration, cost or tokens rather than an outcome."""
        return self.target in _RESOURCE_NAMES

    @property
    def testable(self) -> bool:
        """Has paired values and a significance test."""
        return self.n_pairs > 0 and self.p_value is not None

    @property
    def worse(self) -> bool:
        """The candidate moved in the bad direction (significant or not)."""
        if self.diff is None:
            return False
        return self.diff < 0 if self.higher_is_better else self.diff > 0


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
    # Significance level of ``TargetComparison.significant``.
    alpha: float = 0.05
    cluster_by: str | None = None
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
        self.errored = any(not t.ok for t in trials)

    def value(
        self, target: Target, reduce: Reducer, *, errors_as_failures: bool
    ) -> float | None:
        values: list[float] = []
        for trial in self.trials:
            value = trial_value(trial, target)
            if value is None and errors_as_failures and not trial.ok:
                value = 0.0
            if value is not None:
                values.append(value)
        return reduce(values) if values else None

    def explanation(
        self, target: Target, pick: Literal["lowest", "highest"]
    ) -> str | None:
        if isinstance(target, Measure):
            return None
        found: list[tuple[float, str]] = []
        for trial in self.trials:
            score = trial.score(target)
            value = score.as_float() if score is not None else None
            if score is not None and score.explanation and value is not None:
                found.append((value, score.explanation))
        if not found:
            return None
        chosen = min(found) if pick == "lowest" else max(found)
        return chosen[1]


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


def _pass_fail_scores(*runs: EvaluationRun) -> set[str]:
    kinds: dict[str, set[type]] = {}
    for run in runs:
        for trial in run.trials:
            for score in trial.scores:
                if score.value is not None:
                    kinds.setdefault(score.name, set()).add(type(score.value))
    return {name for name, types in kinds.items() if types == {bool}}


def _default_targets(base: EvaluationRun, candidate: EvaluationRun) -> list[Target]:
    shared = [n for n in _numeric_scores(base) if n in set(_numeric_scores(candidate))]
    targets: list[Target] = [*shared, Measure.ERROR, Measure.DURATION]
    if any(
        t.total_usage.cost_usd is not None for t in (*base.trials, *candidate.trials)
    ):
        targets.append(Measure.COST)
    return targets


def _warnings(
    base: EvaluationRun,
    candidate: EvaluationRun,
    *,
    n_changed: int,
    n_paired: int,
    errored_one_side: int,
) -> list[str]:
    warnings: list[str] = []
    if base.id == candidate.id:
        warnings.append("Both sides are the same run.")
    elif base.config_hash == candidate.config_hash:
        warnings.append(
            "Identical configuration (same task, evaluators and data): differences "
            "measure run-to-run noise."
        )
    if n_paired == 0:
        warnings.append(
            "No example could be paired: the runs share no example id with "
            "unchanged content."
        )
    if base.dataset.fingerprint != candidate.dataset.fingerprint or n_changed:
        warnings.append(
            "The runs evaluated different dataset content; only shared examples with "
            f"unchanged content are compared ({n_changed} changed examples skipped)."
        )
    elif base.dataset.selected_fingerprint != candidate.dataset.selected_fingerprint:
        warnings.append(
            "The runs evaluated different subsets of the dataset "
            f"({', '.join(base.dataset.selection) or 'all'} vs "
            f"{', '.join(candidate.dataset.selection) or 'all'}); only shared "
            "examples are compared."
        )
    base_evaluators = {e.name: e for e in base.evaluators}
    for evaluator in candidate.evaluators:
        old = base_evaluators.get(evaluator.name)
        if old is None:
            continue
        if old.version != evaluator.version:
            warnings.append(
                f"Evaluator {evaluator.name!r} changed version ({old.version} → "
                f"{evaluator.version}): score differences may come from the instrument."
            )
        elif old.config != evaluator.config:
            warnings.append(
                f"Evaluator {evaluator.name!r} changed its configuration without a "
                "version bump: score differences may come from the instrument."
            )
        elif old.source and evaluator.source and old.source != evaluator.source:
            warnings.append(
                f"Evaluator {evaluator.name!r} changed its code without a version "
                "bump: score differences may come from the instrument."
            )
    if base.config.repetitions != candidate.config.repetitions:
        warnings.append(
            f"Repetitions differ ({base.config.repetitions} vs "
            f"{candidate.config.repetitions})."
        )
    if errored_one_side:
        warnings.append(
            f"{errored_one_side} paired examples had task errors in only one run; "
            "pass/fail scores count them as failures, other scores skip them "
            "(see the error target)."
        )
    for run, label in ((base, "base"), (candidate, "candidate")):
        if run.invalid_reason:
            warnings.append(f"The {label} run is invalid: {run.invalid_reason}.")
        if run.status != RunStatus.COMPLETED:
            warnings.append(f"The {label} run did not complete (status {run.status}).")
    return warnings


def _clusters(
    run: EvaluationRun, cluster_by: str | None, example_ids: Sequence[str]
) -> list[Hashable] | None:
    if cluster_by is None:
        return None
    metadata = {e.id: e.metadata for e in run.examples}
    clusters: list[Hashable] = []
    for example_id in example_ids:
        value: Any = metadata.get(example_id, {}).get(cluster_by, example_id)
        clusters.append(value if isinstance(value, Hashable) else repr(value))
    return clusters


def compare(
    base: EvaluationRun,
    candidate: EvaluationRun,
    *,
    targets: Sequence[Target] | None = None,
    top: int = 5,
    reduce: Reducer = mean,
    confidence: float = 0.95,
    cluster_by: str | None = None,
) -> Comparison:
    """
    Paired comparison of two runs on the examples they share.

    Examples are paired by id and only when their content is unchanged;
    repetitions are reduced per example first, and a task error counts as a
    failure of pass/fail scores. Each target reports the mean paired
    difference with a confidence interval, a p-value (exact McNemar for
    pass/fail scores, a paired t-test otherwise — clustered when the base run
    was clustered, or by ``cluster_by``), the minimum detectable effect, and
    the examples that moved most. ``significant`` uses Holm-adjusted p-values
    at ``1 - confidence``, adjusted among the outcome targets (scores, the
    error rate) and separately among resource targets (duration, cost);
    targets where no example changed are left out of the adjustment.
    Examples in sealed splits contribute to the statistics but are never
    listed.
    """
    base_views = _views(base)
    candidate_views = _views(candidate)
    shared = [i for i in base_views if i in candidate_views]
    paired = [i for i in shared if base_views[i].hash == candidate_views[i].hash]
    n_changed = len(shared) - len(paired)
    cluster_key = cluster_by if cluster_by is not None else base.config.cluster_by
    pass_fail = _pass_fail_scores(base, candidate)
    errored_one_side = sum(
        1 for i in paired if base_views[i].errored != candidate_views[i].errored
    )
    comparisons: list[TargetComparison] = []
    for raw_target in (
        targets if targets is not None else _default_targets(base, candidate)
    ):
        target = as_target(raw_target)
        higher_is_better = not (
            isinstance(target, Measure) and target in _LOWER_IS_BETTER
        )
        as_failures = not isinstance(target, Measure) and target in pass_fail
        rows: list[tuple[str, float, float]] = []
        for example_id in paired:
            b = base_views[example_id].value(
                target, reduce, errors_as_failures=as_failures
            )
            c = candidate_views[example_id].value(
                target, reduce, errors_as_failures=as_failures
            )
            if b is not None and c is not None:
                rows.append((example_id, b, c))
        estimate = paired_difference(
            [b for _, b, _ in rows],
            [c for _, _, c in rows],
            clusters=_clusters(base, cluster_key, [i for i, _, _ in rows]),
            confidence=confidence,
        )
        ci_low, ci_high = estimate.ci_low, estimate.ci_high
        if all(0.0 <= v <= 1.0 for _, b, c in rows for v in (b, c)):
            # A difference of shares lies in [-1, 1].
            ci_low = None if ci_low is None else max(-1.0, ci_low)
            ci_high = None if ci_high is None else min(1.0, ci_high)
        sign = 1.0 if higher_is_better else -1.0
        worst, best = (
            ("lowest", "highest") if higher_is_better else ("highest", "lowest")
        )
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
                got_worse = gain < 0
                deltas.append(
                    ExampleDelta(
                        example_id=example_id,
                        base=b,
                        candidate=c,
                        delta=c - b,
                        base_explanation=base_views[example_id].explanation(
                            target, best if got_worse else worst
                        ),
                        candidate_explanation=candidate_views[example_id].explanation(
                            target, worst if got_worse else best
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
    alpha = 1.0 - confidence
    for resource in (False, True):
        family = [
            t
            for t in comparisons
            if t.resource == resource and t.testable and t.improved + t.regressed
        ]
        adjusted = holm_adjust([t.p_value for t in family])
        for comparison, p in zip(family, adjusted, strict=True):
            comparison.p_adjusted = p
            comparison.significant = p is not None and p < alpha
    for comparison in comparisons:
        if comparison.p_adjusted is None:
            comparison.p_adjusted = comparison.p_value
    warnings = _warnings(
        base,
        candidate,
        n_changed=n_changed,
        n_paired=len(paired),
        errored_one_side=errored_one_side,
    )
    if cluster_key is not None and paired:
        clusters = set(_clusters(base, cluster_key, paired) or ())
        if len(clusters) < 2:
            warnings.append(
                f"All paired examples are in one {cluster_key!r} cluster: "
                "nothing can be tested."
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
        alpha=alpha,
        cluster_by=cluster_key,
        warnings=warnings,
        targets=comparisons,
    )


def regressions(
    comparison: Comparison,
    *,
    targets: Sequence[str] | None = None,
    min_effect: float = 0.0,
) -> list[str]:
    """
    Why a regression gate should fail: a significant change in the bad
    direction of at least ``min_effect`` in a gated target — the outcome
    targets (scores and the error rate) unless ``targets`` names others, e.g.
    ``"duration_s"`` — or a gated target that cannot be tested (nothing
    paired, or too few examples or clusters).
    """
    failures: list[str] = []
    if comparison.n_paired_examples == 0:
        failures.append("no example could be paired with the baseline")
    for target in comparison.targets:
        gated = target.target in targets if targets is not None else not target.resource
        if not gated:
            continue
        if target.n_pairs == 0:
            failures.append(f"no paired values for {target.target}")
            continue
        if target.p_value is None:
            failures.append(
                f"{target.target} cannot be tested ({target.n_pairs} paired examples)"
            )
            continue
        diff = target.diff
        if (
            target.significant
            and target.worse
            and diff is not None
            and abs(diff) >= min_effect
        ):
            p = target.p_adjusted
            failures.append(
                f"significant regression in {target.target} "
                f"(Δ={diff:+.3g}, adjusted p={p:.3g})"
                if p is not None
                else f"significant regression in {target.target}"
            )
    if targets is not None:
        known = {t.target for t in comparison.targets}
        failures.extend(
            f"no comparison for gated target {name!r}"
            for name in targets
            if name not in known
        )
    return failures
