"""
Judge validation: a judge (any :class:`Scorer`) becomes the system under
test, applied to stored outputs whose correct verdict is known — labeled by
people, or constructed by perturbing outputs — so its agreement is measured
with the same runs, statistics, sealing and comparisons as any evaluation.
"""

import math
import random
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, override

from pydantic import BaseModel, ValidationError

from ._execution import scorer_sources
from ._util import to_jsonable, user_code_hash, utc_now
from .metrics import (
    ClassRecall,
    CohenKappa,
    ConfusionMatrix,
    Consistency,
    ErrorRate,
    Metric,
    PassRate,
    label_text,
)
from .scorer import (
    EvalContext,
    Scorer,
    ScorerOutput,
    merge_models,
    run_all_or_cancel,
    run_scorer,
)
from .stats import corrected_prevalence, percentile
from .store import RunStore
from .task import Task, TrialContext
from .types import (
    ComponentInfo,
    EvaluationRun,
    Example,
    JudgedOutput,
    MetricResult,
    Score,
    ScoreValue,
    Trial,
    Usage,
)

# ``ComponentInfo.kind`` of the tasks that apply a scorer to stored
# outputs: against labels, and with perturbed outputs.
SCORER_TASK_KIND = "scorer"
PROBE_TASK_KIND = "scorer-probes"

# The name the label-agreement scorer records its scores under.
AGREEMENT = "agreement"

_TRUE_WORDS = frozenset({"true", "yes", "pass", "passed"})
_FALSE_WORDS = frozenset({"false", "no", "fail", "failed"})


class LabelMismatchError(ValueError):
    """A judge's verdict and a label cannot be compared (e.g. pass/fail vs text)."""


def _read(value: ScoreValue, threshold: float | None) -> ScoreValue:
    # Pass/fail words read as booleans; numbers at or above ``threshold`` pass.
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
        return value
    if threshold is not None and not isinstance(value, bool):
        return value >= threshold
    return value


def label_value_text(value: ScoreValue, threshold: float | None = None) -> str:
    """A label as agreement metrics record it (pass/fail words as true/false)."""
    return label_text(_read(value, threshold))


def comparable(
    verdict: ScoreValue, label: ScoreValue, threshold: float | None = None
) -> tuple[str, str]:
    """
    A verdict and a label as the text agreement compares. Pass/fail words
    (true/false, yes/no, pass/fail) read as booleans, numbers at or above
    ``threshold`` as passes, and 0/1 as pass/fail against a boolean. Raises
    :class:`LabelMismatchError` for values of incompatible kinds, which would
    otherwise always disagree.
    """
    judged, expected = _read(verdict, threshold), _read(label, threshold)
    if isinstance(judged, bool) != isinstance(expected, bool):
        other = expected if isinstance(judged, bool) else judged
        if isinstance(other, int | float) and other in {0, 1}:
            if isinstance(judged, bool):
                expected = bool(other)
            else:
                judged = bool(other)
        else:
            raise LabelMismatchError(
                f"Cannot compare the verdict {verdict!r} with the label {label!r}: "
                "a pass/fail verdict needs a pass/fail label (true/false, yes/no, "
                "pass/fail or 1/0); set threshold= for numeric scores"
            )
    elif isinstance(judged, str) != isinstance(expected, str):
        raise LabelMismatchError(
            f"Cannot compare the verdict {verdict!r} with the label {label!r}: "
            "one is a number and the other a text label"
        )
    return label_text(judged), label_text(expected)


def _source_example(
    trial: TrialContext, item: JudgedOutput[Any, Any, Any]
) -> Example[Any, Any]:
    # The judged example under its original id when the record names it.
    original = trial.example.metadata.get("example_id")
    return Example[Any, Any](
        id=str(original or trial.example.id),
        input=item.input,
        reference=item.reference,
        metadata=item.metadata,
    )


@dataclass
class _Spend:
    """What the judge's calls used, across the calls of one trial."""

    usage: list[Usage]
    models: dict[str, list[str]]

    def record(self, trial: TrialContext, name: str) -> None:
        if self.usage:
            trial.usage_by_agent[name] = sum(self.usage, Usage())
        merge_models(trial.models, self.models)


async def _judge(
    scorer: Scorer[Any, Any, Any],
    example: Example[Any, Any],
    output: Any,
    repetition: int,
    spend: _Spend,
) -> list[Score]:
    trial = Trial(
        example_id=example.id,
        repetition=repetition,
        example_hash=example.content_hash,
        output=to_jsonable(output),
        started_at=utc_now(),
        duration_s=0.0,
    )
    ctx = EvalContext(example=example, output=output, trial=trial, usage=spend.usage)
    try:
        return await run_scorer(scorer, ctx)
    finally:
        merge_models(spend.models, ctx.models)


class _JudgeTask[OutT](Task[JudgedOutput[Any, Any, Any], OutT]):
    kind: str = SCORER_TASK_KIND

    def __init__(
        self,
        scorer: Scorer[Any, Any, Any],
        *,
        input_type: Any,
        output_type: Any,
        reference_type: Any,
        name: str | None,
    ) -> None:
        self.scorer = scorer
        self.name = name or scorer.name
        self.version = scorer.version
        self._input_type: Any = JudgedOutput[input_type, output_type, reference_type]
        # Task settings recorded with the judge's identity.
        self.settings: dict[str, Any] = {}

    @property
    def input_type(self) -> Any:
        return self._input_type

    def describe(self) -> ComponentInfo:
        info = self.scorer.describe()
        return ComponentInfo(
            name=self.name,
            kind=self.kind,
            version=info.version,
            config={"scorer": info.model_dump(mode="json"), **self.settings},
            fingerprint=info.fingerprint,
        )

    def source_objects(self) -> list[Any]:
        return scorer_sources([self.scorer])


class ScorerTask(_JudgeTask[list[Score]]):
    """
    Applies ``scorer`` to stored outputs, making the judge the system
    under test: each example's input is a :class:`JudgedOutput` and the
    task's output is the scorer's scores. The judge's model usage and
    models are the trial's, and a judge that fails is a task error.
    Scorers that read the transcript or the session see neither.
    """

    def __init__(
        self,
        scorer: Scorer[Any, Any, Any],
        *,
        input_type: Any = Any,
        output_type: Any = Any,
        reference_type: Any = Any,
        name: str | None = None,
    ) -> None:
        super().__init__(
            scorer,
            input_type=input_type,
            output_type=output_type,
            reference_type=reference_type,
            name=name,
        )

    @property
    def output_type(self) -> Any:
        return list[Score]

    async def run(
        self,
        input: JudgedOutput[Any, Any, Any],  # noqa: A002
        trial: TrialContext,
    ) -> list[Score]:
        spend = _Spend(usage=[], models={})
        try:
            return await _judge(
                self.scorer,
                _source_example(trial, input),
                input.output,
                trial.repetition,
                spend,
            )
        finally:
            spend.record(trial, self.scorer.name)


def _scores_of(output: Any) -> dict[str, Score]:
    scores: dict[str, Score] = {}
    for item in cast("Sequence[Any]", output or []):
        score = item if isinstance(item, Score) else Score.model_validate(item)
        scores[score.name] = score
    return scores


def _filled(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _labels_of(
    reference: Any, scores: Sequence[str], labeled_score: str | None
) -> dict[str, ScoreValue]:
    if isinstance(reference, Mapping):
        mapping = cast("Mapping[str, Any]", reference)
        return {k: v for k, v in mapping.items() if k in scores and _filled(v)}
    if not _filled(reference):
        return {}
    # A single value labels the score its record names, else the one validated.
    target = labeled_score
    if target is None:
        if len(scores) != 1:
            raise ValueError(
                f"The label {reference!r} is a single value but {list(scores)} are "
                "validated; label each score ({score: label}) or name it in "
                "metadata.score"
            )
        target = scores[0]
    return {target: cast("ScoreValue", reference)} if target in scores else {}


class LabelAgreement(Scorer[Any, list[Score], Any]):
    """
    Compares a judge's scores (the output of :class:`ScorerTask`) with the
    example's reference label: a mapping of score names to values, or one
    value for the score its ``metadata.score`` names (or the only one
    validated). For each labeled score ``s`` it records ``s.label``, and the
    judge's verdict ``s.judge`` and ``s.agrees`` (see :func:`comparable`;
    numbers at or above ``threshold`` read as passes). When the judge gave no
    verdict, or failed, those two are unscored, so agreement metrics count the
    item as missing. ``positive`` / ``negative`` name the passing and failing
    labels, recorded for the gate.
    """

    scores_errors = True

    def __init__(
        self,
        scores: Sequence[str],
        *,
        threshold: float | None = None,
        positive: ScoreValue | None = True,
        negative: ScoreValue | None = False,
        name: str = AGREEMENT,
    ) -> None:
        super().__init__(name=name)
        self.scores = list(scores)
        self.threshold = threshold
        self.positive = None if positive is None else label_value_text(positive)
        self.negative = None if negative is None else label_value_text(negative)

    def config(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "threshold": self.threshold,
            "positive": self.positive,
            "negative": self.negative,
        }

    def score(self, ctx: EvalContext[Any, list[Score], Any]) -> ScorerOutput:
        if ctx.reference is None:
            return None
        labels = _labels_of(ctx.reference, self.scores, ctx.metadata.get("score"))
        verdicts = _scores_of(ctx.output) if ctx.trial.ok else {}
        scores: list[Score] = []
        for name, value in labels.items():
            verdict = verdicts.get(name)
            if verdict is None or verdict.value is None:
                label = label_value_text(value, self.threshold)
                if ctx.trial.error is not None:
                    reason, why = "judge_failed", ctx.trial.error.message
                else:
                    reason = (verdict.reason if verdict else None) or "no_verdict"
                    why = "the judge gave no verdict"
                scores.extend(
                    [
                        Score.unscored(
                            f"{name}.agrees",
                            reason=reason,
                            explanation=f"label: {label}; {why}",
                        ),
                        Score.unscored(f"{name}.judge", reason=reason),
                        Score(name=f"{name}.label", value=label),
                    ]
                )
                continue
            judged, label = comparable(verdict.value, value, self.threshold)
            why = f" — {verdict.explanation}" if verdict.explanation else ""
            scores.extend(
                [
                    Score(
                        name=f"{name}.agrees",
                        value=judged == label,
                        explanation=f"judge: {judged}, label: {label}{why}",
                    ),
                    Score(
                        name=f"{name}.judge",
                        value=judged,
                        explanation=verdict.explanation,
                    ),
                    Score(name=f"{name}.label", value=label),
                ]
            )
        return scores or None


def validation_metrics(
    scores: Sequence[str],
    *,
    positive: ScoreValue | None = True,
    negative: ScoreValue | None = False,
    repetitions: int = 1,
) -> list[Metric]:
    """
    Per validated score: accuracy, Cohen's κ, true positive and true negative
    rates (``positive`` / ``negative`` are the passing and failing labels;
    ``None`` skips them, e.g. for multi-class labels), the confusion matrix,
    and the judge's self-consistency when it is sampled more than once.
    Items the judge gave no verdict on, or failed on, count as missing.
    """
    metrics: list[Metric] = []
    for name in scores:
        judged, label = f"{name}.judge", f"{name}.label"
        metrics.extend(
            [
                PassRate(
                    f"{name}.agrees",
                    errors_as_failures=False,
                    name=f"accuracy({name})",
                ),
                CohenKappa(judged, label, name=f"kappa({name})"),
            ]
        )
        if positive is not None:
            metrics.append(
                ClassRecall(
                    judged, label, label_value_text(positive), name=f"tpr({name})"
                )
            )
        if negative is not None:
            metrics.append(
                ClassRecall(
                    judged, label, label_value_text(negative), name=f"tnr({name})"
                )
            )
        metrics.append(ConfusionMatrix(judged, label, name=f"confusion({name})"))
        if repetitions > 1:
            metrics.append(Consistency(judged, name=f"consistency({name})"))
    metrics.append(ErrorRate())
    return metrics


# --- Probes: perturbation sensitivity and invariance ---


type Expectation = Literal["lower", "changed", "same"]


@dataclass(frozen=True)
class Perturbation:
    """
    A change to an output, and what it must do to the judge's verdict:

    - ``"lower"``: a degradation (a dropped key point, an injected error) —
      the judge must fail it, or give a lower number;
    - ``"changed"``: the verdict must differ (for labels without an order);
    - ``"same"``: an irrelevant change (padding, reordering, formatting) —
      the verdict must not move.

    ``apply(item)`` returns the changed output, or ``None`` when the change
    does not apply to that item. Its code is part of the probe run's
    identity.
    """

    name: str
    apply: Callable[[JudgedOutput[Any, Any, Any]], Any]
    expect: Expectation


class ProbeTask(_JudgeTask[dict[str, list[Score] | None]]):
    """
    Judges each stored output as it is and after every applicable
    perturbation; the output maps ``"original"`` and each perturbation's name
    to the judge's scores (``None`` when the perturbation does not apply).
    """

    kind: str = PROBE_TASK_KIND

    def __init__(
        self,
        scorer: Scorer[Any, Any, Any],
        perturbations: Sequence[Perturbation],
        *,
        input_type: Any = Any,
        output_type: Any = Any,
        reference_type: Any = Any,
        name: str | None = None,
    ) -> None:
        super().__init__(
            scorer,
            input_type=input_type,
            output_type=output_type,
            reference_type=reference_type,
            name=name,
        )
        names = [p.name for p in perturbations]
        if "original" in names or len(set(names)) != len(names):
            raise ValueError(
                f"Perturbation names must be unique, not 'original': {names}"
            )
        self.perturbations = list(perturbations)
        self.settings = {
            "probes": {p.name: p.expect for p in perturbations},
            "probe_code": user_code_hash(p.apply for p in perturbations),
        }

    @property
    def output_type(self) -> Any:
        return dict[str, list[Score] | None]

    def source_objects(self) -> list[Any]:
        return [*super().source_objects(), *(p.apply for p in self.perturbations)]

    async def run(
        self,
        input: JudgedOutput[Any, Any, Any],  # noqa: A002
        trial: TrialContext,
    ) -> dict[str, list[Score] | None]:
        example = _source_example(trial, input)
        outputs: dict[str, Any] = {"original": input.output}
        for perturbation in self.perturbations:
            changed = perturbation.apply(input)
            if changed is not None:
                outputs[perturbation.name] = changed
        spend = _Spend(usage=[], models={})
        try:
            judged = await run_all_or_cancel(
                [
                    _judge(self.scorer, example, output, trial.repetition, spend)
                    for output in outputs.values()
                ]
            )
        finally:
            spend.record(trial, self.scorer.name)
        results: dict[str, list[Score] | None] = dict.fromkeys(
            ["original", *(p.name for p in self.perturbations)]
        )
        results.update(zip(outputs, judged, strict=True))
        return results


def _ordinal(value: ScoreValue) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


class ProbeCheck(Scorer[Any, dict[str, list[Score] | None], Any]):
    """
    Scores a :class:`ProbeTask` output: for each validated score ``s`` and
    perturbation ``p``, ``s.p`` passes when the judge's verdict moved as
    ``p`` expects. Not applicable when the perturbation did not apply, when a
    verdict is missing, or — for ``"lower"`` — when the original verdict was
    already a fail. ``"lower"`` needs pass/fail or numeric verdicts (for
    labels, use ``"changed"``).
    """

    def __init__(
        self,
        perturbations: Sequence[Perturbation],
        scores: Sequence[str],
        *,
        threshold: float | None = None,
        name: str = "probes",
    ) -> None:
        super().__init__(name=name)
        self.perturbations = list(perturbations)
        self.scores = list(scores)
        self.threshold = threshold

    def config(self) -> dict[str, Any]:
        return {
            "probes": {p.name: p.expect for p in self.perturbations},
            "scores": self.scores,
            "threshold": self.threshold,
        }

    def _check(
        self, expect: Expectation, before: ScoreValue, after: ScoreValue
    ) -> bool | None:
        before, after = _read(before, self.threshold), _read(after, self.threshold)
        if expect == "same":
            return before == after
        if expect == "changed":
            return before != after
        high, low = _ordinal(before), _ordinal(after)
        if high is None or low is None:
            raise ValueError(
                f"A 'lower' probe needs pass/fail or numeric verdicts, got "
                f"{before!r}; use expect='changed' for labels"
            )
        if before is False:
            # Already a fail: there is nothing to lower.
            return None
        return low < high

    def score(
        self, ctx: EvalContext[Any, dict[str, list[Score] | None], Any]
    ) -> ScorerOutput:
        outputs = ctx.output or {}
        original = _scores_of(outputs.get("original"))
        scores: list[Score] = []
        for name in self.scores:
            before = original.get(name)
            if before is None or before.value is None:
                continue
            for perturbation in self.perturbations:
                changed = outputs.get(perturbation.name)
                if changed is None:
                    continue
                after = _scores_of(changed).get(name)
                if after is None or after.value is None:
                    continue
                passed = self._check(perturbation.expect, before.value, after.value)
                if passed is None:
                    continue
                scores.append(
                    Score(
                        name=f"{name}.{perturbation.name}",
                        value=passed,
                        explanation=(
                            f"expected {perturbation.expect}: {before.value!r} → "
                            f"{after.value!r} — {after.explanation or ''}"
                        ),
                    )
                )
        return scores or None


def probe_metrics(
    perturbations: Sequence[Perturbation], scores: Sequence[str]
) -> list[Metric]:
    """Sensitivity (degradations caught) and invariance (changes ignored) rates."""
    metrics: list[Metric] = []
    for name in scores:
        for perturbation in perturbations:
            kind = "invariance" if perturbation.expect == "same" else "sensitivity"
            metrics.append(
                PassRate(
                    f"{name}.{perturbation.name}",
                    errors_as_failures=False,
                    name=f"{kind}({name}.{perturbation.name})",
                )
            )
    metrics.append(ErrorRate())
    return metrics


# --- What a validation shows: summaries, gates, corrected pass rates ---


class JudgeValidation(BaseModel):
    """A judge's agreement with labels for one score, from a validation run."""

    run_id: str
    score: str
    scorer: ComponentInfo
    # The labels dataset: its name and the fingerprint of what was evaluated.
    labels: str
    labels_fingerprint: str
    # Computed over the sealed (held-out) split only.
    sealed: bool
    # Labeled judgments, and those the judge gave no verdict on or failed on.
    judgments: int
    missing: int
    # Numeric verdicts at or above it were read as passes.
    threshold: float | None = None
    positive: str | None = None
    negative: str | None = None
    accuracy: MetricResult
    kappa: MetricResult
    tpr: MetricResult | None = None
    tnr: MetricResult | None = None

    @property
    def missing_share(self) -> float:
        return self.missing / self.judgments if self.judgments else 0.0

    def rates(self) -> "JudgeErrorRates | None":
        """The judge's error rates, when both classes were labeled."""
        if self.tpr is None or self.tnr is None:
            return None
        if self.tpr.value is None or self.tnr.value is None:
            return None
        return JudgeErrorRates(
            tpr=self.tpr.value,
            tnr=self.tnr.value,
            n_positive=self.tpr.n,
            n_negative=self.tnr.n,
            source=self.run_id,
        )

    def brief(self) -> dict[str, Any]:
        def value(result: MetricResult | None) -> dict[str, Any] | None:
            if result is None:
                return None
            return {
                "value": result.value,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "n": result.n,
            }

        return {
            "run_id": self.run_id,
            "labels": self.labels,
            "labels_fingerprint": self.labels_fingerprint,
            "sealed": self.sealed,
            "judgments": self.judgments,
            "missing": self.missing,
            "accuracy": value(self.accuracy),
            "kappa": value(self.kappa),
            "tpr": value(self.tpr),
            "tnr": value(self.tnr),
        }


def validated_scorer(run: EvaluationRun) -> ComponentInfo | None:
    """The judge a validation run measured, or ``None`` for other runs."""
    if run.task.kind != SCORER_TASK_KIND:
        return None
    try:
        return ComponentInfo.model_validate(run.task.config.get("scorer"))
    except ValidationError:
        return None


def _agreement_config(run: EvaluationRun, score: str) -> dict[str, Any] | None:
    for info in run.scorers:
        scores = info.config.get("scores")
        if info.name == AGREEMENT and isinstance(scores, list) and score in scores:
            return info.config
    return None


def _clusters(run: EvaluationRun) -> dict[str, Hashable] | None:
    key = run.config.cluster_by
    if key is None:
        return None
    clusters: dict[str, Hashable] = {}
    for example in run.examples:
        value = example.metadata.get(key, example.id)
        clusters[example.id] = value if isinstance(value, Hashable) else repr(value)
    return clusters


def summarize_validation(
    run: EvaluationRun, score: str, *, sealed_only: bool = True
) -> JudgeValidation:
    """
    Agreement of the judge a validation run measured with the labels for
    ``score`` — over the sealed split (``sealed_only``), or every trial —
    clustered as the run was.
    """
    scorer = validated_scorer(run)
    config = _agreement_config(run, score)
    if scorer is None or config is None:
        raise ValueError(f"Run {run.id} is not a validation run covering {score!r}")
    trials = [t for t in run.trials if t.sealed] if sealed_only else list(run.trials)
    judged, label = f"{score}.judge", f"{score}.label"
    positive = cast("str | None", config.get("positive", "true"))
    negative = cast("str | None", config.get("negative", "false"))
    clusters = _clusters(run)

    def metric(m: Metric) -> MetricResult:
        return m.compute(trials, clusters=clusters)

    labeled = [
        t
        for t in trials
        if t.score(label) is not None
        or any(f.scorer == AGREEMENT for f in t.scorer_failures)
    ]
    missing = sum(
        1
        for t in labeled
        if (agrees := t.score(f"{score}.agrees")) is None or agrees.value is None
    )
    tpr = (
        metric(ClassRecall(judged, label, positive, name=f"tpr({score})"))
        if positive is not None
        else None
    )
    tnr = (
        metric(ClassRecall(judged, label, negative, name=f"tnr({score})"))
        if negative is not None
        else None
    )
    return JudgeValidation(
        run_id=run.id,
        score=score,
        scorer=scorer,
        labels=run.dataset.name,
        labels_fingerprint=run.dataset.selected_fingerprint,
        sealed=sealed_only,
        judgments=len(labeled),
        missing=missing,
        threshold=cast("float | None", config.get("threshold")),
        positive=positive,
        negative=negative,
        accuracy=metric(
            PassRate(
                f"{score}.agrees",
                errors_as_failures=False,
                name=f"accuracy({score})",
            )
        ),
        kappa=metric(CohenKappa(judged, label, name=f"kappa({score})")),
        tpr=tpr if tpr is not None and (tpr.n or tpr.n_missing) else None,
        tnr=tnr if tnr is not None and (tnr.n or tnr.n_missing) else None,
    )


@dataclass(frozen=True)
class ValidationGate:
    """
    What a judge's validation must show before an evaluation may use it,
    read on the sealed (held-out) split of its labels. Thresholds apply to the
    lower end of each interval (``bound="value"`` for point estimates), so a
    handful of labels cannot pass by luck. At most ``max_missing`` of the
    labeled judgments may lack a verdict (the judge abstained or failed;
    retry failures with ``resume``). ``labels`` names the labels dataset the
    validation must have used.
    """

    min_accuracy: float | None = None
    min_kappa: float | None = None
    min_tpr: float | None = None
    min_tnr: float | None = None
    max_missing: float = 0.0
    labels: str | None = None
    bound: Literal["ci_low", "value"] = "ci_low"

    def failures(self, validation: JudgeValidation) -> list[str]:
        where = f"(validation run {validation.run_id})"
        failures: list[str] = []
        if validation.judgments == 0:
            return [f"{validation.score}: no labeled judgments {where}"]
        if validation.missing_share > self.max_missing:
            failures.append(
                f"{validation.score}: {validation.missing} of "
                f"{validation.judgments} labeled judgments have no verdict "
                f"(max {self.max_missing:.0%}) {where}"
            )
        checks: list[tuple[str, float | None, MetricResult | None]] = [
            ("accuracy", self.min_accuracy, validation.accuracy),
            ("kappa", self.min_kappa, validation.kappa),
            ("tpr", self.min_tpr, validation.tpr),
            ("tnr", self.min_tnr, validation.tnr),
        ]
        for name, minimum, result in checks:
            if minimum is None:
                continue
            value = None if result is None else getattr(result, self.bound)
            if value is None or value < minimum:
                shown = "unknown" if value is None else f"{value:.3f}"
                failures.append(
                    f"{validation.score}: {name} {self.bound} {shown} < {minimum} "
                    f"{where}"
                )
        return failures


def _measured_at(
    run: EvaluationRun, headers: Mapping[str, EvaluationRun]
) -> tuple[Any, Any]:
    # A rescore re-judges its parent's trials: it ranks by when they ran.
    root = run
    while root.kind == "rescore" and root.parent_run_id is not None:
        parent = headers.get(root.parent_run_id)
        if parent is None:
            break
        root = parent
    return (root.created_at, run.created_at)


def find_validation(
    store: RunStore,
    scorers: Sequence[Scorer[Any, Any, Any]],
    score: str,
    *,
    labels: str | None = None,
) -> JudgeValidation | None:
    """
    The newest finished, valid validation run of one of ``scorers`` —
    exactly as they are now (name, version, configuration, fingerprint and
    code) — that covers ``score`` on a sealed split (of the labels dataset
    named ``labels``, when given), summarized over that split. Rescored runs
    rank by when their trials ran.
    """
    identities = [e.describe() for e in scorers]
    headers = {h.id: h for h in store.list_runs()}
    candidates = [
        h
        for h in headers.values()
        if (scorer := validated_scorer(h)) is not None
        and scorer in identities
        and h.completed
        and not h.invalid_reason
        and _agreement_config(h, score) is not None
        and (labels is None or h.dataset.name == labels)
    ]
    candidates.sort(key=lambda h: _measured_at(h, headers), reverse=True)
    for header in candidates:
        run = store.load(header.id)
        if any(t.sealed for t in run.trials):
            return summarize_validation(run, score)
    return None


class UnvalidatedJudgeError(Exception):
    """An evaluation's judge has no validation run that passes its gate."""


def check_validations(
    store: RunStore,
    scorers: Sequence[Scorer[Any, Any, Any]],
    gates: Mapping[str, ValidationGate],
) -> tuple[dict[str, JudgeValidation], list[str]]:
    """``(validations found, gate failures)`` for each gated score."""
    found: dict[str, JudgeValidation] = {}
    failures: list[str] = []
    for score, gate in gates.items():
        validation = find_validation(store, scorers, score, labels=gate.labels)
        if validation is None:
            on = f" of {gate.labels!r}" if gate.labels else ""
            failures.append(
                f"{score}: no finished validation run of the current judge on a "
                f"sealed split{on} (any change to its prompt, model, settings, "
                "version or code needs a new one)"
            )
            continue
        found[score] = validation
        failures.extend(gate.failures(validation))
    return found, failures


@dataclass(frozen=True)
class JudgeErrorRates:
    """A binary judge's measured accuracy: what correcting its pass rate needs."""

    tpr: float
    tnr: float
    # Labeled examples behind each rate.
    n_positive: int
    n_negative: int
    # Where the rates were measured (a validation run id).
    source: str | None = None


def _jeffreys(rng: random.Random, share: float, n: float) -> float:
    # A draw of a share measured as ``share`` of ``n`` units (Jeffreys prior:
    # a rate measured at 0 or 1 still varies).
    successes = min(max(share, 0.0), 1.0) * n
    return rng.betavariate(successes + 0.5, n - successes + 0.5)


class CorrectedPassRate(Metric):
    """
    The pass rate of ``of`` corrected for a binary judge's errors
    (Rogan-Gladen: ``(observed + TNR - 1) / (TPR + TNR - 1)``), with a
    parametric bootstrap interval that carries the uncertainty of the
    observed rate and of both error rates. A judge whose TPR + TNR is at most
    1 is no better than chance and gives no value. Only the judge's verdicts
    are corrected: trials whose task failed stay failures
    (``errors_as_failures``) or are left out.

    It assumes the judge errs on these outputs as it did on the labeled
    ones, so label outputs like the ones being scored (from the same system
    and population).
    """

    def __init__(
        self,
        of: str,
        rates: JudgeErrorRates,
        *,
        threshold: float | None = None,
        errors_as_failures: bool = True,
        name: str | None = None,
        n_resamples: int = 2000,
        seed: int = 0,
        confidence: float = 0.95,
    ) -> None:
        self.of = of
        self.rates = rates
        self.errors_as_failures = errors_as_failures
        self.judged = PassRate(of, threshold=threshold, errors_as_failures=False)
        self.n_resamples = n_resamples
        self.seed = seed
        self.confidence = confidence
        self.name = name or f"corrected_pass_rate({of})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        judged = self.judged.compute(trials, clusters=clusters)
        rates = self.rates
        # Examples whose every trial failed: failures the judge never saw.
        failed = (
            len({t.example_id for t in trials} - {t.example_id for t in trials if t.ok})
            if self.errors_as_failures
            else 0
        )
        total = judged.n + failed
        judged_share = judged.n / total if total else 0.0
        result = MetricResult(
            name=self.name,
            value=None,
            n=total,
            n_missing=judged.n_missing - failed,
            n_na=judged.n_na,
            confidence=self.confidence,
            details={
                "observed": judged.value,
                "judged_share": judged_share,
                "tpr": rates.tpr,
                "tnr": rates.tnr,
                "validation_run": rates.source,
            },
        )
        if judged.value is None or not judged.n:
            return result
        corrected = corrected_prevalence(judged.value, rates.tpr, rates.tnr)
        if corrected is None:
            return result
        result.value = judged_share * corrected
        if not (rates.n_positive and rates.n_negative):
            return result
        p, se = judged.value, judged.stderr
        n_eff = float(judged.n)
        if se is not None and se > 0.0 and 0.0 < p < 1.0:
            n_eff = min(n_eff, p * (1.0 - p) / (se * se))
        rng = random.Random(self.seed)  # noqa: S311
        draws: list[float] = []
        for _ in range(self.n_resamples):
            value = corrected_prevalence(
                _jeffreys(rng, p, n_eff),
                _jeffreys(rng, rates.tpr, rates.n_positive),
                _jeffreys(rng, rates.tnr, rates.n_negative),
            )
            if value is not None:
                draws.append(judged_share * value)
        if len(draws) >= 2:
            draws.sort()
            alpha = (1.0 - self.confidence) / 2.0
            mean = math.fsum(draws) / len(draws)
            result.stderr = math.sqrt(
                math.fsum((d - mean) ** 2 for d in draws) / (len(draws) - 1)
            )
            result.ci_low = percentile(draws, 100.0 * alpha)
            result.ci_high = percentile(draws, 100.0 * (1.0 - alpha))
        return result
