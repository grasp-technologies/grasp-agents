"""
Pairwise (A/B) judging of two runs on the same examples.

A pairwise comparison is itself an :class:`EvaluationRun`: its trials hold
both arms' outputs and its scores come from an order-swapped pairwise judge,
so it is stored, reported, compared and pushed like any other run.
"""

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import starmap
from typing import Any, Literal, override

from pydantic import BaseModel, TypeAdapter

from ._execution import (
    Executor,
    ProgressCallback,
    TrialKey,
    capture_provenance,
    config_hash,
    load_run,
    rehydrate_output,
    resolve_store,
    retype_examples,
    run_all,
)
from ._util import new_run_id, qualified_name, short_hash, utc_now
from .evaluator import EvalContext, Evaluator, EvaluatorOutput
from .metrics import Metric, PassRate
from .stats import binomial_two_sided, mean_estimate
from .store import RunStore
from .types import (
    ComponentInfo,
    EvaluationRun,
    Example,
    MetricResult,
    RunConfig,
    Score,
    Trial,
)

type Winner = Literal["first", "second", "tie"]


class PairwiseVerdict(BaseModel):
    winner: Winner
    explanation: str | None = None


@dataclass(frozen=True)
class PairwiseContext[InT, OutT, RefT]:
    """Two outputs for one example, in the order the judge should see them."""

    example: Example[InT, RefT]
    first: OutT
    second: OutT

    @property
    def input(self) -> InT:
        return self.example.input

    @property
    def reference(self) -> RefT | None:
        return self.example.reference


class PairwiseJudge[InT, OutT, RefT](ABC):
    """
    Decides which of two outputs is better. Judges see the outputs only as
    "first" and "second" — never which arm produced them — and
    :class:`OrderSwapped` asks in both orders to expose position bias.
    """

    name: str = ""
    version: str = "1"
    annotator: Literal["CODE", "LLM", "HUMAN"] = "LLM"

    def __init__(self, name: str | None = None, version: str | None = None) -> None:
        if name is not None:
            self.name = name
        if version is not None:
            self.version = version
        if not self.name:
            self.name = type(self).__name__.lower()

    def config(self) -> dict[str, Any]:
        return {}

    def describe(self) -> ComponentInfo:
        return ComponentInfo(
            name=self.name,
            kind=qualified_name(type(self)),
            version=self.version,
            config=self.config(),
            annotator=self.annotator,
        )

    @abstractmethod
    def judge(
        self, ctx: PairwiseContext[InT, OutT, RefT]
    ) -> PairwiseVerdict | Awaitable[PairwiseVerdict]: ...


class FunctionPairwiseJudge[InT, OutT, RefT](PairwiseJudge[InT, OutT, RefT]):
    def __init__(
        self,
        fn: Callable[
            [PairwiseContext[InT, OutT, RefT]],
            PairwiseVerdict | Awaitable[PairwiseVerdict],
        ],
        *,
        name: str | None = None,
        version: str = "1",
        annotator: Literal["CODE", "LLM", "HUMAN"] = "LLM",
    ) -> None:
        super().__init__(name=name or fn.__name__, version=version)
        self._fn = fn
        self.annotator = annotator

    def judge(
        self, ctx: PairwiseContext[InT, OutT, RefT]
    ) -> PairwiseVerdict | Awaitable[PairwiseVerdict]:
        return self._fn(ctx)


async def _ask(
    judge: PairwiseJudge[Any, Any, Any], ctx: PairwiseContext[Any, Any, Any]
) -> PairwiseVerdict:
    verdict = judge.judge(ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    return verdict


_FORWARD: dict[Winner, str] = {"first": "base", "second": "candidate", "tie": "tie"}
_BACKWARD: dict[Winner, str] = {"first": "candidate", "second": "base", "tie": "tie"}
_PREFERS_CANDIDATE = {"candidate": 1.0, "base": 0.0, "tie": 0.5, "inconsistent": 0.5}


class OrderSwapped[InT, OutT, RefT](Evaluator[InT, Mapping[str, OutT], RefT]):
    """
    Evaluates a pair ``{"base": ..., "candidate": ...}`` with a pairwise judge,
    once in each order. Produces ``<name>.winner`` (base / candidate / tie /
    inconsistent), ``<name>.prefers_candidate`` (1 / ½ / 0; inconsistent
    verdicts count as ties) and ``<name>.position_consistent``.
    """

    def __init__(
        self,
        judge: PairwiseJudge[InT, OutT, RefT],
        *,
        both_orders: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or judge.name, version=judge.version)
        self.judge = judge
        self.both_orders = both_orders
        self.annotator = judge.annotator

    def config(self) -> dict[str, Any]:
        return {
            "judge": self.judge.describe().model_dump(),
            "both_orders": self.both_orders,
        }

    async def evaluate(
        self, ctx: EvalContext[InT, Mapping[str, OutT], RefT]
    ) -> EvaluatorOutput:
        base, candidate = ctx.output["base"], ctx.output["candidate"]
        forward_ctx = PairwiseContext(example=ctx.example, first=base, second=candidate)
        if self.both_orders:
            backward_ctx = PairwiseContext(
                example=ctx.example, first=candidate, second=base
            )
            forward, backward = await asyncio.gather(
                _ask(self.judge, forward_ctx), _ask(self.judge, backward_ctx)
            )
            first, second = _FORWARD[forward.winner], _BACKWARD[backward.winner]
            consistent = first == second
            winner = first if consistent else "inconsistent"
            explanation = (
                f"base first → {first}: {forward.explanation or ''}\n"
                f"candidate first → {second}: {backward.explanation or ''}"
            )
        else:
            forward = await _ask(self.judge, forward_ctx)
            winner, consistent = _FORWARD[forward.winner], None
            explanation = forward.explanation
        scores = [
            Score(name=f"{self.name}.winner", value=winner, explanation=explanation),
            Score(
                name=f"{self.name}.prefers_candidate",
                value=_PREFERS_CANDIDATE[winner],
                explanation=explanation,
            ),
        ]
        if consistent is not None:
            scores.append(
                Score(name=f"{self.name}.position_consistent", value=consistent)
            )
        return scores


class WinRate(Metric):
    """
    Candidate win rate from an :class:`OrderSwapped` judge (ties and
    inconsistent verdicts count as half), with a CI over examples and an
    exact sign test of wins against losses.
    """

    def __init__(self, judge_name: str, *, name: str | None = None) -> None:
        self.judge_name = judge_name
        self.name = name or f"win_rate({judge_name})"

    @override
    def compute(
        self,
        trials: Sequence[Trial],
        *,
        clusters: Mapping[str, Hashable] | None = None,
    ) -> MetricResult:
        per_example: dict[str, list[float]] = {}
        counts = {"candidate": 0, "base": 0, "tie": 0, "inconsistent": 0}
        missing = 0
        for trial in trials:
            score = trial.score(f"{self.judge_name}.winner")
            if score is None or not isinstance(score.value, str):
                missing += 1
                continue
            counts[score.value] = counts.get(score.value, 0) + 1
            per_example.setdefault(trial.example_id, []).append(
                _PREFERS_CANDIDATE.get(score.value, 0.5)
            )
        keys = list(per_example)
        values = [sum(v) / len(v) for v in per_example.values()]
        estimate = mean_estimate(
            values,
            clusters=[clusters.get(k, k) for k in keys]
            if clusters is not None
            else None,
        )
        return MetricResult(
            name=self.name,
            value=estimate.value if values else None,
            n=estimate.n,
            n_missing=missing,
            stderr=estimate.stderr,
            ci_low=None if estimate.ci_low is None else max(0.0, estimate.ci_low),
            ci_high=None if estimate.ci_high is None else min(1.0, estimate.ci_high),
            details={
                **counts,
                "sign_test_p": binomial_two_sided(
                    counts["candidate"], counts["candidate"] + counts["base"]
                ),
            },
        )


async def pairwise(
    base: "str | EvaluationRun",
    candidate: "str | EvaluationRun",
    judge: PairwiseJudge[Any, Any, Any],
    metrics: Sequence[Metric] | None = None,
    *,
    both_orders: bool = True,
    input_type: Any = Any,
    reference_type: Any = Any,
    output_type: Any = Any,
    name: str | None = None,
    concurrency: int = 4,
    store: RunStore | None = None,
    persist: bool = True,
    progress: ProgressCallback | None = None,
) -> EvaluationRun:
    """
    Judge ``candidate`` against ``base`` on every example both ran
    successfully with unchanged content (pairing by example and repetition).
    Stored examples and outputs are re-validated as the given types before
    the judge sees them.
    """
    run_store = resolve_store(store, persist)
    base_run = load_run(run_store, base)
    candidate_run = load_run(run_store, candidate)
    candidate_trials = {t.key: t for t in candidate_run.trials}
    pairs: list[tuple[Trial, Trial]] = []
    for trial in base_run.trials:
        other = candidate_trials.get(trial.key)
        if (
            other is not None
            and trial.ok
            and other.ok
            and trial.example_hash == other.example_hash
        ):
            pairs.append((trial, other))
    evaluator = OrderSwapped(judge, both_orders=both_orders)
    evaluator_info = evaluator.describe()
    paired_ids = sorted({b.example_id for b, _ in pairs})
    dataset = base_run.dataset.model_copy(
        update={
            "selection": [
                *base_run.dataset.selection,
                f"paired_with={candidate_run.id}",
            ],
            "selected_fingerprint": short_hash(
                *(f"{b.example_id}:{b.example_hash}" for b, _ in pairs)
            ),
            "selected_size": len(paired_ids),
        }
    )
    task_info = ComponentInfo(
        name=f"{candidate_run.name} vs {base_run.name}",
        kind="pairwise",
        config={
            "base_run": base_run.id,
            "candidate_run": candidate_run.id,
            "base_task": base_run.task.model_dump(),
            "candidate_task": candidate_run.task.model_dump(),
        },
    )
    run_name = name or f"{candidate_run.name}-vs-{base_run.name}"
    paired_example_ids = set(paired_ids)
    run = EvaluationRun(
        id=new_run_id(run_name),
        name=run_name,
        kind="pairwise",
        created_at=utc_now(),
        dataset=dataset,
        task=task_info,
        evaluators=[evaluator_info],
        config=RunConfig(
            repetitions=base_run.config.repetitions,
            concurrency=concurrency,
            group_by=base_run.config.group_by,
            cluster_by=base_run.config.cluster_by,
            sealed_splits=base_run.config.sealed_splits,
        ),
        provenance=capture_provenance(),
        config_hash=config_hash(
            task_info, [evaluator_info], dataset, base_run.config.repetitions
        ),
        examples=[e for e in base_run.examples if e.id in paired_example_ids],
    )
    if run_store is not None:
        run_store.create(run)
    examples = {
        e.id: e
        for e in retype_examples(
            base_run.examples, input_type=input_type, reference_type=reference_type
        )
    }
    order: list[TrialKey] = [b.key for b, _ in pairs]
    executor = Executor(
        run,
        store=run_store,
        evaluators=[evaluator],
        progress=progress,
        total=len(order),
    )
    adapter: TypeAdapter[Any] = TypeAdapter(output_type)
    semaphore = asyncio.Semaphore(concurrency)

    async def work(base_trial: Trial, candidate_trial: Trial) -> None:
        async with semaphore:
            trial = Trial(
                example_id=base_trial.example_id,
                repetition=base_trial.repetition,
                example_hash=base_trial.example_hash,
                output={"base": base_trial.output, "candidate": candidate_trial.output},
                started_at=utc_now(),
                duration_s=0.0,
                sealed=base_trial.sealed or candidate_trial.sealed,
            )
            example = examples.get(trial.example_id)
            if example is not None:
                output = {
                    "base": rehydrate_output(adapter, base_trial.output),
                    "candidate": rehydrate_output(adapter, candidate_trial.output),
                }
                await executor.score(trial, example, output)
            await executor.record(trial)

    default_metrics: list[Metric] = [WinRate(evaluator.name)]
    if both_orders:
        default_metrics.append(PassRate(f"{evaluator.name}.position_consistent"))
    return await run_all(
        executor,
        list(starmap(work, pairs)),
        metrics=metrics if metrics is not None else default_metrics,
        order=order,
        max_error_rate=None,
    )
