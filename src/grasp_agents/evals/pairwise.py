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

from pydantic import BaseModel

from ._execution import (
    Executor,
    ProgressCallback,
    TrialKey,
    capture_provenance,
    config_hash,
    load_run,
    output_adapter,
    rehydrate_output,
    resolve_store,
    retype_examples,
    run_all,
)
from ._util import code_hash, new_run_id, qualified_name, short_hash, utc_now
from .metrics import Metric, PassRate
from .scorer import (
    EvalContext,
    Scorer,
    ScorerOutput,
    call_off_loop,
    snake_case,
)
from .stats import bounded_mean_estimate, sign_test
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
            self.name = snake_case(type(self).__name__)

    def config(self) -> dict[str, Any]:
        return {}

    def describe(self) -> ComponentInfo:
        return ComponentInfo(
            name=self.name,
            kind=qualified_name(type(self)),
            version=self.version,
            config=self.config(),
            annotator=self.annotator,
            source=code_hash(type(self)),
        )

    @abstractmethod
    def judge(
        self, ctx: PairwiseContext[InT, OutT, RefT]
    ) -> PairwiseVerdict | Awaitable[PairwiseVerdict]: ...


type JudgeFn[InT, OutT, RefT] = Callable[
    [PairwiseContext[InT, OutT, RefT]], PairwiseVerdict | Awaitable[PairwiseVerdict]
]


class FunctionPairwiseJudge[InT, OutT, RefT](PairwiseJudge[InT, OutT, RefT]):
    def __init__(
        self,
        fn: JudgeFn[InT, OutT, RefT],
        *,
        name: str | None = None,
        version: str = "1",
        annotator: Literal["CODE", "LLM", "HUMAN"] = "LLM",
    ) -> None:
        super().__init__(name=name or fn.__name__, version=version)
        self._fn = fn
        self.annotator = annotator

    def describe(self) -> ComponentInfo:
        info = super().describe()
        info.kind = qualified_name(self._fn)
        info.source = code_hash(self._fn)
        return info

    @property
    def fn(self) -> JudgeFn[InT, OutT, RefT]:
        return self._fn

    def judge(
        self, ctx: PairwiseContext[InT, OutT, RefT]
    ) -> PairwiseVerdict | Awaitable[PairwiseVerdict]:
        return self._fn(ctx)


async def _ask(
    judge: PairwiseJudge[Any, Any, Any], ctx: PairwiseContext[Any, Any, Any]
) -> PairwiseVerdict:
    target = judge.fn if isinstance(judge, FunctionPairwiseJudge) else judge.judge
    return await call_off_loop(
        judge.judge, ctx, is_async=inspect.iscoroutinefunction(target)
    )


_FORWARD: dict[Winner, str] = {"first": "base", "second": "candidate", "tie": "tie"}
_BACKWARD: dict[Winner, str] = {"first": "candidate", "second": "base", "tie": "tie"}
_PREFERS_CANDIDATE = {"candidate": 1.0, "base": 0.0, "tie": 0.5, "inconsistent": 0.5}


class OrderSwapped[InT, OutT, RefT](Scorer[InT, Mapping[str, OutT], RefT]):
    """
    Scores a pair ``{"base": ..., "candidate": ...}`` with a pairwise judge,
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

    async def score(
        self, ctx: EvalContext[InT, Mapping[str, OutT], RefT]
    ) -> ScorerOutput:
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
    inconsistent verdicts count as half). Each example's repetitions are
    averaged first: the interval and the exact sign test (examples the
    candidate wins on average against those it loses) treat examples, not
    judgments, as the independent units. Counts in ``details`` are per
    judgment.
    """

    def __init__(
        self, judge_name: str, *, name: str | None = None, confidence: float = 0.95
    ) -> None:
        self.judge_name = judge_name
        self.confidence = confidence
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
        judged: set[str] = set()
        for trial in trials:
            score = trial.score(f"{self.judge_name}.winner")
            if score is None or not isinstance(score.value, str):
                continue
            judged.add(trial.example_id)
            counts[score.value] = counts.get(score.value, 0) + 1
            per_example.setdefault(trial.example_id, []).append(
                _PREFERS_CANDIDATE.get(score.value, 0.5)
            )
        missing = len({t.example_id for t in trials} - judged)
        keys = list(per_example)
        values = [sum(v) / len(v) for v in per_example.values()]
        estimate = bounded_mean_estimate(
            values,
            clusters=[clusters.get(k, k) for k in keys]
            if clusters is not None
            else None,
            observations=sum(len(v) for v in per_example.values()),
            confidence=self.confidence,
        )
        return MetricResult(
            name=self.name,
            value=estimate.value if values else None,
            n=estimate.n,
            n_missing=missing,
            stderr=estimate.stderr,
            ci_low=estimate.ci_low,
            ci_high=estimate.ci_high,
            confidence=self.confidence,
            details={
                **counts,
                "examples_won": sum(1 for v in values if v > 0.5),
                "examples_lost": sum(1 for v in values if v < 0.5),
                "sign_test_p": sign_test([v - 0.5 for v in values]),
            },
        )


def _decided_by_error(judge_name: str, *, base_ok: bool) -> list[Score]:
    winner = "base" if base_ok else "candidate"
    loser = "candidate" if base_ok else "base"
    explanation = f"the {loser}'s task failed on this example"
    return [
        Score(name=f"{judge_name}.winner", value=winner, explanation=explanation),
        Score(
            name=f"{judge_name}.prefers_candidate",
            value=_PREFERS_CANDIDATE[winner],
            explanation=explanation,
        ),
    ]


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
    Judge ``candidate`` against ``base`` on every example both ran with
    unchanged content (pairing by example and repetition). When only one arm's
    task failed, the other wins that pair without asking the judge; pairs
    where both failed are skipped. Stored examples and outputs are
    re-validated as the given types before the judge sees them.
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
            and (trial.ok or other.ok)
            and trial.example_hash == other.example_hash
        ):
            pairs.append((trial, other))
    if not pairs:
        raise ValueError(
            f"Runs {base_run.id} and {candidate_run.id} share no trial with unchanged "
            "content that either ran successfully: nothing to judge"
        )
    examples = {
        e.id: e
        for e in retype_examples(
            base_run.examples, input_type=input_type, reference_type=reference_type
        )
    }
    adapter = output_adapter(output_type)
    scorer = OrderSwapped(judge, both_orders=both_orders)
    scorer_info = scorer.describe()
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
        scorers=[scorer_info],
        config=RunConfig(
            repetitions=base_run.config.repetitions,
            concurrency=concurrency,
            group_by=base_run.config.group_by,
            cluster_by=base_run.config.cluster_by,
            sealed_splits=base_run.config.sealed_splits,
        ),
        provenance=capture_provenance(),
        config_hash=config_hash(
            task_info, [scorer_info], dataset, base_run.config.repetitions
        ),
        examples=[e for e in base_run.examples if e.id in paired_example_ids],
    )
    if run_store is not None:
        run_store.create(run)
    order: list[TrialKey] = [b.key for b, _ in pairs]
    executor = Executor(
        run,
        store=run_store,
        scorers=[scorer],
        progress=progress,
        total=len(order),
    )
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
            if not (base_trial.ok and candidate_trial.ok):
                trial.scores.extend(
                    _decided_by_error(scorer.name, base_ok=base_trial.ok)
                )
                trial.scorers_run.append(scorer.name)
            elif example is not None:
                output = {
                    "base": rehydrate_output(adapter, base_trial.output),
                    "candidate": rehydrate_output(adapter, candidate_trial.output),
                }
                await executor.score(trial, example, output)
            await executor.record(trial)

    default_metrics: list[Metric] = [WinRate(scorer.name)]
    if both_orders:
        default_metrics.append(PassRate(f"{scorer.name}.position_consistent"))
    return await run_all(
        executor,
        list(starmap(work, pairs)),
        metrics=metrics if metrics is not None else default_metrics,
        order=order,
    )
