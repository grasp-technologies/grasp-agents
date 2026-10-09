"""
Judges: scorers backed by grasp-agents processors (an ``LLMAgent``, or any
:class:`Processor`, that reads an output and returns a verdict), and the
evaluations that validate a judge against labels and probe it with changed
outputs.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, Field

from grasp_agents.session_context import SessionContext

from ._util import user_code_hash
from .evaluation import DatasetSource, Evaluation
from .pairwise import PairwiseContext, PairwiseJudge, PairwiseVerdict
from .scorer import ScoreContext, Scorer, ScorerOutput
from .task import ProcessorSource, ProcessorTask, TrialContext
from .types import ComponentInfo, Example, JudgedOutput, Usage
from .validation import (
    LabelAgreement,
    Perturbation,
    ProbeCheck,
    ProbeTask,
    ScorerTask,
    probe_metrics,
    validation_metrics,
)

type Annotator = Literal["CODE", "LLM", "HUMAN"]


class UsageRecorder(Protocol):
    def __call__(
        self, usage: Usage, *, models: Mapping[str, Sequence[str]] | None = None
    ) -> None: ...


class JudgedPair[InT, OutT, RefT](BaseModel):
    """Two outputs for one example, in the order a pairwise judge sees them."""

    input: InT
    first: OutT
    second: OutT
    reference: RefT | None = None
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])


class _JudgeRunner:
    """Runs a judge processor once per call, isolated like a task's trial."""

    def __init__(
        self,
        judge: ProcessorSource[Any, Any],
        *,
        ctx_factory: Callable[[Example[Any, Any]], SessionContext[Any]] | None,
        input_mode: Literal["in_args", "chat"],
    ) -> None:
        self.task: ProcessorTask[Any, Any] = ProcessorTask(
            judge, ctx_factory=ctx_factory, input_mode=input_mode, capture_events=False
        )

    async def run(
        self,
        judge_input: Any,
        example: Example[Any, Any],
        repetition: int,
        record_usage: UsageRecorder,
    ) -> Any:
        call = TrialContext(run_id="", example=example, repetition=repetition)
        try:
            return await self.task.run(judge_input, call)
        finally:
            usage = call.usage
            if not usage.is_empty or call.models:
                record_usage(usage, models=call.models)

    def describe(
        self,
        *,
        name: str,
        version: str,
        config: Mapping[str, Any],
        annotator: Annotator,
        code: list[Any],
    ) -> ComponentInfo:
        info = self.task.describe()
        return ComponentInfo(
            name=name,
            kind=info.kind,
            version=version,
            config=dict(config),
            annotator=annotator,
            fingerprint=info.fingerprint,
            source=user_code_hash([*code, *self.task.source_objects()]),
        )


class ProcessorScorer[InT, OutT, RefT, JudgeInT, JudgeOutT](Scorer[InT, OutT, RefT]):
    """
    A scorer whose judgments come from a :class:`Processor` — typically an
    ``LLMAgent`` with a structured verdict.

    Each call runs a fresh copy of the judge (or a factory's product) in a
    session of its own, as :class:`ProcessorTask` runs tasks, and attributes
    the judge's model usage to the trial. ``to_input(ctx)`` builds the
    judge's input — by default a :class:`JudgedOutput` of the example and the
    output — and ``to_scores(verdict)`` turns its output into scores; without
    it the verdict must already be a score, a scalar or a mapping of them.

    The judge's models, settings, prompts, output schema and tools are part of
    its identity (``describe().fingerprint``), as is the code of ``to_input``,
    ``to_scores`` and custom processor classes: changing any of them makes
    rescoring run it again and invalidates earlier validation runs. A judge
    built by a factory has no fingerprint, so declare its model and prompt in
    ``version`` or ``config``. A judge that raises (including on a verdict
    that does not parse) is a scorer failure, which rescoring retries.
    """

    annotator: Annotator = "LLM"

    def __init__(
        self,
        judge: ProcessorSource[JudgeInT, JudgeOutT],
        *,
        name: str | None = None,
        version: str = "1",
        to_input: Callable[[ScoreContext[InT, OutT, RefT]], JudgeInT] | None = None,
        to_scores: Callable[[JudgeOutT], ScorerOutput] | None = None,
        config: Mapping[str, Any] | None = None,
        ctx_factory: Callable[[Example[Any, Any]], SessionContext[Any]] | None = None,
        input_mode: Literal["in_args", "chat"] = "in_args",
        scores_errors: bool = False,
        annotator: Annotator = "LLM",
    ) -> None:
        self._runner = _JudgeRunner(
            judge, ctx_factory=ctx_factory, input_mode=input_mode
        )
        super().__init__(name=name or self._runner.task.name, version=version)
        self._to_input = to_input
        self._to_scores = to_scores
        self._config = dict(config or {})
        self.scores_errors = scores_errors
        self.annotator = annotator

    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def describe(self) -> ComponentInfo:
        return self._runner.describe(
            name=self.name,
            version=self.version,
            config=self.config(),
            annotator=self.annotator,
            code=[type(self), self._to_input, self._to_scores],
        )

    async def score(self, ctx: ScoreContext[InT, OutT, RefT]) -> ScorerOutput:
        judge_input: Any = (
            self._to_input(ctx)
            if self._to_input is not None
            else JudgedOutput[Any, Any, Any](
                input=ctx.input,
                output=ctx.output,
                reference=ctx.reference,
                metadata=ctx.metadata,
            )
        )
        verdict = await self._runner.run(
            judge_input, ctx.example, ctx.trial.repetition, ctx.record_usage
        )
        if self._to_scores is not None:
            return self._to_scores(verdict)
        return cast("ScorerOutput", verdict)


class ProcessorPairwiseJudge[InT, OutT, RefT, JudgeInT, JudgeOutT](
    PairwiseJudge[InT, OutT, RefT]
):
    """
    A pairwise judge backed by a :class:`Processor`, run in isolation per
    call like :class:`ProcessorScorer`. ``to_input(ctx)`` builds its input
    (by default a :class:`JudgedPair`) and ``to_verdict(output)`` its
    :class:`PairwiseVerdict` (by default the output must be one).
    """

    def __init__(
        self,
        judge: ProcessorSource[JudgeInT, JudgeOutT],
        *,
        name: str | None = None,
        version: str = "1",
        to_input: Callable[[PairwiseContext[InT, OutT, RefT]], JudgeInT] | None = None,
        to_verdict: Callable[[JudgeOutT], PairwiseVerdict] | None = None,
        config: Mapping[str, Any] | None = None,
        ctx_factory: Callable[[Example[Any, Any]], SessionContext[Any]] | None = None,
        input_mode: Literal["in_args", "chat"] = "in_args",
        annotator: Annotator = "LLM",
    ) -> None:
        self._runner = _JudgeRunner(
            judge, ctx_factory=ctx_factory, input_mode=input_mode
        )
        super().__init__(name=name or self._runner.task.name, version=version)
        self._to_input = to_input
        self._to_verdict = to_verdict
        self._config = dict(config or {})
        self.annotator = annotator

    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def describe(self) -> ComponentInfo:
        return self._runner.describe(
            name=self.name,
            version=self.version,
            config=self.config(),
            annotator=self.annotator,
            code=[type(self), self._to_input, self._to_verdict],
        )

    async def judge(self, ctx: PairwiseContext[InT, OutT, RefT]) -> PairwiseVerdict:
        judge_input: Any = (
            self._to_input(ctx)
            if self._to_input is not None
            else JudgedPair[Any, Any, Any](
                input=ctx.input,
                first=ctx.first,
                second=ctx.second,
                reference=ctx.reference,
                metadata=ctx.example.metadata,
            )
        )
        output = await self._runner.run(
            judge_input, ctx.example, ctx.repetition, ctx.record_usage
        )
        if self._to_verdict is not None:
            return self._to_verdict(output)
        if not isinstance(output, PairwiseVerdict):
            raise TypeError(
                f"Pairwise judge {self.name!r} returned {type(output).__name__}; "
                "pass to_verdict= to turn it into a PairwiseVerdict"
            )
        return output


# --- Evaluations of a judge ---


def _score_names(
    judge: Scorer[Any, Any, Any], scores: str | Sequence[str] | None
) -> list[str]:
    if isinstance(scores, str):
        return [scores]
    return list(scores) if scores else [judge.name]


def judge_validation(
    judge: Scorer[Any, Any, Any],
    labels: DatasetSource,
    *,
    scores: str | Sequence[str] | None = None,
    input_type: Any = Any,
    output_type: Any = Any,
    reference_type: Any = Any,
    threshold: float | None = None,
    positive: str | bool | None = True,
    negative: str | bool | None = False,
    repetitions: int = 1,
    name: str | None = None,
    description: str | None = None,
    sealed_splits: Sequence[str] = ("test",),
    cluster_by: str | None = "example_id",
    group_by: Sequence[str] = ("split",),
    concurrency: int = 4,
    timeout_s: float | None = None,
    max_cost_usd: float | None = None,
    tags: Sequence[str] = (),
) -> Evaluation:
    """
    An :class:`Evaluation` measuring how well ``judge`` agrees with labels.

    ``labels`` is a dataset (a file, a ``phoenix:`` reference or a loader) of
    :class:`JudgedOutput` inputs — an output with the example it answers, as
    ``grasp-evals labels sample`` writes them — whose reference is the
    correct verdict: ``{score: value}``, or one value for the score its
    ``metadata.score`` names. ``scores`` are the judge's score names to
    validate (by default the one named after it); ``input_type``,
    ``output_type`` and ``reference_type`` are those of the evaluation it
    judges, so it sees typed values. Verdicts and labels compare as described
    in :func:`comparable` (pass/fail words and 0/1 read as booleans; numbers
    at or above ``threshold`` pass).

    Each run applies the judge, exactly as it is now, to every labeled output
    (``repetitions`` times to measure its self-consistency) and reports per
    score the accuracy, Cohen's κ, TPR and TNR (``positive`` / ``negative``
    are the passing and failing labels; ``None`` for labels without a pass)
    and the confusion matrix, per split (``group_by``), with intervals
    clustered on the original example (several labeled outputs of one example
    are not independent). Iterate on the dev split — ``show --failures``
    lists the disagreements with the judge's explanation, ``compare`` pairs
    two judge versions — and evaluate the sealed test split once, at the end:
    that is the run a :class:`ValidationGate` reads.
    """
    names = _score_names(judge, scores)
    return Evaluation(
        name=name or f"{judge.name}-validation",
        description=description or f"Agreement of the {judge.name} judge with labels",
        task=ScorerTask(
            judge,
            input_type=input_type,
            output_type=output_type,
            reference_type=reference_type,
        ),
        dataset=labels,
        scorers=[
            LabelAgreement(
                names, threshold=threshold, positive=positive, negative=negative
            )
        ],
        metrics=validation_metrics(
            names, positive=positive, negative=negative, repetitions=repetitions
        ),
        repetitions=repetitions,
        concurrency=concurrency,
        timeout_s=timeout_s,
        max_cost_usd=max_cost_usd,
        group_by=group_by,
        cluster_by=cluster_by,
        sealed_splits=sealed_splits,
        tags=["judge-validation", *tags],
    )


def judge_probes(
    judge: Scorer[Any, Any, Any],
    items: DatasetSource,
    perturbations: Sequence[Perturbation],
    *,
    scores: str | Sequence[str] | None = None,
    input_type: Any = Any,
    output_type: Any = Any,
    reference_type: Any = Any,
    threshold: float | None = None,
    repetitions: int = 1,
    name: str | None = None,
    description: str | None = None,
    sealed_splits: Sequence[str] = (),
    cluster_by: str | None = "example_id",
    group_by: Sequence[str] = (),
    concurrency: int = 4,
    timeout_s: float | None = None,
    max_cost_usd: float | None = None,
    tags: Sequence[str] = (),
) -> Evaluation:
    """
    An :class:`Evaluation` probing ``judge`` with changed outputs: every
    :class:`Perturbation` must move the verdict as it expects — a degradation
    lowers it (sensitivity), an irrelevant change leaves it (invariance). No
    labels are needed; ``items`` holds :class:`JudgedOutput` inputs (e.g.
    :func:`judged_outputs` of a run), best ones whose outputs are good, so a
    degradation has room to show. Reports one rate per score and
    perturbation. Probing a labels dataset shows its examples one by one, so
    seal its test split (``sealed_splits=("test",)``) or probe the dev split
    only.
    """
    names = _score_names(judge, scores)
    return Evaluation(
        name=name or f"{judge.name}-probes",
        description=description
        or f"Sensitivity and invariance of the {judge.name} judge",
        task=ProbeTask(
            judge,
            perturbations,
            input_type=input_type,
            output_type=output_type,
            reference_type=reference_type,
        ),
        dataset=items,
        scorers=[ProbeCheck(perturbations, names, threshold=threshold)],
        metrics=probe_metrics(perturbations, names),
        repetitions=repetitions,
        concurrency=concurrency,
        timeout_s=timeout_s,
        max_cost_usd=max_cost_usd,
        group_by=group_by,
        cluster_by=cluster_by,
        sealed_splits=sealed_splits,
        tags=["judge-probes", *tags],
    )
