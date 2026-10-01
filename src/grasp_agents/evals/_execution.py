import asyncio
import logging
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import TypeAdapter, ValidationError

from grasp_agents.session_context import SessionContext
from grasp_agents.types.events import Event

from ._util import canonical_json, git_state, short_hash, to_jsonable, utc_now
from .evaluator import EvalContext, Evaluator, run_evaluator
from .metrics import Metric, compute_metrics
from .report import render_run_markdown
from .store import LocalRunStore, RunStore
from .task import Task, TrialContext
from .types import (
    ComponentInfo,
    DatasetRef,
    ErrorInfo,
    EvaluationRun,
    EvaluatorFailure,
    Example,
    Provenance,
    RunCounts,
    RunStatus,
    Score,
    Trial,
    Usage,
)

logger = logging.getLogger(__name__)

_tracer = trace.get_tracer("grasp_agents.evals")

type TrialKey = tuple[str, int]


@dataclass(frozen=True)
class TrialProgress:
    run_id: str
    trial: Trial
    done: int
    total: int
    task_errors: int
    cost_usd: float


type ProgressCallback = Callable[[TrialProgress], None]


def capture_provenance(
    observed_models: Mapping[str, list[str]] | None = None,
) -> Provenance:
    commit, branch, dirty, diff_hash = git_state(Path.cwd())
    try:
        grasp_version: str | None = package_version("grasp_agents")
    except PackageNotFoundError:
        grasp_version = None
    return Provenance(
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        git_diff_hash=diff_hash,
        python=platform.python_version(),
        grasp_agents=grasp_version,
        observed_models=dict(observed_models or {}),
    )


def config_hash(
    task: ComponentInfo,
    evaluators: Sequence[ComponentInfo],
    dataset: DatasetRef,
    repetitions: int,
) -> str:
    """Identity of what a run measures: same hash ⇔ like-for-like runs."""
    return short_hash(
        canonical_json(
            {
                "task": task,
                "evaluators": sorted(evaluators, key=lambda e: e.name),
                "dataset": dataset.selected_fingerprint,
                "repetitions": repetitions,
            }
        )
    )


def retype_examples(
    examples: Sequence[Example[Any, Any]],
    *,
    input_type: Any = Any,
    reference_type: Any = Any,
) -> list[Example[Any, Any]]:
    """Re-validate stored examples (plain JSON after a reload) as typed ones."""
    if input_type is Any and reference_type is Any:
        return list(examples)
    inputs: TypeAdapter[Any] = TypeAdapter(input_type)
    references: TypeAdapter[Any] = TypeAdapter(reference_type)
    return [
        e.model_copy(
            update={
                "input": inputs.validate_python(e.input),
                "reference": None
                if e.reference is None
                else references.validate_python(e.reference),
            }
        )
        for e in examples
    ]


def rehydrate_output(adapter: TypeAdapter[Any], output: Any) -> Any:
    """Typed form of a stored output; the stored JSON if it no longer validates."""
    try:
        return adapter.validate_python(output)
    except ValidationError:
        logger.warning("Stored output no longer matches the task's output type")
        return output


def _trace_id(span: trace.Span) -> str | None:
    context = span.get_span_context()
    if not span.is_recording() or not context.trace_id:
        return None
    return format(context.trace_id, "032x")


async def execute_trial(
    task: Task[Any, Any],
    run: EvaluationRun,
    example: Example[Any, Any],
    repetition: int,
    *,
    timeout_s: float | None,
    sealed_splits: frozenset[str],
    capture_events: bool,
) -> tuple[Trial, Any, list[Event[Any]], SessionContext[Any] | None]:
    """Run the task once; failures are recorded on the trial, never raised."""
    trial_ctx = TrialContext(run_id=run.id, example=example, repetition=repetition)
    started_at = utc_now()
    start = time.perf_counter()
    output: Any = None
    error: ErrorInfo | None = None
    attributes = {
        "grasp.eval.run_id": run.id,
        "grasp.eval.name": run.name,
        "grasp.eval.example_id": example.id,
        "grasp.eval.repetition": repetition,
        "openinference.span.kind": "CHAIN",
    }
    with _tracer.start_as_current_span(
        f"eval.trial[{example.id}#{repetition}]", attributes=attributes
    ) as span:
        scope = asyncio.timeout(timeout_s)
        try:
            async with scope:
                output = await task.run(example.input, trial_ctx)
        except TimeoutError as exc:
            error = (
                ErrorInfo(
                    type="TimeoutError",
                    message=f"TimeoutError: trial exceeded its {timeout_s:g}s timeout",
                )
                if scope.expired() and timeout_s is not None
                else ErrorInfo.from_exception(exc)
            )
        except Exception as exc:
            error = ErrorInfo.from_exception(exc)
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, error.message))
        trace_id = _trace_id(span)
    trial = Trial(
        example_id=example.id,
        repetition=repetition,
        example_hash=example.content_hash(),
        output=None if error is not None else to_jsonable(output),
        error=error,
        started_at=started_at,
        duration_s=time.perf_counter() - start,
        usage=trial_ctx.usage,
        usage_by_agent=trial_ctx.usage_by_agent,
        models=trial_ctx.models,
        trace_id=trace_id,
        measurements=trial_ctx.measurements,
        sealed=bool(sealed_splits.intersection(example.splits)),
    )
    events = trial_ctx.events if capture_events else []
    return trial, output, events, trial_ctx.session


class Executor:
    """Scores, persists and finalizes the trials of one run."""

    def __init__(
        self,
        run: EvaluationRun,
        *,
        store: RunStore | None,
        evaluators: Sequence[Evaluator[Any, Any, Any]],
        capture_events: bool = True,
        max_cost_usd: float | None = None,
        progress: ProgressCallback | None = None,
        total: int = 0,
    ) -> None:
        self.run = run
        self.store = store
        self.evaluators = list(evaluators)
        self.capture_events = capture_events
        self.max_cost_usd = max_cost_usd
        self.progress = progress
        self.total = total
        self.budget_exhausted = False
        self._lock = asyncio.Lock()
        self._trials: dict[TrialKey, Trial] = {t.key: t for t in run.trials}
        self._costs: dict[TrialKey, float] = {
            t.key: t.total_usage.cost_usd or 0.0 for t in run.trials
        }

    @property
    def spent_usd(self) -> float:
        return sum(self._costs.values())

    def get(self, key: TrialKey) -> Trial | None:
        return self._trials.get(key)

    def over_budget(self) -> bool:
        if self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd:
            self.budget_exhausted = True
        return self.budget_exhausted

    async def score(
        self,
        trial: Trial,
        example: Example[Any, Any],
        output: Any,
        events: Sequence[Event[Any]] = (),
        session: SessionContext[Any] | None = None,
    ) -> None:
        pending = [
            e
            for e in self.evaluators
            if e.name not in trial.evaluated and (trial.ok or e.evaluates_errors)
        ]
        if not pending:
            return
        ctx = EvalContext(
            example=example, output=output, trial=trial, events=events, session=session
        )
        outcomes = await asyncio.gather(*(self._run_one(e, ctx) for e in pending))
        names = {s.name for s in trial.scores}
        for evaluator, outcome in zip(pending, outcomes, strict=True):
            trial.evaluator_failures = [
                f for f in trial.evaluator_failures if f.evaluator != evaluator.name
            ]
            if isinstance(outcome, ErrorInfo):
                trial.evaluator_failures.append(
                    EvaluatorFailure(evaluator=evaluator.name, error=outcome)
                )
                continue
            clashes = sorted({s.name for s in outcome} & names)
            if clashes:
                trial.evaluator_failures.append(
                    EvaluatorFailure(
                        evaluator=evaluator.name,
                        error=ErrorInfo(
                            type="DuplicateScoreName",
                            message=f"DuplicateScoreName: already produced {clashes}",
                        ),
                    )
                )
                continue
            trial.scores.extend(outcome)
            names.update(s.name for s in outcome)
            trial.evaluated.append(evaluator.name)
        if ctx.usage:
            trial.evaluator_usage = sum(ctx.usage, trial.evaluator_usage)

    @staticmethod
    async def _run_one(
        evaluator: Evaluator[Any, Any, Any], ctx: EvalContext[Any, Any, Any]
    ) -> list[Score] | ErrorInfo:
        try:
            return await run_evaluator(evaluator, ctx)
        except Exception as exc:
            logger.debug(
                "Evaluator %s failed on example %s: %s",
                evaluator.name,
                ctx.example.id,
                exc,
            )
            return ErrorInfo.from_exception(exc)

    async def record(
        self, trial: Trial, events: Sequence[Event[Any]] | None = None
    ) -> None:
        async with self._lock:
            self._trials[trial.key] = trial
            self._costs[trial.key] = trial.total_usage.cost_usd or 0.0
            if self.store is not None:
                self.store.append_trial(
                    self.run.id, trial, events if self.capture_events else None
                )
            if self.progress is not None:
                self.progress(
                    TrialProgress(
                        run_id=self.run.id,
                        trial=trial,
                        done=len(self._trials),
                        total=self.total,
                        task_errors=sum(1 for t in self._trials.values() if not t.ok),
                        cost_usd=self.spent_usd,
                    )
                )

    def finalize(
        self,
        status: RunStatus,
        *,
        metrics: Sequence[Metric] | None,
        order: Sequence[TrialKey],
        max_error_rate: float | None,
    ) -> EvaluationRun:
        run = self.run
        index = {key: i for i, key in enumerate(order)}
        run.trials = sorted(
            self._trials.values(), key=lambda t: index.get(t.key, len(index))
        )
        trials = run.trials
        run.counts = RunCounts(
            examples=len({key[0] for key in order}),
            trials_expected=len(order),
            trials_done=len(trials),
            task_errors=sum(1 for t in trials if not t.ok),
            evaluator_failures=sum(len(t.evaluator_failures) for t in trials),
            unscored=sum(1 for t in trials for s in t.scores if not s.scored),
        )
        run.usage = sum((t.total_usage for t in trials), Usage())
        observed: dict[str, list[str]] = {
            agent: list(models)
            for agent, models in run.provenance.observed_models.items()
        }
        for trial in trials:
            for agent, models in trial.models.items():
                known = observed.setdefault(agent, [])
                known.extend(m for m in models if m not in known)
        run.provenance.observed_models = observed
        run.metrics = compute_metrics(
            trials,
            metrics,
            examples=run.examples,
            group_by=run.config.group_by,
            cluster_by=run.config.cluster_by,
        )
        run.invalid_reason = None
        if max_error_rate is not None and trials:
            rate = run.counts.task_errors / len(trials)
            if rate > max_error_rate:
                run.invalid_reason = (
                    f"task error rate {rate:.1%} exceeds max_error_rate "
                    f"{max_error_rate:.1%}"
                )
        run.status = status
        run.finished_at = utc_now()
        if self.store is not None:
            self.store.save(run)
            self.store.write_report(run.id, render_run_markdown(run))
        return run


def resolve_store(store: RunStore | None, persist: bool) -> RunStore | None:
    if store is not None:
        return store
    return LocalRunStore() if persist else None


def load_run(store: RunStore | None, ref: "str | EvaluationRun") -> EvaluationRun:
    if isinstance(ref, EvaluationRun):
        if not ref.trials and store is not None:
            return store.load(ref.id)
        return ref
    if store is None:
        raise ValueError("A run id needs a store to load it from")
    return store.load(store.resolve(ref))


async def run_all(
    executor: Executor,
    coroutines: Sequence[Any],
    *,
    metrics: Sequence[Metric] | None,
    order: Sequence[TrialKey],
    max_error_rate: float | None,
) -> EvaluationRun:
    try:
        async with asyncio.TaskGroup() as group:
            for coroutine in coroutines:
                group.create_task(coroutine)
    except asyncio.CancelledError:
        executor.finalize(
            RunStatus.CANCELLED,
            metrics=metrics,
            order=order,
            max_error_rate=max_error_rate,
        )
        raise
    except BaseException:
        executor.finalize(
            RunStatus.FAILED,
            metrics=metrics,
            order=order,
            max_error_rate=max_error_rate,
        )
        raise
    status = RunStatus.PARTIAL if executor.budget_exhausted else RunStatus.COMPLETED
    return executor.finalize(
        status, metrics=metrics, order=order, max_error_rate=max_error_rate
    )
