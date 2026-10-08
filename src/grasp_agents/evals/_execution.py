import asyncio
import json
import logging
import operator
import platform
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from itertools import starmap
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import ConfigDict, PydanticUserError, TypeAdapter, ValidationError

from grasp_agents.session_context import SessionContext
from grasp_agents.types.events import Event

from ._util import (
    canonical_json,
    git_state,
    short_hash,
    source_hash,
    to_jsonable,
    utc_now,
)
from .metrics import MetricsSpec, compute_metrics
from .report import render_run_markdown
from .scorer import (
    FunctionScorer,
    ScoreContext,
    Scorer,
    merge_models,
    run_scorer,
)
from .store import LocalRunStore, RunStore
from .task import Task, TrialContext
from .types import (
    ComponentInfo,
    DatasetRef,
    ErrorInfo,
    EvaluationRun,
    Example,
    Provenance,
    RunCounts,
    RunStatus,
    Score,
    ScorerFailure,
    Trial,
    Usage,
    with_record,
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


def scorer_sources(scorers: Iterable[Scorer[Any, Any, Any]]) -> list[Any]:
    """Classes and functions whose source files define ``scorers``."""
    return [e.fn if isinstance(e, FunctionScorer) else type(e) for e in scorers]


def capture_provenance(
    observed_models: Mapping[str, list[str]] | None = None,
    *,
    sources: Iterable[Any] = (),
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
        source_hash=source_hash(sources),
        python=platform.python_version(),
        grasp_agents=grasp_version,
        observed_models=dict(observed_models or {}),
    )


def config_hash(
    task: ComponentInfo,
    scorers: Sequence[ComponentInfo],
    dataset: DatasetRef,
    repetitions: int,
) -> str:
    """Identity of what a run measures: same hash ⇔ like-for-like runs."""
    return short_hash(
        canonical_json(
            {
                "task": task,
                "scorers": sorted(
                    (e.model_dump(exclude={"source"}) for e in scorers),
                    key=operator.itemgetter("name"),
                ),
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
    # Same stored content, now typed: the identity and record are unchanged.
    return [
        with_record(
            e.model_copy(
                update={
                    "input": inputs.validate_python(e.input),
                    "reference": None
                    if e.reference is None
                    else references.validate_python(e.reference),
                    "content_hash": e.content_hash,
                }
            ),
            e.record,
        )
        for e in examples
    ]


def output_adapter(output_type: Any) -> TypeAdapter[Any]:
    """Validator of stored outputs: plain JSON, with bytes base64-encoded."""
    try:
        return TypeAdapter(output_type, config=ConfigDict(val_json_bytes="base64"))
    except PydanticUserError:
        # Models, dataclasses and TypedDicts carry their own configuration.
        return TypeAdapter(output_type)


def rehydrate_output(adapter: TypeAdapter[Any], output: Any) -> Any:
    """Typed form of a stored output; the stored JSON if it no longer validates."""
    try:
        return adapter.validate_json(json.dumps(output))
    except ValidationError:
        logger.warning("Stored output no longer matches the task's output type")
        return output


def warn_if_untyped(output_type: Any, outputs: Iterable[Any]) -> None:
    """Warn once when stored outputs reach scorers as plain JSON objects."""
    if output_type is not Any:
        return
    if any(isinstance(o, dict | list) for o in outputs):
        logger.warning(
            "The task's output type is unknown, so scorers receive stored "
            "outputs as plain JSON; pass output_type= to re-validate them"
        )


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
    stored: Any = None
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
        if error is None:
            try:
                stored = to_jsonable(output)
            except Exception as exc:
                error = ErrorInfo(
                    type="OutputNotSerializable",
                    message=(
                        "OutputNotSerializable: the task's output cannot be stored "
                        f"as JSON ({type(exc).__name__}: {exc})"
                    ),
                )
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, error.message))
        trace_id = _trace_id(span)
    trial = Trial(
        example_id=example.id,
        repetition=repetition,
        example_hash=example.content_hash,
        output=None if error is not None else stored,
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
    """Scores, persists, budgets and finalizes the trials of one run."""

    def __init__(
        self,
        run: EvaluationRun,
        *,
        store: RunStore | None,
        scorers: Sequence[Scorer[Any, Any, Any]],
        capture_events: bool = True,
        max_cost_usd: float | None = None,
        scorer_timeout_s: float | None = None,
        progress: ProgressCallback | None = None,
        total: int = 0,
    ) -> None:
        self.run = run
        self.store = store
        self.scorers = list(scorers)
        self.capture_events = capture_events
        self.max_cost_usd = max_cost_usd
        self.scorer_timeout_s = scorer_timeout_s
        self.progress = progress
        self.total = total
        self.budget_exhausted = False
        self._lock = asyncio.Lock()
        self._admission = asyncio.Condition()
        self._in_flight = 0
        self._trials: dict[TrialKey, Trial] = {t.key: t for t in run.trials}
        # What this run has spent so far, including attempts later superseded
        # (a resumed run continues its own total).
        self._base_usage = run.usage
        self._new_usage = Usage()
        self._trial_costs: list[float] = [
            t.total_usage.cost_usd or 0.0 for t in run.trials
        ]
        self._unpriced: set[str] = set()

    @property
    def spent_usd(self) -> float:
        return (self._base_usage + self._new_usage).cost_usd or 0.0

    def get(self, key: TrialKey) -> Trial | None:
        return self._trials.get(key)

    def add_task_usage(self, trial: Trial) -> None:
        self._new_usage += trial.usage
        for agent, usage in trial.usage_by_agent.items():
            if usage.total_tokens and usage.cost_usd is None:
                self._unpriced.add(agent)

    async def admit(self) -> bool:
        """
        Whether another trial may start under ``max_cost_usd``. Trials already
        running count at the highest cost a trial has had so far. Until a
        trial has cost anything, one runs at first and the number running
        doubles with each finished trial, so a quick failure does not open
        the whole batch.
        """
        if self.max_cost_usd is None:
            return True
        async with self._admission:
            while True:
                if self.spent_usd >= self.max_cost_usd:
                    self.budget_exhausted = True
                    return False
                if self._in_flight == 0:
                    break
                highest = max(self._trial_costs, default=0.0)
                if highest > 0.0:
                    committed = self.spent_usd + (self._in_flight + 1) * highest
                    if committed <= self.max_cost_usd:
                        break
                elif self._in_flight < 1 << min(len(self._trial_costs), 30):
                    break
                await self._admission.wait()
            self._in_flight += 1
            return True

    async def release(self) -> None:
        if self.max_cost_usd is None:
            return
        async with self._admission:
            self._in_flight -= 1
            self._admission.notify_all()

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
            for e in self.scorers
            if e.name not in trial.scorers_run and (trial.ok or e.scores_errors)
        ]
        if not pending:
            return
        contexts = [
            ScoreContext(
                example=example,
                output=output,
                trial=trial,
                events=events,
                session=session,
            )
            for _ in pending
        ]
        outcomes = await asyncio.gather(
            *starmap(self._run_one, zip(pending, contexts, strict=True))
        )
        names = {s.name for s in trial.scores}
        for scorer, ctx, outcome in zip(pending, contexts, outcomes, strict=True):
            trial.scorer_failures = [
                f for f in trial.scorer_failures if f.scorer != scorer.name
            ]
            usage = sum(ctx.usage, Usage())
            if not usage.is_empty:
                trial.scorer_usage[scorer.name] = usage
                self._new_usage += usage
                if usage.total_tokens and usage.cost_usd is None:
                    self._unpriced.add(f"scorer {scorer.name}")
            merge_models(trial.models, ctx.models)
            if isinstance(outcome, ErrorInfo):
                trial.scorer_failures.append(
                    ScorerFailure(scorer=scorer.name, error=outcome)
                )
                continue
            clashes = sorted({s.name for s in outcome} & names)
            if clashes:
                trial.scorer_failures.append(
                    ScorerFailure(
                        scorer=scorer.name,
                        error=ErrorInfo(
                            type="DuplicateScoreName",
                            message=f"DuplicateScoreName: already produced {clashes}",
                        ),
                    )
                )
                continue
            trial.scores.extend(outcome)
            names.update(s.name for s in outcome)
            trial.scorers_run.append(scorer.name)

    async def _run_one(
        self, scorer: Scorer[Any, Any, Any], ctx: ScoreContext[Any, Any, Any]
    ) -> list[Score] | ErrorInfo:
        try:
            return await run_scorer(scorer, ctx, timeout_s=self.scorer_timeout_s)
        except TimeoutError:
            return ErrorInfo(
                type="TimeoutError",
                message=(
                    "TimeoutError: scorer exceeded its "
                    f"{self.scorer_timeout_s:g}s timeout"
                ),
            )
        except Exception as exc:
            logger.debug(
                "Scorer %s failed on example %s: %s",
                scorer.name,
                ctx.example.id,
                exc,
            )
            return ErrorInfo.from_exception(exc)

    async def record(
        self, trial: Trial, events: Sequence[Event[Any]] | None = None
    ) -> None:
        async with self._lock:
            self._trials[trial.key] = trial
            self._trial_costs.append(trial.total_usage.cost_usd or 0.0)
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

    def _invalid_reason(self, trials: Sequence[Trial], expected: int) -> str | None:
        if expected and not trials:
            return "no trial ran"
        errors = sum(1 for t in trials if not t.ok)
        if trials and errors == len(trials):
            return "every trial failed"
        limit = self.run.config.max_error_rate
        if limit is not None and trials:
            rate = errors / len(trials)
            if rate > limit:
                return f"task error rate {rate:.1%} exceeds max_error_rate {limit:.1%}"
        return None

    def finalize(
        self,
        status: RunStatus,
        *,
        metrics: MetricsSpec,
        order: Sequence[TrialKey],
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
            scorer_failures=sum(len(t.scorer_failures) for t in trials),
            unscored=sum(1 for t in trials for s in t.scores if not s.scored),
        )
        run.usage = self._base_usage + self._new_usage
        observed: dict[str, list[str]] = {
            agent: list(models)
            for agent, models in run.provenance.observed_models.items()
        }
        for trial in trials:
            for agent, models in trial.models.items():
                known = observed.setdefault(agent, [])
                known.extend(m for m in models if m not in known)
        run.provenance.observed_models = observed
        if self._unpriced and self.max_cost_usd is not None:
            agents = sorted(self._unpriced)
            logger.warning(
                "No price is known for the models of %s: max_cost_usd cannot "
                "limit their spend",
                ", ".join(agents),
            )
            run.metadata = {**run.metadata, "unpriced_agents": agents}
        run.invalid_reason = self._invalid_reason(trials, len(order))
        run.status = status
        run.finished_at = utc_now()
        try:
            run.metrics = compute_metrics(
                trials,
                metrics,
                examples=run.examples,
                group_by=run.config.group_by,
                cluster_by=run.config.cluster_by,
                expected=order,
            )
        except Exception as exc:
            run.metrics = []
            run.status = RunStatus.FAILED
            run.invalid_reason = f"computing metrics failed: {exc}"
            if self.store is not None:
                self.store.save(run)
            raise
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
    metrics: MetricsSpec,
    order: Sequence[TrialKey],
) -> EvaluationRun:
    """
    Run the trial coroutines and finalize the run: ``completed`` when every
    expected trial exists, ``partial`` when some never ran.
    """
    try:
        async with asyncio.TaskGroup() as group:
            for coroutine in coroutines:
                group.create_task(coroutine)
    except asyncio.CancelledError:
        executor.finalize(RunStatus.CANCELLED, metrics=metrics, order=order)
        raise
    except BaseExceptionGroup as failures:
        executor.finalize(RunStatus.FAILED, metrics=metrics, order=order)
        errors = [
            e for e in failures.exceptions if not isinstance(e, asyncio.CancelledError)
        ]
        if errors and len({type(e) for e in errors}) == 1:
            raise errors[0] from failures
        raise
    except BaseException:
        executor.finalize(RunStatus.FAILED, metrics=metrics, order=order)
        raise
    done = sum(1 for key in order if executor.get(key) is not None)
    status = RunStatus.COMPLETED if done >= len(order) else RunStatus.PARTIAL
    return executor.finalize(status, metrics=metrics, order=order)
