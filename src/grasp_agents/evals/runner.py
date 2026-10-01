import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from grasp_agents.processors.processor import Processor

from ._execution import (
    Executor,
    ProgressCallback,
    TrialKey,
    capture_provenance,
    config_hash,
    execute_trial,
    load_run,
    rehydrate_output,
    resolve_store,
    retype_examples,
    run_all,
)
from ._util import new_run_id, utc_now
from .dataset import Dataset
from .evaluator import Evaluator
from .metrics import Metric
from .store import RunStore
from .task import Task, TaskFn, as_task
from .types import EvaluationRun, Example, RunConfig, RunStatus, Trial

if TYPE_CHECKING:
    from grasp_agents.types.events import Event

logger = logging.getLogger(__name__)

type TaskLike[InT, OutT] = (
    Task[InT, OutT] | Processor[InT, OutT, Any] | TaskFn[InT, OutT]
)


class ResumeError(ValueError):
    pass


def _check_evaluators(evaluators: Sequence[Evaluator[Any, Any, Any]]) -> None:
    seen: set[str] = set()
    for evaluator in evaluators:
        if evaluator.name in seen:
            raise ValueError(f"Duplicate evaluator name {evaluator.name!r}")
        seen.add(evaluator.name)


async def evaluate[InT, OutT, RefT](
    task: TaskLike[InT, OutT],
    dataset: Dataset[InT, RefT],
    evaluators: Sequence[Evaluator[InT, OutT, RefT]] = (),
    metrics: Sequence[Metric] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    repetitions: int = 1,
    concurrency: int = 4,
    timeout_s: float | None = None,
    max_cost_usd: float | None = None,
    max_error_rate: float | None = None,
    group_by: Sequence[str] = (),
    cluster_by: str | None = None,
    sealed_splits: Sequence[str] = (),
    score: bool = True,
    capture_events: bool = True,
    store: RunStore | None = None,
    persist: bool = True,
    resume: "str | EvaluationRun | None" = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    evaluation: str | None = None,
    progress: ProgressCallback | None = None,
) -> EvaluationRun:
    """
    Run ``task`` on every example of ``dataset`` (x ``repetitions``) and score
    each trial with ``evaluators``.

    ``task`` may be a :class:`Task`, any :class:`Processor` (run in an isolated
    session per trial) or an async function. Trials run with bounded
    ``concurrency``; each is persisted as soon as it is scored, so an
    interrupted run can be continued with ``resume=<run id>``, which re-runs
    only missing or failed trials (and evaluators that did not finish) and
    refuses if the configuration or code changed. ``max_cost_usd`` stops
    scheduling new trials once spent (the run ends ``partial``);
    ``max_error_rate`` marks a run with too many task errors invalid.
    ``metrics=None`` picks defaults from the scores produced. The run is
    stored under ``$GRASP_EVALS_DIR`` (``./.evals``) unless ``persist=False``
    or another ``store`` is given.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    the_task = as_task(task)
    evaluator_list: list[Evaluator[Any, Any, Any]] = list(evaluators)
    _check_evaluators(evaluator_list)
    run_store = resolve_store(store, persist)
    dataset_ref = dataset.ref()
    task_info = the_task.describe()
    evaluator_infos = [e.describe() for e in evaluator_list]
    run_hash = config_hash(task_info, evaluator_infos, dataset_ref, repetitions)
    provenance = capture_provenance()
    sealed = frozenset(sealed_splits)

    events_by_key: dict[TrialKey, list[Event[Any]]] = {}
    if resume is not None:
        run = load_run(run_store, resume)
        if run.config_hash != run_hash:
            raise ResumeError(
                f"Cannot resume {run.id}: its task, evaluators, dataset or repetitions "
                "differ from this call. Start a new run, or rescore the old one."
            )
        old = run.provenance
        same_code = (old.git_commit, old.git_diff_hash) == (
            provenance.git_commit,
            provenance.git_diff_hash,
        )
        if old.git_commit is not None and not same_code:
            raise ResumeError(
                f"Cannot resume {run.id}: the code changed since it started "
                f"({(old.git_commit or '')[:12]} → "
                f"{(provenance.git_commit or '')[:12]}, or uncommitted changes "
                "differ). Start a new run instead."
            )
        run.status = RunStatus.RUNNING
        run.finished_at = None
        run.metadata = {
            **run.metadata,
            "resumed_at": [*run.metadata.get("resumed_at", []), utc_now().isoformat()],
        }
        if run_store is not None:
            run_store.save(run)
            events_by_key = run_store.load_events(run.id)
    else:
        run_name = name or the_task.name
        run = EvaluationRun(
            id=new_run_id(run_name),
            name=run_name,
            created_at=utc_now(),
            description=description,
            evaluation=evaluation,
            dataset=dataset_ref,
            task=task_info,
            evaluators=evaluator_infos,
            config=RunConfig(
                repetitions=repetitions,
                concurrency=concurrency,
                timeout_s=timeout_s,
                max_cost_usd=max_cost_usd,
                max_error_rate=max_error_rate,
                score=score,
                group_by=list(group_by),
                cluster_by=cluster_by,
                sealed_splits=sorted(sealed),
            ),
            provenance=provenance,
            config_hash=run_hash,
            tags=list(tags),
            metadata=dict(metadata or {}),
            examples=list(dataset),
        )
        if run_store is not None:
            run_store.create(run)

    order: list[TrialKey] = [(e.id, r) for e in dataset for r in range(repetitions)]
    executor = Executor(
        run,
        store=run_store,
        evaluators=evaluator_list if score else [],
        capture_events=capture_events,
        max_cost_usd=max_cost_usd,
        progress=progress,
        total=len(order),
    )
    adapter: TypeAdapter[Any] = TypeAdapter(the_task.output_type)
    semaphore = asyncio.Semaphore(concurrency)

    async def work(example: Example[Any, Any], repetition: int) -> None:
        async with semaphore:
            prior = executor.get((example.id, repetition))
            if prior is not None and prior.ok:
                if score and any(e.name not in prior.evaluated for e in evaluator_list):
                    output = rehydrate_output(adapter, prior.output)
                    events = events_by_key.get(prior.key, [])
                    await executor.score(prior, example, output, events)
                    await executor.record(prior)
                return
            if executor.over_budget():
                return
            trial, output, events, session = await execute_trial(
                the_task,
                run,
                example,
                repetition,
                timeout_s=timeout_s,
                sealed_splits=sealed,
                capture_events=capture_events,
            )
            await executor.score(trial, example, output, events, session)
            await executor.record(trial, events)

    return await run_all(
        executor,
        [work(e, r) for e in dataset for r in range(repetitions)],
        metrics=metrics,
        order=order,
        max_error_rate=max_error_rate,
    )


def _without(trial: Trial, replaced: set[str]) -> Trial:
    return trial.model_copy(
        deep=True,
        update={
            "scores": [s for s in trial.scores if s.evaluator not in replaced],
            "evaluated": [e for e in trial.evaluated if e not in replaced],
            "evaluator_failures": [
                f for f in trial.evaluator_failures if f.evaluator not in replaced
            ],
        },
    )


async def rescore(
    run: "str | EvaluationRun",
    evaluators: Sequence[Evaluator[Any, Any, Any]],
    metrics: Sequence[Metric] | None = None,
    *,
    input_type: Any = Any,
    reference_type: Any = Any,
    output_type: Any = Any,
    name: str | None = None,
    concurrency: int = 4,
    group_by: Sequence[str] | None = None,
    cluster_by: str | None = None,
    store: RunStore | None = None,
    persist: bool = True,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> EvaluationRun:
    """
    Score a finished run's stored outputs again, without re-running the task.

    Creates a child run (``parent_run_id``) that copies the parent's trials and
    transcripts. ``evaluators`` replace same-named evaluators of the parent;
    the parent's other scores are kept. Stored examples and outputs are
    re-validated as ``input_type`` / ``reference_type`` / ``output_type`` before
    evaluators see them. The parent is never modified.
    """
    _check_evaluators(evaluators)
    run_store = resolve_store(store, persist)
    parent = load_run(run_store, run)
    events_by_key: dict[TrialKey, list[Event[Any]]] = (
        run_store.load_events(parent.id) if run_store is not None else {}
    )
    replaced = {e.name for e in evaluators}
    evaluator_infos = [i for i in parent.evaluators if i.name not in replaced] + [
        e.describe() for e in evaluators
    ]
    config = parent.config.model_copy(
        update={
            "score": True,
            "concurrency": concurrency,
            "group_by": list(group_by)
            if group_by is not None
            else parent.config.group_by,
            "cluster_by": cluster_by
            if cluster_by is not None
            else parent.config.cluster_by,
        }
    )
    child_name = name or parent.name
    child = EvaluationRun(
        id=new_run_id(child_name),
        name=child_name,
        kind="rescore",
        created_at=utc_now(),
        description=parent.description,
        evaluation=parent.evaluation,
        parent_run_id=parent.id,
        dataset=parent.dataset,
        task=parent.task,
        evaluators=evaluator_infos,
        config=config,
        provenance=capture_provenance(parent.provenance.observed_models),
        config_hash=config_hash(
            parent.task, evaluator_infos, parent.dataset, parent.config.repetitions
        ),
        tags=[*parent.tags, *tags],
        metadata={**parent.metadata, **(metadata or {})},
        examples=list(parent.examples),
    )
    if run_store is not None:
        run_store.create(child)
    typed = retype_examples(
        parent.examples, input_type=input_type, reference_type=reference_type
    )
    examples = {e.id: e for e in typed}
    order: list[TrialKey] = [t.key for t in parent.trials]
    executor = Executor(
        child,
        store=run_store,
        evaluators=evaluators,
        progress=progress,
        total=len(order),
    )
    adapter: TypeAdapter[Any] = TypeAdapter(output_type)
    semaphore = asyncio.Semaphore(concurrency)

    async def work(parent_trial: Trial) -> None:
        async with semaphore:
            trial = _without(parent_trial, replaced)
            example = examples.get(trial.example_id)
            events = events_by_key.get(trial.key, [])
            if example is not None:
                output = rehydrate_output(adapter, trial.output) if trial.ok else None
                await executor.score(trial, example, output, events)
            await executor.record(trial, events)

    return await run_all(
        executor,
        [work(t) for t in parent.trials],
        metrics=metrics,
        order=order,
        max_error_rate=parent.config.max_error_rate,
    )
