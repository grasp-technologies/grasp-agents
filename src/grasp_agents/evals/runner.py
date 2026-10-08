import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from grasp_agents.processors.processor import Processor

from ._execution import (
    Executor,
    ProgressCallback,
    TrialKey,
    capture_provenance,
    config_hash,
    execute_trial,
    load_run,
    output_adapter,
    rehydrate_output,
    resolve_store,
    retype_examples,
    run_all,
    scorer_sources,
    warn_if_untyped,
)
from ._util import new_run_id, utc_now
from .dataset import Dataset
from .metrics import Metric
from .scorer import Scorer
from .store import RunStore
from .task import Task, TaskFn, as_task
from .types import (
    ComponentInfo,
    EvaluationRun,
    Example,
    Provenance,
    RunConfig,
    RunStatus,
    Trial,
)

if TYPE_CHECKING:
    from grasp_agents.types.events import Event

logger = logging.getLogger(__name__)

type TaskLike[InT, OutT] = (
    Task[InT, OutT] | Processor[InT, OutT, Any] | TaskFn[InT, OutT]
)

# Settings a resumed run may change: they affect how trials are executed and
# aggregated, not what a trial measures. Changes are recorded on the run.
_ADJUSTABLE = (
    "concurrency",
    "timeout_s",
    "scorer_timeout_s",
    "max_cost_usd",
    "max_error_rate",
    "score",
    "group_by",
    "cluster_by",
)


class ResumeError(ValueError):
    pass


class SealedSelectionError(ValueError):
    pass


def _check_scorers(scorers: Sequence[Scorer[Any, Any, Any]]) -> None:
    seen: set[str] = set()
    for scorer in scorers:
        if scorer.name in seen:
            raise ValueError(f"Duplicate scorer name {scorer.name!r}")
        seen.add(scorer.name)


def check_sealed(dataset: Dataset[Any, Any], sealed: frozenset[str]) -> None:
    """
    Sealed splits must exist, and a selection must include each sealed split
    it touches whole: a subset would report its examples' results one by
    one (``--ids <sealed id>``).
    """
    if not sealed:
        return
    known = set(dataset.origin.splits)
    unknown = sorted(sealed - known)
    if unknown:
        raise ValueError(
            f"Sealed splits {unknown} are not splits of {dataset.name!r} "
            f"(splits: {sorted(known) or 'none'})"
        )
    for split in sorted(sealed):
        selected = {e.id for e in dataset if split in e.splits}
        if not selected:
            continue
        whole = {e.id for e in dataset.origin if split in e.splits}
        if selected != whole:
            raise SealedSelectionError(
                f"The selection covers {len(selected)} of the {len(whole)} examples "
                f"of sealed split {split!r}. Sealed splits are evaluated whole, so "
                "their per-example results cannot be inferred; drop the ids, limit, "
                "sample or filter (or select the split alone)."
            )


def _code_changes(old: Provenance, new: Provenance) -> list[str]:
    changes: list[str] = []
    if old.git_commit is not None:
        if old.git_commit != new.git_commit:
            changes.append(
                f"commit {old.git_commit[:12]} → {(new.git_commit or 'none')[:12]}"
            )
        elif old.git_diff_hash != new.git_diff_hash:
            changes.append(
                "uncommitted changes differ "
                f"({old.git_diff_hash or 'none'} → {new.git_diff_hash or 'none'})"
            )
    if old.source_hash is not None and old.source_hash != new.source_hash:
        changes.append("source files of the task or scorers changed")
    return changes


def _resume_record(
    previous: EvaluationRun,
    *,
    run_hash: str,
    config: RunConfig,
    provenance: Provenance,
    force: bool,
) -> dict[str, Any]:
    if previous.config_hash != run_hash:
        raise ResumeError(
            f"Cannot resume {previous.id}: the task, scorers, selected examples or "
            "repetitions differ from the original run. Pass the same selection "
            "(split, ids, limit, sample) and repetitions as the original run, or "
            "start a new one."
        )
    if sorted(previous.config.sealed_splits) != sorted(config.sealed_splits):
        raise ResumeError(
            f"Cannot resume {previous.id}: its sealed splits "
            f"{previous.config.sealed_splits} differ from {config.sealed_splits}"
        )
    changes = _code_changes(previous.provenance, provenance)
    if changes and not force:
        raise ResumeError(
            f"Cannot resume {previous.id}: the code changed since it started "
            f"({'; '.join(changes)}). Start a new run, or resume with force=True "
            "if the change cannot affect results."
        )
    overrides = {
        name: [getattr(previous.config, name), getattr(config, name)]
        for name in _ADJUSTABLE
        if getattr(previous.config, name) != getattr(config, name)
    }
    record: dict[str, Any] = {"at": utc_now().isoformat()}
    if overrides:
        record["changed_settings"] = overrides
    if changes:
        record["forced_past"] = changes
    return record


def _retry_run(
    parent: EvaluationRun,
    *,
    task_info: ComponentInfo,
    scorer_infos: list[ComponentInfo],
    config: RunConfig,
    provenance: Provenance,
    run_hash: str,
    tags: Sequence[str],
    metadata: Mapping[str, Any] | None,
    evaluation: str | None,
) -> EvaluationRun:
    return EvaluationRun(
        id=new_run_id(parent.name),
        name=parent.name,
        kind="retry",
        created_at=utc_now(),
        description=parent.description,
        evaluation=evaluation or parent.evaluation,
        parent_run_id=parent.id,
        dataset=parent.dataset,
        task=task_info,
        scorers=scorer_infos,
        config=config,
        provenance=provenance,
        config_hash=run_hash,
        tags=[*parent.tags, *tags],
        metadata={**parent.metadata, **(metadata or {})},
        examples=list(parent.examples),
        trials=[t.model_copy(deep=True) for t in parent.trials if t.ok],
    )


def _nothing_to_redo(
    run: EvaluationRun,
    order: Sequence[TrialKey],
    scorers: Sequence[Scorer[Any, Any, Any]],
    score: bool,
) -> bool:
    trials = {t.key: t for t in run.trials}
    for key in order:
        trial = trials.get(key)
        if trial is None or not trial.ok:
            return False
        if score and any(e.name not in trial.scorers_run for e in scorers):
            return False
    return True


def _model_drift(trial: Trial, known: Mapping[str, set[str]]) -> str | None:
    for agent, models in trial.models.items():
        before = known.get(agent)
        new = set(models) - before if before else set[str]()
        if new:
            return (
                f"agent {agent!r} now answers with {sorted(new)}, but the run used "
                f"{sorted(before or ())}"
            )
    return None


async def evaluate[InT, OutT, RefT](
    task: TaskLike[InT, OutT],
    dataset: Dataset[InT, RefT],
    scorers: Sequence[Scorer[InT, OutT, RefT]] = (),
    metrics: Sequence[Metric] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    repetitions: int = 1,
    concurrency: int = 4,
    timeout_s: float | None = None,
    scorer_timeout_s: float | None = None,
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
    force: bool = False,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    evaluation: str | None = None,
    progress: ProgressCallback | None = None,
) -> EvaluationRun:
    """
    Run ``task`` on every example of ``dataset`` (x ``repetitions``) and score
    each trial with ``scorers``.

    ``task`` may be a :class:`Task`, any :class:`Processor` (run in an isolated
    session per trial) or an async function. Trials run with bounded
    ``concurrency``; each is persisted as soon as it is scored.

    ``resume=<run>`` continues a run with the same task, scorers, selected
    examples and repetitions: it runs only missing or failed trials (and
    scorers that did not finish). A run that did not complete is continued
    in place; a completed one keeps its results — its failed trials are
    retried in a new ``retry`` run that copies the rest. Resuming refuses if
    the code changed (git state, or the task's and scorers' source files;
    ``force=True`` overrides) and stops if an agent answers with a model the
    run did not use.

    ``max_cost_usd`` stops scheduling new trials once spent (the run ends
    ``partial``); trials already running finish, so the total can exceed it
    by about one trial's cost each. ``max_error_rate`` marks a run with too
    many task errors invalid (as is a run in which every trial failed).
    ``metrics=None`` picks defaults from the scores produced. The run is
    stored under ``$GRASP_EVALS_DIR`` (``./.evals``) unless ``persist=False``
    or another ``store`` is given.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if len(dataset) == 0:
        raise ValueError(f"The selection from {dataset.name!r} has no examples")
    the_task = as_task(task)
    scorer_list: list[Scorer[Any, Any, Any]] = list(scorers)
    _check_scorers(scorer_list)
    sealed = frozenset(sealed_splits)
    check_sealed(dataset, sealed)
    run_store = resolve_store(store, persist)
    dataset_ref = dataset.ref()
    task_info = the_task.describe()
    scorer_infos = [e.describe() for e in scorer_list]
    run_hash = config_hash(task_info, scorer_infos, dataset_ref, repetitions)
    adapter = output_adapter(the_task.output_type)
    provenance = capture_provenance(
        sources=[*the_task.source_objects(), *scorer_sources(scorer_list)]
    )
    config = RunConfig(
        repetitions=repetitions,
        concurrency=concurrency,
        timeout_s=timeout_s,
        scorer_timeout_s=scorer_timeout_s,
        max_cost_usd=max_cost_usd,
        max_error_rate=max_error_rate,
        score=score,
        group_by=list(group_by),
        cluster_by=cluster_by,
        sealed_splits=sorted(sealed),
    )
    order: list[TrialKey] = [(e.id, r) for e in dataset for r in range(repetitions)]

    events_by_key: dict[TrialKey, list[Event[Any]]] = {}
    known_models: dict[str, set[str]] = {}
    if resume is not None:
        previous = load_run(run_store, resume)
        record = _resume_record(
            previous,
            run_hash=run_hash,
            config=config,
            provenance=provenance,
            force=force,
        )
        if run_store is not None:
            events_by_key = run_store.load_events(previous.id)
        # From the trials too: a run that crashed never wrote its totals.
        for agent, models in (
            *previous.provenance.observed_models.items(),
            *(item for t in previous.trials for item in t.models.items()),
        ):
            known_models.setdefault(agent, set()).update(models)
        if previous.completed:
            if _nothing_to_redo(previous, order, scorer_list, score):
                logger.info("Run %s has nothing left to retry", previous.id)
                return previous
            run = _retry_run(
                previous,
                task_info=task_info,
                scorer_infos=scorer_infos,
                config=config,
                provenance=provenance,
                run_hash=run_hash,
                tags=tags,
                metadata=metadata,
                evaluation=evaluation,
            )
            run.metadata = {**run.metadata, "resumed": [record]}
            if run_store is not None:
                run_store.create(run)
                for trial in run.trials:
                    run_store.append_trial(run.id, trial, events_by_key.get(trial.key))
        else:
            run = previous
            run.status = RunStatus.RUNNING
            run.finished_at = None
            run.invalid_reason = None
            run.config = config
            run.metadata = {
                **run.metadata,
                "resumed": [*run.metadata.get("resumed", []), record],
            }
            if run_store is not None:
                run_store.save(run)
        if score:
            warn_if_untyped(
                the_task.output_type, (t.output for t in run.trials if t.ok)
            )
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
            scorers=scorer_infos,
            config=config,
            provenance=provenance,
            config_hash=run_hash,
            tags=list(tags),
            metadata=dict(metadata or {}),
            examples=list(dataset),
        )
        if run_store is not None:
            run_store.create(run)

    executor = Executor(
        run,
        store=run_store,
        scorers=scorer_list if score else [],
        capture_events=capture_events,
        max_cost_usd=max_cost_usd,
        scorer_timeout_s=scorer_timeout_s,
        progress=progress,
        total=len(order),
    )
    semaphore = asyncio.Semaphore(concurrency)
    check_models = resume is not None and not force

    async def work(example: Example[Any, Any], repetition: int) -> None:
        async with semaphore:
            prior = executor.get((example.id, repetition))
            if prior is not None and prior.ok:
                if score and any(e.name not in prior.scorers_run for e in scorer_list):
                    output = rehydrate_output(adapter, prior.output)
                    events = events_by_key.get(prior.key, [])
                    await executor.score(prior, example, output, events)
                    await executor.record(prior)
                return
            if not await executor.admit():
                return
            try:
                trial, output, events, session = await execute_trial(
                    the_task,
                    run,
                    example,
                    repetition,
                    timeout_s=timeout_s,
                    sealed_splits=sealed,
                    capture_events=capture_events,
                )
                executor.add_task_usage(trial)
                drift = _model_drift(trial, known_models) if check_models else None
                if drift is not None:
                    raise ResumeError(f"Cannot resume {run.id}: {drift}")
                await executor.score(trial, example, output, events, session)
                await executor.record(trial, events)
            finally:
                await executor.release()

    return await run_all(
        executor,
        [work(e, r) for e in dataset for r in range(repetitions)],
        metrics=metrics,
        order=order,
    )


def _without(trial: Trial, replaced: set[str]) -> Trial:
    return trial.model_copy(
        deep=True,
        update={
            "scores": [s for s in trial.scores if s.scorer not in replaced],
            "scorers_run": [e for e in trial.scorers_run if e not in replaced],
            "scorer_failures": [
                f for f in trial.scorer_failures if f.scorer not in replaced
            ],
            "scorer_usage": {
                k: v for k, v in trial.scorer_usage.items() if k not in replaced
            },
        },
    )


async def rescore(
    run: "str | EvaluationRun",
    scorers: Sequence[Scorer[Any, Any, Any]],
    metrics: Sequence[Metric] | None = None,
    *,
    input_type: Any = Any,
    reference_type: Any = Any,
    output_type: Any = Any,
    name: str | None = None,
    concurrency: int = 4,
    scorer_timeout_s: float | None = None,
    group_by: Sequence[str] | None = None,
    cluster_by: str | None = None,
    rerun: bool = False,
    store: RunStore | None = None,
    persist: bool = True,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> EvaluationRun:
    """
    Score a run's stored outputs again, without re-running the task.

    Creates a child run (``parent_run_id``) that copies the parent's trials and
    transcripts; the parent is never modified. A scorer identical to the
    parent's (same name, version, configuration and code) keeps its scores and
    only fills the trials where it failed or did not run; a new or changed one
    replaces the parent's scores under its name. Only the scorer's own
    function or class is compared: when a helper, closure or data file it
    relies on changes, bump its version or pass ``rerun=True``, which re-runs
    every given scorer (also to measure a judge's self-consistency). The
    parent's other scorers' scores are kept.

    Stored examples and outputs are re-validated as ``input_type`` /
    ``reference_type`` / ``output_type`` before scorers see them. A parent
    that did not complete yields a ``partial`` child.
    """
    _check_scorers(scorers)
    run_store = resolve_store(store, persist)
    parent = load_run(run_store, run)
    if parent.status == RunStatus.RUNNING:
        raise ValueError(
            f"Run {parent.id} has not finished; resume it (or let it finish) first"
        )
    events_by_key: dict[TrialKey, list[Event[Any]]] = (
        run_store.load_events(parent.id) if run_store is not None else {}
    )
    typed = retype_examples(
        parent.examples, input_type=input_type, reference_type=reference_type
    )
    adapter = output_adapter(output_type)
    warn_if_untyped(output_type, (t.output for t in parent.trials if t.ok))
    previous = {info.name: info for info in parent.scorers}
    given = {e.name for e in scorers}
    replaced = {
        e.name for e in scorers if rerun or previous.get(e.name) != e.describe()
    }
    scorer_infos = [i for i in parent.scorers if i.name not in given] + [
        e.describe() for e in scorers
    ]
    config = parent.config.model_copy(
        update={
            "score": True,
            "concurrency": concurrency,
            "scorer_timeout_s": scorer_timeout_s,
            "group_by": list(group_by)
            if group_by is not None
            else parent.config.group_by,
            "cluster_by": cluster_by
            if cluster_by is not None
            else parent.config.cluster_by,
        }
    )
    order: list[TrialKey] = [
        (e.id, r) for e in parent.examples for r in range(parent.config.repetitions)
    ]
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
        scorers=scorer_infos,
        config=config,
        provenance=capture_provenance(
            parent.provenance.observed_models, sources=scorer_sources(scorers)
        ),
        config_hash=config_hash(
            parent.task, scorer_infos, parent.dataset, parent.config.repetitions
        ),
        tags=[*parent.tags, *tags],
        metadata={**parent.metadata, **(metadata or {})},
        examples=list(parent.examples),
    )
    if run_store is not None:
        run_store.create(child)
    examples = {e.id: e for e in typed}
    executor = Executor(
        child,
        store=run_store,
        scorers=scorers,
        scorer_timeout_s=scorer_timeout_s,
        progress=progress,
        total=len(order),
    )
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
        executor, [work(t) for t in parent.trials], metrics=metrics, order=order
    )
