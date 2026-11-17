"""
Runs of outputs produced elsewhere — by another agent, a person or a vendor —
scored like any run, so they compare and push like runs of a task.
"""

import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ._execution import (
    Executor,
    ProgressCallback,
    TrialKey,
    capture_provenance,
    config_hash,
    output_adapter,
    rehydrate_output,
    resolve_store,
    run_all,
    scorer_sources,
)
from ._util import new_run_id, to_jsonable, utc_now
from .dataset import Dataset, DatasetError
from .metrics import MetricsSpec
from .runner import check_scorers, check_sealed, leave_out_sealed
from .scorer import Scorer
from .store import RunStore
from .types import (
    ComponentInfo,
    ErrorInfo,
    EvaluationRun,
    RunConfig,
    Trial,
    Usage,
)

# ``ComponentInfo.kind`` of the task of an imported run.
IMPORTED_TASK_KIND = "imported"


class ImportedTrial(BaseModel):
    """
    One output produced outside grasp-evals: a line of an outputs file.
    ``error`` marks a failed attempt (its output is ignored); the rest is
    what was observed, recorded as a task's trial would be.
    """

    model_config = ConfigDict(extra="forbid")

    example_id: str
    repetition: int = Field(default=0, ge=0)
    output: Any = None
    error: str | None = None
    duration_s: float = Field(default=0.0, ge=0.0)
    cost_usd: float | None = None
    usage: Usage | None = None
    # Models per agent, as a task's trial records them.
    models: dict[str, list[str]] = Field(default_factory=dict[str, list[str]])
    measurements: dict[str, float] = Field(default_factory=dict[str, float])
    trace_id: str | None = None
    started_at: datetime | None = None

    def total_usage(self) -> Usage:
        usage = self.usage or Usage()
        if self.cost_usd is None:
            return usage
        return usage.model_copy(update={"cost_usd": self.cost_usd})


def read_outputs(path: str | Path) -> list[ImportedTrial]:
    """The :class:`ImportedTrial` records of a JSONL file (``//`` comments)."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"{source}: cannot read ({exc.strerror or exc})") from exc
    records: list[ImportedTrial] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            records.append(ImportedTrial.model_validate(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{source}:{lineno}: invalid JSON ({exc.msg})") from exc
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'record'}: {e['msg']}"
                for e in exc.errors()
            )
            raise DatasetError(f"{source}:{lineno}: {problems}") from exc
    return records


def _trial(
    record: ImportedTrial, example_hash: str, *, sealed: bool, output: Any
) -> Trial:
    usage = record.total_usage()
    error = (
        None
        if record.error is None
        else ErrorInfo(type="ImportedError", message=record.error)
    )
    started = record.started_at or utc_now() - timedelta(seconds=record.duration_s)
    return Trial(
        example_id=record.example_id,
        repetition=record.repetition,
        example_hash=example_hash,
        output=None if error is not None else output,
        error=error,
        started_at=started,
        duration_s=record.duration_s,
        usage=usage,
        models={agent: list(names) for agent, names in record.models.items()},
        trace_id=record.trace_id,
        measurements=dict(record.measurements),
        sealed=sealed,
    )


async def import_outputs(
    outputs: Iterable[ImportedTrial | Mapping[str, Any]],
    dataset: Dataset[Any, Any],
    scorers: Sequence[Scorer[Any, Any, Any]] = (),
    metrics: MetricsSpec = None,
    *,
    task: str,
    task_version: str | None = None,
    task_config: Mapping[str, Any] | None = None,
    output_type: Any = Any,
    repetitions: int | None = None,
    name: str | None = None,
    description: str | None = None,
    concurrency: int = 4,
    scorer_timeout_s: float | None = None,
    group_by: Sequence[str] = (),
    cluster_by: str | None = None,
    sealed_splits: Sequence[str] = (),
    include_sealed: bool = False,
    store: RunStore | None = None,
    persist: bool = True,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    evaluation: str | None = None,
    progress: ProgressCallback | None = None,
) -> EvaluationRun:
    """
    A run (``kind="import"``) of outputs produced elsewhere for the examples
    of ``dataset``, scored with ``scorers`` as :func:`rescore` scores stored
    outputs. The task is identified by ``task``, ``task_version`` and
    ``task_config`` only, so declare what produced the outputs there (a
    model, an agent version).

    Outputs are matched to examples by id and repetition, and re-validated as
    ``output_type`` before scorers see them. Outputs for examples of the
    dataset outside the selection are ignored; an id the dataset lacks, or a
    repeated (example, repetition), is an error. ``repetitions`` defaults to
    the most any example has; examples missing some leave the run
    ``partial``. Sealed splits are handled as in :func:`evaluate`; one that is
    included needs an output (or error) for each of its examples and
    repetitions.
    """
    records = [
        r if isinstance(r, ImportedTrial) else ImportedTrial.model_validate(r)
        for r in outputs
    ]
    scorer_list = list(scorers)
    check_scorers(scorer_list, metrics)
    sealed = frozenset(sealed_splits)
    selected = leave_out_sealed(dataset, sealed, include_sealed=include_sealed)
    check_sealed(selected, sealed)
    unknown = sorted({r.example_id for r in records} - {e.id for e in dataset.origin})
    if unknown:
        raise DatasetError(
            f"Outputs for {len(unknown)} examples {dataset.name!r} lacks: "
            f"{unknown[:10]}"
        )
    examples = {e.id: e for e in selected}
    chosen: dict[TrialKey, ImportedTrial] = {}
    for record in records:
        if record.example_id not in examples:
            continue
        key = (record.example_id, record.repetition)
        if key in chosen:
            raise DatasetError(
                f"Two outputs for example {record.example_id!r} repetition "
                f"{record.repetition}"
            )
        chosen[key] = record
    if not chosen:
        raise DatasetError(f"No output is for the selection from {dataset.name!r}")
    reps = (
        repetitions
        if repetitions is not None
        else 1 + max(repetition for _, repetition in chosen)
    )
    if reps < 1:
        raise ValueError("repetitions must be >= 1")
    beyond = sorted(f"{i}#{r}" for i, r in chosen if r >= reps)
    if beyond:
        raise DatasetError(f"Outputs beyond {reps} repetitions: {beyond[:10]}")
    for split in sorted(sealed):
        keys = [(e.id, r) for e in selected if split in e.splits for r in range(reps)]
        lacking = sum(1 for key in keys if key not in chosen)
        if lacking:
            raise DatasetError(
                f"Sealed split {split!r} is evaluated whole, but {lacking} of its "
                f"{len(keys)} (example, repetition) pairs have no output (record a "
                "failed one with error)"
            )
    run_store = resolve_store(store, persist)
    task_info = ComponentInfo(
        name=task,
        kind=IMPORTED_TASK_KIND,
        version=task_version,
        config=dict(task_config or {}),
    )
    scorer_infos = [s.describe() for s in scorer_list]
    dataset_ref = selected.ref()
    run_name = name or task
    run = EvaluationRun(
        id=new_run_id(run_name),
        name=run_name,
        kind="import",
        created_at=utc_now(),
        description=description,
        evaluation=evaluation,
        dataset=dataset_ref,
        task=task_info,
        scorers=scorer_infos,
        config=RunConfig(
            repetitions=reps,
            concurrency=concurrency,
            scorer_timeout_s=scorer_timeout_s,
            group_by=list(group_by),
            cluster_by=cluster_by,
            sealed_splits=sorted(sealed),
        ),
        provenance=capture_provenance(sources=scorer_sources(scorer_list)),
        config_hash=config_hash(task_info, scorer_infos, dataset_ref, reps),
        tags=list(tags),
        metadata=dict(metadata or {}),
        examples=list(selected),
    )
    if run_store is not None:
        run_store.create(run)
    order: list[TrialKey] = [(e.id, r) for e in selected for r in range(reps)]
    adapter = output_adapter(output_type)
    executor = Executor(
        run,
        store=run_store,
        scorers=scorer_list,
        scorer_timeout_s=scorer_timeout_s,
        progress=progress,
        total=len(order),
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def work(record: ImportedTrial) -> None:
        async with semaphore:
            example = examples[record.example_id]
            stored = to_jsonable(record.output)
            trial = _trial(
                record,
                example.content_hash,
                sealed=bool(sealed.intersection(example.splits)),
                output=stored,
            )
            executor.add_task_usage(trial)
            output = rehydrate_output(adapter, stored) if trial.ok else None
            await executor.score(trial, example, output)
            await executor.record(trial)

    return await run_all(
        executor, [work(r) for r in chosen.values()], metrics=metrics, order=order
    )
