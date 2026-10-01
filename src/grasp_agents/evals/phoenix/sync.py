"""
Phoenix as the shared home of datasets and the UI for runs.

Local run records stay canonical: datasets are pulled into an immutable
local cache by version, and finished runs are pushed as Phoenix experiments
(one experiment per run, one Phoenix run per trial, one evaluation per score).
"""

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter

from grasp_agents.evals._util import canonical_json, short_hash, slugify, to_jsonable
from grasp_agents.evals.dataset import Dataset
from grasp_agents.evals.store import RunStore, default_evals_dir
from grasp_agents.evals.types import (
    EvaluationRun,
    Example,
    PhoenixLink,
    Score,
    Trial,
)

from .client import EXTERNAL_EXAMPLE_IDS, AnnotatorKind, PhoenixClient

logger = logging.getLogger(__name__)

# Example metadata key carrying what Phoenix records cannot hold natively:
# our stable id, split membership, and which fields were wrapped into an
# object (Phoenix inputs/outputs must be JSON objects).
GRASP_KEY = "grasp_example"
_VALUE_KEY = "value"


class StaleDatasetError(RuntimeError):
    pass


class PhoenixCompatibilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class PhoenixRecord:
    id: str
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]
    splits: list[str]

    def content(self) -> str:
        return canonical_json(
            {"input": self.input, "output": self.output, "metadata": self.metadata}
        )


def to_phoenix_record(example: Example[Any, Any]) -> PhoenixRecord:
    wrapped: list[str] = []
    payload = to_jsonable(example.input)
    if not isinstance(payload, dict):
        payload = {_VALUE_KEY: payload}
        wrapped.append("input")
    output: Any = {}
    if example.reference is not None:
        output = to_jsonable(example.reference)
        if not isinstance(output, dict):
            output = {_VALUE_KEY: output}
            wrapped.append("reference")
    metadata = cast("dict[str, Any]", to_jsonable(example.metadata))
    metadata[GRASP_KEY] = {
        "id": example.id,
        "wrapped": wrapped,
        "has_reference": example.reference is not None,
        "splits": list(example.splits),
    }
    return PhoenixRecord(
        id=example.id,
        input=cast("dict[str, Any]", payload),
        output=cast("dict[str, Any]", output),
        metadata=metadata,
        splits=list(example.splits),
    )


def record_id(record: Mapping[str, Any]) -> str:
    """Our example id for a Phoenix example (its own id if it never was ours)."""
    metadata = cast("dict[str, Any]", record.get("metadata") or {})
    info = cast("dict[str, Any]", metadata.get(GRASP_KEY) or {})
    return str(info.get("id") or record["id"])


def record_node_id(record: Mapping[str, Any]) -> str:
    """The id Phoenix expects when logging runs against this example."""
    return str(record.get("node_id") or record["id"])


def from_phoenix_record(
    record: Mapping[str, Any],
    *,
    input_adapter: TypeAdapter[Any],
    reference_adapter: TypeAdapter[Any],
) -> Example[Any, Any]:
    metadata = dict(cast("dict[str, Any]", record.get("metadata") or {}))
    info = cast("dict[str, Any]", metadata.pop(GRASP_KEY, None) or {})
    wrapped = set(cast("list[str]", info.get("wrapped") or []))
    raw_input = cast("dict[str, Any]", record.get("input") or {})
    raw_output = cast("dict[str, Any]", record.get("output") or {})
    value = raw_input.get(_VALUE_KEY) if "input" in wrapped else raw_input
    has_reference = bool(info.get("has_reference", bool(raw_output)))
    reference: Any = None
    if has_reference:
        reference = raw_output.get(_VALUE_KEY) if "reference" in wrapped else raw_output
    return Example[Any, Any](
        id=record_id(record),
        input=input_adapter.validate_python(value),
        reference=None
        if reference is None
        else reference_adapter.validate_python(reference),
        metadata=metadata,
        splits=list(cast("list[str]", info.get("splits") or [])),
    )


# --- Datasets ---


def _cache_path(root: Path, name: str, version_id: str) -> Path:
    return (
        root / "phoenix" / slugify(name, max_len=80) / f"{short_hash(version_id)}.jsonl"
    )


async def pull_dataset(
    client: PhoenixClient,
    name: str,
    *,
    version: str | None = None,
    input_type: Any = Any,
    reference_type: Any = Any,
    cache_dir: str | Path | None = None,
) -> Dataset[Any, Any]:
    """
    A Phoenix dataset version as a :class:`Dataset` (latest when ``version``
    is omitted). Versions are immutable, so each is cached under
    ``cache_dir`` (default ``<evals dir>/datasets``) and later pulls of a
    pinned version need no network.
    """
    root = (
        Path(cache_dir) if cache_dir is not None else default_evals_dir() / "datasets"
    )
    input_adapter: TypeAdapter[Any] = TypeAdapter(input_type)
    reference_adapter: TypeAdapter[Any] = TypeAdapter(reference_type)
    if version is not None:
        cached = _cache_path(root, name, version)
        if cached.exists():
            header, *lines = cached.read_text(encoding="utf-8").splitlines()
            meta = json.loads(header)
            return _build(
                meta,
                [json.loads(line) for line in lines],
                input_adapter,
                reference_adapter,
            )
    found = await client.find_dataset(name)
    if found is None:
        raise LookupError(f"No Phoenix dataset named {name!r} at {client.base_url}")
    dataset_id = str(found["id"])
    version_id, records = await client.dataset_examples(dataset_id, version)
    meta = {
        "name": name,
        "dataset_id": dataset_id,
        "version_id": version_id,
        "description": found.get("description"),
    }
    path = _cache_path(root, name, version_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [meta, *records]) + "\n",
        encoding="utf-8",
    )
    return _build(meta, records, input_adapter, reference_adapter)


def _build(
    meta: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    input_adapter: TypeAdapter[Any],
    reference_adapter: TypeAdapter[Any],
) -> Dataset[Any, Any]:
    examples = [
        from_phoenix_record(
            r, input_adapter=input_adapter, reference_adapter=reference_adapter
        )
        for r in records
    ]
    return Dataset(
        examples,
        name=str(meta["name"]),
        version=str(meta["version_id"]),
        source=f"phoenix:{meta['dataset_id']}",
        description=cast("str | None", meta.get("description")),
    )


def _upload_fields(records: Sequence[PhoenixRecord]) -> dict[str, Any]:
    return {
        "inputs": [r.input for r in records],
        "outputs": [r.output for r in records],
        "metadata": [r.metadata for r in records],
        "splits": [r.splits or None for r in records],
    }


async def push_dataset(
    client: PhoenixClient,
    dataset: Dataset[Any, Any],
    *,
    name: str | None = None,
    base_version: str | None = None,
    force: bool = False,
) -> tuple[str, str]:
    """
    Make the Phoenix dataset ``name`` hold exactly ``dataset``; returns
    ``(dataset_id, version_id)``.

    Replacing an existing dataset deletes its examples that ``dataset`` lacks,
    so it must be based on the latest version: pass the ``base_version`` you
    pulled (a pulled dataset carries it), or ``force=True``. Servers older
    than 15.0 cannot replace or edit examples over REST; there, new examples
    are appended and changed ones are refused.
    """
    target = name or dataset.name
    records = [to_phoenix_record(e) for e in dataset]
    fields = _upload_fields(records)
    external = await client.supports(EXTERNAL_EXAMPLE_IDS)
    ids = [r.id for r in records] if external else None
    existing = await client.find_dataset(target)
    if existing is None:
        return await client.upload_dataset(
            action="create",
            name=target,
            description=dataset.description,
            example_ids=ids,
            **fields,
        )
    dataset_id = str(existing["id"])
    latest, remote = await client.dataset_examples(dataset_id)
    if external:
        base = base_version or (
            dataset.version if dataset.source == f"phoenix:{dataset_id}" else None
        )
        if not force and base != latest:
            raise StaleDatasetError(
                f"Phoenix dataset {target!r} is at version {latest}, but this copy is "
                f"based on {base or 'no pulled version'}. Pull the latest version, "
                "merge your changes and push again (or force)."
            )
        return await client.upload_dataset(
            action="update",
            name=target,
            description=dataset.description,
            example_ids=ids,
            **fields,
        )
    remote_content = {
        record_id(r): canonical_json(
            {
                "input": r.get("input"),
                "output": r.get("output"),
                "metadata": r.get("metadata"),
            }
        )
        for r in remote
    }
    changed = [
        r.id
        for r in records
        if r.id in remote_content and remote_content[r.id] != r.content()
    ]
    if changed:
        raise PhoenixCompatibilityError(
            f"Phoenix {'.'.join(map(str, await client.server_version()))} cannot edit "
            f"examples over REST, and {len(changed)} changed (e.g. {changed[:3]}). "
            "Upgrade Phoenix to >= 15 or push under a new dataset name."
        )
    new = [r for r in records if r.id not in remote_content]
    if not new:
        return dataset_id, latest
    return await client.upload_dataset(
        action="append", name=target, **_upload_fields(new)
    )


# --- Runs ---


def _score_result(score: Score) -> dict[str, Any]:
    value = score.value
    if value is None:
        return {"error": f"unscored: {score.reason or 'no value'}"}
    if isinstance(value, bool):
        return {"score": 1.0 if value else 0.0, "label": "pass" if value else "fail"}
    if isinstance(value, str):
        return {"label": value}
    return {"score": float(value)}


def _experiment_metadata(run: EvaluationRun) -> dict[str, Any]:
    return {
        "grasp_run_id": run.id,
        "kind": run.kind,
        "evaluation": run.evaluation,
        "parent_run_id": run.parent_run_id,
        "task": run.task.model_dump(mode="json"),
        "evaluators": {e.name: e.version for e in run.evaluators},
        "metrics": {
            m.name: {"value": m.value, "ci": [m.ci_low, m.ci_high], "n": m.n}
            for m in run.metrics
        },
        "counts": run.counts.model_dump(),
        "dataset": {
            "fingerprint": run.dataset.fingerprint,
            "selection": run.dataset.selection,
        },
        "config_hash": run.config_hash,
        "git_commit": run.provenance.git_commit,
        "git_dirty": run.provenance.git_dirty,
        "invalid_reason": run.invalid_reason,
    }


async def _ensure_dataset(client: PhoenixClient, run: EvaluationRun) -> tuple[str, str]:
    source = run.dataset.source or ""
    if source.startswith("phoenix:") and run.dataset.version:
        return source.removeprefix("phoenix:"), run.dataset.version
    records = [to_phoenix_record(e) for e in run.examples]
    existing = await client.find_dataset(run.dataset.name)
    if existing is None:
        external = await client.supports(EXTERNAL_EXAMPLE_IDS)
        return await client.upload_dataset(
            action="create",
            name=run.dataset.name,
            example_ids=[r.id for r in records] if external else None,
            **_upload_fields(records),
        )
    dataset_id = str(existing["id"])
    latest, remote = await client.dataset_examples(dataset_id)
    remote_content = {
        record_id(r): canonical_json(
            {
                "input": r.get("input"),
                "output": r.get("output"),
                "metadata": r.get("metadata"),
            }
        )
        for r in remote
    }
    changed = [
        r.id
        for r in records
        if r.id in remote_content and remote_content[r.id] != r.content()
    ]
    if changed:
        raise PhoenixCompatibilityError(
            f"Phoenix dataset {run.dataset.name!r} holds different content for "
            f"{len(changed)} of this run's examples (e.g. {changed[:3]}); push the "
            "dataset first (grasp-evals datasets push) so the run can reference it."
        )
    new = [r for r in records if r.id not in remote_content]
    if not new:
        return dataset_id, latest
    external = await client.supports(EXTERNAL_EXAMPLE_IDS)
    return await client.upload_dataset(
        action="append",
        name=run.dataset.name,
        example_ids=[r.id for r in new] if external else None,
        **_upload_fields(new),
    )


def _trial_key(trial: Trial) -> str:
    return f"{trial.example_id}#{trial.repetition}"


async def push_run(
    client: PhoenixClient,
    run: EvaluationRun,
    *,
    store: RunStore | None = None,
    concurrency: int = 8,
) -> PhoenixLink:
    """
    Mirror a finished run into Phoenix as an experiment.

    The dataset is resolved (or created from the run's examples), then every
    trial is logged as a Phoenix run and every score as an evaluation. Progress
    is recorded on ``run.phoenix`` (and saved to ``store``), so an interrupted
    push resumes where it stopped and a repeated push adds nothing.
    """
    if not run.finished:
        raise ValueError(f"Run {run.id} is still running; push it when it finishes")
    link = run.phoenix
    if link is None or link.base_url != client.base_url:
        dataset_id, version_id = await _ensure_dataset(client, run)
        link = PhoenixLink(
            base_url=client.base_url,
            dataset_id=dataset_id,
            dataset_version_id=version_id,
        )
        run.phoenix = link
    _, remote = await client.dataset_examples(link.dataset_id, link.dataset_version_id)
    node_ids = {record_id(r): record_node_id(r) for r in remote}
    missing = sorted({t.example_id for t in run.trials} - set(node_ids))
    if missing:
        raise PhoenixCompatibilityError(
            f"Phoenix dataset version {link.dataset_version_id} lacks examples "
            f"{missing[:5]}"
        )
    if link.experiment_id is None:
        experiment = await client.create_experiment(
            link.dataset_id,
            version_id=link.dataset_version_id,
            name=run.id,
            description=run.description or run.name,
            metadata=_experiment_metadata(run),
            repetitions=run.config.repetitions,
        )
        link.experiment_id = str(experiment["id"])
        if store is not None:
            store.save(run)
    experiment_id = link.experiment_id
    annotators: dict[str, AnnotatorKind] = {
        e.name: e.annotator or "CODE" for e in run.evaluators
    }
    versions = {e.name: e.version for e in run.evaluators}
    logged = set(link.logged_trials)
    existing_runs: dict[tuple[str, int], str] | None = None
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    async def existing_run_id(node_id: str, repetition: int) -> str | None:
        nonlocal existing_runs
        async with lock:
            if existing_runs is None:
                existing_runs = {
                    (str(r["dataset_example_id"]), int(r["repetition_number"])): str(
                        r["id"]
                    )
                    for r in await client.list_runs(experiment_id)
                }
        return existing_runs.get((node_id, repetition))

    async def push_trial(trial: Trial) -> None:
        async with semaphore:
            node_id = node_ids[trial.example_id]
            ended = trial.started_at + timedelta(seconds=trial.duration_s)
            error = None if trial.error is None else trial.error.message
            run_id = await client.create_run(
                experiment_id,
                dataset_example_id=node_id,
                output=trial.output,
                repetition_number=trial.repetition + 1,
                start_time=trial.started_at,
                end_time=ended,
                trace_id=trial.trace_id,
                error=error,
            )
            if run_id is None:
                run_id = await existing_run_id(node_id, trial.repetition + 1)
                if run_id is None:
                    raise PhoenixCompatibilityError(
                        f"Phoenix reported an existing run for {_trial_key(trial)} "
                        "but does not list it"
                    )
            for score in trial.scores:
                evaluator = score.evaluator or score.name
                await client.upsert_evaluation(
                    experiment_run_id=run_id,
                    name=score.name,
                    annotator_kind=annotators.get(evaluator, "CODE"),
                    start_time=ended,
                    end_time=ended,
                    explanation=score.explanation,
                    metadata={
                        "evaluator": evaluator,
                        "evaluator_version": versions.get(evaluator),
                        "reason": score.reason,
                        **cast("dict[str, Any]", to_jsonable(score.metadata)),
                    },
                    trace_id=trial.trace_id,
                    **_score_result(score),
                )
            for failure in trial.evaluator_failures:
                await client.upsert_evaluation(
                    experiment_run_id=run_id,
                    name=failure.evaluator,
                    annotator_kind=annotators.get(failure.evaluator, "CODE"),
                    start_time=ended,
                    end_time=ended,
                    error=failure.error.message,
                )
            async with lock:
                logged.add(_trial_key(trial))

    pending = [t for t in run.trials if _trial_key(t) not in logged]
    try:
        async with asyncio.TaskGroup() as group:
            for trial in pending:
                group.create_task(push_trial(trial))
    finally:
        link.logged_trials = sorted(logged)
        if store is not None:
            store.save(run)
    return link


def experiment_url(link: PhoenixLink) -> str | None:
    if link.experiment_id is None:
        return None
    return (
        f"{link.base_url}/datasets/{link.dataset_id}/compare"
        f"?experimentId={link.experiment_id}"
    )
