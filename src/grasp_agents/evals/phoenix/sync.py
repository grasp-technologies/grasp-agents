"""
Phoenix as the shared home of datasets and the UI for runs.

Local run records stay canonical: datasets are pulled into a local cache by
version, and finished runs are pushed as Phoenix experiments (one experiment
per run, one Phoenix run per trial, one evaluation per score). Our example
ids travel as Phoenix's external example ids.
"""

import asyncio
import base64
import binascii
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, TypeAdapter

from grasp_agents.evals._util import canonical_json, short_hash, slugify, to_jsonable
from grasp_agents.evals.dataset import Dataset, DatasetError
from grasp_agents.evals.store import RunStore, default_evals_dir
from grasp_agents.evals.types import (
    EvaluationRun,
    Example,
    ExampleRecord,
    PhoenixLink,
    Score,
    ScoreReason,
    Trial,
    content_digest,
    with_record,
)

from .client import AnnotatorKind, PhoenixClient, PhoenixError

logger = logging.getLogger(__name__)

# Example metadata key for what Phoenix records cannot hold natively: split
# membership (versioned with the example, unlike Phoenix splits) and which
# fields were wrapped into an object (Phoenix inputs/outputs must be objects).
GRASP_KEY = "grasp_example"
_VALUE_KEY = "value"
_SOURCE_PREFIX = "phoenix:"


class StaleDatasetError(RuntimeError):
    pass


class DatasetPushError(RuntimeError):
    pass


def phoenix_source(base_url: str, dataset_id: str) -> str:
    """``Dataset.source`` of a dataset pulled from ``base_url``."""
    return f"{_SOURCE_PREFIX}{base_url}/datasets/{dataset_id}"


def parse_phoenix_source(source: str | None) -> tuple[str, str] | None:
    """``(base_url, dataset_id)`` of a source written by :func:`phoenix_source`."""
    if not source or not source.startswith(_SOURCE_PREFIX):
        return None
    base_url, sep, dataset_id = source.removeprefix(_SOURCE_PREFIX).rpartition(
        "/datasets/"
    )
    return (base_url, dataset_id) if sep and base_url and dataset_id else None


@dataclass(frozen=True)
class PhoenixRecord:
    id: str
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]
    splits: list[str]

    def content(self) -> str:
        return _content(self.input, self.output, self.metadata)


def _content(raw_input: Any, raw_output: Any, metadata: Any) -> str:
    plain = {
        k: v
        for k, v in cast("dict[str, Any]", metadata or {}).items()
        if k != GRASP_KEY
    }
    return canonical_json(
        {"input": raw_input or {}, "output": raw_output or {}, "metadata": plain}
    )


def to_phoenix_record(example: Example[Any, Any]) -> PhoenixRecord:
    content = example.record
    wrapped: list[str] = []
    payload = to_jsonable(content.input)
    if not isinstance(payload, dict):
        payload = {_VALUE_KEY: payload}
        wrapped.append("input")
    output: Any = {}
    grasp: dict[str, Any] = {}
    if content.reference is not None:
        output = to_jsonable(content.reference)
        if not isinstance(output, dict):
            output = {_VALUE_KEY: output}
            wrapped.append("reference")
        elif not output:
            grasp["empty_reference"] = True
    metadata = dict(cast("dict[str, Any]", to_jsonable(content.metadata)))
    if wrapped:
        grasp["wrapped"] = wrapped
    if example.splits:
        grasp["splits"] = list(example.splits)
    if grasp:
        metadata[GRASP_KEY] = grasp
    return PhoenixRecord(
        id=example.id,
        input=cast("dict[str, Any]", payload),
        output=cast("dict[str, Any]", output),
        metadata=metadata,
        splits=list(example.splits),
    )


def record_id(record: Mapping[str, Any]) -> str:
    """Our id of a Phoenix example: its external id, else its Phoenix id."""
    return str(record["id"])


def record_node_id(record: Mapping[str, Any]) -> str:
    """The id Phoenix expects when logging runs against this example."""
    return str(record.get("node_id") or record["id"])


def _is_example_node_id(value: str) -> bool:
    # Phoenix-born examples carry Phoenix's own (global) id; sending it to
    # another dataset would reference that dataset's rows.
    try:
        decoded = base64.b64decode(value, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    kind, _, number = decoded.partition(":")
    return kind == "DatasetExample" and number.isdigit()


def _unpack(record: Mapping[str, Any]) -> ExampleRecord:
    # The example content a Phoenix row holds, as our example stored it.
    metadata = dict(cast("dict[str, Any]", record.get("metadata") or {}))
    info = cast("dict[str, Any]", metadata.pop(GRASP_KEY, None) or {})
    wrapped = set(cast("list[str]", info.get("wrapped") or []))
    raw_input = cast("dict[str, Any]", record.get("input") or {})
    raw_output = cast("dict[str, Any]", record.get("output") or {})
    value = raw_input.get(_VALUE_KEY) if "input" in wrapped else raw_input
    reference: Any = None
    if raw_output or info.get("empty_reference"):
        reference = raw_output.get(_VALUE_KEY) if "reference" in wrapped else raw_output
    return ExampleRecord(value, reference, metadata)


def from_phoenix_record(
    record: Mapping[str, Any],
    *,
    input_adapter: TypeAdapter[Any],
    reference_adapter: TypeAdapter[Any],
) -> Example[Any, Any]:
    content = _unpack(record)
    example = Example[Any, Any](
        id=record_id(record),
        input=input_adapter.validate_python(content.input),
        reference=None
        if content.reference is None
        else reference_adapter.validate_python(content.reference),
        metadata=content.metadata,
        splits=_splits_of(record),
        content_hash=content_digest(*content),
    )
    return with_record(example, content)


# --- Datasets ---


def _cache_path(root: Path, base_url: str, name: str, version_id: str) -> Path:
    return (
        root
        / "phoenix"
        / short_hash(base_url, length=8)
        / slugify(name, max_len=80)
        / f"{short_hash(version_id)}.jsonl"
    )


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{id(text):x}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read_cache(path: Path, base_url: str, version_id: str) -> list[Any] | None:
    try:
        header, *lines = path.read_text(encoding="utf-8").splitlines()
        meta = json.loads(header)
        records = [json.loads(line) for line in lines]
    except (OSError, ValueError):
        return None
    if meta.get("base_url") != base_url or meta.get("version_id") != version_id:
        return None
    if len(records) != meta.get("count"):
        return None
    return [meta, records]


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
    ``cache_dir`` (default ``<evals dir>/datasets``), keyed by server and
    version: later pulls of a pinned version need no network.
    """
    root = (
        Path(cache_dir) if cache_dir is not None else default_evals_dir() / "datasets"
    )
    input_adapter: TypeAdapter[Any] = TypeAdapter(input_type)
    reference_adapter: TypeAdapter[Any] = TypeAdapter(reference_type)
    if version is not None:
        cached = _read_cache(
            _cache_path(root, client.base_url, name, version), client.base_url, version
        )
        if cached is not None:
            meta, records = cached
            return _build(meta, records, input_adapter, reference_adapter)
    await client.check_server()
    if await client.find_dataset(name) is None:
        raise DatasetError(f"No Phoenix dataset named {name!r} at {client.base_url}")
    remote = await client.call(
        client.sdk.datasets.get_dataset(
            dataset=name, version_id=version, timeout=int(client.timeout_s)
        )
    )
    records = [dict(r) for r in remote.examples]
    meta = {
        "name": remote.name,
        "dataset_id": remote.id,
        "version_id": remote.version_id,
        "description": remote.description,
        "base_url": client.base_url,
        "count": len(records),
    }
    _write_atomic(
        _cache_path(root, client.base_url, name, remote.version_id),
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [meta, *records]) + "\n",
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
        source=phoenix_source(str(meta["base_url"]), str(meta["dataset_id"])),
        description=cast("str | None", meta.get("description")),
    )


class DatasetPushResult(BaseModel):
    name: str
    dataset_id: str
    version_id: str
    created: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0


def _upload(record: PhoenixRecord, example_id: str | None) -> dict[str, Any]:
    upload: dict[str, Any] = {
        "input": record.input,
        "output": record.output,
        "metadata": record.metadata,
    }
    if record.splits:
        upload["splits"] = record.splits
    if example_id is not None:
        upload["id"] = example_id
    return upload


async def push_dataset(
    client: PhoenixClient,
    dataset: Dataset[Any, Any],
    *,
    name: str | None = None,
    base_version: str | None = None,
    force: bool = False,
    allow_deletes: bool = False,
) -> DatasetPushResult:
    """
    Make the Phoenix dataset ``name`` hold exactly ``dataset`` (a new version
    when anything changed).

    The push must be based on the latest version — pass the ``base_version``
    you pulled (a pulled dataset carries it) — so it never overwrites
    someone else's changes; ``force=True`` skips the check. Examples the
    Phoenix dataset has but ``dataset`` lacks are deleted only with
    ``allow_deletes=True``, and a derived subset (``split``, ``filter``, …)
    is refused unless forced, since pushing it would delete the rest.
    """
    if dataset.selection and not force:
        raise DatasetPushError(
            f"This is a subset of {dataset.name!r} ({', '.join(dataset.selection)}); "
            "pushing it would remove the other examples from Phoenix. Push the "
            "whole dataset, or force."
        )
    await client.check_server()
    target = name or dataset.name
    records = [to_phoenix_record(e) for e in dataset]
    existing = await client.find_dataset(target)
    timeout = int(client.timeout_s)
    if existing is None:
        created = await client.upsert_dataset(
            name=target,
            examples=[
                _upload(r, None if _is_example_node_id(r.id) else r.id) for r in records
            ],
            description=dataset.description,
        )
        return DatasetPushResult(
            name=target,
            dataset_id=created.id,
            version_id=created.version_id,
            created=len(records),
        )
    dataset_id = str(existing["id"])
    remote = await client.call(
        client.sdk.datasets.get_dataset(dataset=dataset_id, timeout=timeout)
    )
    pulled_from = parse_phoenix_source(dataset.source)
    base = base_version or (
        dataset.version if pulled_from == (client.base_url, dataset_id) else None
    )
    if not force and base != remote.version_id:
        raise StaleDatasetError(
            f"Phoenix dataset {target!r} is at version {remote.version_id}, but this "
            f"copy is based on {base or 'no pulled version'}. Pull the latest version, "
            "merge your changes and push again (or force)."
        )
    remote_by_id = {record_id(r): cast("Mapping[str, Any]", r) for r in remote.examples}
    local_ids = {r.id for r in records}
    deleted = sorted(set(remote_by_id) - local_ids)
    if deleted and not allow_deletes:
        raise DatasetPushError(
            f"Pushing would delete {len(deleted)} examples of Phoenix dataset "
            f"{target!r} that this copy lacks (e.g. {deleted[:5]}); allow deletes to "
            "remove them."
        )
    created_n = updated_n = unchanged_n = 0
    for record in records:
        remote_record = remote_by_id.get(record.id)
        if remote_record is None:
            created_n += 1
        elif (
            record.content()
            == _content(
                remote_record.get("input"),
                remote_record.get("output"),
                remote_record.get("metadata"),
            )
            and _splits_of(remote_record) == record.splits
        ):
            unchanged_n += 1
        else:
            updated_n += 1
    if not created_n and not updated_n and not deleted:
        return DatasetPushResult(
            name=target,
            dataset_id=dataset_id,
            version_id=remote.version_id,
            unchanged=unchanged_n,
        )
    uploaded = await client.upsert_dataset(
        name=target,
        examples=[
            _upload(
                r,
                r.id if r.id in remote_by_id or not _is_example_node_id(r.id) else None,
            )
            for r in records
        ],
        description=dataset.description,
    )
    return DatasetPushResult(
        name=target,
        dataset_id=dataset_id,
        version_id=uploaded.version_id,
        created=created_n,
        updated=updated_n,
        deleted=len(deleted),
        unchanged=unchanged_n,
    )


def _splits_of(record: Mapping[str, Any]) -> list[str]:
    metadata = cast("dict[str, Any]", record.get("metadata") or {})
    info = cast("dict[str, Any]", metadata.get(GRASP_KEY) or {})
    return list(cast("list[str]", info.get("splits") or []))


# --- Runs ---


def _evaluation_fields(score: Score) -> dict[str, Any]:
    value = score.value
    fields: dict[str, Any] = {"explanation": score.explanation}
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        reason = score.reason or ScoreReason.NON_FINITE_VALUE
        return {**fields, "error": f"unscored: {reason}"}
    if isinstance(value, bool):
        return {
            **fields,
            "score": 1.0 if value else 0.0,
            "label": "pass" if value else "fail",
        }
    if isinstance(value, str):
        return {**fields, "label": value}
    return {**fields, "score": float(value)}


def _experiment_metadata(run: EvaluationRun, withheld: int) -> dict[str, Any]:
    return {
        "grasp_run_id": run.id,
        "kind": run.kind,
        "status": str(run.status),
        "evaluation": run.evaluation,
        "parent_run_id": run.parent_run_id,
        "task": run.task.model_dump(mode="json"),
        "scorers": {e.name: e.version for e in run.scorers},
        "metrics": {
            m.name: {"value": m.value, "ci": [m.ci_low, m.ci_high], "n": m.n}
            for m in run.metrics
        },
        "counts": run.counts.model_dump(),
        "sealed_trials_withheld": withheld,
        "dataset": {
            "fingerprint": run.dataset.fingerprint,
            "selection": run.dataset.selection,
        },
        "config_hash": run.config_hash,
        "git_commit": run.provenance.git_commit,
        "git_dirty": run.provenance.git_dirty,
        "invalid_reason": run.invalid_reason,
    }


def _shareable(run: EvaluationRun) -> list[Example[Any, Any]]:
    sealed = set(run.config.sealed_splits)
    return [e for e in run.examples if not sealed.intersection(e.splits)]


async def _resolve_dataset(
    client: PhoenixClient, run: EvaluationRun
) -> tuple[str, str]:
    source = parse_phoenix_source(run.dataset.source)
    if source is not None and source[0] == client.base_url and run.dataset.version:
        return source[1], run.dataset.version
    # A run on a local file (or on another server's dataset): reuse a Phoenix
    # dataset of the same name only if it holds exactly these examples.
    examples = _shareable(run)
    records = [to_phoenix_record(e) for e in examples]
    existing = await client.find_dataset(run.dataset.name)
    timeout = int(client.timeout_s)
    if existing is None and not records:
        raise DatasetPushError(
            f"Every example of run {run.id} is in a sealed split, so there is no "
            f"Phoenix dataset {run.dataset.name!r} to attach it to: push a run on "
            "the shareable splits first"
        )
    if existing is None:
        created = await client.upsert_dataset(
            name=run.dataset.name,
            examples=[
                _upload(r, None if _is_example_node_id(r.id) else r.id) for r in records
            ],
        )
        return created.id, created.version_id
    dataset_id = str(existing["id"])
    remote = await client.call(
        client.sdk.datasets.get_dataset(dataset=dataset_id, timeout=timeout)
    )
    remote_hashes = {
        record_id(r): content_digest(*_unpack(r))
        for r in cast("list[Mapping[str, Any]]", remote.examples)
    }
    differing = [e.id for e in examples if remote_hashes.get(e.id) != e.content_hash]
    if differing:
        raise DatasetPushError(
            f"Phoenix dataset {run.dataset.name!r} at {client.base_url} does not hold "
            f"{len(differing)} of this run's examples as evaluated (e.g. "
            f"{differing[:3]}). Push the dataset first (grasp-evals datasets push), or "
            "run on a dataset pulled from this server."
        )
    return dataset_id, remote.version_id


def _trial_key(trial: Trial) -> str:
    return f"{trial.example_id}#{trial.repetition}"


def _digest(trial: Trial) -> str:
    return short_hash(
        canonical_json(
            {
                "output": trial.output,
                "error": None if trial.error is None else trial.error.message,
                "scores": [
                    [s.name, s.value, s.explanation, s.reason] for s in trial.scores
                ],
                "failures": [
                    [f.scorer, f.error.message] for f in trial.scorer_failures
                ],
            }
        )
    )


async def _experiment_for(
    client: PhoenixClient, run: EvaluationRun, link: PhoenixLink, withheld: int
) -> tuple[str, bool]:
    """The run's experiment id, and whether it already existed."""
    if link.experiment_id is not None:
        return link.experiment_id, True
    # An earlier push may have created it without recording the id (a lost
    # response, or the same server under another URL spelling).
    experiments = await client.call(
        client.sdk.experiments.list(
            dataset_id=link.dataset_id, timeout=int(client.timeout_s)
        )
    )
    for experiment in experiments:
        metadata = cast("Mapping[str, Any]", experiment.get("metadata") or {})
        if metadata.get("grasp_run_id") == run.id:
            return str(experiment["id"]), True
    created = await client.call(
        client.sdk.experiments.create(
            dataset_id=link.dataset_id,
            dataset_version_id=link.dataset_version_id,
            experiment_name=run.id,
            experiment_description=run.description or run.name,
            experiment_metadata=_experiment_metadata(run, withheld),
            repetitions=run.config.repetitions,
            timeout=int(client.timeout_s),
        )
    )
    return str(created["id"]), False


async def push_run(
    client: PhoenixClient,
    run: EvaluationRun,
    *,
    store: RunStore | None = None,
    concurrency: int = 8,
) -> PhoenixLink:
    """
    Mirror a finished run into Phoenix as an experiment.

    The dataset is resolved (a dataset pulled from this server, or one of the
    same name holding exactly the run's examples, created when missing); then
    every trial is logged as a Phoenix run and every score as an evaluation.
    Trials in sealed splits are withheld (their count is in the experiment
    metadata). Progress is recorded on ``run.phoenix`` (and saved to
    ``store``): an interrupted push resumes where it stopped, and pushing again
    after the run was resumed logs only the trials that changed.
    """
    if not run.finished:
        raise ValueError(f"Run {run.id} is still running; push it when it finishes")
    await client.check_server()
    link = run.phoenix
    if link is None or link.base_url != client.base_url:
        dataset_id, version_id = await _resolve_dataset(client, run)
        link = PhoenixLink(
            base_url=client.base_url,
            dataset_id=dataset_id,
            dataset_version_id=version_id,
        )
        run.phoenix = link
    remote = await client.call(
        client.sdk.datasets.get_dataset(
            dataset=link.dataset_id,
            version_id=link.dataset_version_id,
            timeout=int(client.timeout_s),
        )
    )
    node_ids = {record_id(r): record_node_id(r) for r in remote.examples}
    trials = [t for t in run.trials if not t.sealed]
    withheld = len(run.trials) - len(trials)
    missing = sorted({t.example_id for t in trials} - set(node_ids))
    if missing:
        raise DatasetPushError(
            f"Phoenix dataset version {link.dataset_version_id} lacks examples "
            f"{missing[:5]}"
        )
    experiment_id, existed = await _experiment_for(client, run, link, withheld)
    if link.experiment_id is None:
        link.experiment_id = experiment_id
        if store is not None:
            store.save(run)
    annotators: dict[str, AnnotatorKind] = {
        e.name: e.annotator or "CODE" for e in run.scorers
    }
    versions = {e.name: e.version for e in run.scorers}
    logged = dict(link.logged_trials)
    known_runs: dict[tuple[str, int], str] = {}
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    async def existing_run_id(node_id: str, repetition: int) -> str:
        async with lock:
            if (node_id, repetition) not in known_runs:
                experiment = await client.call(
                    client.sdk.experiments.get_experiment(experiment_id=experiment_id)
                )
                known_runs.update(
                    {
                        (
                            str(r["dataset_example_id"]),
                            int(r["repetition_number"]),
                        ): str(r["id"])
                        for r in experiment["task_runs"]
                    }
                )
            found = known_runs.get((node_id, repetition))
        if found is None:
            raise DatasetPushError(
                f"Phoenix reports a run for example {node_id} repetition {repetition} "
                "of this experiment but does not list it"
            )
        return found

    async def push_trial(trial: Trial) -> None:
        async with semaphore:
            node_id = node_ids[trial.example_id]
            ended = trial.started_at + timedelta(seconds=trial.duration_s)
            repetition = trial.repetition + 1
            try:
                logged_run = await client.call(
                    client.sdk.experiments.log_run(
                        experiment_id=experiment_id,
                        dataset_example_id=node_id,
                        output=trial.output,
                        start_time=trial.started_at,
                        end_time=ended,
                        repetition_number=repetition,
                        trace_id=trial.trace_id,
                        error=None if trial.error is None else trial.error.message,
                        timeout=int(client.timeout_s),
                    )
                )
                run_id = str(logged_run["id"])
            except PhoenixError as exc:
                # A successful run of this trial is already logged; its scores
                # are updated below.
                if exc.status != 409:
                    raise
                run_id = await existing_run_id(node_id, repetition)
            for score in trial.scores:
                scorer = score.scorer or score.name
                await client.call(
                    client.sdk.experiments.log_evaluation(
                        experiment_run_id=run_id,
                        name=score.name,
                        annotator_kind=annotators.get(scorer, "CODE"),
                        start_time=ended,
                        end_time=ended,
                        metadata={
                            "scorer": scorer,
                            "scorer_version": versions.get(scorer),
                            "reason": score.reason,
                            **cast("dict[str, Any]", to_jsonable(score.metadata)),
                        },
                        trace_id=trial.trace_id,
                        timeout=int(client.timeout_s),
                        **_evaluation_fields(score),
                    )
                )
            for failure in trial.scorer_failures:
                await client.call(
                    client.sdk.experiments.log_evaluation(
                        experiment_run_id=run_id,
                        name=failure.scorer,
                        annotator_kind=annotators.get(failure.scorer, "CODE"),
                        start_time=ended,
                        end_time=ended,
                        error=failure.error.message,
                        timeout=int(client.timeout_s),
                    )
                )
            async with lock:
                logged[_trial_key(trial)] = _digest(trial)

    pending = [t for t in trials if logged.get(_trial_key(t)) != _digest(t)]
    try:
        async with asyncio.TaskGroup() as group:
            for trial in pending:
                group.create_task(push_trial(trial))
    except BaseExceptionGroup as failures:
        # Report the first failure itself (e.g. a PhoenixError), not the group.
        errors = [
            e for e in failures.exceptions if not isinstance(e, asyncio.CancelledError)
        ]
        if errors:
            raise errors[0] from failures
        raise
    finally:
        link.logged_trials = dict(sorted(logged.items()))
        if store is not None:
            store.save(run)
    if existed and pending:
        await client.update_experiment_metadata(
            experiment_id, _experiment_metadata(run, withheld)
        )
    return link


def experiment_url(link: PhoenixLink) -> str | None:
    if link.experiment_id is None:
        return None
    return (
        f"{link.base_url}/datasets/{link.dataset_id}/compare"
        f"?experimentId={link.experiment_id}"
    )
