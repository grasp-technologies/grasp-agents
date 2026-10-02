"""
The labeling loop: choose stored outputs for people to label, collect their
labels — from filled-in files or Phoenix annotations — into a labels dataset,
and validate judges against it with :func:`judge_validation`.
"""

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from ._util import canonical_json, short_hash, utc_now
from .dataset import Dataset, DatasetError
from .types import EvaluationRun, JudgedOutput, ScoreValue, Trial, input_digest
from .validation import (
    EVALUATOR_TASK_KIND,
    PROBE_TASK_KIND,
    LabelMismatchError,
    comparable,
    label_value_text,
)

logger = logging.getLogger(__name__)

# Runs whose outputs are not a system's answers (a judge's verdicts, pairs).
_NOT_SAMPLEABLE = {
    EVALUATOR_TASK_KIND: "a judge validation",
    PROBE_TASK_KIND: "a probe",
}


def label_split(example_id: str, test_share: float) -> str:
    """
    ``"test"`` for a stable ``test_share`` of example ids, ``"dev"`` for the
    rest: every output of one example lands in the same split, in every
    labeling round at the same share.
    """
    bucket = int(short_hash(example_id, length=8), 16) / 16**8
    return "test" if bucket < test_share else "dev"


def _judged_record(run: EvaluationRun, trial: Trial) -> dict[str, Any] | None:
    example = run.example(trial.example_id)
    if example is None:
        return None
    content = example.record
    item: dict[str, Any] = {"input": content.input, "output": trial.output}
    if content.reference is not None:
        item["reference"] = content.reference
    if content.metadata:
        item["metadata"] = content.metadata
    metadata: dict[str, Any] = {
        "example_id": trial.example_id,
        "repetition": trial.repetition,
        "run_id": run.id,
    }
    if trial.trace_id:
        metadata["trace_id"] = trial.trace_id
    return {"id": input_digest(item), "input": item, "metadata": metadata}


def _check_sampleable(run: EvaluationRun) -> None:
    what = _NOT_SAMPLEABLE.get(run.task.kind) or (
        "a pairwise" if run.kind == "pairwise" else None
    )
    if what is not None:
        raise ValueError(
            f"Run {run.id} is {what} run: its outputs are verdicts, not answers "
            "to label; sample the run of the system the judge scores"
        )


def _candidates(run: EvaluationRun) -> list[tuple[dict[str, Any], Trial]]:
    # Sealed trials are never exported: labels are seen by people and agents.
    _check_sampleable(run)
    seen: set[str] = set()
    found: list[tuple[dict[str, Any], Trial]] = []
    for trial in run.trials:
        if trial.sealed or not trial.ok:
            continue
        record = _judged_record(run, trial)
        if record is None or record["id"] in seen:
            continue
        seen.add(record["id"])
        found.append((record, trial))
    return found


def judged_outputs(
    run: EvaluationRun,
    *,
    input_type: Any = Any,
    output_type: Any = Any,
    reference_type: Any = Any,
    name: str | None = None,
) -> Dataset[JudgedOutput[Any, Any, Any], Any]:
    """
    ``run``'s successful outputs with the examples they answer, as a dataset
    of :class:`JudgedOutput` inputs (e.g. for :func:`judge_probes`). Sealed
    trials are left out, and identical outputs for one example appear once.
    """
    records = [record for record, _ in _candidates(run)]
    return Dataset.from_records(
        records,
        input_type=JudgedOutput[input_type, output_type, reference_type],
        name=name or f"{run.name}-outputs",
        source=f"run:{run.id}",
    )


def _verdict(trial: Trial, name: str | None) -> ScoreValue | None:
    score = trial.score(name) if name is not None else None
    return None if score is None else score.value


def _require_score(run: EvaluationRun, name: str | None, option: str) -> None:
    if name is not None and name not in run.score_names():
        raise ValueError(
            f"No trial of {run.id} has a score named {name!r} ({option}); its "
            f"scores: {run.score_names()}"
        )


def _disagree(
    verdict: ScoreValue | None, other: ScoreValue | None, against: str
) -> bool:
    if verdict is None or other is None:
        return False
    try:
        first, second = comparable(verdict, other)
    except LabelMismatchError as exc:
        raise ValueError(f"--against {against!r}: {exc}") from exc
    return first != second


def _interleave(queues: Sequence[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    order: list[dict[str, Any]] = []
    pending = [list(q) for q in queues]
    while any(pending):
        for queue in pending:
            if queue:
                order.append(queue.pop(0))
    return order


def sample_for_labeling(
    run: EvaluationRun,
    n: int,
    *,
    score: str | None = None,
    against: str | None = None,
    strata: str | None = None,
    exclude: Iterable[str] = (),
    test_share: float = 0.4,
    splits: Mapping[str, str] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Up to ``n`` of ``run``'s outputs for people to label, as dataset records
    whose ``reference`` is left empty: fill it in with the correct verdict
    for ``score`` (or ``{score: verdict}`` for several), then
    :func:`import_labels`. The judge's verdicts are not included, so labels
    are blind.

    Outputs where ``score`` and ``against`` (another score of the same trials,
    e.g. a cheap check) disagree come first. The rest alternate between the
    verdicts ``score`` took — TPR and TNR need both passes and fails labeled —
    and, within a verdict, between the values of the metadata key ``strata``,
    in a seeded random order; outputs the judge did not score come last.
    Sealed trials, failed trials and ``exclude`` (ids already labeled or
    requested) are never chosen. ``score`` need not exist yet (labels for a
    judge still to be built); ``against`` must. Each output goes to the split ``splits``
    gives its example (from earlier rounds), else to ``test`` for a stable
    ``test_share`` of the examples (see :func:`label_split`), else ``dev``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 <= test_share <= 1.0:
        raise ValueError("test_share must be between 0 and 1")
    if score is not None and score not in run.score_names():
        # Fine when labeling for a judge that does not exist yet.
        logger.warning(
            "No trial of %s has a score named %r: the sample is not spread over "
            "its verdicts",
            run.id,
            score,
        )
    _require_score(run, against, "against")
    skip = set(exclude)

    def rank(record: dict[str, Any]) -> str:
        return short_hash(str(seed), record["id"])

    disagreements: list[dict[str, Any]] = []
    by_verdict: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record, trial in _candidates(run):
        if record["id"] in skip:
            continue
        verdict = _verdict(trial, score)
        if against is not None and _disagree(
            verdict, _verdict(trial, against), against
        ):
            disagreements.append(record)
            continue
        metadata = cast("dict[str, Any]", record["input"].get("metadata") or {})
        stratum = "" if strata is None else json.dumps(metadata.get(strata))
        key = "" if verdict is None else label_value_text(verdict)
        by_verdict.setdefault(key, {}).setdefault(stratum, []).append(record)
    disagreements.sort(key=rank)
    verdict_queues = {
        key: _interleave([sorted(strata_map[s], key=rank) for s in sorted(strata_map)])
        for key, strata_map in by_verdict.items()
    }
    scored = [verdict_queues[k] for k in sorted(verdict_queues) if k]
    chosen = [*disagreements, *_interleave(scored), *verdict_queues.get("", [])][:n]
    known = splits or {}
    requests: list[dict[str, Any]] = []
    for record in chosen:
        metadata = record["metadata"]
        if score is not None:
            metadata["score"] = score
        example_id = metadata["example_id"]
        requests.append(
            {
                "id": record["id"],
                "input": record["input"],
                "reference": None,
                "metadata": metadata,
                "splits": [
                    known.get(example_id) or label_split(example_id, test_share)
                ],
            }
        )
    return requests


def known_splits(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """The split each labeled example went to, from earlier label records."""
    found: dict[str, str] = {}
    for record in records:
        metadata = cast("Mapping[str, Any]", record.get("metadata") or {})
        example_id = metadata.get("example_id")
        splits = cast("Sequence[str]", record.get("splits") or [])
        if example_id is not None and splits:
            found.setdefault(str(example_id), splits[0])
    return found


_FIELD_ORDER = ("id", "input", "reference", "metadata", "splits")


def write_records(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write dataset records as JSONL (atomically), keeping empty references."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {k: record[k] for k in _FIELD_ORDER if k in record}, ensure_ascii=False
        )
        for record in records
    ]
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    tmp.replace(target)
    return target


def _filled(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _check_label(value: Any, where: str) -> None:
    def scalar(v: Any) -> bool:
        return isinstance(v, bool | int | float | str)

    if scalar(value):
        return
    if isinstance(value, Mapping) and all(
        isinstance(k, str) and scalar(v)
        for k, v in cast("Mapping[Any, Any]", value).items()
    ):
        return
    raise DatasetError(
        f"{where}: a label is true/false, a number, a text label, or "
        f"{{score: label}}; got {value!r}"
    )


def _labels(record: Mapping[str, Any]) -> Any:
    """
    A record's filled-in labels: ``{score: value}`` when the record names its
    score (or labels several), the bare value otherwise, ``None`` when empty.
    """
    reference = record.get("reference")
    if isinstance(reference, Mapping):
        filled = {
            k: v for k, v in cast("Mapping[str, Any]", reference).items() if _filled(v)
        }
        return filled or None
    if not _filled(reference):
        return None
    metadata = cast("Mapping[str, Any]", record.get("metadata") or {})
    score = metadata.get("score")
    return {str(score): reference} if score is not None else reference


def _same(a: Any, b: Any) -> bool:
    # Typed equality: True, 1 and 1.0 are different labels.
    return canonical_json(a) == canonical_json(b)


class LabelImport(BaseModel):
    """What :func:`import_labels` did to the labels dataset."""

    path: str
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    # Records whose reference was still empty.
    unlabeled: int = 0
    # Records labeled differently than the dataset already is (kept as they
    # were unless ``replace``).
    conflicts: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    total: int = 0


def _per_score(value: Any, held: Iterable[str]) -> dict[str, Any]:
    # Provenance per score; a single value applied to the scores already held.
    if isinstance(value, dict):
        return dict(cast("dict[str, Any]", value))
    return {} if value is None else dict.fromkeys(held, value)


def _provenance(
    metadata: dict[str, Any],
    scores: Iterable[str],
    labeler: Any,
    labeled_at: Any,
    held: Iterable[str] = (),
) -> None:
    # Who labeled each score, and when.
    held = list(held)
    by = _per_score(metadata.get("labeler"), held)
    at = _per_score(metadata.get("labeled_at"), held)
    for score in scores:
        if labeler is not None:
            by[score] = labeler
        at[score] = labeled_at
    metadata["labeler"] = by
    metadata["labeled_at"] = at


def merge_labels(
    records: Iterable[Mapping[str, Any]],
    into: str | Path,
    *,
    labeler: str | None = None,
    replace: bool = False,
) -> LabelImport:
    """
    Add labeled ``records`` to the labels dataset file ``into`` (a ``.jsonl``
    file, created when missing). Labels are kept per score —
    ``{score: value}``, a record's single value going to the score its
    ``metadata.score`` names — with who labeled each (``metadata.labeler``:
    the record's own, else ``labeler``) and when. A score already labeled
    the same way is left alone; one labeled differently is a conflict, kept
    as it was unless ``replace``. The file is not written when nothing
    changed.
    """
    target = Path(into)
    if target.suffix != ".jsonl":
        raise DatasetError(f"{target}: a labels dataset is a .jsonl file")
    current: dict[str, dict[str, Any]] = {}
    if target.exists():
        for record in Dataset.load(target).to_records():
            current[str(record["id"])] = record
    result = LabelImport(path=str(target))
    now = utc_now().isoformat()
    for record in records:
        incoming = _labels(record)
        if incoming is None:
            result.unlabeled += 1
            continue
        record_id = str(record["id"])
        _check_label(incoming, f"record {record_id!r}")
        metadata = dict(cast("Mapping[str, Any]", record.get("metadata") or {}))
        who = metadata.pop("labeler", None) or labeler
        when = metadata.pop("labeled_at", None) or now
        existing = current.get(record_id)
        if existing is None:
            if isinstance(incoming, dict):
                _provenance(metadata, cast("dict[str, Any]", incoming), who, when)
            else:
                metadata["labeler"], metadata["labeled_at"] = who, when
            current[record_id] = {
                **dict(record),
                "reference": incoming,
                "metadata": metadata,
            }
            result.added += 1
            continue
        if existing.get("input") != record.get("input"):
            raise DatasetError(
                f"Record {record_id!r} labels a different output than the one "
                f"{target} holds under that id"
            )
        held: Any = _labels(existing)
        changed: list[str] = []
        clashes: list[str] = []
        merged: Any
        if held is None:
            merged = incoming
            changed = (
                list(cast("dict[str, Any]", incoming))
                if isinstance(incoming, dict)
                else [""]
            )
        elif isinstance(held, dict) and isinstance(incoming, dict):
            merged = dict(cast("dict[str, Any]", held))
            for key, value in cast("dict[str, Any]", incoming).items():
                if key not in merged:
                    merged[key] = value
                    changed.append(key)
                elif not _same(merged[key], value):
                    clashes.append(key)
                    if replace:
                        merged[key] = value
                        changed.append(key)
        elif _same(held, incoming):
            merged = cast("Any", held)
        else:
            clashes.append("")
            merged = incoming if replace else cast("Any", held)
            if replace:
                changed.append("")
        if clashes:
            result.conflicts.append(
                {
                    "id": record_id,
                    "scores": [c for c in clashes if c],
                    "existing": held,
                    "new": incoming,
                    "replaced": replace,
                }
            )
        if not changed:
            result.unchanged += 1
            continue
        kept = dict(cast("Mapping[str, Any]", existing.get("metadata") or {}))
        if isinstance(merged, dict):
            before = (
                list(cast("dict[str, Any]", held)) if isinstance(held, dict) else []
            )
            _provenance(kept, [c for c in changed if c], who, when, before)
        else:
            kept["labeler"], kept["labeled_at"] = who, when
        existing["reference"] = merged
        existing["metadata"] = kept
        result.updated += 1
    result.total = len(current)
    if result.added or result.updated:
        write_records(current.values(), target)
    return result


def import_labels(
    paths: Sequence[str | Path],
    into: str | Path,
    *,
    labeler: str | None = None,
    replace: bool = False,
) -> LabelImport:
    """
    Merge filled-in label files (as :func:`sample_for_labeling` writes them)
    into the labels dataset ``into``; see :func:`merge_labels`.
    """
    records = [r for path in paths for r in Dataset.load(path).to_records()]
    return merge_labels(records, into, labeler=labeler, replace=replace)


def label_requests(path: str | Path) -> list[dict[str, Any]]:
    """The records of a to-label file, labeled or not."""
    return Dataset.load(path).to_records()


_TRUE_LABELS = frozenset({"true", "yes", "pass", "passed"})
_FALSE_LABELS = frozenset({"false", "no", "fail", "failed"})


def annotation_value(
    label: str | None,
    score: float | None,
    *,
    true_labels: Iterable[str] = (),
    false_labels: Iterable[str] = (),
) -> ScoreValue | None:
    """
    A label from an annotation's result: pass/fail words (true/false, yes/no,
    pass/fail, and any ``true_labels`` / ``false_labels``) as booleans, other
    labels as text, else the numeric score.
    """
    if label is not None and label.strip():
        word = label.strip().lower()
        if word in _TRUE_LABELS or word in {w.lower() for w in true_labels}:
            return True
        if word in _FALSE_LABELS or word in {w.lower() for w in false_labels}:
            return False
        return label
    return score
