import json
import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from grasp_agents.types.events import Event

from ._util import to_jsonable
from .types import EvaluationRun, Example, Trial

logger = logging.getLogger(__name__)

EVALS_DIR_ENV = "GRASP_EVALS_DIR"
DEFAULT_EVALS_DIR = ".evals"
_TAIL_BLOCK = 1 << 16


class RunNotFoundError(LookupError):
    pass


class RunStore(Protocol):
    """Persistence for evaluation runs. Implementations must be append-safe."""

    def create(self, run: EvaluationRun) -> None:
        """Persist a new run's header and examples."""
        ...

    def save(self, run: EvaluationRun) -> None:
        """Rewrite a run's header (status, counts, metrics, links)."""
        ...

    def append_trial(
        self,
        run_id: str,
        trial: Trial,
        events: Sequence[Event[Any]] | None = None,
    ) -> None:
        """Append a finished trial; a later trial with the same key supersedes it."""
        ...

    def write_report(self, run_id: str, text: str) -> None: ...

    def load(
        self, run_id: str, *, trials: bool = True, examples: bool = True
    ) -> EvaluationRun: ...

    def load_events(self, run_id: str) -> dict[tuple[str, int], list[Event[Any]]]: ...

    def list_runs(
        self, *, name: str | None = None, limit: int | None = None
    ) -> list[EvaluationRun]:
        """Run headers, newest first."""
        ...

    def resolve(self, ref: str) -> str:
        """
        Run id for an id, unique id prefix, ``latest`` or ``latest:<name>``
        (a run name, or the attribute of the evaluation spec it came from).
        """
        ...


def default_evals_dir() -> Path:
    return Path(os.environ.get(EVALS_DIR_ENV) or DEFAULT_EVALS_DIR)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """
    Records of an append-only JSONL file. A last line cut short by a crash is
    skipped (and trimmed before the next append) rather than making the whole
    file unreadable.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if not line.endswith("\n"):
                    logger.warning(
                        "Skipping the incomplete last record of %s (line %d)",
                        path,
                        lineno,
                    )
                    return
                raise ValueError(f"{path}:{lineno}: corrupt record ({exc})") from exc


def _trim_partial_tail(path: Path) -> None:
    """
    Repair a file whose last write was cut short by a crash: a complete record
    that only lacks its newline is kept, anything else after the last newline
    is dropped.
    """
    if not path.exists():
        return
    with path.open("rb+") as fh:
        end = fh.seek(0, os.SEEK_END)
        if end == 0:
            return
        fh.seek(end - 1)
        if fh.read(1) == b"\n":
            return
        tail = 0
        position = end
        while position > 0:
            start = max(0, position - _TAIL_BLOCK)
            fh.seek(start)
            newline = fh.read(position - start).rfind(b"\n")
            if newline >= 0:
                tail = start + newline + 1
                break
            position = start
        fh.seek(tail)
        try:
            json.loads(fh.read(end - tail))
        except ValueError:
            fh.truncate(tail)
        else:
            fh.seek(end)
            fh.write(b"\n")


def _example_line(example: Example[Any, Any]) -> dict[str, Any]:
    # Examples are kept as stored in their dataset, so the run can be matched
    # with that dataset (e.g. in Phoenix) by content.
    content = example.record
    return {
        "id": example.id,
        "input": content.input,
        "reference": content.reference,
        "metadata": content.metadata,
        "splits": list(example.splits),
        "content_hash": example.content_hash,
    }


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class LocalRunStore:
    """
    Runs as directories under ``<root>/runs/<run_id>/``::

        run.json            header: status, provenance, config, metrics
        examples.jsonl      the examples evaluated (the run is self-contained)
        trials.jsonl        one trial per line, appended as trials finish
        transcripts.jsonl   per-trial event streams (when captured)
        report.md           rendered summary

    ``root`` defaults to ``$GRASP_EVALS_DIR`` or ``./.evals``.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_evals_dir()
        self.runs_dir = self.root / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def create(self, run: EvaluationRun) -> None:
        directory = self.run_dir(run.id)
        directory.mkdir(parents=True, exist_ok=False)
        _atomic_write(directory / "run.json", run.model_dump_json(indent=2))
        with (directory / "examples.jsonl").open("w", encoding="utf-8") as fh:
            for example in run.examples:
                fh.write(json.dumps(_example_line(example), ensure_ascii=False) + "\n")

    def save(self, run: EvaluationRun) -> None:
        _atomic_write(self.run_dir(run.id) / "run.json", run.model_dump_json(indent=2))

    def append_trial(
        self,
        run_id: str,
        trial: Trial,
        events: Sequence[Event[Any]] | None = None,
    ) -> None:
        directory = self.run_dir(run_id)
        self._append(directory / "trials.jsonl", trial.model_dump_json())
        if events:
            record = {
                "example_id": trial.example_id,
                "repetition": trial.repetition,
                "events": to_jsonable(list(events)),
            }
            self._append(
                directory / "transcripts.jsonl", json.dumps(record, ensure_ascii=False)
            )

    def _append(self, path: Path, line: str) -> None:
        _trim_partial_tail(path)
        _append_line(path, line)

    def write_report(self, run_id: str, text: str) -> None:
        _atomic_write(self.run_dir(run_id) / "report.md", text)

    def load(
        self, run_id: str, *, trials: bool = True, examples: bool = True
    ) -> EvaluationRun:
        directory = self.run_dir(run_id)
        header = directory / "run.json"
        if not header.exists():
            raise RunNotFoundError(f"No run {run_id!r} in {self.runs_dir}")
        run = EvaluationRun.model_validate_json(header.read_text(encoding="utf-8"))
        if run.id != directory.name:
            raise RunNotFoundError(
                f"{directory} holds run {run.id!r}: a run directory must be named "
                "after its run id (copying a run under another name is not supported)"
            )
        if trials:
            latest: dict[tuple[str, int], Trial] = {}
            for record in _iter_jsonl(directory / "trials.jsonl"):
                trial = Trial.model_validate(record)
                latest[trial.key] = trial
            run.trials = list(latest.values())
        if examples:
            run.examples = [
                Example[Any, Any].model_validate(r)
                for r in _iter_jsonl(directory / "examples.jsonl")
            ]
        return run

    def load_events(self, run_id: str) -> dict[tuple[str, int], list[Event[Any]]]:
        events: dict[tuple[str, int], list[Event[Any]]] = {}
        for record in _iter_jsonl(self.run_dir(run_id) / "transcripts.jsonl"):
            key = (str(record["example_id"]), int(record["repetition"]))
            events[key] = [Event[Any].model_validate(e) for e in record["events"]]
        return events

    def _run_ids(self) -> list[str]:
        if not self.runs_dir.exists():
            return []
        return sorted(
            (p.name for p in self.runs_dir.iterdir() if (p / "run.json").exists()),
            reverse=True,
        )

    def list_runs(
        self, *, name: str | None = None, limit: int | None = None
    ) -> list[EvaluationRun]:
        headers = [
            self.load(run_id, trials=False, examples=False)
            for run_id in self._run_ids()
        ]
        if name is not None:
            headers = [h for h in headers if _matches(h, name)]
        headers.sort(key=lambda r: r.created_at, reverse=True)
        return headers if limit is None else headers[:limit]

    def resolve(self, ref: str) -> str:
        if ref == "latest" or ref.startswith("latest:"):
            name = ref.partition(":")[2] or None
            runs = self.list_runs(name=name, limit=1)
            if not runs:
                scope = f" named {name!r}" if name else ""
                raise RunNotFoundError(f"No runs{scope} in {self.runs_dir}")
            return runs[0].id
        ids = self._run_ids()
        if ref in ids:
            return ref
        matches = [i for i in ids if i.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise RunNotFoundError(f"Ambiguous run reference {ref!r}: {matches[:5]}")
        raise RunNotFoundError(f"No run matching {ref!r} in {self.runs_dir}")


def _matches(run: EvaluationRun, name: str) -> bool:
    if run.name == name:
        return True
    spec = run.evaluation or ""
    return bool(spec) and spec.rpartition(":")[2] == name


def store_for_ref(
    ref: str, root: str | Path | None = None
) -> tuple[LocalRunStore, str]:
    """
    ``(store, run_id)`` for a run reference that may also be a path to a run
    directory (``.../runs/<run_id>``), which may live outside the default root.
    """
    path = Path(ref)
    if (path / "run.json").exists():
        return LocalRunStore(path.resolve().parent.parent), path.resolve().name
    store = LocalRunStore(root)
    return store, store.resolve(ref)
