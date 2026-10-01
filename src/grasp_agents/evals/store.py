import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from grasp_agents.types.events import Event

from ._util import to_jsonable
from .types import EvaluationRun, Example, Trial

EVALS_DIR_ENV = "GRASP_EVALS_DIR"
DEFAULT_EVALS_DIR = ".evals"


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
        """Run id for an id, unique id prefix, ``latest`` or ``latest:<name>``."""
        ...


def default_evals_dir() -> Path:
    return Path(os.environ.get(EVALS_DIR_ENV) or DEFAULT_EVALS_DIR)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


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
                fh.write(json.dumps(to_jsonable(example), ensure_ascii=False) + "\n")

    def save(self, run: EvaluationRun) -> None:
        _atomic_write(self.run_dir(run.id) / "run.json", run.model_dump_json(indent=2))

    def append_trial(
        self,
        run_id: str,
        trial: Trial,
        events: Sequence[Event[Any]] | None = None,
    ) -> None:
        directory = self.run_dir(run_id)
        with (directory / "trials.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(trial.model_dump_json() + "\n")
        if events:
            record = {
                "example_id": trial.example_id,
                "repetition": trial.repetition,
                "events": to_jsonable(list(events)),
            }
            with (directory / "transcripts.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

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
            headers = [h for h in headers if h.name == name]
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
