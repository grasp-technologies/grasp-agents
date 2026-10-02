import importlib
import importlib.util
import inspect
import logging
import sys
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

from grasp_agents.processors.processor import Processor

from ._execution import ProgressCallback
from ._util import SPEC_MODULE_PREFIX, short_hash, utc_now
from .dataset import Dataset, DatasetCheck, DatasetError, DatasetProblem
from .evaluator import Evaluator
from .metrics import MetricsSpec, default_metrics
from .online import (
    TraceQuery,
    TraceSource,
    annotate_run,
    annotations_location,
    annotations_pending,
    collect_items,
    evaluate_traces,
    extract_trials,
    resolve_window,
    sample_items,
)
from .runner import evaluate as evaluate_task
from .runner import rescore
from .store import LocalRunStore, RunStore
from .task import Task, as_task
from .types import EvaluationRun, Example, TraceWindow, Trial, input_digest
from .validation import (
    CorrectedPassRate,
    UnvalidatedJudgeError,
    ValidationGate,
    check_validations,
)

logger = logging.getLogger(__name__)

PHOENIX_PREFIX = "phoenix:"


def parse_phoenix_ref(ref: str) -> tuple[str, str | None]:
    """``(name, version)`` of ``[phoenix:]NAME[@VERSION]``."""
    body = ref.removeprefix(PHOENIX_PREFIX)
    name, at, version = body.rpartition("@")
    if not at:
        return body, None
    return name, version or None


async def load_phoenix_dataset(
    ref: str,
    *,
    input_type: Any = Any,
    reference_type: Any = Any,
    base_url: str | None = None,
    cache_dir: str | Path | None = None,
) -> Dataset[Any, Any]:
    """
    Pull ``phoenix:NAME[@VERSION]`` from ``base_url`` (``$PHOENIX_BASE_URL``),
    caching it in ``cache_dir`` (default ``<evals dir>/datasets``).
    """
    from .phoenix import (  # noqa: PLC0415
        PhoenixClient,
        pull_dataset,
    )

    name, version = parse_phoenix_ref(ref)
    async with PhoenixClient(base_url) as client:
        return await pull_dataset(
            client,
            name,
            version=version,
            input_type=input_type,
            reference_type=reference_type,
            cache_dir=cache_dir,
        )


type TaskSource = (
    Task[Any, Any]
    | Processor[Any, Any, Any]
    | Callable[[], Task[Any, Any] | Processor[Any, Any, Any]]
)
type DatasetSource = (
    Dataset[Any, Any]
    | str
    | Path
    | Callable[[], Dataset[Any, Any] | Awaitable[Dataset[Any, Any]]]
)


@dataclass(frozen=True)
class TraceExport:
    """Examples made from production traces, and what was left out."""

    dataset: Dataset[Any, Any]
    # Items in the window, and those sampled from them.
    items: int
    sampled: int
    # Examples whose input another one (or an excluded id) already had.
    duplicates: int
    skipped: list[str]
    failures: dict[str, str]


@dataclass
class Evaluation:
    """
    A named, importable evaluation: the question it answers, the task, where
    its data comes from, how outputs are judged and which numbers to report.

    Definitions are cheap to import: ``task`` may be a zero-argument factory
    (called on the first run, so no clients are built at import time) and
    ``dataset`` a path, a ``phoenix:NAME[@VERSION]`` reference or a loader.
    Run with :meth:`run`, or from the CLI as ``grasp-evals run
    package.module:attr``.

    With ``traces``, the same evaluators also score what the system did in
    production (:meth:`run_online`, ``grasp-evals online``), and production
    inputs become dataset examples (:meth:`dataset_from_traces`). An
    evaluation of production traces only needs no ``task`` or ``dataset``.
    """

    name: str
    task: TaskSource | None = None
    dataset: DatasetSource | None = None
    evaluators: Sequence[Evaluator[Any, Any, Any]] = ()
    metrics: MetricsSpec = None
    description: str | None = None
    # Types used to validate dataset files and stored outputs; they default to
    # the task's (a processor's ``in_type`` / ``out_type``).
    input_type: Any = None
    reference_type: Any = Any
    output_type: Any = None
    repetitions: int = 1
    concurrency: int = 4
    timeout_s: float | None = None
    evaluator_timeout_s: float | None = None
    max_cost_usd: float | None = None
    max_error_rate: float | None = None
    group_by: Sequence[str] = ()
    cluster_by: str | None = None
    # Held-out splits: reports, ``show``, ``compare`` and Phoenix pushes give
    # only their aggregates, and a run must include each one whole. This keeps
    # an agent iterating through the CLI from fitting to their examples; it is
    # not access control — the dataset file and run records contain them.
    sealed_splits: Sequence[str] = ()
    # Integrity checks run on the dataset before every run.
    dataset_checks: Mapping[str, DatasetCheck] | Sequence[DatasetCheck] = ()
    # Scores whose judge must have passed validation on held-out labels before
    # this evaluation scores anything: the newest validation run of the judge
    # exactly as it is now must meet the gate. Their pass rates are also
    # reported corrected for the judge's measured errors.
    validation_gates: Mapping[str, ValidationGate] = field(
        default_factory=dict[str, ValidationGate]
    )
    # Production spans this evaluation scores online.
    traces: TraceQuery | None = None
    tags: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])
    # Import spec this definition was loaded from (set by ``load_evaluation``).
    spec: str | None = None
    _task: Task[Any, Any] | None = field(default=None, init=False, repr=False)

    def build_task(self) -> Task[Any, Any]:
        if self._task is None:
            source = self.task
            if source is None:
                raise SpecError(
                    f"Evaluation {self.name!r} has no task: it only scores "
                    "production traces (run it online)"
                )
            if isinstance(source, Task | Processor):
                self._task = as_task(
                    cast("Task[Any, Any] | Processor[Any, Any, Any]", source)
                )
            else:
                self._task = as_task(source())
        return self._task

    @property
    def resolved_input_type(self) -> Any:
        if self.input_type is not None:
            return self.input_type
        if self.task is None:
            return Any
        return self.build_task().input_type

    @property
    def resolved_output_type(self) -> Any:
        if self.output_type is not None:
            return self.output_type
        if self.task is None:
            return Any
        return self.build_task().output_type

    async def load_dataset(
        self,
        path: str | Path | None = None,
        *,
        phoenix_url: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> Dataset[Any, Any]:
        source = path if path is not None else self.dataset
        if source is None:
            raise SpecError(
                f"Evaluation {self.name!r} has no dataset: pass one, or make one "
                "from production traces"
            )
        if isinstance(source, Dataset):
            return cast("Dataset[Any, Any]", source)
        if isinstance(source, str) and source.startswith(PHOENIX_PREFIX):
            return await load_phoenix_dataset(
                source,
                input_type=self.resolved_input_type,
                reference_type=self.reference_type,
                base_url=phoenix_url,
                cache_dir=cache_dir,
            )
        if isinstance(source, str | Path):
            return Dataset.load(
                source,
                input_type=self.resolved_input_type,
                reference_type=self.reference_type,
            )
        loaded = source()
        if inspect.isawaitable(loaded):
            loaded = await loaded
        return loaded

    def check_dataset(self, dataset: Dataset[Any, Any]) -> list[DatasetProblem]:
        return dataset.check(self.dataset_checks) if self.dataset_checks else []

    def check_judges(
        self, store: RunStore | None = None, *, allow_unvalidated: bool = False
    ) -> tuple[MetricsSpec, dict[str, Any]]:
        """
        Check :attr:`validation_gates` against the validation runs in
        ``store``: the metrics to report (with corrected pass rates for
        validated binary judges) and what to record on the run. Raises
        :class:`UnvalidatedJudgeError` unless ``allow_unvalidated``.
        """
        if not self.validation_gates:
            return self.metrics, {}
        found, failures = check_validations(
            store if store is not None else LocalRunStore(),
            self.evaluators,
            self.validation_gates,
        )
        if failures and not allow_unvalidated:
            listed = "\n".join(f"  {failure}" for failure in failures)
            raise UnvalidatedJudgeError(
                f"{self.name}: judges are not validated:\n{listed}\nRun their "
                "validation evaluations first, or allow unvalidated judges."
            )
        record: dict[str, Any] = {
            "judge_validations": {s: v.brief() for s, v in found.items()},
            "unvalidated_judges": failures,
        }
        corrected = [
            CorrectedPassRate(score, rates, threshold=validation.threshold)
            for score, validation in found.items()
            if (rates := validation.rates()) is not None
        ]
        metrics = self.metrics
        if not corrected:
            return metrics, record
        if metrics is not None and not callable(metrics):
            return [*metrics, *corrected], record
        base = metrics

        def with_corrected(trials: Sequence[Trial]) -> list[Any]:
            chosen = default_metrics(trials) if base is None else list(base(trials))
            return [*chosen, *corrected]

        return with_corrected, record

    async def select(
        self,
        *,
        dataset: Dataset[Any, Any] | str | Path | None = None,
        split: str | None = None,
        ids: Sequence[str] | None = None,
        sample: int | None = None,
        seed: int = 0,
        limit: int | None = None,
        check: bool = True,
        phoenix_url: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> Dataset[Any, Any]:
        """The dataset subset a run would evaluate (after integrity checks)."""
        data = (
            dataset
            if isinstance(dataset, Dataset)
            else await self.load_dataset(
                dataset, phoenix_url=phoenix_url, cache_dir=cache_dir
            )
        )
        if check:
            problems = self.check_dataset(data)
            if problems:
                listed = "\n".join(
                    f"  {p.example_id}: [{p.check}] {p.message}" for p in problems[:20]
                )
                raise DatasetError(
                    f"{len(problems)} dataset problem(s) in {data.name!r}:\n{listed}"
                )
        if split is not None:
            data = data.split(split)
        if ids:
            data = data.select(ids)
        if sample is not None:
            data = data.sample(sample, seed=seed)
        if limit is not None:
            data = data.head(limit)
        if len(data) == 0:
            raise DatasetError(
                f"The selection from {data.name!r} is empty "
                f"({', '.join(data.selection) or 'no examples'})"
            )
        return data

    async def run(
        self,
        *,
        dataset: Dataset[Any, Any] | str | Path | None = None,
        split: str | None = None,
        ids: Sequence[str] | None = None,
        sample: int | None = None,
        seed: int = 0,
        limit: int | None = None,
        repetitions: int | None = None,
        concurrency: int | None = None,
        timeout_s: float | None = None,
        evaluator_timeout_s: float | None = None,
        max_cost_usd: float | None = None,
        score: bool = True,
        name: str | None = None,
        store: RunStore | None = None,
        persist: bool = True,
        resume: "str | EvaluationRun | None" = None,
        force: bool = False,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        progress: ProgressCallback | None = None,
        check: bool = True,
        phoenix_url: str | None = None,
        allow_unvalidated: bool = False,
    ) -> EvaluationRun:
        metrics, judges = (
            self.check_judges(store, allow_unvalidated=allow_unvalidated)
            if score
            else (self.metrics, {})
        )
        data = await self.select(
            dataset=dataset,
            split=split,
            ids=ids,
            sample=sample,
            seed=seed,
            limit=limit,
            check=check,
            phoenix_url=phoenix_url,
            # Pulled datasets are cached next to the runs.
            cache_dir=store.root / "datasets"
            if isinstance(store, LocalRunStore)
            else None,
        )
        return await evaluate_task(
            self.build_task(),
            data,
            self.evaluators,
            metrics,
            name=name or self.name,
            description=self.description,
            repetitions=repetitions if repetitions is not None else self.repetitions,
            concurrency=concurrency if concurrency is not None else self.concurrency,
            timeout_s=timeout_s if timeout_s is not None else self.timeout_s,
            evaluator_timeout_s=evaluator_timeout_s
            if evaluator_timeout_s is not None
            else self.evaluator_timeout_s,
            max_cost_usd=max_cost_usd
            if max_cost_usd is not None
            else self.max_cost_usd,
            max_error_rate=self.max_error_rate,
            group_by=self.group_by,
            cluster_by=self.cluster_by,
            sealed_splits=self.sealed_splits,
            score=score,
            store=store,
            persist=persist,
            resume=resume,
            force=force,
            tags=[*self.tags, *tags],
            metadata={**self.metadata, **(metadata or {}), **judges},
            evaluation=self.spec,
            progress=progress,
        )

    async def rescore(
        self,
        run: "str | EvaluationRun",
        *,
        rerun: bool = False,
        store: RunStore | None = None,
        persist: bool = True,
        concurrency: int | None = None,
        progress: ProgressCallback | None = None,
        allow_unvalidated: bool = False,
    ) -> EvaluationRun:
        """
        Re-score a stored run of this evaluation with its current evaluators:
        changed ones run again, unchanged ones only where they failed or did
        not run (all of them with ``rerun=True``). Gated judges must be
        validated, as for :meth:`run`.
        """
        metrics, judges = self.check_judges(store, allow_unvalidated=allow_unvalidated)
        return await rescore(
            run,
            self.evaluators,
            metrics,
            input_type=self.resolved_input_type,
            reference_type=self.reference_type,
            output_type=self.resolved_output_type,
            concurrency=concurrency if concurrency is not None else self.concurrency,
            evaluator_timeout_s=self.evaluator_timeout_s,
            group_by=self.group_by,
            cluster_by=self.cluster_by,
            rerun=rerun,
            store=store,
            persist=persist,
            metadata=judges,
            progress=progress,
        )

    # --- Online ---

    def trace_query(self) -> TraceQuery:
        if self.traces is None:
            raise SpecError(f"Evaluation {self.name!r} defines no traces to read")
        return self.traces

    async def run_online(
        self,
        *,
        source: TraceSource | None = None,
        phoenix_url: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        now: datetime | None = None,
        annotate: bool = True,
        concurrency: int | None = None,
        max_cost_usd: float | None = None,
        store: RunStore | None = None,
        persist: bool = True,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        progress: ProgressCallback | None = None,
        allow_unvalidated: bool = False,
    ) -> EvaluationRun | None:
        """
        Score the production spans :attr:`traces` selects in a window — from
        ``start`` (default: where the previous online run ended) to ``end``
        (default: the completion buffer before ``now``) — and write the scores
        back as annotations — first those earlier runs on the same store
        failed to write (best effort). ``None`` when the window is empty.
        Traces are read from ``source``, by default the Phoenix server at
        ``phoenix_url`` (``$PHOENIX_BASE_URL``); gated judges must be
        validated, as for :meth:`run`. Raises :class:`AnnotationError`, with
        the scored run, when its annotations cannot be written. Run one at a
        time per evaluation and store: overlapping runs score a window twice.
        """
        query = self.trace_query()
        run_store = (
            store if store is not None else (LocalRunStore() if persist else None)
        )
        runs = run_store.list_runs(name=self.name) if run_store is not None else []
        window = resolve_window(
            query, name=self.name, runs=runs, start=start, end=end, now=now
        )
        if window is None:
            return None
        metrics, judges = self.check_judges(
            run_store, allow_unvalidated=allow_unvalidated
        )
        async with open_trace_source(source, phoenix_url) as traces:
            if annotate and run_store is not None:
                await _retry_pending(traces, runs, run_store)
            return await evaluate_traces(
                traces,
                query,
                self.evaluators,
                metrics,
                window=window,
                name=self.name,
                output_type=self.resolved_output_type,
                description=self.description,
                concurrency=concurrency
                if concurrency is not None
                else self.concurrency,
                evaluator_timeout_s=self.evaluator_timeout_s,
                max_cost_usd=max_cost_usd
                if max_cost_usd is not None
                else self.max_cost_usd,
                max_error_rate=self.max_error_rate,
                group_by=self.group_by,
                cluster_by=self.cluster_by,
                annotate=annotate,
                store=run_store,
                persist=persist,
                tags=[*self.tags, *tags],
                metadata={**self.metadata, **(metadata or {}), **judges},
                evaluation=self.spec,
                progress=progress,
            )

    async def dataset_from_traces(
        self,
        *,
        start: datetime,
        end: datetime | None = None,
        source: TraceSource | None = None,
        phoenix_url: str | None = None,
        status: Literal["ok", "error"] | None = None,
        sample_rate: float | None = None,
        max_items: int | None = None,
        exclude: Iterable[Example[Any, Any]] = (),
    ) -> TraceExport:
        """
        Production inputs as dataset examples: :attr:`traces`' extractor over
        the items in ``[start, end)`` (``end`` defaults to now), optionally
        only failed (``status="error"``) or succeeded ones and sampled. Each
        example keeps its trace provenance in its metadata (and a failure's
        error); examples are identified by their input, so a repeated input —
        or the input of an example in ``exclude`` — is left out.
        """
        query = self.trace_query()
        changes: dict[str, Any] = {}
        if status is not None:
            changes["status"] = status
        if sample_rate is not None:
            changes["sample_rate"] = sample_rate
        if max_items is not None:
            changes["max_items"] = max_items
        query = replace(query, **changes)
        window = TraceWindow(start=start, end=end or utc_now())
        async with open_trace_source(source, phoenix_url) as traces:
            items = await collect_items(traces, query, window)
            sampled = sample_items(
                items,
                rate=query.sample_rate,
                max_items=query.max_items,
                strata=query.strata,
            )
            extraction = await extract_trials(sampled, query.extractor)
        seen = {input_digest(e.record.input) for e in exclude}
        examples: list[Example[Any, Any]] = []
        duplicates = 0
        trials = {t.example_id: t for t in extraction.trials}
        for found in extraction.examples:
            trial = trials[found.id]
            metadata = dict(found.metadata)
            if trial.error is not None:
                metadata["error"] = trial.error.message
            example = Example[Any, Any](input=found.input, metadata=metadata)
            if example.id in seen:
                duplicates += 1
                continue
            seen.add(example.id)
            examples.append(example)
        return TraceExport(
            dataset=Dataset(
                examples,
                name=f"{self.name}-traces",
                source=traces.location,
                description=(
                    f"Inputs of {query.project} traces "
                    f"{window.start.isoformat()} to {window.end.isoformat()}"
                ),
            ),
            items=len(items),
            sampled=len(sampled),
            duplicates=duplicates,
            skipped=extraction.skipped,
            failures=extraction.failures,
        )


async def _retry_pending(
    source: TraceSource, runs: Sequence[EvaluationRun], store: RunStore
) -> None:
    # Earlier online runs whose annotations failed to write to this store.
    for earlier in runs:
        if (
            earlier.kind != "online"
            or not annotations_pending(earlier)
            or annotations_location(earlier) != source.location
        ):
            continue
        try:
            await annotate_run(source, store.load(earlier.id), store=store)
        except Exception as exc:
            logger.warning(
                "Annotations of %s are still not written: %s", earlier.id, exc
            )


@asynccontextmanager
async def open_trace_source(
    source: TraceSource | None = None, phoenix_url: str | None = None
) -> AsyncGenerator[TraceSource]:
    """``source``, or else the Phoenix server at ``phoenix_url`` ($PHOENIX_BASE_URL)."""
    if source is not None:
        yield source
        return
    from .phoenix import PhoenixClient, PhoenixTraceSource  # noqa: PLC0415

    async with PhoenixClient(phoenix_url) as client:
        yield PhoenixTraceSource(client)


class SpecError(LookupError):
    """An evaluation spec (``MODULE:ATTR`` or ``file.py:ATTR``) cannot be resolved."""


def _importable_name(path: Path) -> str | None:
    # A spec file that is importable is imported under that name: it runs
    # once, its relative imports work, and its objects have the same identity
    # whichever way they are reached. The longest name wins (the package
    # path, not a directory inside the package that is also on sys.path).
    bases = sorted(
        {Path(entry or ".").resolve() for entry in sys.path},
        key=lambda base: len(base.parts),
    )
    for base in bases:
        try:
            parts = path.with_suffix("").relative_to(base).parts
        except ValueError:
            continue
        if not parts or not all(part.isidentifier() for part in parts):
            continue
        # The top level must be a regular package: a namespace directory (a
        # checkout's ``src``) would import a second copy of the package.
        if len(parts) > 1 and not (base / parts[0] / "__init__.py").exists():
            continue
        name = ".".join(parts)
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if found is not None and found.origin and Path(found.origin).resolve() == path:
            return name
    return None


def _import_module(target: str) -> ModuleType:
    path = Path(target)
    if path.suffix != ".py":
        return importlib.import_module(target)
    if not path.exists():
        raise SpecError(f"No spec file {target!r}")
    resolved = path.resolve()
    module_name = (
        f"{SPEC_MODULE_PREFIX}{resolved.stem}_{short_hash(str(resolved), length=8)}"
    )
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    name = _importable_name(resolved)
    if name is None:
        directory = str(resolved.parent)
        if directory not in sys.path:
            # Sibling modules of the spec file are importable from it...
            sys.path.append(directory)
        # ...and import it under its plain name, unless another module has it.
        name = _importable_name(resolved)
    if name is not None:
        return importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise SpecError(f"Cannot import {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def _normalized(spec: str) -> str:
    target, sep, attr = spec.partition(":")
    path = Path(target)
    if path.suffix == ".py" and path.exists():
        target = str(path.resolve())
    return f"{target}{sep}{attr}"


def list_evaluations(target: str) -> dict[str, Evaluation]:
    """The :class:`Evaluation` objects defined at module level in ``target``."""
    module = _import_module(target)
    return {
        attr: value
        for attr, value in vars(module).items()
        if isinstance(value, Evaluation)
    }


def load_object(spec: str) -> Any:
    """Resolve ``package.module:attr`` or ``path/to/file.py:attr``."""
    target, sep, attr = spec.partition(":")
    if not sep or not attr:
        raise SpecError(f"Expected MODULE:ATTR, got {spec!r}")
    module = _import_module(target)
    if not hasattr(module, attr):
        raise SpecError(f"{target} has no attribute {attr!r}")
    return getattr(module, attr)


def load_evaluation(spec: str) -> Evaluation:
    """
    Resolve ``package.module:attr`` or ``path/to/file.py:attr`` to an
    :class:`Evaluation`. ``attr`` may name a zero-argument factory; without
    ``:attr`` the module must define exactly one evaluation.
    """
    target, _, attr = spec.partition(":")
    module = _import_module(target)
    if attr:
        value = getattr(module, attr, None)
        if value is None:
            raise SpecError(f"{target} has no attribute {attr!r}")
        if not isinstance(value, Evaluation) and callable(value):
            value = cast("Callable[[], Any]", value)()
        if not isinstance(value, Evaluation):
            raise SpecError(f"{spec} is not an Evaluation (got {type(value).__name__})")
    else:
        found = {a: v for a, v in vars(module).items() if isinstance(v, Evaluation)}
        if len(found) != 1:
            raise SpecError(
                f"{target} defines {len(found)} evaluations {sorted(found)}; "
                f"use {target}:<attr>"
            )
        attr, value = next(iter(found.items()))
    value.spec = _normalized(f"{target}:{attr}")
    return value
