import importlib
import importlib.util
import inspect
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from grasp_agents.processors.processor import Processor

from ._execution import ProgressCallback
from ._util import SPEC_MODULE_PREFIX, short_hash
from .dataset import Dataset, DatasetCheck, DatasetError, DatasetProblem
from .metrics import MetricsSpec, default_metrics
from .runner import evaluate as evaluate_task
from .runner import leave_out_sealed, rescore
from .scorer import Scorer
from .store import LocalRunStore, RunStore
from .task import Task, as_task
from .types import EvaluationRun, Trial
from .validation import (
    CorrectedPassRate,
    UnvalidatedJudgeError,
    ValidationGate,
    check_validations,
    outdated_judgments,
)

PHOENIX_PREFIX = "phoenix:"
_SCHEMES = ("http://", "https://")


@dataclass(frozen=True)
class PhoenixRef:
    """
    A Phoenix dataset: its ``name``, optionally pinned to a ``version`` —
    which only means something on the server that issued it, so a pinned
    reference should name that server (``base_url``). Written as
    ``phoenix:[https://HOST/]NAME[@VERSION]``.
    """

    name: str
    version: str | None = None
    base_url: str | None = None

    def __str__(self) -> str:
        server = f"{self.base_url}/" if self.base_url else ""
        version = f"@{self.version}" if self.version else ""
        return f"{PHOENIX_PREFIX}{server}{self.name}{version}"

    def server(self, base_url: str | None = None) -> str | None:
        """
        The server to pull from: the one this reference names, else
        ``base_url`` (else ``$PHOENIX_BASE_URL``, read by the client).
        Raises ``ValueError`` when both are given and differ.
        """
        if self.base_url is None:
            return base_url
        from .phoenix import (  # ruff: ignore[import-outside-top-level]
            normalize_base_url,
        )

        if base_url is not None and normalize_base_url(base_url) != normalize_base_url(
            self.base_url
        ):
            raise ValueError(f"{self} names the server {self.base_url}, not {base_url}")
        return self.base_url


def parse_phoenix_ref(ref: str) -> PhoenixRef:
    """
    The :class:`PhoenixRef` of ``[phoenix:][https://HOST/]NAME[@VERSION]``.
    A server is a scheme and host (with a port, no path); everything after
    it up to the last ``@`` is the dataset name.
    """
    body = ref.removeprefix(PHOENIX_PREFIX)
    base_url: str | None = None
    if body.startswith(_SCHEMES):
        scheme, _, rest = body.partition("://")
        host, _, body = rest.partition("/")
        if not host or not body:
            raise ValueError(
                f"Expected phoenix:{scheme}://HOST/NAME[@VERSION], got {ref!r}"
            )
        base_url = f"{scheme}://{host}"
    name, at, version = body.rpartition("@")
    if not at:
        return PhoenixRef(body, None, base_url)
    return PhoenixRef(name, version or None, base_url)


async def load_phoenix_dataset(
    ref: "str | PhoenixRef",
    *,
    input_type: Any = Any,
    reference_type: Any = Any,
    base_url: str | None = None,
    cache_dir: str | Path | None = None,
) -> Dataset[Any, Any]:
    """
    Pull a Phoenix dataset (``phoenix:[https://HOST/]NAME[@VERSION]`` or a
    :class:`PhoenixRef`) from the server it names, else from ``base_url``
    (``$PHOENIX_BASE_URL``), caching pinned versions in ``cache_dir``.
    """
    from .phoenix import (  # ruff: ignore[import-outside-top-level]
        PhoenixClient,
        pull_dataset,
    )

    parsed = ref if isinstance(ref, PhoenixRef) else parse_phoenix_ref(ref)
    async with PhoenixClient(parsed.server(base_url)) as client:
        return await pull_dataset(
            client,
            parsed.name,
            version=parsed.version,
            input_type=input_type,
            reference_type=reference_type,
            cache_dir=cache_dir,
        )


async def load_dataset_source(
    source: "DatasetSource",
    *,
    input_type: Any = Any,
    reference_type: Any = Any,
    phoenix_url: str | None = None,
    cache_dir: str | Path | None = None,
) -> Dataset[Any, Any]:
    """
    The dataset ``source`` names: a :class:`Dataset`, a file, a
    ``phoenix:`` reference or a (possibly async) loader.
    """
    if isinstance(source, Dataset):
        return cast("Dataset[Any, Any]", source)
    if isinstance(source, PhoenixRef) or (
        isinstance(source, str) and source.startswith(PHOENIX_PREFIX)
    ):
        return await load_phoenix_dataset(
            source,
            input_type=input_type,
            reference_type=reference_type,
            base_url=phoenix_url,
            cache_dir=cache_dir,
        )
    if isinstance(source, str | Path):
        return Dataset.load(
            source, input_type=input_type, reference_type=reference_type
        )
    loaded = source()
    if inspect.isawaitable(loaded):
        loaded = await loaded
    return loaded


type TaskSource = (
    Task[Any, Any]
    | Processor[Any, Any, Any]
    | Callable[[], Task[Any, Any] | Processor[Any, Any, Any]]
)
type DatasetSource = (
    Dataset[Any, Any]
    | str
    | Path
    | PhoenixRef
    | Callable[[], Dataset[Any, Any] | Awaitable[Dataset[Any, Any]]]
)


@dataclass
class Evaluation:
    """
    A named, importable evaluation: the question it answers, the task, where
    its data comes from, how outputs are judged and which numbers to report.

    Definitions are cheap to import: ``task`` may be a zero-argument factory
    (called on the first run, so no clients are built at import time) and
    ``dataset`` a path, a Phoenix reference
    (``phoenix:[https://HOST/]NAME[@VERSION]`` or :class:`PhoenixRef`) or a
    loader.
    Run with :meth:`run`, or from the CLI as ``grasp-evals run
    package.module:attr``.
    """

    name: str
    task: TaskSource
    dataset: DatasetSource
    scorers: Sequence[Scorer[Any, Any, Any]] = ()
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
    scorer_timeout_s: float | None = None
    max_cost_usd: float | None = None
    max_error_rate: float | None = None
    group_by: Sequence[str] = ()
    cluster_by: str | None = None
    # Held-out splits: left out of a run unless it names one (``split=``) or
    # includes them (``include_sealed=True``); reports, ``show``, ``compare``
    # and Phoenix pushes give only their aggregates, and a run must include
    # each one whole. This keeps an agent iterating through the CLI from
    # fitting to their examples; it is not access control — the dataset file
    # and run records contain them.
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
    # For a dataset of judged outputs (labels, probe items): the dataset of
    # the examples they answer. An item whose example is gone, or whose input
    # or reference changed since it was judged, is a dataset problem.
    judged_dataset: "DatasetSource | None" = None
    tags: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])
    # Import spec this definition was loaded from (set by ``load_evaluation``).
    spec: str | None = None
    _task: Task[Any, Any] | None = field(default=None, init=False, repr=False)

    def build_task(self) -> Task[Any, Any]:
        if self._task is None:
            source = self.task
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
        return self.build_task().input_type

    @property
    def resolved_output_type(self) -> Any:
        if self.output_type is not None:
            return self.output_type
        return self.build_task().output_type

    async def load_dataset(
        self,
        path: str | Path | None = None,
        *,
        phoenix_url: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> Dataset[Any, Any]:
        return await load_dataset_source(
            path if path is not None else self.dataset,
            input_type=self.resolved_input_type,
            reference_type=self.reference_type,
            phoenix_url=phoenix_url,
            cache_dir=cache_dir,
        )

    def check_dataset(self, dataset: Dataset[Any, Any]) -> list[DatasetProblem]:
        return dataset.check(self.dataset_checks) if self.dataset_checks else []

    async def dataset_problems(
        self,
        dataset: Dataset[Any, Any],
        *,
        phoenix_url: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> list[DatasetProblem]:
        """
        Problems :meth:`select` refuses a dataset for: ``dataset_checks``,
        and judged outputs that no longer match :attr:`judged_dataset`.
        """
        problems = self.check_dataset(dataset)
        if self.judged_dataset is not None:
            judged = await load_dataset_source(
                self.judged_dataset, phoenix_url=phoenix_url, cache_dir=cache_dir
            )
            problems.extend(outdated_judgments(dataset, judged))
        return problems

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
            self.scorers,
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
        include_sealed: bool = False,
        check: bool = True,
        phoenix_url: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> Dataset[Any, Any]:
        """
        The dataset subset a run would evaluate (after integrity checks):
        ``split``, then ``ids``, ``sample`` and ``limit``. Sealed splits are
        left out unless ``split`` names one or ``include_sealed``.
        """
        data = (
            dataset
            if isinstance(dataset, Dataset)
            else await self.load_dataset(
                dataset, phoenix_url=phoenix_url, cache_dir=cache_dir
            )
        )
        if check:
            problems = await self.dataset_problems(
                data, phoenix_url=phoenix_url, cache_dir=cache_dir
            )
            if problems:
                listed = "\n".join(
                    f"  {p.example_id}: [{p.check}] {p.message}" for p in problems[:20]
                )
                raise DatasetError(
                    f"{len(problems)} dataset problem(s) in {data.name!r}:\n{listed}"
                )
        if split is not None:
            data = data.split(split)
        data = leave_out_sealed(
            data, frozenset(self.sealed_splits), include_sealed=include_sealed
        )
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
        include_sealed: bool = False,
        repetitions: int | None = None,
        concurrency: int | None = None,
        timeout_s: float | None = None,
        scorer_timeout_s: float | None = None,
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
            include_sealed=include_sealed,
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
            self.scorers,
            metrics,
            name=name or self.name,
            description=self.description,
            repetitions=repetitions if repetitions is not None else self.repetitions,
            concurrency=concurrency if concurrency is not None else self.concurrency,
            timeout_s=timeout_s if timeout_s is not None else self.timeout_s,
            scorer_timeout_s=scorer_timeout_s
            if scorer_timeout_s is not None
            else self.scorer_timeout_s,
            max_cost_usd=max_cost_usd
            if max_cost_usd is not None
            else self.max_cost_usd,
            max_error_rate=self.max_error_rate,
            group_by=self.group_by,
            cluster_by=self.cluster_by,
            sealed_splits=self.sealed_splits,
            include_sealed=include_sealed,
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
        split: str | None = None,
        ids: Sequence[str] | None = None,
        dataset: DatasetSource | None = None,
        rerun: bool = False,
        store: RunStore | None = None,
        persist: bool = True,
        concurrency: int | None = None,
        progress: ProgressCallback | None = None,
        allow_unvalidated: bool = False,
        phoenix_url: str | None = None,
    ) -> EvaluationRun:
        """
        Re-score a stored run of this evaluation with its current scorers:
        changed ones run again, unchanged ones only where they failed or did
        not run (all of them with ``rerun=True``). ``split`` and ``ids``
        narrow it to those examples; ``dataset`` (a file, a ``phoenix:``
        reference, ...) scores against that version of the dataset, after
        this evaluation's dataset checks (see :func:`rescore`). Gated judges
        must be validated, as for :meth:`run`.
        """
        metrics, judges = self.check_judges(store, allow_unvalidated=allow_unvalidated)
        newer: Dataset[Any, Any] | None = None
        if dataset is not None:
            newer = await load_dataset_source(
                dataset,
                input_type=self.resolved_input_type,
                reference_type=self.reference_type,
                phoenix_url=phoenix_url,
            )
            problems = await self.dataset_problems(newer, phoenix_url=phoenix_url)
            if problems:
                listed = "\n".join(
                    f"  {p.example_id}: [{p.check}] {p.message}" for p in problems[:20]
                )
                raise DatasetError(
                    f"{len(problems)} dataset problem(s) in {newer.name!r}:\n{listed}"
                )
        return await rescore(
            run,
            self.scorers,
            metrics,
            input_type=self.resolved_input_type,
            reference_type=self.reference_type,
            output_type=self.resolved_output_type,
            split=split,
            ids=ids,
            dataset=newer,
            concurrency=concurrency if concurrency is not None else self.concurrency,
            scorer_timeout_s=self.scorer_timeout_s,
            group_by=self.group_by,
            cluster_by=self.cluster_by,
            rerun=rerun,
            store=store,
            persist=persist,
            metadata=judges,
            progress=progress,
        )


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
