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
from .evaluator import Evaluator
from .metrics import Metric
from .runner import evaluate as evaluate_task
from .runner import rescore
from .store import LocalRunStore, RunStore
from .task import Task, as_task
from .types import EvaluationRun

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
    """

    name: str
    task: TaskSource
    dataset: DatasetSource
    evaluators: Sequence[Evaluator[Any, Any, Any]] = ()
    metrics: Sequence[Metric] | None = None
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
        source: DatasetSource = path if path is not None else self.dataset
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
    ) -> EvaluationRun:
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
            self.metrics,
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
            metadata={**self.metadata, **(metadata or {})},
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
    ) -> EvaluationRun:
        """
        Re-score a stored run of this evaluation with its current evaluators:
        changed ones run again, unchanged ones only where they failed or did
        not run (all of them with ``rerun=True``).
        """
        return await rescore(
            run,
            self.evaluators,
            self.metrics,
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
