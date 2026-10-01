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
from ._util import short_hash
from .dataset import Dataset, DatasetCheck, DatasetError, DatasetProblem
from .evaluator import Evaluator
from .metrics import Metric
from .runner import evaluate as evaluate_task
from .runner import rescore
from .store import RunStore
from .task import Task, as_task
from .types import EvaluationRun

PHOENIX_PREFIX = "phoenix:"


async def load_phoenix_dataset(
    ref: str, *, input_type: Any = Any, reference_type: Any = Any
) -> Dataset[Any, Any]:
    """Pull ``phoenix:NAME[@VERSION]`` from the Phoenix at ``$PHOENIX_BASE_URL``."""
    from .phoenix import (  # noqa: PLC0415
        PhoenixClient,
        pull_dataset,
    )

    name, _, version = ref.removeprefix(PHOENIX_PREFIX).partition("@")
    async with PhoenixClient() as client:
        return await pull_dataset(
            client,
            name,
            version=version or None,
            input_type=input_type,
            reference_type=reference_type,
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
    ``dataset`` a path or a loader. Run with :meth:`run`, or from the CLI as
    ``grasp-evals run package.module:attr``.
    """

    name: str
    task: TaskSource
    dataset: DatasetSource
    evaluators: Sequence[Evaluator[Any, Any, Any]] = ()
    metrics: Sequence[Metric] | None = None
    description: str | None = None
    # Types used to validate dataset files; the input type defaults to the
    # task's (a processor's ``in_type``).
    input_type: Any = None
    reference_type: Any = Any
    repetitions: int = 1
    concurrency: int = 4
    timeout_s: float | None = None
    max_cost_usd: float | None = None
    max_error_rate: float | None = None
    group_by: Sequence[str] = ()
    cluster_by: str | None = None
    # Held-out splits: reported in aggregate only, so an agent iterating on
    # the task cannot fit to their examples.
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

    async def load_dataset(self, path: str | Path | None = None) -> Dataset[Any, Any]:
        source: DatasetSource = path if path is not None else self.dataset
        if isinstance(source, Dataset):
            return cast("Dataset[Any, Any]", source)
        if isinstance(source, str) and source.startswith(PHOENIX_PREFIX):
            return await load_phoenix_dataset(
                source,
                input_type=self.resolved_input_type,
                reference_type=self.reference_type,
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
    ) -> Dataset[Any, Any]:
        """The dataset subset a run would evaluate (after integrity checks)."""
        data = (
            dataset
            if isinstance(dataset, Dataset)
            else await self.load_dataset(dataset)
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
        max_cost_usd: float | None = None,
        score: bool = True,
        name: str | None = None,
        store: RunStore | None = None,
        persist: bool = True,
        resume: "str | EvaluationRun | None" = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        progress: ProgressCallback | None = None,
        check: bool = True,
    ) -> EvaluationRun:
        data = await self.select(
            dataset=dataset,
            split=split,
            ids=ids,
            sample=sample,
            seed=seed,
            limit=limit,
            check=check,
        )
        return await evaluate_task(
            self.build_task(),
            data,
            self.evaluators,
            self.metrics,
            name=name or self.name,
            description=self.description,
            repetitions=repetitions or self.repetitions,
            concurrency=concurrency or self.concurrency,
            timeout_s=timeout_s if timeout_s is not None else self.timeout_s,
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
            tags=[*self.tags, *tags],
            metadata={**self.metadata, **(metadata or {})},
            evaluation=self.spec,
            progress=progress,
        )

    async def rescore(
        self,
        run: "str | EvaluationRun",
        *,
        store: RunStore | None = None,
        persist: bool = True,
        concurrency: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> EvaluationRun:
        """Re-score a stored run of this evaluation with its current evaluators."""
        return await rescore(
            run,
            self.evaluators,
            self.metrics,
            input_type=self.resolved_input_type,
            reference_type=self.reference_type,
            output_type=self.build_task().output_type,
            concurrency=concurrency or self.concurrency,
            group_by=self.group_by,
            cluster_by=self.cluster_by,
            store=store,
            persist=persist,
            progress=progress,
        )


def _import_module(target: str) -> ModuleType:
    path = Path(target)
    if path.suffix == ".py" and path.exists():
        resolved = path.resolve()
        module_name = (
            f"_grasp_evals_{resolved.stem}_{short_hash(str(resolved), length=8)}"
        )
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {target}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


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
        raise LookupError(f"Expected MODULE:ATTR, got {spec!r}")
    module = _import_module(target)
    if not hasattr(module, attr):
        raise LookupError(f"{target} has no attribute {attr!r}")
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
            raise LookupError(f"{target} has no attribute {attr!r}")
        if not isinstance(value, Evaluation) and callable(value):
            value = cast("Callable[[], Any]", value)()
        if not isinstance(value, Evaluation):
            raise TypeError(f"{spec} is not an Evaluation (got {type(value).__name__})")
    else:
        found = [v for v in vars(module).values() if isinstance(v, Evaluation)]
        if len(found) != 1:
            names = [a for a, v in vars(module).items() if isinstance(v, Evaluation)]
            raise LookupError(
                f"{target} defines {len(found)} evaluations {names}; "
                f"use {target}:<attr>"
            )
        value = found[0]
    value.spec = spec
    return value
