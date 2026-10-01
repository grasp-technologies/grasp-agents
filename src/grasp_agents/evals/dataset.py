import json
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, cast, overload

import yaml
from pydantic import BaseModel, TypeAdapter, create_model

from ._util import short_hash, to_jsonable
from .types import DatasetRef, Example

_RECORD_KEYS = frozenset({"id", "input", "reference", "metadata", "splits"})
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})


class DatasetError(ValueError):
    pass


class DatasetProblem(BaseModel):
    example_id: str
    check: str
    message: str


type DatasetCheck = Callable[[Example[Any, Any]], str | Sequence[str] | None]
"""Returns the problems found in one example (``None`` / empty when it is fine)."""


class Dataset[InT, RefT]:
    """
    An ordered, id-addressable collection of :class:`Example` s.

    Derived datasets (``split``, ``sample``, ``select``, …) remember the source
    they came from and the steps applied, so a run records exactly which subset
    of which dataset version it evaluated.
    """

    def __init__(
        self,
        examples: Iterable[Example[InT, RefT]],
        *,
        name: str = "dataset",
        version: str | None = None,
        source: str | None = None,
        description: str | None = None,
    ) -> None:
        self._examples: list[Example[InT, RefT]] = list(examples)
        self._by_id: dict[str, Example[InT, RefT]] = {}
        for example in self._examples:
            if example.id in self._by_id:
                raise DatasetError(f"Duplicate example id {example.id!r} in {name!r}")
            self._by_id[example.id] = example
        self.name = name
        self.version = version
        self.source = source
        self.description = description
        self._origin: Dataset[InT, RefT] | None = None
        self._selection: list[str] = []

    # --- Collection protocol ---

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[Example[InT, RefT]]:
        return iter(self._examples)

    @overload
    def __getitem__(self, key: int) -> Example[InT, RefT]: ...
    @overload
    def __getitem__(self, key: str) -> Example[InT, RefT]: ...
    def __getitem__(self, key: int | str) -> Example[InT, RefT]:
        if isinstance(key, str):
            return self._by_id[key]
        return self._examples[key]

    def __contains__(self, example_id: object) -> bool:
        return example_id in self._by_id

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, examples={len(self)}, "
            f"fingerprint={self.fingerprint!r})"
        )

    @property
    def examples(self) -> Sequence[Example[InT, RefT]]:
        return tuple(self._examples)

    @property
    def ids(self) -> list[str]:
        return [example.id for example in self._examples]

    def get(self, example_id: str) -> Example[InT, RefT] | None:
        return self._by_id.get(example_id)

    @property
    def splits(self) -> list[str]:
        names: dict[str, None] = {}
        for example in self._examples:
            for split in example.splits:
                names.setdefault(split, None)
        return list(names)

    # --- Identity ---

    @cached_property
    def fingerprint(self) -> str:
        """Content hash of the examples, independent of their order."""
        parts = sorted(f"{e.id}:{e.content_hash()}" for e in self._examples)
        return short_hash(*parts)

    @property
    def origin(self) -> "Dataset[InT, RefT]":
        """The dataset this one was derived from (itself when not derived)."""
        return self._origin or self

    @property
    def selection(self) -> list[str]:
        return list(self._selection)

    def ref(self) -> DatasetRef:
        origin = self.origin
        return DatasetRef(
            name=origin.name,
            source=origin.source,
            version=origin.version,
            fingerprint=origin.fingerprint,
            size=len(origin),
            selection=self.selection,
            selected_fingerprint=self.fingerprint,
            selected_size=len(self),
        )

    # --- Selection ---

    def _derive(
        self, examples: Iterable[Example[InT, RefT]], step: str
    ) -> "Dataset[InT, RefT]":
        derived = Dataset(
            examples,
            name=self.name,
            version=self.version,
            source=self.source,
            description=self.description,
        )
        derived._origin = self.origin
        derived._selection = [*self._selection, step]
        return derived

    def select(self, ids: Iterable[str]) -> "Dataset[InT, RefT]":
        wanted = list(dict.fromkeys(ids))
        missing = [i for i in wanted if i not in self._by_id]
        if missing:
            raise DatasetError(f"Unknown example ids in {self.name!r}: {missing}")
        return self._derive((self._by_id[i] for i in wanted), f"ids={len(wanted)}")

    def split(self, name: str) -> "Dataset[InT, RefT]":
        return self._derive(
            (e for e in self._examples if name in e.splits), f"split={name}"
        )

    def exclude_splits(self, names: Iterable[str]) -> "Dataset[InT, RefT]":
        excluded = set(names)
        if not excluded:
            return self
        return self._derive(
            (e for e in self._examples if not excluded.intersection(e.splits)),
            f"exclude_splits={','.join(sorted(excluded))}",
        )

    def filter(
        self, predicate: Callable[[Example[InT, RefT]], bool], *, description: str
    ) -> "Dataset[InT, RefT]":
        return self._derive(
            (e for e in self._examples if predicate(e)), f"filter={description}"
        )

    def head(self, n: int) -> "Dataset[InT, RefT]":
        return self._derive(self._examples[:n], f"head={n}")

    def sample(self, n: int, *, seed: int = 0) -> "Dataset[InT, RefT]":
        """``n`` examples drawn without replacement; deterministic for a seed."""
        if n >= len(self._examples):
            return self._derive(self._examples, f"sample={n} seed={seed}")
        rng = random.Random(seed)  # noqa: S311
        chosen = set(rng.sample(range(len(self._examples)), n))
        return self._derive(
            (e for i, e in enumerate(self._examples) if i in chosen),
            f"sample={n} seed={seed}",
        )

    # --- Checks ---

    def check(
        self, checks: Mapping[str, DatasetCheck] | Sequence[DatasetCheck]
    ) -> list[DatasetProblem]:
        named: Mapping[str, DatasetCheck] = (
            checks
            if isinstance(checks, Mapping)
            else {getattr(c, "__name__", f"check_{i}"): c for i, c in enumerate(checks)}
        )
        problems: list[DatasetProblem] = []
        for example in self._examples:
            for check_name, check in named.items():
                try:
                    found = check(example)
                except Exception as exc:
                    found = f"check raised {type(exc).__name__}: {exc}"
                messages = [found] if isinstance(found, str) else list(found or [])
                problems.extend(
                    DatasetProblem(example_id=example.id, check=check_name, message=m)
                    for m in messages
                )
        return problems

    # --- Serialization ---

    def to_records(self) -> list[dict[str, Any]]:
        return [_to_record(e) for e in self._examples]

    def save(self, path: str | Path) -> Path:
        """Write as JSONL, or as YAML/JSON (``{name, description, examples}``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.to_records()
        if path.suffix == ".jsonl":
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                encoding="utf-8",
            )
            return path
        document: dict[str, Any] = {"name": self.name}
        if self.description:
            document["description"] = self.description
        document["examples"] = records
        if path.suffix in _YAML_SUFFIXES:
            path.write_text(
                yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        else:
            path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return path

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        input_type: Any = Any,
        reference_type: Any = Any,
        name: str = "dataset",
        version: str | None = None,
        source: str | None = None,
        description: str | None = None,
    ) -> "Dataset[Any, Any]":
        input_adapter: TypeAdapter[Any] = TypeAdapter(input_type)
        reference_adapter: TypeAdapter[Any] = TypeAdapter(reference_type)
        examples: list[Example[Any, Any]] = []
        for index, record in enumerate(records):
            unknown = set(record) - _RECORD_KEYS
            if unknown:
                raise DatasetError(
                    f"{name}: record {index} has unknown keys {sorted(unknown)}; "
                    f"expected a subset of {sorted(_RECORD_KEYS)}"
                )
            if "input" not in record:
                raise DatasetError(f"{name}: record {index} has no 'input'")
            try:
                reference = record.get("reference")
                examples.append(
                    Example[Any, Any](
                        id=str(record.get("id") or ""),
                        input=input_adapter.validate_python(record["input"]),
                        reference=(
                            None
                            if reference is None
                            else reference_adapter.validate_python(reference)
                        ),
                        metadata=dict(record.get("metadata") or {}),
                        splits=list(record.get("splits") or []),
                    )
                )
            except ValueError as exc:
                raise DatasetError(
                    f"{name}: record {index} (id={record.get('id')!r}) is invalid: "
                    f"{exc}"
                ) from exc
        return Dataset(
            examples,
            name=name,
            version=version,
            source=source,
            description=description,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        input_type: Any = Any,
        reference_type: Any = Any,
        name: str | None = None,
    ) -> "Dataset[Any, Any]":
        """Load a ``.jsonl``, ``.json`` or ``.yaml`` dataset file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        document_name: str | None = None
        description: str | None = None
        records: list[Mapping[str, Any]]
        if path.suffix == ".jsonl":
            records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("//")
            ]
        else:
            loaded: Any = (
                yaml.safe_load(text)
                if path.suffix in _YAML_SUFFIXES
                else json.loads(text)
            )
            if isinstance(loaded, list):
                records = cast("list[Mapping[str, Any]]", loaded)
            elif isinstance(loaded, dict) and "examples" in loaded:
                document = cast("dict[str, Any]", loaded)
                document_name = document.get("name")
                description = document.get("description")
                records = document["examples"]
            else:
                raise DatasetError(
                    f"{path}: expected a list of examples or an object with 'examples'"
                )
        return cls.from_records(
            records,
            input_type=input_type,
            reference_type=reference_type,
            name=name or document_name or path.stem,
            source=str(path),
            description=description,
        )


def _to_record(example: Example[Any, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"id": example.id, "input": to_jsonable(example.input)}
    if example.reference is not None:
        record["reference"] = to_jsonable(example.reference)
    if example.metadata:
        record["metadata"] = to_jsonable(example.metadata)
    if example.splits:
        record["splits"] = list(example.splits)
    return record


def example_json_schema(
    input_type: Any = Any, reference_type: Any = Any
) -> dict[str, Any]:
    """JSON Schema of one dataset record, for authoring and validating files."""
    model = create_model(
        "ExampleRecord",
        id=(str, ""),
        input=(input_type, ...),
        reference=(reference_type | None, None),
        metadata=(dict[str, Any], {}),
        splits=(list[str], []),
    )
    return model.model_json_schema()
