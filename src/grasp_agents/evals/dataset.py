import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, cast, overload

import yaml
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    create_model,
)

from ._util import short_hash
from .types import (
    DatasetRef,
    Example,
    ExampleRecord,
    content_digest,
    input_digest,
    with_record,
)

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_SUFFIXES = frozenset({".jsonl", ".json", *_YAML_SUFFIXES})


class DatasetError(ValueError):
    pass


class DatasetProblem(BaseModel):
    example_id: str
    check: str
    message: str


type DatasetCheck = Callable[[Example[Any, Any]], str | Sequence[str] | None]
"""Returns the problems found in one example (``None`` / empty when it is fine)."""


class _Record(BaseModel):
    """One dataset record as written in a file."""

    model_config = ConfigDict(extra="forbid")

    id: StrictStr | StrictInt | None = None
    input: Any
    reference: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])
    splits: list[StrictStr] = Field(default_factory=list[StrictStr])


def _is_model(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _alias_keys(alias: str | AliasPath | AliasChoices | None) -> list[str]:
    if alias is None:
        return []
    if isinstance(alias, str):
        return [alias]
    if isinstance(alias, AliasPath):
        first = alias.path[0] if alias.path else None
        return [first] if isinstance(first, str) else []
    return [key for choice in alias.choices for key in _alias_keys(choice)]


def _field_names(model: type[BaseModel]) -> set[str]:
    names: set[str] = set()
    for name, info in model.model_fields.items():
        names.add(name)
        if info.alias:
            names.add(info.alias)
        names.update(_alias_keys(info.validation_alias))
    return names


def _unknown_fields(tp: Any, raw: Any) -> list[str]:
    # Models that set ``extra`` decide for themselves; for the rest an unknown
    # key is almost always a typo that pydantic would silently drop.
    if not _is_model(tp) or not isinstance(raw, Mapping):
        return []
    model = cast("type[BaseModel]", tp)
    if model.model_config.get("extra") is not None:
        return []
    known = _field_names(model)
    return sorted(str(k) for k in cast("Mapping[Any, Any]", raw) if str(k) not in known)


def _explain(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in error['loc']) or 'record'}: {error['msg']}"
        for error in exc.errors()
    )


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
        parts = sorted(f"{e.id}:{e.content_hash}" for e in self._examples)
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
            elsewhere = [i for i in missing if i in self.origin]
            if elsewhere and self._selection:
                raise DatasetError(
                    f"Example ids {elsewhere} exist in {self.name!r} but are not in "
                    f"the selected subset ({', '.join(self._selection)})"
                )
            raise DatasetError(f"Unknown example ids in {self.name!r}: {missing}")
        return self._derive((self._by_id[i] for i in wanted), f"ids={len(wanted)}")

    def split(self, name: str) -> "Dataset[InT, RefT]":
        if name not in self.splits:
            raise DatasetError(
                f"No split {name!r} in {self.name!r}; splits: {self.splits or 'none'}"
            )
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
        if n < 0:
            raise ValueError("head: n must be >= 0")
        return self._derive(self._examples[:n], f"head={n}")

    def sample(self, n: int, *, seed: int = 0) -> "Dataset[InT, RefT]":
        """
        ``n`` examples drawn without replacement. The draw depends only on the
        seed and the example ids — not on their order or the Python version.
        """
        if n < 0:
            raise ValueError("sample: n must be >= 0")

        def rank(example: Example[InT, RefT]) -> str:
            return hashlib.sha256(f"{seed}:{example.id}".encode()).hexdigest()

        chosen = {e.id for e in sorted(self._examples, key=rank)[:n]}
        return self._derive(
            (e for e in self._examples if e.id in chosen), f"sample={n} seed={seed}"
        )

    # --- Checks ---

    def check(
        self, checks: Mapping[str, DatasetCheck] | Sequence[DatasetCheck]
    ) -> list[DatasetProblem]:
        named: dict[str, DatasetCheck] = {}
        if isinstance(checks, Mapping):
            named = dict(checks)
        else:
            for index, check in enumerate(checks):
                base = getattr(check, "__name__", "") or f"check_{index}"
                key, copy = base, 2
                while key in named:
                    key, copy = f"{base}#{copy}", copy + 1
                named[key] = check
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
        """
        Write as JSONL, or as YAML/JSON (``{name, description, examples}``).
        Each example is written as :attr:`Example.record` — loaded ones as they
        were read — so reloading keeps every content hash.
        """
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

    @overload
    @classmethod
    def from_records[I, R](
        cls,
        records: Iterable[Any],
        *,
        input_type: type[I],
        reference_type: type[R],
        name: str = ...,
        version: str | None = ...,
        source: str | None = ...,
        description: str | None = ...,
        locations: Sequence[str] | None = ...,
    ) -> "Dataset[I, R]": ...
    @overload
    @classmethod
    def from_records[I](
        cls,
        records: Iterable[Any],
        *,
        input_type: type[I],
        name: str = ...,
        version: str | None = ...,
        source: str | None = ...,
        description: str | None = ...,
        locations: Sequence[str] | None = ...,
    ) -> "Dataset[I, Any]": ...
    @overload
    @classmethod
    def from_records(
        cls,
        records: Iterable[Any],
        *,
        input_type: Any = ...,
        reference_type: Any = ...,
        name: str = ...,
        version: str | None = ...,
        source: str | None = ...,
        description: str | None = ...,
        locations: Sequence[str] | None = ...,
    ) -> "Dataset[Any, Any]": ...
    @classmethod
    def from_records(
        cls,
        records: Iterable[Any],
        *,
        input_type: Any = Any,
        reference_type: Any = Any,
        name: str = "dataset",
        version: str | None = None,
        source: str | None = None,
        description: str | None = None,
        locations: Sequence[str] | None = None,
    ) -> "Dataset[Any, Any]":
        """
        Build a dataset from plain records (``{id, input, reference,
        metadata, splits}``), validating inputs and references as the given
        types. Ids and content hashes come from the records as written.
        """
        input_adapter: TypeAdapter[Any] = TypeAdapter(input_type)
        reference_adapter: TypeAdapter[Any] = TypeAdapter(reference_type)
        examples: list[Example[Any, Any]] = []
        first_seen: dict[str, str] = {}
        for index, raw in enumerate(records):
            where = locations[index] if locations is not None else f"record {index}"
            examples.append(
                _parse_record(
                    raw,
                    where=f"{name}: {where}" if locations is None else where,
                    input_type=input_type,
                    reference_type=reference_type,
                    input_adapter=input_adapter,
                    reference_adapter=reference_adapter,
                )
            )
            example_id = examples[-1].id
            if example_id in first_seen:
                raise DatasetError(
                    f"{where}: duplicate example id {example_id!r} "
                    f"(first at {first_seen[example_id]})"
                )
            first_seen[example_id] = where
        return Dataset(
            examples,
            name=name,
            version=version,
            source=source,
            description=description,
        )

    @overload
    @classmethod
    def load[I, R](
        cls,
        path: str | Path,
        *,
        input_type: type[I],
        reference_type: type[R],
        name: str | None = ...,
    ) -> "Dataset[I, R]": ...
    @overload
    @classmethod
    def load[I](
        cls, path: str | Path, *, input_type: type[I], name: str | None = ...
    ) -> "Dataset[I, Any]": ...
    @overload
    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        input_type: Any = ...,
        reference_type: Any = ...,
        name: str | None = ...,
    ) -> "Dataset[Any, Any]": ...
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
        if path.suffix not in _SUFFIXES:
            raise DatasetError(
                f"{path}: unsupported dataset format (use .jsonl, .json or .yaml)"
            )
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatasetError(f"{path}: cannot read ({exc.strerror or exc})") from exc
        document_name: str | None = None
        description: str | None = None
        records: list[Any] = []
        locations: list[str] = []
        if path.suffix == ".jsonl":
            for lineno, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DatasetError(
                        f"{path}:{lineno}: invalid JSON ({exc.msg}, column {exc.colno})"
                    ) from exc
                locations.append(f"{path}:{lineno}")
        else:
            try:
                loaded: Any = (
                    yaml.safe_load(text)
                    if path.suffix in _YAML_SUFFIXES
                    else json.loads(text)
                )
            except (yaml.YAMLError, json.JSONDecodeError) as exc:
                raise DatasetError(f"{path}: cannot parse ({exc})") from exc
            if isinstance(loaded, list):
                records = cast("list[Any]", loaded)
            elif isinstance(loaded, dict) and "examples" in loaded:
                document = cast("dict[str, Any]", loaded)
                document_name = document.get("name")
                description = document.get("description")
                examples = document["examples"]
                if not isinstance(examples, list):
                    raise DatasetError(f"{path}: 'examples' must be a list")
                records = cast("list[Any]", examples)
            else:
                raise DatasetError(
                    f"{path}: expected a list of examples or an object with 'examples'"
                )
            locations = [f"{path}: examples[{i}]" for i in range(len(records))]
        return cls.from_records(
            records,
            input_type=input_type,
            reference_type=reference_type,
            name=name or document_name or path.stem,
            source=str(path),
            description=description,
            locations=locations,
        )


def _parse_record(
    raw: Any,
    *,
    where: str,
    input_type: Any,
    reference_type: Any,
    input_adapter: TypeAdapter[Any],
    reference_adapter: TypeAdapter[Any],
) -> Example[Any, Any]:
    if not isinstance(raw, Mapping):
        raise DatasetError(
            f"{where}: expected an object with 'input', got {type(raw).__name__}"
        )
    try:
        record = _Record.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(f"{where}: {_explain(exc)}") from exc
    label = f"{where} (id={record.id!r})" if record.id is not None else where
    for field, tp, value in (
        ("input", input_type, record.input),
        ("reference", reference_type, record.reference),
    ):
        unknown = _unknown_fields(tp, value)
        if unknown:
            known = sorted(_field_names(cast("type[BaseModel]", tp)))
            raise DatasetError(
                f"{label}: {field} has unknown fields {unknown}; expected {known}"
            )
    try:
        value = input_adapter.validate_python(record.input)
        reference = (
            None
            if record.reference is None
            else reference_adapter.validate_python(record.reference)
        )
    except ValidationError as exc:
        raise DatasetError(f"{label}: {_explain(exc)}") from exc
    try:
        example_id = (
            str(record.id)
            if record.id not in {None, ""}
            else input_digest(record.input)
        )
        content_hash = content_digest(record.input, record.reference, record.metadata)
    except TypeError as exc:
        raise DatasetError(f"{label}: {exc}") from exc
    example = Example[Any, Any](
        id=example_id,
        input=value,
        reference=reference,
        metadata=record.metadata,
        splits=list(record.splits),
        content_hash=content_hash,
    )
    return with_record(
        example, ExampleRecord(record.input, record.reference, record.metadata)
    )


def _to_record(example: Example[Any, Any]) -> dict[str, Any]:
    content = example.record
    record: dict[str, Any] = {"id": example.id, "input": content.input}
    if content.reference is not None:
        record["reference"] = content.reference
    if content.metadata:
        record["metadata"] = content.metadata
    if example.splits:
        record["splits"] = list(example.splits)
    return record


def _closed(tp: Any) -> Any:
    # The schema mirrors the loader, which rejects unknown keys of models that
    # leave ``extra`` unset.
    if not _is_model(tp):
        return tp
    model = cast("type[BaseModel]", tp)
    if model.model_config.get("extra") is not None:
        return model
    config = cast("ConfigDict", {**model.model_config, "extra": "forbid"})
    return type(
        model.__name__,
        (model,),
        {"model_config": config, "__module__": model.__module__},
    )


def example_json_schema(
    input_type: Any = Any, reference_type: Any = Any
) -> dict[str, Any]:
    """JSON Schema of one dataset record: exactly what the loader accepts."""
    model = create_model(
        "ExampleRecord",
        __config__=ConfigDict(extra="forbid"),
        id=(StrictStr | StrictInt | None, None),
        input=(_closed(input_type), ...),
        reference=(_closed(reference_type) | None, None),
        metadata=(dict[str, Any], Field(default_factory=dict[str, Any])),
        splits=(list[StrictStr], Field(default_factory=list[StrictStr])),
    )
    return model.model_json_schema()
