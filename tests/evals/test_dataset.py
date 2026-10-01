import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import AliasChoices, BaseModel, Field

from grasp_agents.evals import Dataset, DatasetError, Example, example_json_schema


class Question(BaseModel):
    text: str
    level: int = 1


def _dataset() -> Dataset[Question, str]:
    return Dataset(
        [
            Example(id="q1", input=Question(text="2+2"), reference="4", splits=["dev"]),
            Example(id="q2", input=Question(text="3+3"), reference="6", splits=["dev"]),
            Example(
                id="q3",
                input=Question(text="capital of France"),
                reference="Paris",
                metadata={"topic": "geo"},
                splits=["test"],
            ),
        ],
        name="qa",
    )


class TestExample:
    def test_default_id_is_input_hash(self) -> None:
        a = Example[str, None](input="hello")
        b = Example[str, None](input="hello")
        c = Example[str, None](input="other")
        assert a.id
        assert a.id == b.id
        assert a.id != c.id

    def test_content_hash_ignores_splits(self) -> None:
        a = Example[str, str](id="x", input="i", reference="r")
        b = Example[str, str](id="x", input="i", reference="r", splits=["test"])
        c = Example[str, str](id="x", input="i", reference="changed")
        assert a.content_hash == b.content_hash
        assert a.content_hash != c.content_hash


class TestDataset:
    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(DatasetError, match="Duplicate"):
            Dataset(
                [
                    Example[str, None](id="a", input="1"),
                    Example[str, None](id="a", input="2"),
                ]
            )

    def test_access(self) -> None:
        ds = _dataset()
        assert len(ds) == 3
        assert ds[0].id == "q1"
        assert ds["q3"].reference == "Paris"
        assert "q2" in ds
        assert ds.splits == ["dev", "test"]

    def test_fingerprint_is_order_independent_and_content_sensitive(self) -> None:
        ds = _dataset()
        reordered = Dataset(list(reversed(ds.examples)), name="qa")
        assert ds.fingerprint == reordered.fingerprint
        changed = Dataset(
            [*ds.examples[:2], ds["q3"].model_copy(update={"reference": "Lyon"})]
        )
        assert changed.fingerprint != ds.fingerprint

    def test_selection_is_recorded(self) -> None:
        ds = _dataset()
        derived = ds.split("dev").head(1)
        assert derived.ids == ["q1"]
        ref = derived.ref()
        assert ref.name == "qa"
        assert ref.size == 3
        assert ref.fingerprint == ds.fingerprint
        assert ref.selection == ["split=dev", "head=1"]
        assert ref.selected_size == 1
        assert ref.selected_fingerprint == derived.fingerprint
        assert derived.origin is ds

    def test_sample_is_deterministic(self) -> None:
        ds = Dataset([Example[int, None](id=str(i), input=i) for i in range(30)])
        first = ds.sample(5, seed=7)
        second = ds.sample(5, seed=7)
        other = ds.sample(5, seed=8)
        assert first.ids == second.ids
        assert first.ids != other.ids
        assert len(first) == 5
        assert first.selection == ["sample=5 seed=7"]

    def test_select_unknown_ids(self) -> None:
        with pytest.raises(DatasetError, match="Unknown"):
            _dataset().select(["q1", "nope"])

    def test_exclude_splits(self) -> None:
        assert _dataset().exclude_splits(["test"]).ids == ["q1", "q2"]

    def test_checks_report_problems(self) -> None:
        def needs_reference(example: Example[Any, Any]) -> str | None:
            return None if example.reference else "missing reference"

        def explodes(example: Example[Any, Any]) -> None:
            raise RuntimeError("bad check")

        ds = Dataset(
            [
                Example[str, str](id="a", input="x", reference="y"),
                Example[str, str](id="b", input="z"),
            ]
        )
        problems = ds.check({"needs_reference": needs_reference, "explodes": explodes})
        assert {(p.example_id, p.check) for p in problems} == {
            ("b", "needs_reference"),
            ("a", "explodes"),
            ("b", "explodes"),
        }


class TestFiles:
    @pytest.mark.parametrize("suffix", [".jsonl", ".yaml", ".json"])
    def test_round_trip_typed(self, tmp_path: Path, suffix: str) -> None:
        ds = _dataset()
        path = ds.save(tmp_path / f"qa{suffix}")
        loaded = Dataset.load(path, input_type=Question, reference_type=str)
        assert loaded.ids == ds.ids
        assert isinstance(loaded["q1"].input, Question)
        assert loaded["q3"].metadata == {"topic": "geo"}
        assert loaded.fingerprint == ds.fingerprint
        assert loaded.name == "qa"
        assert loaded.source == str(path)

    def test_invalid_input_names_the_record(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"id": "x", "input": {"level": 2}}) + "\n")
        with pytest.raises(DatasetError, match="id='x'"):
            Dataset.load(path, input_type=Question)

    def test_unknown_keys_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"input": "x", "expected": "y"}) + "\n")
        with pytest.raises(DatasetError, match="expected: Extra inputs"):
            Dataset.load(path)

    def test_untyped_load_keeps_json(self, tmp_path: Path) -> None:
        path = tmp_path / "raw.jsonl"
        path.write_text(json.dumps({"input": {"a": 1}}) + "\n")
        loaded = Dataset.load(path)
        assert loaded[0].input == {"a": 1}

    def test_json_schema_describes_records(self) -> None:
        schema = example_json_schema(Question, str)
        assert schema["required"] == ["input"]
        assert "Question" in json.dumps(schema)


def _write_jsonl(path: Path, records: list[Any]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


class TestStrictRecords:
    def test_splits_must_be_a_list(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "d.jsonl", [{"input": "x", "splits": "test"}])
        with pytest.raises(DatasetError, match="splits"):
            Dataset.load(path)

    def test_integer_ids_are_kept_and_floats_refused(self, tmp_path: Path) -> None:
        ok = _write_jsonl(tmp_path / "ok.jsonl", [{"id": 0, "input": "a"}])
        assert Dataset.load(ok).ids == ["0"]
        bad = tmp_path / "bad.yaml"
        bad.write_text("- id: 1.10\n  input: a\n")
        with pytest.raises(DatasetError, match="id"):
            Dataset.load(bad)

    def test_misspelled_input_fields_are_refused(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "d.jsonl", [{"id": "x", "input": {"text": "q", "levle": 2}}]
        )
        with pytest.raises(DatasetError, match="unknown fields \\['levle'\\]"):
            Dataset.load(path, input_type=Question)

    def test_models_that_allow_extras_keep_them(self, tmp_path: Path) -> None:
        class Open(BaseModel, extra="allow"):
            text: str

        path = _write_jsonl(
            tmp_path / "d.jsonl", [{"id": "x", "input": {"text": "q", "more": 1}}]
        )
        assert Dataset.load(path, input_type=Open)["x"].input.more == 1  # type: ignore[attr-defined]

    def test_errors_point_at_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text('{"input": "a"}\n{"input": "b"\n')
        with pytest.raises(DatasetError, match=r"d\.jsonl:2: invalid JSON"):
            Dataset.load(path)
        dup = _write_jsonl(tmp_path / "dup.jsonl", [{"input": "a"}, {"input": "a"}])
        with pytest.raises(
            DatasetError, match=r"dup\.jsonl:2: duplicate.*first at .*:1"
        ):
            Dataset.load(dup)

    def test_schema_matches_the_loader(self, tmp_path: Path) -> None:
        jsonschema = pytest.importorskip("jsonschema")
        schema = example_json_schema(Question, str)
        valid = {"id": 3, "input": {"text": "q"}, "splits": ["dev"]}
        invalid = [
            {"input": {"text": "q"}, "expected": "y"},
            {"input": {"text": "q"}, "splits": "dev"},
            {"input": {"text": "q", "levle": 1}},
        ]
        jsonschema.validate(valid, schema)
        Dataset.load(_write_jsonl(tmp_path / "v.jsonl", [valid]), input_type=Question)
        for i, record in enumerate(invalid):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(record, schema)
            with pytest.raises(DatasetError):
                Dataset.load(
                    _write_jsonl(tmp_path / f"i{i}.jsonl", [record]),
                    input_type=Question,
                )


class TestIdentity:
    def test_new_defaulted_input_fields_keep_ids_and_hashes(
        self, tmp_path: Path
    ) -> None:
        class Before(BaseModel):
            text: str

        class After(BaseModel):
            text: str
            hint: str | None = None

        path = _write_jsonl(tmp_path / "d.jsonl", [{"input": {"text": "q"}}])
        old = Dataset.load(path, input_type=Before)
        new = Dataset.load(path, input_type=After)
        assert old.ids == new.ids
        assert old.fingerprint == new.fingerprint

    def test_hashing_is_canonical(self) -> None:
        from grasp_agents.evals._util import canonical_json

        assert canonical_json({"s": {3, 1, 2}}) == canonical_json({"s": {2, 3, 1}})
        assert canonical_json(float("nan")) != canonical_json(None)
        with pytest.raises(TypeError):
            canonical_json(object())

    def test_examples_are_immutable_and_copies_rehash(self) -> None:
        example = Example[str, str](id="x", input="q", reference="a")
        with pytest.raises(ValueError, match="frozen"):
            example.reference = "b"  # type: ignore[misc]
        changed = example.model_copy(update={"reference": "b"})
        assert changed.content_hash != example.content_hash
        assert changed.id == "x"


class TestSelectionRules:
    def test_sample_ignores_example_order(self) -> None:
        ds = Dataset([Example(id=f"e{i}", input=i) for i in range(30)])
        reordered = Dataset(list(reversed(ds.examples)))
        assert sorted(ds.sample(7, seed=3).ids) == sorted(
            reordered.sample(7, seed=3).ids
        )

    def test_negative_sizes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            _dataset().head(-1)
        with pytest.raises(ValueError, match="n must be"):
            _dataset().sample(-1)

    def test_unknown_split_lists_the_known_ones(self) -> None:
        with pytest.raises(DatasetError, match="splits: \\['dev', 'test'\\]"):
            _dataset().split("tset")

    def test_ids_outside_the_subset_are_explained(self) -> None:
        with pytest.raises(DatasetError, match="not in the selected subset"):
            _dataset().split("dev").select(["q3"])

    def test_checks_with_the_same_name_all_run(self) -> None:
        problems = _dataset().check([lambda e: "first", lambda e: "second"])
        assert {p.message for p in problems} == {"first", "second"}


class Draft(BaseModel):
    text: str
    weight: float = 1.0
    tags: set[str] = Field(default_factory=set[str])


def test_saving_keeps_content_hashes(tmp_path: Path) -> None:
    source = tmp_path / "drafts.jsonl"
    source.write_text(
        # An explicit default, and an int coerced to float on loading.
        '{"id": "a", "input": {"text": "x", "weight": 1.0}}\n'
        '{"id": "b", "input": {"text": "y", "weight": 2}}\n',
        encoding="utf-8",
    )
    loaded = Dataset.load(source, input_type=Draft)
    built = Dataset(
        [Example(id="c", input=Draft(text="z", tags={"q", "p"}))], name="built"
    )
    for original in (loaded, built):
        for suffix in (".jsonl", ".yaml", ".json"):
            path = original.save(tmp_path / "copies" / f"{original.name}{suffix}")
            copy = Dataset.load(path, input_type=Draft)
            assert [e.content_hash for e in copy] == [e.content_hash for e in original]
            assert copy.fingerprint == original.fingerprint


class Aliased(BaseModel):
    question: str = Field(validation_alias=AliasChoices("q", "question"))


def test_alias_keys_are_known_fields(tmp_path: Path) -> None:
    path = tmp_path / "aliased.jsonl"
    path.write_text('{"id": "a", "input": {"q": "why?"}}\n', encoding="utf-8")
    assert Dataset.load(path, input_type=Aliased)[0].input.question == "why?"
