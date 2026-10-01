import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

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
        assert a.content_hash() == b.content_hash()
        assert a.content_hash() != c.content_hash()


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
        with pytest.raises(DatasetError, match="unknown keys"):
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
