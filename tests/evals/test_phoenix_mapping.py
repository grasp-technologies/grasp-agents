from typing import Any

from pydantic import BaseModel, TypeAdapter

from grasp_agents.evals import Example
from grasp_agents.evals.phoenix import from_phoenix_record, to_phoenix_record
from grasp_agents.evals.phoenix.sync import GRASP_KEY, record_id, record_node_id


class Question(BaseModel):
    text: str


def _round_trip(
    example: Example[Any, Any], input_type: Any, reference_type: Any
) -> Example[Any, Any]:
    record = to_phoenix_record(example)
    payload = {
        "id": "RGF0YXNldEV4YW1wbGU6MQ==",
        "input": record.input,
        "output": record.output,
        "metadata": record.metadata,
    }
    return from_phoenix_record(
        payload,
        input_adapter=TypeAdapter(input_type),
        reference_adapter=TypeAdapter(reference_type),
    )


def test_object_inputs_are_stored_as_is() -> None:
    example = Example(
        id="q1",
        input=Question(text="2+2"),
        reference={"answer": 4},
        metadata={"topic": "math"},
        splits=["dev"],
    )
    record = to_phoenix_record(example)
    assert record.input == {"text": "2+2"}
    assert record.output == {"answer": 4}
    assert record.metadata["topic"] == "math"
    assert record.metadata[GRASP_KEY]["id"] == "q1"
    back = _round_trip(example, Question, dict[str, int])
    assert back == example


def test_scalars_are_wrapped_and_unwrapped() -> None:
    example = Example[str, str](id="s", input="hello", reference="HELLO")
    record = to_phoenix_record(example)
    assert record.input == {"value": "hello"}
    assert record.output == {"value": "HELLO"}
    assert _round_trip(example, str, str) == example


def test_missing_reference_survives() -> None:
    example = Example[int, int](id="n", input=3)
    record = to_phoenix_record(example)
    assert record.output == {}
    back = _round_trip(example, int, int)
    assert back.reference is None
    assert back == example


def test_foreign_records_use_phoenix_ids() -> None:
    record = {
        "id": "ext-7",
        "node_id": "RGF0YXNldEV4YW1wbGU6Nw==",
        "input": {"question": "why?"},
        "output": {"answer": "because"},
        "metadata": {"source": "ui"},
    }
    example = from_phoenix_record(
        record, input_adapter=TypeAdapter(Any), reference_adapter=TypeAdapter(Any)
    )
    assert example.id == "ext-7"
    assert example.input == {"question": "why?"}
    assert example.reference == {"answer": "because"}
    assert record_id(record) == "ext-7"
    assert record_node_id(record) == "RGF0YXNldEV4YW1wbGU6Nw=="


def test_content_identity_is_stable() -> None:
    a = to_phoenix_record(
        Example[str, None](id="x", input="i", metadata={"b": 1, "a": 2})
    )
    b = to_phoenix_record(
        Example[str, None](id="x", input="i", metadata={"a": 2, "b": 1})
    )
    assert a.content() == b.content()
