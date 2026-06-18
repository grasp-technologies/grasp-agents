"""Tests for grasp_agents.utils.schema utilities."""

from pydantic import BaseModel, Field

from grasp_agents.utils.schema import exclude_fields


class FullInput(BaseModel):
    query: str
    api_key: str
    timeout: int = 30


def test_exclude_single_field():
    Reduced = exclude_fields(FullInput, {"api_key"})
    assert "query" in Reduced.model_fields
    assert "timeout" in Reduced.model_fields
    assert "api_key" not in Reduced.model_fields


def test_exclude_multiple_fields():
    Reduced = exclude_fields(FullInput, {"api_key", "timeout"})
    assert sorted(Reduced.model_fields) == ["query"]


def test_exclude_empty_set():
    Reduced = exclude_fields(FullInput, set())
    assert sorted(Reduced.model_fields) == sorted(FullInput.model_fields)


def test_generated_model_name():
    Reduced = exclude_fields(FullInput, {"api_key"})
    assert Reduced.__name__ == "FullInput_llm"


def test_preserves_defaults():
    Reduced = exclude_fields(FullInput, {"api_key"})
    instance = Reduced(query="test")  # type: ignore[call-arg]
    assert instance.timeout == 30  # type: ignore[attr-defined]


def test_preserves_field_descriptions():
    class Annotated(BaseModel):
        query: str = Field(description="Search query")
        secret: str = Field(description="Hidden")

    Reduced = exclude_fields(Annotated, {"secret"})
    assert Reduced.model_fields["query"].description == "Search query"


def test_json_schema_excludes_field():
    Reduced = exclude_fields(FullInput, {"api_key"})
    schema = Reduced.model_json_schema()
    assert "api_key" not in schema.get("properties", {})
    assert "query" in schema.get("properties", {})
