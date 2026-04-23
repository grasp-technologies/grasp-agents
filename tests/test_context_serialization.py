"""Unit tests for :mod:`grasp_agents.durability.context_serialization`."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from grasp_agents.durability import (
    ContextKind,
    rehydrate_context,
    serialize_context,
)


class _MyState(BaseModel):
    pathway_id: str = ""
    count: int = 0


@dataclass
class _DCState:
    pathway_id: str = ""
    count: int = 0


def test_none_is_omitted() -> None:
    kind, data = serialize_context(None)
    assert kind == ContextKind.OMITTED
    assert data is None


def test_mapping_round_trip() -> None:
    state = {"a": 1, "b": [2, 3]}
    kind, data = serialize_context(state)
    assert kind == ContextKind.MAPPING
    assert data == {"a": 1, "b": [2, 3]}
    restored = rehydrate_context(kind, data, current_state={})
    assert restored == {"a": 1, "b": [2, 3]}


def test_pydantic_round_trip() -> None:
    state = _MyState(pathway_id="p-1", count=5)
    kind, data = serialize_context(state)
    assert kind == ContextKind.PYDANTIC
    assert data == {"pathway_id": "p-1", "count": 5}
    # Caller seeds ctx.state with an empty instance for type-inference.
    restored = rehydrate_context(kind, data, current_state=_MyState())
    assert isinstance(restored, _MyState)
    assert restored.pathway_id == "p-1"
    assert restored.count == 5


def test_dataclass_round_trip() -> None:
    state = _DCState(pathway_id="dc", count=9)
    kind, data = serialize_context(state)
    assert kind == ContextKind.DATACLASS
    assert data == {"pathway_id": "dc", "count": 9}
    restored = rehydrate_context(kind, data, current_state=_DCState())
    assert isinstance(restored, _DCState)
    assert restored.pathway_id == "dc"


def test_unsupported_type_is_omitted() -> None:
    class _Random:
        def __init__(self) -> None:
            self.x = 1

    kind, data = serialize_context(_Random())
    assert kind == ContextKind.OMITTED
    assert data is None


def test_rehydrate_omitted_leaves_current() -> None:
    sentinel = object()
    assert rehydrate_context(ContextKind.OMITTED, None, sentinel) is sentinel


def test_rehydrate_custom_leaves_current() -> None:
    sentinel = object()
    assert rehydrate_context(ContextKind.CUSTOM, {"foo": 1}, sentinel) is sentinel


def test_rehydrate_pydantic_without_seed_leaves_current() -> None:
    """
    If the caller didn't seed ``ctx.state`` with an instance of the
    expected pydantic type, the framework can't infer the target type
    and leaves ``current_state`` untouched.
    """
    restored = rehydrate_context(
        ContextKind.PYDANTIC, {"pathway_id": "p", "count": 1}, current_state=None
    )
    assert restored is None


def test_rehydrate_none_kind_leaves_current() -> None:
    assert rehydrate_context(None, None, 42) == 42
