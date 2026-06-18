"""AutoInstanceAttributesMixin resolution with PEP 695 type parameters."""

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from grasp_agents.utils.generics import AutoInstanceAttributesMixin


class Base[T, U](AutoInstanceAttributesMixin):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "elem_type",
        1: "meta_type",
    }

    elem_type: type[Any]
    meta_type: type[Any]


class Concrete(Base[int, str]):
    pass


def test_subclass_resolution() -> None:
    obj = Concrete()
    assert obj.elem_type is int
    assert obj.meta_type is str


def test_alias_resolution() -> None:
    alias = Base[bytes, float]
    obj = alias()
    assert obj.elem_type is bytes
    assert obj.meta_type is float


def test_unresolved_params_fall_back_to_object() -> None:
    class Partial[T](Base[T, str]):
        pass

    obj = Partial()
    assert obj.elem_type is object
    assert obj.meta_type is str


def test_intermediate_with_map_redeclaration_resolves_chain() -> None:
    # An intermediate generic subclass must redeclare the map (mirroring
    # LLMAgent over Processor) for its own subclasses to resolve.
    class Mid[T](Base[T, str]):
        _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
            0: "elem_type",
            1: "meta_type",
        }

    class Leaf(Mid[bool]):
        pass

    obj = Leaf()
    assert obj.elem_type is bool
    # meta_type stays unresolved relative to Mid (its parameterization binds
    # only position 0); Base's binding is not substituted through the chain.
    assert obj.meta_type is object


def test_pydantic_generic_consumer_with_private_attr() -> None:
    # A pydantic consumer maps generic args to *declared private* attributes
    # (pydantic forbids setattr of undeclared names); the mixin writes the
    # resolved type into ``__pydantic_private__``.
    class PydBase[T](AutoInstanceAttributesMixin, BaseModel):
        _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
            0: "_payload_type"
        }
        _payload_type: type[Any] = object
        value: T

    class PydConcrete(PydBase[int]):
        pass

    assert PydConcrete(value=1)._payload_type is int
    assert PydBase[str](value="x")._payload_type is str


def test_pydantic_consumer_without_private_attrs_raises_cleanly() -> None:
    # ``__pydantic_private__`` is None (not absent) when a model declares no
    # private attributes. Resolution must not crash on the None itself; the
    # undeclared target attr then fails with pydantic's own error.
    class PydBase[T](AutoInstanceAttributesMixin, BaseModel):
        _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
            0: "payload_type"
        }
        value: T

    class PydConcrete(PydBase[int]):
        pass

    with pytest.raises(ValueError, match="payload_type"):
        PydConcrete(value=1)


def test_bounded_param_resolution() -> None:
    class Bounded[M: BaseModel](AutoInstanceAttributesMixin):
        _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
            0: "model_type"
        }

        model_type: type[Any]

    class Payload(BaseModel):
        pass

    class BoundedConcrete(Bounded[Payload]):
        pass

    assert BoundedConcrete().model_type is Payload


# ---------------------------------------------------------------------------
# Subscriptions must not poison the origin class
# ---------------------------------------------------------------------------


def test_alias_survives_other_parameterizations() -> None:
    """Store-alias-then-instantiate-later, with interleaved subscriptions."""
    alias = Base[bytes, float]
    # A different parameterization in between (e.g. another agent factory,
    # or a runtime `Base[Any, Any]` check) must not clobber the alias.
    _ = Base[int, str]
    _ = Base[Any, Any]

    obj = alias()
    assert obj.elem_type is bytes
    assert obj.meta_type is float


def test_runtime_any_subscription_does_not_poison_origin() -> None:
    _ = Base[Any, Any]
    # Class-statement subclasses keep their own resolution.
    assert Concrete().elem_type is int
    # The origin class's own (unparameterized) mapping is untouched.
    assert getattr(Base, "_resolved_instance_attr_types", {}) in ({}, None) or (
        Base._resolved_instance_attr_types.get("elem_type") is not Any
    )


def test_repeated_subscription_returns_same_class() -> None:
    assert Base[int, str] is Base[int, str]


def test_partially_concrete_subscription_resolves_concrete_args() -> None:
    # e.g. AgentTool building LLMAgent[Input, str, CtxT] with CtxT still free.
    def build[T]() -> Any:
        return Base[bytes, T]  # type: ignore[valid-type]

    specialized = build()
    obj = specialized()
    assert obj.elem_type is bytes
    assert obj.meta_type is object  # unresolved position falls back


def test_concurrent_parameterizations_are_isolated() -> None:
    import threading

    errors: list[str] = []

    def worker(elem: type, meta: type) -> None:
        for _ in range(50):
            obj = Base[elem, meta]()  # type: ignore[valid-type]
            if obj.elem_type is not elem or obj.meta_type is not meta:
                errors.append(f"got {obj.elem_type}/{obj.meta_type}")

    threads = [
        threading.Thread(target=worker, args=(int, str)),
        threading.Thread(target=worker, args=(bytes, float)),
        threading.Thread(target=worker, args=(dict, list)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_dunder_introspection_preserved() -> None:
    specialized = Base[int, str]
    assert specialized.__args__ == (int, str)  # type: ignore[attr-defined]
    assert specialized.__origin__ is Base  # type: ignore[attr-defined]
    # And it is a real subclass — isinstance works (it raised on the alias).
    assert issubclass(specialized, Base)
    assert isinstance(specialized(), Base)
