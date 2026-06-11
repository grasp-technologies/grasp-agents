from __future__ import annotations

import threading
import types
from typing import Any, ClassVar, Self, TypeVar, cast, get_args

# Concrete specializations built by ``__class_getitem__``, keyed by
# ``(origin class, params)`` so repeated subscriptions return the same class.
_SPECIALIZATIONS: dict[tuple[type, tuple[Any, ...]], type] = {}
_SPECIALIZATIONS_LOCK = threading.Lock()


def _param_name(param: Any) -> str:
    return getattr(param, "__name__", None) or repr(param)


class AutoInstanceAttributesMixin:
    """
    A **runtime convenience mix-in** that automatically exposes the concrete
    types supplied to a *generic* base class as **instance attributes**.

    Example:
    -------
    from typing import ClassVar
    from grasp_agents.utils.generics import AutoInstanceAttributesMixin

    class MyBase[T, U](AutoInstanceAttributesMixin):
        _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
            0: "elem_type",
            1: "meta_type",
        }

    class Concrete(MyBase[int, str]):
        ...

    Alias = MyBase[bytes, float]   # "late" specialization

    print(Concrete().elem_type)    # <class 'int'>
    print(Alias().meta_type)       # <class 'float'>

    Resolution looks for parameterizations of the **nearest base that declares
    ``_generic_arg_to_instance_attr_map`` in its own ``__dict__``**. An
    intermediate subclass that is itself generic in a mapped position
    (``class Mid[T](MyBase[T, str])``) must therefore redeclare the map —
    otherwise types bound by *its* subclasses resolve to ``object``.

    """

    # Configure this on your *generic* base class
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {}

    # Filled automatically for every concrete specialization
    _resolved_instance_attr_types: ClassVar[dict[str, type]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._set_resolved_generic_instance_attributes()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Runs when a *class statement* defines a new subclass."""
        super().__init_subclass__(**kwargs)
        cls._resolved_instance_attr_types = cls._compute_resolved_attrs(cls)

    @classmethod
    def __class_getitem__(cls, params: Any) -> type[Self]:
        """
        Run when someone writes ``SomeGeneric[ConcreteTypes]``.

        A concrete subscription returns a real (cached) subclass carrying its
        own resolved-type mapping. The typing alias must NOT be annotated in
        place: a plain class's alias forwards attribute writes to the ORIGIN
        class, so every subscription would overwrite one shared global —
        breaking the store-alias-then-instantiate-later pattern and racing
        across threads.
        """
        params_tuple: tuple[Any, ...] = (
            cast("tuple[Any, ...]", params) if isinstance(params, tuple) else (params,)
        )
        if all(isinstance(p, TypeVar) for p in params_tuple):
            # Pure re-parameterization (generic base lists, annotations) —
            # nothing concrete to resolve. Partially-concrete subscriptions
            # (e.g. ``LLMAgent[Input, str, CtxT]``) fall through: their
            # concrete positions must resolve.
            return cast("type[Self]", super().__class_getitem__(params))  # type: ignore[misc]

        try:
            cache_key = (cls, params_tuple)
            with _SPECIALIZATIONS_LOCK:
                cached = _SPECIALIZATIONS.get(cache_key)
        except TypeError:  # unhashable parameter
            cache_key = None
            cached = None
        if cached is not None:
            return cast("type[Self]", cached)

        alias = super().__class_getitem__(params)  # type: ignore[misc]
        if isinstance(alias, type):
            # Already a real class (e.g. a pydantic generic specialization) —
            # annotating it is safe and shared with nothing else.
            alias._resolved_instance_attr_types = cls._compute_resolved_attrs(  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]
                alias
            )
            specialized = alias
        else:
            # ``__init_subclass__`` resolves the attrs from ``__orig_bases__``.
            name = f"{cls.__name__}[{', '.join(_param_name(p) for p in params_tuple)}]"
            specialized = types.new_class(name, (cast("type", alias),))
            specialized.__module__ = cls.__module__
            # Direct ``__args__`` / ``__origin__`` introspection still works
            # (``typing.get_args`` does not — it only reads alias types).
            specialized.__args__ = params_tuple  # pyright: ignore[reportAttributeAccessIssue]
            specialized.__origin__ = cls  # pyright: ignore[reportAttributeAccessIssue]

        if cache_key is not None:
            with _SPECIALIZATIONS_LOCK:
                specialized = _SPECIALIZATIONS.setdefault(cache_key, specialized)
        return cast("type[Self]", specialized)

    @staticmethod
    def _compute_resolved_attrs(_cls: type) -> dict[str, type]:
        """
        Walks the MRO, finds the first generic base that defines
        `_generic_arg_to_instance_attr_map`, and resolves concrete types.
        """
        target_generic_base: Any | None = None
        attr_mapping: dict[int, str] | None = None

        # Locate the mapping
        for mro_cls in _cls.mro():
            if mro_cls in {AutoInstanceAttributesMixin, object}:
                continue
            if (
                hasattr(mro_cls, "__parameters__")
                and mro_cls.__parameters__  # type: ignore[has-type]
                and "_generic_arg_to_instance_attr_map" in mro_cls.__dict__
            ):
                target_generic_base = mro_cls
                attr_mapping = cast(
                    "dict[int, str]",
                    mro_cls._generic_arg_to_instance_attr_map,  # type: ignore[attr-defined] # noqa: SLF001
                )
                break

        if target_generic_base is None or attr_mapping is None:
            return {}

        resolved: dict[str, type] = {}

        def _add_to_resolved(generic_args: tuple[type | TypeVar, ...]) -> None:
            for index, attr_name in attr_mapping.items():
                if attr_name in resolved:
                    continue
                if index < len(generic_args):
                    arg = generic_args[index]
                    if not isinstance(arg, TypeVar):
                        resolved[attr_name] = arg

        def _all_resolved() -> bool:
            return all(name in resolved for name in attr_mapping.values())

        # Scenario 1: _cls itself is the direct parameterization (handles aliases).
        # e.g., _cls is MyBase[bytes, float]. Its __origin__ is MyBase.
        if getattr(_cls, "__origin__", None) is target_generic_base:
            _add_to_resolved(get_args(_cls))

        # Scenario 2: Check MRO for subclasses or more complex structures.
        # This also acts as a fallback if Scenario 1 didn't resolve all attributes.
        if not _all_resolved():
            for mro_candidate in _cls.mro():
                # Pydantic-specific check first
                pydantic_generic_metadata = getattr(
                    mro_candidate, "__pydantic_generic_metadata__", None
                )
                if (
                    pydantic_generic_metadata
                    and pydantic_generic_metadata.get("origin") is target_generic_base
                ):
                    _add_to_resolved(pydantic_generic_metadata.get("args", ()))
                    if _all_resolved():
                        break

                # Fall back to standard generic introspection if the
                # Pydantic check did not fully resolve.
                if not _all_resolved():
                    mro_candidate_origin = getattr(mro_candidate, "__origin__", None)
                    if mro_candidate_origin is target_generic_base:
                        _add_to_resolved(get_args(mro_candidate))
                        if _all_resolved():
                            break

                if not _all_resolved():
                    mro_candidate_orig_bases = getattr(
                        mro_candidate, "__orig_bases__", []
                    )
                    for param_base in mro_candidate_orig_bases:
                        param_base_origin = getattr(param_base, "__origin__", None)
                        if param_base_origin is target_generic_base:
                            _add_to_resolved(get_args(param_base))
                            if _all_resolved():
                                break

                if _all_resolved():
                    break

        return resolved

    def _set_resolved_generic_instance_attributes(self) -> None:
        attr_names = self._generic_arg_to_instance_attr_map.values()
        resolved_attr_types = getattr(
            self.__class__, "_resolved_instance_attr_types", {}
        )
        # ``__pydantic_private__`` is ``None`` (not absent) on pydantic models
        # that declare no private attributes.
        pyd_private: dict[str, Any] = getattr(self, "__pydantic_private__", None) or {}

        for attr_name in attr_names:
            if attr_name in resolved_attr_types:
                attr_type = resolved_attr_types[attr_name]
                if attr_type is Any:
                    attr_type = object
            else:
                attr_type = object

            if attr_name in pyd_private:
                pyd_private[attr_name] = attr_type
            else:
                setattr(self, attr_name, attr_type)
