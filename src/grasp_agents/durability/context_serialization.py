"""
Optional machine-rehydration of ``SessionContext.state`` across session
checkpoint round-trips.

The framework's default stance is that application state is rebuilt by the
caller and passed at ``SessionContext`` construction — the persistent source of
truth is the app's own database, not the checkpoint. Serialization is
therefore **opt-in**: set ``SessionContext(serialize_state=True)`` for tests,
notebooks, and simple workloads where state is a plain container and
round-tripping it through the session checkpoint is convenient.

We record *how* the state was serialized alongside the payload so the
restore path knows what to do with it:

- ``OMITTED`` — state is ``None`` or an unrecognized type; load leaves
  ``ctx.state`` untouched. The caller rebuilds state itself.
- ``MAPPING`` — state is a ``dict`` (or dict-compatible). Round-trips as
  a JSON object; restored as ``dict``.
- ``PYDANTIC`` — state is a Pydantic ``BaseModel``. Round-trips via
  ``model_dump(mode="json")``; restored by calling
  ``type(current_state).model_validate(...)`` — so the caller must seed
  ``ctx.state`` with an empty instance (or rely on the default) before
  loading so we know the target type.
- ``DATACLASS`` — state is a dataclass instance. Same idea: caller seeds
  the type; we rehydrate via ``type(current_state)(**data)``.

The kind is auto-detected at save time by inspection; only the opt-in
toggle (``SessionContext.serialize_state``) is configured.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from dataclasses import fields as dataclass_fields
from enum import StrEnum
from typing import Any, cast, get_type_hints

from pydantic import BaseModel, TypeAdapter

logger = logging.getLogger(__name__)


class ContextKind(StrEnum):
    OMITTED = "omitted"
    MAPPING = "mapping"
    PYDANTIC = "pydantic"
    DATACLASS = "dataclass"


def serialize_context(state: Any) -> tuple[ContextKind, Any]:
    """
    Classify ``state`` and return ``(kind, json-safe payload)``.

    Unsupported types (arbitrary classes without pydantic/dataclass
    support) fall through to ``OMITTED`` — the caller is then
    responsible for reconstruction on resume.
    """
    if state is None:
        return ContextKind.OMITTED, None

    if isinstance(state, BaseModel):
        return ContextKind.PYDANTIC, state.model_dump(mode="json")

    # Dataclass check precedes Mapping because a dataclass could in
    # principle also satisfy Mapping via a user-defined __iter__/__getitem__.
    if is_dataclass(state) and not isinstance(state, type):
        return ContextKind.DATACLASS, asdict(state)

    if isinstance(state, Mapping):
        return ContextKind.MAPPING, dict(cast("Mapping[str, Any]", state))

    return ContextKind.OMITTED, None


def rehydrate_context(
    kind: ContextKind | None,
    data: Any,
    current_state: Any,
) -> Any:
    """
    Invert :func:`serialize_context`. Returns the rehydrated state, or
    ``current_state`` unchanged when ``kind`` is ``OMITTED`` / ``None``
    or when we can't infer the target type.

    ``current_state`` is consulted to learn the pydantic / dataclass
    type — callers should seed ``ctx.state`` with a fresh instance of
    the expected type before loading for automatic rehydration to work.
    """
    if kind is None or kind == ContextKind.OMITTED:
        return current_state

    if kind == ContextKind.MAPPING:
        return dict(data) if data is not None else {}

    if kind == ContextKind.PYDANTIC:
        if isinstance(current_state, BaseModel):
            return type(current_state).model_validate(data)
        return current_state

    if is_dataclass(current_state) and not isinstance(current_state, type):
        # ``TypeAdapter`` (not ``cls(**data)``) so nested dataclasses are
        # constructed rather than left as dicts, and ``init=False`` fields
        # don't crash the constructor.
        cls = type(current_state)
        try:
            restored = TypeAdapter(cls).validate_python(data)
            _restore_init_false_fields(cls, restored, data)
        except Exception:
            logger.warning(
                "Failed to rehydrate %s state from checkpoint; "
                "keeping the current state",
                cls.__name__,
                exc_info=True,
            )
            return current_state
        return restored

    return current_state


def _restore_init_false_fields(cls: type, restored: Any, data: Any) -> None:
    """
    Re-apply persisted values of ``init=False`` fields.

    Dataclass construction (and pydantic validation, which honors it)
    ignores inputs for ``init=False`` fields, so they would silently
    revert to their defaults on resume.
    """
    if not isinstance(data, Mapping):
        return
    data_map = cast("Mapping[str, Any]", data)
    init_false = [f for f in dataclass_fields(cls) if not f.init and f.name in data_map]
    if not init_false:
        return
    hints = get_type_hints(cls)
    for f in init_false:
        value: Any = TypeAdapter(hints.get(f.name, Any)).validate_python(  # type: ignore[arg-type]
            data_map[f.name]
        )
        object.__setattr__(restored, f.name, value)  # noqa: PLC2801  # frozen-safe
