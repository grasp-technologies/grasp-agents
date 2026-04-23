"""
Optional machine-rehydration of ``RunContext.state`` across checkpoint
round-trips.

The framework's default stance is that application state is rebuilt on
resume via ``@agent.add_state_builder`` — the persistent source of truth
is the app's own database, not the checkpoint. But for tests, notebooks,
and simple workloads where state is a plain container, it is handy to
have the framework serialize and restore it automatically.

We record *how* the state was serialized alongside the payload so the
restore path knows what to do with it:

- ``OMITTED`` — state is ``None`` (or intentionally not persisted); load
  leaves ``ctx.state`` untouched. ``state_builder`` does the work.
- ``MAPPING`` — state is a ``dict`` (or dict-compatible). Round-trips as
  a JSON object; restored as ``dict``.
- ``PYDANTIC`` — state is a Pydantic ``BaseModel``. Round-trips via
  ``model_dump(mode="json")``; restored by calling
  ``type(current_state).model_validate(...)`` — so the caller must seed
  ``ctx.state`` with an empty instance (or rely on the default) before
  loading so we know the target type.
- ``DATACLASS`` — state is a dataclass instance. Same idea: caller seeds
  the type; we rehydrate via ``type(current_state)(**data)``.
- ``CUSTOM`` — the framework can't rehydrate without a user codec; the
  payload is kept but ``ctx.state`` is left to ``state_builder``.

The kind is detected at save time by inspection; no configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel


class ContextKind(StrEnum):
    OMITTED = "omitted"
    MAPPING = "mapping"
    PYDANTIC = "pydantic"
    DATACLASS = "dataclass"
    CUSTOM = "custom"


def serialize_context(state: Any) -> tuple[ContextKind, Any]:
    """
    Classify ``state`` and return ``(kind, json-safe payload)``.

    Unsupported types (arbitrary classes without pydantic/dataclass
    support) fall through to ``OMITTED`` — ``state_builder`` is then
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
    ``current_state`` unchanged when the kind is not machine-rehydratable
    (``OMITTED`` / ``CUSTOM``) or when we can't infer the target type.

    ``current_state`` is consulted to learn the pydantic / dataclass
    type — callers should seed ``ctx.state`` with a fresh instance of
    the expected type before loading for automatic rehydration to work.
    """
    if kind is None or kind in {ContextKind.OMITTED, ContextKind.CUSTOM}:
        return current_state
    if kind == ContextKind.MAPPING:
        return dict(data) if data is not None else {}
    if kind == ContextKind.PYDANTIC:
        if isinstance(current_state, BaseModel):
            return type(current_state).model_validate(data)
        return current_state
    if is_dataclass(current_state) and not isinstance(current_state, type):
        return type(current_state)(**data)
    return current_state
