"""Utilities for deriving modified Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo


def exclude_fields(model: type[BaseModel], fields: set[str]) -> type[BaseModel]:
    """
    Create a new Pydantic model with *fields* removed.

    Useful for building a reduced ``llm_in_type`` from a tool's full
    ``in_type`` — e.g. hiding fields that a :class:`ToolInputConverter`
    will inject from context.

    >>> from pydantic import BaseModel
    >>> class Full(BaseModel):
    ...     query: str
    ...     api_key: str
    >>> Reduced = exclude_fields(Full, {"api_key"})
    >>> sorted(Reduced.model_fields)
    ['query']
    """
    kept: dict[str, tuple[type, FieldInfo]] = {}
    for name, info in model.model_fields.items():
        if name not in fields:
            kept[name] = (info.annotation, info)  # type: ignore[assignment]

    return create_model(
        f"{model.__name__}_llm",
        **kept,  # type: ignore[call-overload]
    )
