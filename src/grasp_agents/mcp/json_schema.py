from __future__ import annotations

import keyword
import re
from enum import Enum, IntEnum, StrEnum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, create_model


def json_schema_to_pydantic(
    schema: dict[str, Any],
    model_name: str,
) -> type[BaseModel]:
    """
    Create a Pydantic model from a JSON Schema dict.

    Supports the common subset of JSON Schema used for MCP tool inputs:
    object properties, required/optional fields, nested objects, arrays,
    enums, const, local $ref with caching, anyOf/oneOf as unions, and
    allOf as a shallow merge. Not a full JSON Schema implementation.

    Non-required fields are modeled as ``T | None = None`` for Pydantic
    ergonomics, even when the schema does not explicitly allow null.
    """
    defs = schema.get("$defs", schema.get("defs", {}))
    cache: dict[str, Any] = {}
    return _schema_to_model(model_name, schema, defs, set(), cache)


def _schema_to_model(
    name: str,
    schema: dict[str, Any],
    defs: dict[str, Any],
    resolving: set[str],
    cache: dict[str, Any],
) -> type[BaseModel]:
    """Recursively build a Pydantic model from a JSON Schema object."""
    properties: dict[str, Any] = schema.get("properties", {})
    required_fields: set[str] = set(schema.get("required", []))

    field_definitions: dict[str, Any] = {}
    for field_name, prop_schema in properties.items():
        is_required = field_name in required_fields
        annotation = _resolve_type(
            field_name, prop_schema, defs, name, resolving, cache
        )

        has_default = "default" in prop_schema
        default = prop_schema.get("default")
        description = prop_schema.get("description")
        field_kwargs: dict[str, Any] = {}
        if description:
            field_kwargs["description"] = description

        if is_required:
            field_definitions[field_name] = (
                annotation,
                Field(**field_kwargs) if field_kwargs else ...,
            )
        elif has_default:
            field_definitions[field_name] = (
                annotation | None,
                Field(default=default, **field_kwargs),
            )
        else:
            field_definitions[field_name] = (
                annotation | None,
                Field(default=None, **field_kwargs),
            )

    return create_model(name, **field_definitions)  # type: ignore[call-overload]


def _resolve_type(
    field_name: str,
    prop_schema: dict[str, Any],
    defs: dict[str, Any],
    parent_name: str,
    resolving: set[str],
    cache: dict[str, Any],
) -> Any:
    """Resolve a JSON Schema property to a Python type annotation."""
    # $ref
    if "$ref" in prop_schema:
        return _resolve_ref(prop_schema["$ref"], defs, parent_name, resolving, cache)

    # const → Literal
    if "const" in prop_schema:
        return Literal[prop_schema["const"]]  # type: ignore[valid-type]

    # anyOf / oneOf → Union
    variants = prop_schema.get("anyOf") or prop_schema.get("oneOf")
    if variants:
        types = tuple(
            _resolve_schema_node(
                f"{parent_name}_{field_name}_v{i}",
                v,
                defs,
                parent_name,
                resolving,
                cache,
            )
            for i, v in enumerate(variants)
        )
        return Union[types]  # type: ignore[valid-type]  # noqa: UP007

    # allOf → merge into single model
    if "allOf" in prop_schema:
        return _resolve_allof(
            field_name,
            prop_schema["allOf"],
            defs,
            parent_name,
            resolving,
            cache,
        )

    return _resolve_schema_node(
        field_name, prop_schema, defs, parent_name, resolving, cache
    )


def _resolve_ref(
    ref: str,
    defs: dict[str, Any],
    parent_name: str,
    resolving: set[str],
    cache: dict[str, Any],
) -> Any:
    """Resolve a $ref, returning a cached type if already built."""
    ref_name = ref.rsplit("/", maxsplit=1)[-1]

    if ref_name in cache:
        return cache[ref_name]

    ref_schema = defs.get(ref_name)
    if ref_schema is None:
        return dict[str, Any]

    # Guard against recursive $ref
    if ref_name in resolving:
        return dict[str, Any]

    result = _resolve_schema_node(
        ref_name, ref_schema, defs, parent_name, resolving, cache
    )
    cache[ref_name] = result
    return result


def _resolve_schema_node(
    name: str,
    schema: dict[str, Any],
    defs: dict[str, Any],
    parent_name: str,
    resolving: set[str],
    cache: dict[str, Any],
) -> Any:
    """Resolve a schema node (not a field — could be a $ref target, etc.)."""
    # $ref
    if "$ref" in schema:
        return _resolve_ref(schema["$ref"], defs, parent_name, resolving, cache)

    # const → Literal
    if "const" in schema:
        return Literal[schema["const"]]  # type: ignore[valid-type]

    # enum
    if "enum" in schema:
        return _make_enum(name, schema["enum"])

    # allOf → merge
    if "allOf" in schema:
        return _resolve_allof(
            name, schema["allOf"], defs, parent_name, resolving, cache
        )

    json_type = schema.get("type", "string")

    # object with properties → nested model
    if json_type == "object" and "properties" in schema:
        nested_name = _to_pascal(name)
        if nested_name in resolving:
            return dict[str, Any]
        resolving.add(nested_name)
        try:
            return _schema_to_model(nested_name, schema, defs, resolving, cache)
        finally:
            resolving.discard(nested_name)

    # array
    if json_type == "array":
        items_schema = schema.get("items", {"type": "string"})
        item_type = _resolve_schema_node(
            f"{name}_item",
            items_schema,
            defs,
            parent_name,
            resolving,
            cache,
        )
        return list[item_type]  # type: ignore[valid-type]

    # primitives
    type_map: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict[str, Any],  # type: ignore[dict-item]
    }
    return type_map.get(json_type, str)


def _resolve_allof(
    name: str,
    subschemas: list[dict[str, Any]],
    defs: dict[str, Any],
    parent_name: str,
    resolving: set[str],
    cache: dict[str, Any],
) -> Any:
    """Merge allOf subschemas into a single model."""
    merged: dict[str, Any] = {}
    for sub in subschemas:
        if "$ref" in sub:
            ref_name = sub["$ref"].split("/")[-1]
            sub = defs.get(ref_name, sub)  # noqa: PLW2901
        merged = _merge_schemas(merged, sub)

    model_name = _to_pascal(f"{parent_name}_{name}")
    return _schema_to_model(model_name, merged, defs, resolving, cache)


def _merge_schemas(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge two JSON Schema object dicts."""
    result = {**a, **b}
    if "properties" in a or "properties" in b:
        result["properties"] = {
            **a.get("properties", {}),
            **b.get("properties", {}),
        }
    req_a = set(a.get("required", []))
    req_b = set(b.get("required", []))
    if req_a or req_b:
        result["required"] = list(req_a | req_b)
    return result


# --- Enum helpers ---

_IDENT_RE = re.compile(r"[^a-zA-Z0-9_]")


def _safe_member_name(value: Any) -> str:
    """Turn an arbitrary enum value into a valid Python identifier."""
    s = _IDENT_RE.sub("_", str(value))
    if not s or s[0].isdigit():
        s = f"v_{s}"
    if keyword.iskeyword(s):
        s = f"{s}_"
    return s


def _make_enum(name: str, values: list[Any]) -> type[Enum]:
    """Build a StrEnum, IntEnum, or plain Enum depending on value types."""
    enum_name = _to_pascal(name)

    all_str = all(isinstance(v, str) for v in values)
    all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in values)

    members: dict[str, Any] = {}
    for i, v in enumerate(values):
        member_name = _safe_member_name(v)
        if member_name in members:
            member_name = f"{member_name}_{i}"
        members[member_name] = v

    if all_str:
        return StrEnum(enum_name, members)  # type: ignore[return-value]
    if all_int:
        return IntEnum(enum_name, members)  # type: ignore[return-value]
    # Mixed types — plain Enum
    return Enum(enum_name, members)  # type: ignore[return-value]


def _to_pascal(name: str) -> str:
    """Convert snake_case or kebab-case to PascalCase."""
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_"))
