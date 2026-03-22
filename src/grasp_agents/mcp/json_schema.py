from __future__ import annotations

from typing import Any

from pydantic import BaseModel, create_model


def _json_schema_type_to_annotation(
    prop_schema: dict[str, Any],
    required: bool,
) -> Any:
    """Map a JSON Schema property to a Python type annotation."""
    if "enum" in prop_schema:
        # For enums, just use str (enum values are typically strings)
        base: type[Any] = str
        return base if required else base | None

    json_type = prop_schema.get("type", "string")

    type_map: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    if json_type == "array":
        item_type = _json_schema_type_to_annotation(
            prop_schema.get("items", {"type": "string"}),
            required=True,
        )
        base = list[item_type]  # type: ignore[valid-type]
    elif json_type == "object":
        base = dict[str, Any]
    else:
        base = type_map.get(json_type, str)

    if not required:
        return base | None
    return base


def json_schema_to_pydantic(
    schema: dict[str, Any],
    model_name: str,
) -> type[BaseModel]:
    """Create a Pydantic model from a JSON Schema dict.

    Handles: string, number, integer, boolean, array, object, enums,
    optional fields. Nested objects become dict[str, Any].
    """
    properties: dict[str, Any] = schema.get("properties", {})
    required_fields: set[str] = set(schema.get("required", []))

    field_definitions: dict[str, Any] = {}
    for name, prop_schema in properties.items():
        is_required = name in required_fields
        annotation = _json_schema_type_to_annotation(prop_schema, required=is_required)

        if is_required:
            field_definitions[name] = (annotation, ...)
        else:
            field_definitions[name] = (annotation, None)

    return create_model(model_name, **field_definitions)  # type: ignore[call-overload]
