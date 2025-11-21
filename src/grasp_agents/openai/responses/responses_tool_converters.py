"""
OpenAI Responses API — Tool Converters

This module provides thin adapters to convert our internal BaseTool objects and
tool choice into the exact shapes expected by the Responses API.

Why a separate adapter?
- Chat and Responses APIs use different tool parameter schemas.
- Chat: {"type":"function", "function": { name, description, parameters, strict }}
- Responses: flat FunctionToolParam at the top level:
    {
      "type": "function",
      "name": str,
      "description": str | None,
      "parameters": dict | None,   # JSON Schema
      "strict": bool | None
    }

These helpers return the Responses shapes so you can pass them directly to
client.responses.create/parse(..., tools=[...]).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, TypeAdapter

from ...typing.tool import BaseTool, NamedToolChoice, ToolChoice
from .. import (
    OpenAIResponsesParseableToolParam,  # alias to the Responses parseable ToolParam
    OpenAIResponsesToolParam,  # full ToolParam union type
    OpenAIResponseToolChoice,  # ToolChoice union type for Responses
)


def _model_json_schema(model_cls: type[BaseModel]) -> dict[str, Any] | None:
    """
    Return a JSON schema dict for a Pydantic v2 model class.

    Falls back to TypeAdapter for robustness when model_json_schema is missing.
    """
    try:
        return model_cls.model_json_schema()  # type: ignore[attr-defined]
    except Exception:
        try:
            return TypeAdapter(model_cls).json_schema()  # type: ignore[arg-type]
        except Exception:
            return None


def to_responses_function_tool(
    tool: BaseTool[BaseModel, Any, Any], *, strict: bool | None = None
) -> OpenAIResponsesParseableToolParam:
    """
    Convert our BaseTool to a Responses FunctionToolParam (ParseableToolParam).

    The Responses API expects a flat function tool shape. We extract the tool
    metadata (name, description) and its input JSON schema from the tool's
    Pydantic input model (tool.in_type).
    """
    schema = _model_json_schema(tool.in_type)

    # When strict mode is enabled, the Responses API requires
    # the top-level object schema to explicitly set
    # "additionalProperties": false.
    strict_val = True if strict is None else strict
    if strict_val and isinstance(schema, dict):
        if (schema.get("type") == "object"):
            # Copy to avoid mutating cached references
            schema = dict(schema)
            if schema.get("additionalProperties") is not False:
                schema["additionalProperties"] = False

    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": schema,
        "strict": strict_val,
    }


def to_responses_tool_choice(tool_choice: ToolChoice) -> OpenAIResponseToolChoice:
    """
    Map our ToolChoice → Responses ToolChoice.

    - "auto" | "none" | "required" → {"tool_choice": "auto"|"none"|"required"}
    - NamedToolChoice(name=...) → {"type": "function", "function": {"name": ...}}
    """
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            # For Responses API, tool_choice can be the literal string
            # 'auto' | 'none' | 'required'.
            return tool_choice
        raise ValueError(f"Unsupported ToolChoice: {tool_choice}")
    if isinstance(tool_choice, NamedToolChoice):
        # Force-call a specific function tool by name
        return {"type": "function", "name": tool_choice.name}
    raise ValueError(f"Unsupported ToolChoice object: {tool_choice!r}")


def to_responses_tools(
    tools: Mapping[str, BaseTool[BaseModel, Any, Any]], *, strict: bool | None = None
) -> list[OpenAIResponsesParseableToolParam]:
    """
    Convert a mapping of tools → list of Responses parseable tool params.

    Use this to build the 'tools' argument for client.responses.create/parse.
    """
    return [to_responses_function_tool(t, strict=strict) for t in tools.values()]
