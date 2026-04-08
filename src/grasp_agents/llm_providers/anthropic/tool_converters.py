"""Convert grasp-agents tool types → Anthropic API tool params."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from anthropic.types import (
    ToolChoiceAnyParam,
    ToolChoiceAutoParam,
    ToolChoiceNoneParam,
    ToolChoiceParam,
    ToolChoiceToolParam,
    ToolParam,
)
from grasp_agents.types.tool import BaseTool, NamedToolChoice, ToolChoice


def to_api_tool(tool: BaseTool[BaseModel, Any, Any]) -> ToolParam:
    return ToolParam(
        name=tool.name,
        description=tool.description,
        input_schema=tool.llm_in_type.model_json_schema(),
    )


def to_api_tool_choice(tool_choice: ToolChoice) -> ToolChoiceParam:
    if isinstance(tool_choice, NamedToolChoice):
        return ToolChoiceToolParam(type="tool", name=tool_choice.name)
    if tool_choice == "auto":
        return ToolChoiceAutoParam(type="auto")
    if tool_choice == "required":
        return ToolChoiceAnyParam(type="any")
    if tool_choice == "none":
        return ToolChoiceNoneParam(type="none")
    return ToolChoiceAutoParam(type="auto")
