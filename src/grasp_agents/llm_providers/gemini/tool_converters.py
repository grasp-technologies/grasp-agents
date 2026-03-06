"""Convert grasp-agents tool types → Google Gemini API tool params."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.genai.types import FunctionCallingConfigMode
from pydantic import BaseModel

from grasp_agents.types.tool import NamedToolChoice, ToolChoice

from . import (
    GeminiFunctionCallingConfig,
    GeminiFunctionDeclaration,
    GeminiTool,
    GeminiToolConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from grasp_agents.types.tool import BaseTool


def to_api_tools(
    tools: Mapping[str, BaseTool[BaseModel, Any, Any]],
) -> GeminiTool:
    """Wrap all tool declarations in a single Gemini Tool object."""
    declarations = [
        GeminiFunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.in_type.model_json_schema(),
        )
        for tool in tools.values()
    ]
    return GeminiTool(function_declarations=declarations)


def to_api_tool_config(tool_choice: ToolChoice) -> GeminiToolConfig:
    if isinstance(tool_choice, NamedToolChoice):
        return GeminiToolConfig(
            function_calling_config=GeminiFunctionCallingConfig(
                mode=FunctionCallingConfigMode.ANY,
                allowed_function_names=[tool_choice.name],
            )
        )
    if tool_choice == "auto":
        return GeminiToolConfig(
            function_calling_config=GeminiFunctionCallingConfig(
                mode=FunctionCallingConfigMode.AUTO,
            )
        )
    if tool_choice == "required":
        return GeminiToolConfig(
            function_calling_config=GeminiFunctionCallingConfig(
                mode=FunctionCallingConfigMode.ANY,
            )
        )
    if tool_choice == "none":
        return GeminiToolConfig(
            function_calling_config=GeminiFunctionCallingConfig(
                mode=FunctionCallingConfigMode.NONE,
            )
        )
    return GeminiToolConfig(
        function_calling_config=GeminiFunctionCallingConfig(
            mode=FunctionCallingConfigMode.AUTO,
        )
    )
