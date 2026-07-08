"""Convert grasp-agents tool types → Google Gemini API tool params."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.genai.types import FunctionCallingConfigMode
from pydantic import BaseModel

from grasp_agents.tools.base import NamedToolChoice, ToolChoice

from . import (
    GeminiFunctionCallingConfig,
    GeminiFunctionDeclaration,
    GeminiTool,
    GeminiToolConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from grasp_agents.tools.base import BaseTool


def to_api_tools(
    tools: Mapping[str, BaseTool[BaseModel, Any, Any]],
) -> GeminiTool:
    """Wrap all tool declarations in a single Gemini Tool object."""
    declarations = [
        GeminiFunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.llm_in_type.model_json_schema(),
        )
        for tool in tools.values()
    ]
    return GeminiTool(function_declarations=declarations)


def to_api_tool_config(
    tool_choice: ToolChoice, strict: bool | None = None
) -> GeminiToolConfig:
    """
    Map a :class:`ToolChoice` to a Gemini function-calling config.

    Gemini has no per-tool ``strict`` flag; provider-side schema enforcement
    for function-call arguments is a *mode*. ``ANY`` (used for ``"required"``
    and named tools) already constrains argument decoding; with ``strict``,
    the free-choice ``"auto"`` mode uses ``VALIDATED`` instead of ``AUTO``
    (Preview; documented for Gemini 3+ — older models accept the request
    but enforcement there is not documented).
    """
    if isinstance(tool_choice, NamedToolChoice):
        return GeminiToolConfig(
            function_calling_config=GeminiFunctionCallingConfig(
                mode=FunctionCallingConfigMode.ANY,
                allowed_function_names=[tool_choice.name],
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
    # "auto" and any unrecognized value
    return GeminiToolConfig(
        function_calling_config=GeminiFunctionCallingConfig(
            mode=(
                FunctionCallingConfigMode.VALIDATED
                if strict
                else FunctionCallingConfigMode.AUTO
            ),
        )
    )
