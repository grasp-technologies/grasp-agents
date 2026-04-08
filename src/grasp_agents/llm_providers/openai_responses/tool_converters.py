from typing import Any

from openai.lib._pydantic import to_strict_json_schema
from openai.types.responses.function_tool_param import (
    FunctionToolParam as OpenAIFunctionToolParam,
)
from openai.types.responses.response_create_params import (
    ToolChoice as OpenAIToolChoice,
)
from pydantic import BaseModel

from grasp_agents.types.tool import BaseTool, NamedToolChoice, ToolChoice


def to_api_tool(
    tool: BaseTool[BaseModel, Any, Any], strict: bool | None = None
) -> OpenAIFunctionToolParam:
    return OpenAIFunctionToolParam(
        type="function",
        name=tool.name,
        description=tool.description,
        parameters=to_strict_json_schema(tool.llm_in_type),
        strict=True if strict is None else strict,
    )


def to_api_tool_choice(tool_choice: ToolChoice) -> OpenAIToolChoice:
    if isinstance(tool_choice, NamedToolChoice):
        return {"type": "function", "name": tool_choice.name}
    return tool_choice
