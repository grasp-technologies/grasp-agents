from typing import Any

# The OpenAI SDK exposes its strict-schema converter only in this private
# module; it is the same function the SDK itself uses for structured outputs.
from openai.lib._pydantic import to_strict_json_schema  # noqa: PLC2701
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
    # The strict transformation rewrites the schema (optionals become
    # required, ``additionalProperties: false``) — only apply it when strict
    # mode is actually requested.
    return OpenAIFunctionToolParam(
        type="function",
        name=tool.name,
        description=tool.description,
        parameters=(
            to_strict_json_schema(tool.llm_in_type)
            if strict
            else tool.llm_in_type.model_json_schema()
        ),
        strict=bool(strict),
    )


def to_api_tool_choice(tool_choice: ToolChoice) -> OpenAIToolChoice:
    if isinstance(tool_choice, NamedToolChoice):
        return {"type": "function", "name": tool_choice.name}
    return tool_choice
