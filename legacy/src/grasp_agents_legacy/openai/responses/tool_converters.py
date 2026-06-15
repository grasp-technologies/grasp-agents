from typing import Any

from openai import pydantic_function_tool
from openai.lib._pydantic import to_strict_json_schema
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam as OpenAIChatToolParam,
)
from openai.types.responses.response_create_params import (
    ToolChoice as OpenAIResponseToolChoice,
)
from openai.types.responses.tool_param import ToolParam as OpenAIResponseToolParam
from pydantic import BaseModel

from ...typing.tool import BaseTool, NamedToolChoice, ToolChoice


def to_api_tool(
    tool: BaseTool[BaseModel, Any, Any], strict: bool | None = None
) -> OpenAIResponseToolParam:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": to_strict_json_schema(tool.in_type),
        "strict": True if strict is None else strict,
    }


def to_api_tool_choice(tool_choice: ToolChoice) -> OpenAIResponseToolChoice:
    if isinstance(tool_choice, NamedToolChoice):
        return {"type": "function", "name": tool_choice.name}
    return tool_choice  # type: ignore[return-value]
