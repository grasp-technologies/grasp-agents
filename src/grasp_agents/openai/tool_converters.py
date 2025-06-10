from typing import Any

from pydantic import BaseModel

from ..typing.tool import BaseTool, ToolChoice
from . import (
    OpenAIFunctionDefinition,
    OpenAINamedToolChoiceFunction,
    OpenAINamedToolChoiceParam,
    OpenAIToolChoiceOptionParam,
    OpenAIToolParam,
)


def to_api_tool(tool: BaseTool[BaseModel, Any, Any]) -> OpenAIToolParam:
    function = OpenAIFunctionDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.in_type.model_json_schema(),
        strict=tool.strict,
    )
    if tool.strict is None:
        function.pop("strict")

    return OpenAIToolParam(type="function", function=function)


def to_api_tool_choice(tool_choice: ToolChoice) -> OpenAIToolChoiceOptionParam:
    if isinstance(tool_choice, BaseTool):
        return OpenAINamedToolChoiceParam(
            type="function",
            function=OpenAINamedToolChoiceFunction(name=tool_choice.name),
        )
    return tool_choice
