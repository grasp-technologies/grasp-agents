from typing import Any

from openai import pydantic_function_tool
from pydantic import BaseModel

from ..typing.tool import BaseTool, NamedToolChoice, ToolChoice
from . import (
    OpenAIFunctionDefinition,
    OpenAINamedToolChoiceFunction,
    OpenAINamedToolChoiceParam,
    OpenAIToolChoiceOptionParam,
    OpenAIToolParam,
)


def to_api_tool(
    tool: BaseTool[BaseModel, Any, Any], strict: bool | None = None
) -> OpenAIToolParam:
    if strict:
        return pydantic_function_tool(
            model=tool.in_type, name=tool.name, description=tool.description
        )

    function = OpenAIFunctionDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.in_type.model_json_schema(),
        strict=strict,
    )
    if strict is None:
        function.pop("strict")

    return OpenAIToolParam(type="function", function=function)


def to_api_tool_choice(tool_choice: ToolChoice) -> OpenAIToolChoiceOptionParam:
    if isinstance(tool_choice, NamedToolChoice):
        return OpenAINamedToolChoiceParam(
            type="function",
            function=OpenAINamedToolChoiceFunction(name=tool_choice.name),
        )
    return tool_choice
