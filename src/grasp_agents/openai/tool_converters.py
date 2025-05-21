from typing import Any

from pydantic import BaseModel

from ..typing.tool import BaseTool, ToolChoice
from . import (
    ChatCompletionFunctionDefinition,
    ChatCompletionNamedToolChoiceFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)


def to_api_tool(
    tool: BaseTool[BaseModel, Any, Any],
) -> ChatCompletionToolParam:
    function = ChatCompletionFunctionDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.in_schema.model_json_schema(),
        strict=tool.strict,
    )
    if tool.strict is None:
        function.pop("strict")

    return ChatCompletionToolParam(type="function", function=function)


def to_api_tool_choice(
    tool_choice: ToolChoice,
) -> ChatCompletionToolChoiceOptionParam:
    if isinstance(tool_choice, BaseTool):
        return ChatCompletionNamedToolChoiceParam(
            type="function",
            function=ChatCompletionNamedToolChoiceFunction(name=tool_choice.name),
        )
    return tool_choice
