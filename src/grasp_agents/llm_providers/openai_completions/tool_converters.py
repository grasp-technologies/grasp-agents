from typing import Any

from openai import pydantic_function_tool
from openai.types.chat.chat_completion_named_tool_choice_param import (
    ChatCompletionNamedToolChoiceParam,
)
from openai.types.chat.chat_completion_named_tool_choice_param import (
    Function as OpenAINamedToolChoiceFunction,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params import (
    FunctionDefinition as ChatCompletionFunctionDefinition,
)
from pydantic import BaseModel

from grasp_agents.types.tool import BaseTool, NamedToolChoice, ToolChoice


def to_api_tool(
    tool: BaseTool[BaseModel, Any, Any], strict: bool | None = None
) -> ChatCompletionToolParam:
    if strict:
        return pydantic_function_tool(
            model=tool.llm_in_type, name=tool.name, description=tool.description
        )

    function = ChatCompletionFunctionDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.llm_in_type.model_json_schema(),
        strict=strict,
    )
    if strict is None:
        function.pop("strict")

    return ChatCompletionToolParam(type="function", function=function)


def to_api_tool_choice(
    tool_choice: ToolChoice,
) -> ChatCompletionToolChoiceOptionParam:
    if isinstance(tool_choice, NamedToolChoice):
        return ChatCompletionNamedToolChoiceParam(
            type="function",
            function=OpenAINamedToolChoiceFunction(name=tool_choice.name),
        )
    return tool_choice
