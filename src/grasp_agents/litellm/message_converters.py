from ..typing.message import (
    AssistantMessage,
)
from ..typing.tool import ToolCall
from . import LiteLLMCompletionMessage, LiteLLMFunction, LiteLLMToolCall


def from_api_assistant_message(
    api_message: LiteLLMCompletionMessage, name: str | None = None
) -> AssistantMessage:
    tool_calls = None
    if api_message.tool_calls is not None:
        tool_calls = [
            ToolCall(
                id=tool_call.id,
                tool_name=tool_call.function.name,  # type: ignore
                tool_arguments=tool_call.function.arguments,
            )
            for tool_call in api_message.tool_calls
        ]

    return AssistantMessage(
        content=api_message.content,
        tool_calls=tool_calls,
        name=name,
        thinking_blocks=getattr(api_message, "thinking_blocks", None),
        reasoning_content=getattr(api_message, "reasoning_content", None),
        annotations=getattr(api_message, "annotations", None),
        provider_specific_fields=api_message.provider_specific_fields,
        refusal=getattr(api_message, "refusal", None),
    )


def to_api_assistant_message(
    message: AssistantMessage,
) -> LiteLLMCompletionMessage:
    api_tool_calls = None
    if message.tool_calls is not None:
        api_tool_calls = [
            LiteLLMToolCall(
                type="function",
                id=tool_call.id,
                function=LiteLLMFunction(
                    name=tool_call.tool_name,
                    arguments=tool_call.tool_arguments,
                ),
            )
            for tool_call in message.tool_calls
        ]

    api_message = LiteLLMCompletionMessage(role="assistant", content=message.content)

    if api_tool_calls:
        api_message.tool_calls = api_tool_calls

    for key in [
        "thinking_blocks",
        "reasoning_content",
        "annotations",
        "provider_specific_fields",
        "refusal",
    ]:
        if getattr(message, key):
            api_message[key] = getattr(message, key)

    return api_message
