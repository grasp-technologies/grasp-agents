from typing import TypeAlias

from ..typing.message import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    Usage,
    UserMessage,
)
from ..typing.tool import ToolCall
from . import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolCallFunction,
    ChatCompletionToolMessageParam,
    ChatCompletionUsage,
    ChatCompletionUserMessageParam,
)
from .content_converters import from_api_content, to_api_content

OpenAIMessage: TypeAlias = (
    ChatCompletionAssistantMessageParam
    | ChatCompletionToolMessageParam
    | ChatCompletionUserMessageParam
    | ChatCompletionDeveloperMessageParam
    | ChatCompletionSystemMessageParam
    | ChatCompletionFunctionMessageParam
)


def from_api_user_message(
    api_message: ChatCompletionUserMessageParam, model_id: str | None = None
) -> UserMessage:
    content = from_api_content(api_message["content"])

    return UserMessage(content=content, model_id=model_id)


def to_api_user_message(message: UserMessage) -> ChatCompletionUserMessageParam:
    api_content = to_api_content(message.content)

    return ChatCompletionUserMessageParam(role="user", content=api_content)


def from_api_assistant_message(
    api_message: ChatCompletionMessage,
    api_usage: ChatCompletionUsage | None = None,
    model_id: str | None = None,
) -> AssistantMessage:
    usage = None
    if api_usage is not None:
        reasoning_tokens = None
        cached_tokens = None

        if api_usage.completion_tokens_details is not None:
            reasoning_tokens = api_usage.completion_tokens_details.reasoning_tokens
        if api_usage.prompt_tokens_details is not None:
            cached_tokens = api_usage.prompt_tokens_details.cached_tokens

        input_tokens = api_usage.prompt_tokens - (cached_tokens or 0)
        output_tokens = api_usage.completion_tokens - (reasoning_tokens or 0)

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
        )

    tool_calls = None
    if api_message.tool_calls is not None:
        tool_calls = [
            ToolCall(
                id=tool_call.id,
                tool_name=tool_call.function.name,
                tool_arguments=tool_call.function.arguments,
            )
            for tool_call in api_message.tool_calls
        ]

    return AssistantMessage(
        content=api_message.content,
        usage=usage,
        tool_calls=tool_calls,
        refusal=api_message.refusal,
        model_id=model_id,
    )


def to_api_assistant_message(
    message: AssistantMessage,
) -> ChatCompletionAssistantMessageParam:
    api_tool_calls = None
    if message.tool_calls is not None:
        api_tool_calls = [
            ChatCompletionMessageToolCallParam(
                type="function",
                id=tool_call.id,
                function=ChatCompletionToolCallFunction(
                    name=tool_call.tool_name,
                    arguments=tool_call.tool_arguments,
                ),
            )
            for tool_call in message.tool_calls
        ]

    api_message = ChatCompletionAssistantMessageParam(
        role="assistant",
        content=message.content,
        tool_calls=api_tool_calls or [],
        refusal=message.refusal,
    )
    if message.content is None:
        # Some API providers return None in the generated content without errors,
        # even though None in the input content is not accepted.
        api_message["content"] = "<empty>"
    if api_tool_calls is None:
        api_message.pop("tool_calls")
    if message.refusal is None:
        api_message.pop("refusal")

    return api_message


def from_api_system_message(
    api_message: ChatCompletionSystemMessageParam,
    model_id: str | None = None,
) -> SystemMessage:
    return SystemMessage(content=api_message["content"], model_id=model_id)  # type: ignore


def to_api_system_message(
    message: SystemMessage,
) -> ChatCompletionSystemMessageParam:
    return ChatCompletionSystemMessageParam(role="system", content=message.content)


def from_api_tool_message(
    api_message: ChatCompletionToolMessageParam, model_id: str | None = None
) -> ToolMessage:
    return ToolMessage(
        content=api_message["content"],  # type: ignore
        tool_call_id=api_message["tool_call_id"],
        model_id=model_id,
    )


def to_api_tool_message(message: ToolMessage) -> ChatCompletionToolMessageParam:
    return ChatCompletionToolMessageParam(
        role="tool", content=message.content, tool_call_id=message.tool_call_id
    )
