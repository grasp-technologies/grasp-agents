from typing import TypeAlias

from ..typing.content import Content
from ..typing.message import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from ..typing.tool import ToolCall
from . import (
    OpenAIAssistantMessageParam,
    OpenAIDeveloperMessageParam,
    OpenAIFunctionMessageParam,
    OpenAIMessage,
    OpenAISystemMessageParam,
    OpenAIToolCallFunction,
    OpenAIToolCallParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)
from .content_converters import from_api_content, to_api_content

OpenAIMessageType: TypeAlias = (
    OpenAIAssistantMessageParam
    | OpenAIToolMessageParam
    | OpenAIUserMessageParam
    | OpenAIDeveloperMessageParam
    | OpenAISystemMessageParam
    | OpenAIFunctionMessageParam
)


def from_api_user_message(
    api_message: OpenAIUserMessageParam, name: str | None = None
) -> UserMessage:
    content = from_api_content(api_message["content"])
    name = api_message.get("name")

    return UserMessage(content=content, name=name)


def to_api_user_message(message: UserMessage) -> OpenAIUserMessageParam:
    api_content = (
        to_api_content(message.content)
        if isinstance(message.content, Content)
        else message.content
    )
    api_name = message.name
    api_message = OpenAIUserMessageParam(role="user", content=api_content)
    if api_name is not None:
        api_message["name"] = api_name

    return api_message


def from_api_assistant_message(
    api_message: OpenAIMessage, name: str | None = None
) -> AssistantMessage:
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
        tool_calls=tool_calls,
        refusal=api_message.refusal,
        name=name,
    )


def to_api_assistant_message(
    message: AssistantMessage,
) -> OpenAIAssistantMessageParam:
    api_tool_calls = None
    if message.tool_calls is not None:
        api_tool_calls = [
            OpenAIToolCallParam(
                type="function",
                id=tool_call.id,
                function=OpenAIToolCallFunction(
                    name=tool_call.tool_name,
                    arguments=tool_call.tool_arguments,
                ),
            )
            for tool_call in message.tool_calls
        ]

    api_message = OpenAIAssistantMessageParam(role="assistant", content=message.content)

    if message.name is not None:
        api_message["name"] = message.name
    if api_tool_calls is not None:
        api_message["tool_calls"] = api_tool_calls or []
    if message.refusal is not None:
        api_message["refusal"] = message.refusal

    # TODO: hack
    if message.content is None:
        # Some API providers return None in the generated content without errors,
        # even though None in the input content is not accepted.
        api_message["content"] = "<empty>"

    return api_message


def from_api_system_message(
    api_message: OpenAISystemMessageParam, name: str | None = None
) -> SystemMessage:
    return SystemMessage(content=api_message["content"], name=name)  # type: ignore


def to_api_system_message(message: SystemMessage) -> OpenAISystemMessageParam:
    api_message = OpenAISystemMessageParam(role="system", content=message.content)
    if message.name is not None:
        api_message["name"] = message.name

    return api_message


def from_api_tool_message(
    api_message: OpenAIToolMessageParam, name: str | None = None
) -> ToolMessage:
    return ToolMessage(
        content=api_message["content"],  # type: ignore
        tool_call_id=api_message["tool_call_id"],
        name=name,
    )


def to_api_tool_message(message: ToolMessage) -> OpenAIToolMessageParam:
    return OpenAIToolMessageParam(
        role="tool", content=message.content, tool_call_id=message.tool_call_id
    )
