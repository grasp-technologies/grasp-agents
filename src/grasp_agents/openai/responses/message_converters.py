from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_input_message_item import (
    ResponseInputMessageItem as OpenAIResponseInputMessage,
)
from openai.types.responses.response_output_message import (
    ResponseOutputMessage as OpenAIResponseOutputMessage,
)

from grasp_agents.typing.content import Content, ContentPartText
from grasp_agents.typing.message import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

from .content_converters import from_api_content, to_api_content


def from_api_user_message(
    api_message: OpenAIResponseInputMessage, name: str | None = None
) -> UserMessage:
    content = from_api_content(api_message.content)
    return UserMessage(content=content, name=name)


def to_api_user_message(message: UserMessage) -> EasyInputMessageParam:
    api_content = (
        to_api_content(message.content)
        if isinstance(message.content, Content)
        else message.content
    )
    return EasyInputMessageParam(role="user", content=api_content, type="message")


def from_api_assistant_message(
    api_message: OpenAIResponseOutputMessage, name: str | None = None
) -> AssistantMessage:
    items = api_message.content

    text_parts: list[str] = []
    refusal: str | None = None

    for item in items or []:
        if item.type == "output_text":
            text_parts.append(item.text)
        elif item.type == "refusal":
            refusal = item.refusal

    content_str = "".join(text_parts) if text_parts else None

    return AssistantMessage(content=content_str, refusal=refusal, name=name)


def to_api_assistant_message(
    message: AssistantMessage,
) -> EasyInputMessageParam:
    if isinstance(message.content, str):
        api_content = message.content
    else:
        api_content = "<empty>" if message.content is None else str(message.content)
    return EasyInputMessageParam(role="assistant", content=api_content, type="message")


def from_api_system_message(
    api_message: OpenAIResponseInputMessage, name: str | None = None
) -> SystemMessage:
    content_obj = from_api_content(api_message.content)
    content_str = "".join(
        part.data for part in content_obj.parts if isinstance(part, ContentPartText)
    )
    return SystemMessage(content=content_str, name=name)


def to_api_system_message(message: SystemMessage) -> EasyInputMessageParam:
    return EasyInputMessageParam(role="system", content=message.content, type="message")


def from_api_tool_message(
    api_message: FunctionCallOutput, name: str | None = None
) -> ToolMessage:
    return ToolMessage(
        content=api_message["output"],  # type: ignore
        tool_call_id=api_message["call_id"],
        name=name,
    )


def to_api_tool_message(message: ToolMessage) -> FunctionCallOutput:
    return FunctionCallOutput(
        type="function_call_output",
        output=message.content,
        call_id=message.tool_call_id,
    )
