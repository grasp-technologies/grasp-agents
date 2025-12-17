from litellm.types.llms.openai import (
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
)
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
)
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_input_message_item import (
    ResponseInputMessageItem as OpenAIResponseInputMessage,
)
from openai.types.responses.response_reasoning_item_param import Summary

from grasp_agents.typing.content import Content, ContentPartText
from grasp_agents.typing.message import (
    AssistantMessage,
    RedactedThinkingBlock,
    SystemMessage,
    ThinkingBlock,
    ToolMessage,
    UserMessage,
)
from grasp_agents.typing.tool import ToolCall

from .content_converters import from_api_content, to_api_content


def from_api_responses_output_message(
    output: ResponseOutputMessage,
) -> tuple[list[str], list[ChatCompletionAnnotation], str | None]:
    content: list[str] = []
    refusal: str | None = None
    annotations: list[ChatCompletionAnnotation] = []
    raw_contents = output.content
    for raw_content in raw_contents:
        if isinstance(raw_content, ResponseOutputText):
            content.append(raw_content.text)
            if raw_content.annotations:
                annotations.extend(
                    [
                        ChatCompletionAnnotation(
                            type="url_citation",
                            url_citation=ChatCompletionAnnotationURLCitation(
                                **api_annotation.model_dump()
                            ),
                        )
                        for api_annotation in raw_content.annotations
                    ]
                )
        else:
            refusal = raw_content.refusal
    return content, annotations, refusal


def from_api_user_message(
    api_message: OpenAIResponseInputMessage, name: str | None = None
) -> UserMessage:
    content = from_api_content(api_message.content)
    return UserMessage(content=content, name=name)


def to_api_user_message(message: UserMessage) -> ResponseInputParam:
    input_message: ResponseInputParam = []
    api_content = (
        to_api_content(message.content)
        if isinstance(message.content, Content)
        else message.content
    )
    usr_message = EasyInputMessageParam(
        role="user", content=api_content, type="message"
    )
    input_message.append(usr_message)
    return input_message


def from_api_assistant_message(
    raw_completion: OpenAIResponse, name: str | None = None
) -> AssistantMessage:
    output_items = raw_completion.output
    content: list[str] = []
    refusal: str | None = None
    reasoning_id: str | None = None
    encrypted_content = None
    reasoning_summary: list[str] = []
    tool_calls: list[ToolCall] = []
    annotations: list[ChatCompletionAnnotation] = []
    for output in output_items:
        if isinstance(output, ResponseOutputMessage):
            content_output, annotation_output, refusal = (
                from_api_responses_output_message(output)
            )
            content.extend(content_output)
            annotations.extend(annotation_output)
        if isinstance(output, ResponseReasoningItem):
            raw_summaries = output.summary
            encrypted_content = output.encrypted_content
            reasoning_id = output.id
            for raw_summary in raw_summaries:
                reasoning_summary.append(raw_summary.text)
        if isinstance(output, ResponseFunctionToolCall):
            tool = ToolCall(
                id=output.call_id,
                tool_arguments=output.arguments,
                tool_name=output.name,
            )
            tool_calls.append(tool)
    thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock] = [
        ThinkingBlock(type="thinking", thinking=item) for item in reasoning_summary
    ]
    if encrypted_content:
        thinking_blocks.append(
            RedactedThinkingBlock(type="redacted_thinking", data=encrypted_content)
        )
    return AssistantMessage(
        content=" ".join(content),
        annotations=annotations,
        reasoning_content=" ".join(reasoning_summary),
        tool_calls=tool_calls,
        refusal=refusal,
        response_id=raw_completion.id,
        thinking_blocks=thinking_blocks,
        reasoning_id=reasoning_id,
    )


def to_api_assistant_message(
    message: AssistantMessage,
) -> ResponseInputParam:
    input_message: ResponseInputParam = []
    if message.content:
        text_message = EasyInputMessageParam(
            role="assistant", content=message.content, type="message"
        )
        input_message.append(text_message)
    if message.tool_calls:
        tool_calls = [
            ResponseFunctionToolCallParam(
                arguments=item.tool_arguments,
                call_id=item.id,
                name=item.tool_name,
                type="function_call",
            )
            for item in message.tool_calls
        ]
        input_message.extend(tool_calls)
    if message.reasoning_id:
        reasoning = ResponseReasoningItemParam(
            id=message.reasoning_id,
            summary=[
                Summary(text=text, type="summary_text")
                for text in message.thinking_summaries
            ],
            type="reasoning",
            encrypted_content=message.encrypted_content,
        )
        input_message.append(reasoning)
    return input_message


def from_api_system_message(
    api_message: OpenAIResponseInputMessage, name: str | None = None
) -> SystemMessage:
    content_obj = from_api_content(api_message.content)
    content_str = "".join(
        part.data for part in content_obj.parts if isinstance(part, ContentPartText)
    )
    return SystemMessage(content=content_str, name=name)


def to_api_system_message(message: SystemMessage) -> ResponseInputParam:
    input_message: ResponseInputParam = []
    sys_message = EasyInputMessageParam(
        role="system", content=message.content or "", type="message"
    )
    input_message.append(sys_message)
    return input_message


def from_api_tool_message(
    api_message: FunctionCallOutput, name: str | None = None
) -> ToolMessage:
    return ToolMessage(
        content=api_message["output"],  # type: ignore
        tool_call_id=api_message["call_id"],
        name=name,
    )


# this is conversion from call output result
def to_api_tool_message(message: ToolMessage) -> ResponseInputParam:
    input_message: ResponseInputParam = []
    input_message.append(
        FunctionCallOutput(
            type="function_call_output",
            output=message.content,
            call_id=message.tool_call_id,
        )
    )
    return input_message
