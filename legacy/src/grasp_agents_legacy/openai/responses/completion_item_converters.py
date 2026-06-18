from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
)

from ...typing.completion_item import CompletionItem, Reasoning, TextOutput
from ...typing.message import ToolCall
from .message_converters import from_api_responses_output_message


def from_api_completion_item(
    api_chunk: ResponseOutputItemDoneEvent, name: str | None = None
) -> CompletionItem:
    item = api_chunk.item
    if isinstance(item, ResponseReasoningItem):
        summaries = [reasoning.text for reasoning in item.summary]
        encrypted_content = item.encrypted_content
        return Reasoning(summaries=summaries, encrypted_content=encrypted_content)
    if isinstance(item, ResponseFunctionToolCall):
        tool_call = ToolCall(
            tool_arguments=item.arguments, tool_name=item.name, id=item.call_id
        )
        return tool_call
    if isinstance(item, ResponseOutputMessage):
        content_item, annotations, refusal = from_api_responses_output_message(item)
        return TextOutput(
            content=" ".join(content_item), annotations=annotations, refusal=refusal
        )
    raise TypeError(
        f"Cannot convert event of type '{type(api_chunk).__name__}' to CompletionItem"
    )
