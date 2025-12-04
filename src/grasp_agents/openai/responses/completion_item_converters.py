from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
    ResponseTextDoneEvent,
)

from ...typing.completion_item import CompletionItem
from ...typing.message import AssistantMessage, Role, ToolCall


def from_api_completion_item(
    api_chunk: ResponseOutputItemDoneEvent, name: str | None = None
) -> CompletionItem:
    item = api_chunk.item
    if isinstance(item, ResponseReasoningItem):
        summaries = [reasoning.text for reasoning in item.summary]
        return CompletionItem(
            id=item.id,
            item=AssistantMessage(
                content="",
                thinking_blocks=[
                    {"type": "thinking", "thinking": item} for item in summaries
                ],
                reasoning_content=" ".join(summaries),
            ),
            role=Role.ASSISTANT,
        )
    if isinstance(item, ResponseFunctionToolCall):
        tool_call = ToolCall(
            tool_arguments=item.arguments, tool_name=item.name, id=item.call_id
        )
        return CompletionItem(
            id=item.id, item=tool_call, role=Role.ASSISTANT, name=name
        )
    if isinstance(item, ResponseTextDoneEvent):
        return CompletionItem(
            id=item.id,
            item=AssistantMessage(content=item.text),
            role=Role.ASSISTANT,
        )
    raise TypeError(
        f"Cannot convert this type of event {type(api_chunk)} to the completion item"
    )
