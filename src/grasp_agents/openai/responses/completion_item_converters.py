from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseStreamEvent,
)

from ...typing.completion_item import CompletionItem
from ...typing.message import AssistantMessage, Role, ToolCall


def from_api_completion_item(
    event: ResponseStreamEvent, name: str | None = None
) -> CompletionItem:
    if isinstance(event, ResponseReasoningSummaryTextDoneEvent):
        return CompletionItem(
            item=AssistantMessage(content=event.text), role=Role.ASSISTANT
        )

    if isinstance(event, ResponseOutputItemDoneEvent) and isinstance(
        event.item, ResponseFunctionToolCall
    ):
        item = event.item
        tool_call = ToolCall(
            tool_arguments=item.arguments, tool_name=item.name, id=item.call_id
        )
        return CompletionItem(item=tool_call, role=Role.ASSISTANT, name=name)
    raise TypeError(
        f"Cannot convert this type of event {type(event)} to the completion item"
    )


def is_supported_stream_event(event: ResponseStreamEvent) -> bool:
    return isinstance(
        event,
        (
            ResponseReasoningSummaryTextDoneEvent,
            ResponseOutputItemDoneEvent,
        ),
    )
