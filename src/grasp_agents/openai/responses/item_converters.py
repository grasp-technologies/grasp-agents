"""Convert OpenAI Responses API output items → internal item types."""

from __future__ import annotations

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_output_item import ResponseOutputItem

from ...typing.items import (
    FunctionToolCallItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
)


def from_openai_output_item(item: ResponseOutputItem) -> OutputItem | None:
    if isinstance(item, ResponseOutputMessage):
        return OutputMessageItem.model_validate(item)
    if isinstance(item, ResponseFunctionToolCall):
        return FunctionToolCallItem.model_validate(item)
    if isinstance(item, ResponseReasoningItem):
        return ReasoningItem.model_validate(item)
    return None
