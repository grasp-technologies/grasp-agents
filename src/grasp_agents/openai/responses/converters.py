from __future__ import annotations

from typing import Any

from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam as OpenAIChatToolParam,
)
from openai.types.responses import (
    ParsedResponse,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseInputItemParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses.response_create_params import (
    ToolChoice as OpenAIResponseToolChoice,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_input_message_item import (
    ResponseInputMessageItem as OpenAIResponseInputMessage,
)
from openai.types.responses.response_output_message import (
    ResponseOutputMessage as OpenAIResponseOutputMessage,
)
from openai.types.responses.tool_param import ToolParam as OpenAIResponseToolParam
from pydantic import BaseModel

from grasp_agents.typing.completion import Completion, Usage
from grasp_agents.typing.completion_chunk import CompletionChunk
from grasp_agents.typing.completion_item import CompletionItem
from grasp_agents.typing.content import Content
from grasp_agents.typing.converters import Converters
from grasp_agents.typing.message import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from grasp_agents.typing.tool import BaseTool, ToolChoice

from .chunk_converters import (
    from_api_completion_chunk,
    to_completion_chunk,
    ResponseApiChunk,
)
from .completion_converters import (
    completion_from_response,
    from_response_usage,
    to_api_completion,
)
from .completion_item_converters import from_api_completion_item
from .content_converters import from_api_content, to_api_content
from .message_converters import (
    from_api_assistant_message,
    from_api_system_message,
    from_api_tool_message,
    from_api_user_message,
    to_api_assistant_message,
    to_api_system_message,
    to_api_tool_message,
    to_api_user_message,
)
from .tool_converters import to_api_tool, to_api_tool_choice


class OpenAIResponsesConverters(Converters):
    @staticmethod
    def to_system_message(
        system_message: SystemMessage, **kwargs: Any
    ) -> ResponseInputItemParam:
        return to_api_system_message(system_message, **kwargs)

    @staticmethod
    def to_user_message(
        user_message: UserMessage, **kwargs: Any
    ) -> ResponseInputItemParam:
        return to_api_user_message(user_message, **kwargs)

    @staticmethod
    def to_assistant_message(
        assistant_message: AssistantMessage, **kwargs: Any
    ) -> ResponseInputItemParam:
        return to_api_assistant_message(assistant_message, **kwargs)

    @staticmethod
    def from_system_message(
        raw_message: OpenAIResponseInputMessage,
        **kwargs: Any,
    ) -> SystemMessage:
        return from_api_system_message(raw_message, **kwargs)

    @staticmethod
    def from_user_message(
        raw_message: OpenAIResponseInputMessage,
        **kwargs: Any,
    ) -> UserMessage:
        return from_api_user_message(raw_message, **kwargs)

    @staticmethod
    def from_assistant_message(
        raw_message: OpenAIResponseOutputMessage,
        **kwargs: Any,
    ) -> AssistantMessage:
        return from_api_assistant_message(raw_message, **kwargs)

    @staticmethod
    def to_tool(
        tool: BaseTool[BaseModel, Any, Any], strict: bool | None = None, **kwargs: Any
    ) -> OpenAIResponseToolParam | OpenAIChatToolParam:
        return to_api_tool(tool=tool, strict=strict, **kwargs)

    @staticmethod
    def to_tool_choice(
        tool_choice: ToolChoice, **kwargs: Any
    ) -> OpenAIResponseToolChoice:
        return to_api_tool_choice(tool_choice=tool_choice, **kwargs)

    @staticmethod
    def to_tool_message(tool_message: ToolMessage, **kwargs: Any) -> FunctionCallOutput:
        return to_api_tool_message(tool_message, **kwargs)

    @staticmethod
    def from_tool_message(
        raw_message: FunctionCallOutput, **kwargs: Any
    ) -> ToolMessage:
        return from_api_tool_message(raw_message, **kwargs)

    @staticmethod
    def to_content(content: Content, **kwargs: Any) -> Any:
        return to_api_content(content, **kwargs)

    @staticmethod
    def from_content(raw_content: Any, **kwargs: Any) -> Content:
        return from_api_content(raw_content, **kwargs)

    @staticmethod
    def to_completion(completion: Completion, **kwargs: Any) -> Any:
        return to_api_completion(completion, **kwargs)

    @staticmethod
    def from_completion(
        raw_completion: ParsedResponse[Any] | OpenAIResponse,
        name: str | None = None,
        **kwargs: Any,
    ) -> Completion:
        return completion_from_response(raw_completion, name=name, **kwargs)

    @staticmethod
    def from_completion_usage(raw_usage: ResponseUsage, **kwargs: Any) -> Usage:
        return from_response_usage(raw_usage, **kwargs)

    @staticmethod
    def to_completion_chunk(chunk: CompletionChunk, **kwargs: Any) -> Any:
        return to_completion_chunk(chunk, **kwargs)

    @staticmethod
    def from_completion_chunk(
        raw_chunk: ResponseApiChunk, name: str | None = None, **kwargs: Any
    ) -> CompletionChunk:
        return from_api_completion_chunk(raw_chunk, name=name, **kwargs)

    @staticmethod
    def from_api_item(
        raw_event: ResponseOutputItemDoneEvent, name: str | None = None, **kwargs: Any
    ) -> CompletionItem:
        return from_api_completion_item(raw_event, name=name, **kwargs)
