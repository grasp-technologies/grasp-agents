from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from ..openai.content_converters import from_api_content, to_api_content
from ..openai.message_converters import (
    from_api_system_message,
    from_api_tool_message,
    from_api_user_message,
    to_api_system_message,
    to_api_tool_message,
    to_api_user_message,
)
from ..openai.tool_converters import to_api_tool, to_api_tool_choice
from ..typing.completion import Completion, Usage
from ..typing.completion_chunk import CompletionChunk
from ..typing.content import Content
from ..typing.converters import Converters
from ..typing.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from ..typing.tool import BaseTool, ToolChoice
from . import (
    LiteLLMCompletion,
    LiteLLMCompletionChunk,
    LiteLLMCompletionMessage,
    LiteLLMUsage,
    OpenAIContentPartParam,
    OpenAISystemMessageParam,
    OpenAIToolChoiceOptionParam,
    OpenAIToolMessageParam,
    OpenAIToolParam,
    OpenAIUserMessageParam,
)
from .completion_chunk_converters import from_api_completion_chunk
from .completion_converters import (
    from_api_completion,
    from_api_completion_usage,
    to_api_completion,
)
from .message_converters import from_api_assistant_message, to_api_assistant_message


class LiteLLMConverters(Converters):
    @staticmethod
    def to_completion(completion: Completion, **kwargs: Any) -> LiteLLMCompletion:
        return to_api_completion(completion, **kwargs)

    @staticmethod
    def from_completion(
        raw_completion: LiteLLMCompletion, name: str | None = None, **kwargs: Any
    ) -> Completion:
        return from_api_completion(raw_completion, name=name, **kwargs)

    @staticmethod
    def to_completion_chunk(
        chunk: CompletionChunk, **kwargs: Any
    ) -> LiteLLMCompletionChunk:
        raise NotImplementedError

    @staticmethod
    def from_completion_chunk(
        raw_chunk: LiteLLMCompletionChunk, name: str | None = None, **kwargs: Any
    ) -> CompletionChunk:
        return from_api_completion_chunk(raw_chunk, name=name, **kwargs)

    @staticmethod
    def from_assistant_message(
        raw_message: LiteLLMCompletionMessage, name: str | None = None, **kwargs: Any
    ) -> AssistantMessage:
        return from_api_assistant_message(raw_message, name=name, **kwargs)

    @staticmethod
    def to_assistant_message(
        assistant_message: AssistantMessage, **kwargs: Any
    ) -> LiteLLMCompletionMessage:
        return to_api_assistant_message(assistant_message, **kwargs)

    # The remaining converters are the same as OpenAIConverters

    @staticmethod
    def to_system_message(
        system_message: SystemMessage, **kwargs: Any
    ) -> OpenAISystemMessageParam:
        return to_api_system_message(system_message, **kwargs)

    @staticmethod
    def from_system_message(
        raw_message: OpenAISystemMessageParam, name: str | None = None, **kwargs: Any
    ) -> SystemMessage:
        return from_api_system_message(raw_message, name=name, **kwargs)

    @staticmethod
    def to_user_message(
        user_message: UserMessage, **kwargs: Any
    ) -> OpenAIUserMessageParam:
        return to_api_user_message(user_message, **kwargs)

    @staticmethod
    def from_user_message(
        raw_message: OpenAIUserMessageParam, name: str | None = None, **kwargs: Any
    ) -> UserMessage:
        return from_api_user_message(raw_message, name=name, **kwargs)

    @staticmethod
    def from_completion_usage(raw_usage: LiteLLMUsage, **kwargs: Any) -> Usage:
        return from_api_completion_usage(raw_usage, **kwargs)

    @staticmethod
    def to_tool_message(
        tool_message: ToolMessage, **kwargs: Any
    ) -> OpenAIToolMessageParam:
        return to_api_tool_message(tool_message, **kwargs)

    @staticmethod
    def from_tool_message(
        raw_message: OpenAIToolMessageParam, name: str | None = None, **kwargs: Any
    ) -> ToolMessage:
        return from_api_tool_message(raw_message, name=name, **kwargs)

    @staticmethod
    def to_tool(tool: BaseTool[BaseModel, Any, Any], **kwargs: Any) -> OpenAIToolParam:
        return to_api_tool(tool, **kwargs)

    @staticmethod
    def to_tool_choice(
        tool_choice: ToolChoice, **kwargs: Any
    ) -> OpenAIToolChoiceOptionParam:
        return to_api_tool_choice(tool_choice, **kwargs)

    @staticmethod
    def to_content(content: Content, **kwargs: Any) -> Iterable[OpenAIContentPartParam]:
        return to_api_content(content, **kwargs)

    @staticmethod
    def from_content(
        raw_content: str | Iterable[OpenAIContentPartParam], **kwargs: Any
    ) -> Content:
        return from_api_content(raw_content, **kwargs)
