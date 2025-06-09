from collections.abc import AsyncIterator, Iterable
from typing import Any

from pydantic import BaseModel

from ..typing.completion import Completion, CompletionChunk
from ..typing.content import Content
from ..typing.converters import Converters
from ..typing.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from ..typing.tool import BaseTool, ToolChoice
from . import (
    OpenAIAssistantMessageParam,
    OpenAIAsyncStream,  # type: ignore[import]
    OpenAICompletion,
    OpenAICompletionChunk,
    OpenAICompletionUsage,
    OpenAIContentPartParam,
    OpenAIMessage,
    OpenAISystemMessageParam,
    OpenAIToolChoiceOptionParam,
    OpenAIToolMessageParam,
    OpenAIToolParam,
    OpenAIUserMessageParam,
)
from .completion_converters import (
    from_api_completion,
    from_api_completion_chunk,
    from_api_completion_chunk_iterator,
    to_api_completion,
)
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


class OpenAIConverters(Converters):
    @staticmethod
    def to_system_message(
        system_message: SystemMessage, **kwargs: Any
    ) -> OpenAISystemMessageParam:
        return to_api_system_message(system_message, **kwargs)

    @staticmethod
    def from_system_message(
        raw_message: OpenAISystemMessageParam,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> SystemMessage:
        return from_api_system_message(raw_message, model_id=model_id, **kwargs)

    @staticmethod
    def to_user_message(
        user_message: UserMessage, **kwargs: Any
    ) -> OpenAIUserMessageParam:
        return to_api_user_message(user_message, **kwargs)

    @staticmethod
    def from_user_message(
        raw_message: OpenAIUserMessageParam,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> UserMessage:
        return from_api_user_message(raw_message, model_id=model_id, **kwargs)

    @staticmethod
    def to_assistant_message(
        assistant_message: AssistantMessage, **kwargs: Any
    ) -> OpenAIAssistantMessageParam:
        return to_api_assistant_message(assistant_message, **kwargs)

    @staticmethod
    def from_assistant_message(
        raw_message: OpenAIMessage,
        raw_usage: OpenAICompletionUsage,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        return from_api_assistant_message(
            raw_message, raw_usage, model_id=model_id, **kwargs
        )

    @staticmethod
    def to_tool_message(
        tool_message: ToolMessage, **kwargs: Any
    ) -> OpenAIToolMessageParam:
        return to_api_tool_message(tool_message, **kwargs)

    @staticmethod
    def from_tool_message(
        raw_message: OpenAIToolMessageParam,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> ToolMessage:
        return from_api_tool_message(raw_message, model_id=model_id, **kwargs)

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
        raw_content: str | Iterable[OpenAIContentPartParam],
        **kwargs: Any,
    ) -> Content:
        return from_api_content(raw_content, **kwargs)

    @staticmethod
    def to_completion(completion: Completion, **kwargs: Any) -> OpenAICompletion:
        return to_api_completion(completion, **kwargs)

    @staticmethod
    def from_completion(
        raw_completion: OpenAICompletion,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> Completion:
        return from_api_completion(raw_completion, model_id=model_id, **kwargs)

    @staticmethod
    def to_completion_chunk(
        chunk: CompletionChunk, **kwargs: Any
    ) -> OpenAICompletionChunk:
        raise NotImplementedError

    @staticmethod
    def from_completion_chunk(
        raw_chunk: OpenAICompletionChunk,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> CompletionChunk:
        return from_api_completion_chunk(raw_chunk, model_id=model_id, **kwargs)

    @staticmethod
    def from_completion_chunk_iterator(  # type: ignore[override]
        raw_chunk_iterator: OpenAIAsyncStream[OpenAICompletionChunk],
        model_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        return from_api_completion_chunk_iterator(
            raw_chunk_iterator, model_id=model_id, **kwargs
        )
