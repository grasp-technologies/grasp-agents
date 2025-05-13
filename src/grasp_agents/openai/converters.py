from collections.abc import AsyncIterator, Iterable
from typing import Any

from pydantic import BaseModel

from ..typing.completion import Completion, CompletionChunk
from ..typing.content import Content
from ..typing.converters import Converters
from ..typing.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from ..typing.tool import BaseTool, ToolChoice
from . import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionAsyncStream,  # type: ignore[import]
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionMessage,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUsage,
    ChatCompletionUserMessageParam,
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
    ) -> ChatCompletionSystemMessageParam:
        return to_api_system_message(system_message, **kwargs)

    @staticmethod
    def from_system_message(
        raw_message: ChatCompletionSystemMessageParam,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> SystemMessage:
        return from_api_system_message(raw_message, model_id=model_id, **kwargs)

    @staticmethod
    def to_user_message(
        user_message: UserMessage, **kwargs: Any
    ) -> ChatCompletionUserMessageParam:
        return to_api_user_message(user_message, **kwargs)

    @staticmethod
    def from_user_message(
        raw_message: ChatCompletionUserMessageParam,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> UserMessage:
        return from_api_user_message(raw_message, model_id=model_id, **kwargs)

    @staticmethod
    def to_assistant_message(
        assistant_message: AssistantMessage, **kwargs: Any
    ) -> ChatCompletionAssistantMessageParam:
        return to_api_assistant_message(assistant_message, **kwargs)

    @staticmethod
    def from_assistant_message(
        raw_message: ChatCompletionMessage,
        raw_usage: ChatCompletionUsage,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        return from_api_assistant_message(
            raw_message, raw_usage, model_id=model_id, **kwargs
        )

    @staticmethod
    def to_tool_message(
        tool_message: ToolMessage, **kwargs: Any
    ) -> ChatCompletionToolMessageParam:
        return to_api_tool_message(tool_message, **kwargs)

    @staticmethod
    def from_tool_message(
        raw_message: ChatCompletionToolMessageParam,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> ToolMessage:
        return from_api_tool_message(raw_message, model_id=model_id, **kwargs)

    @staticmethod
    def to_tool(
        tool: BaseTool[BaseModel, Any, Any], **kwargs: Any
    ) -> ChatCompletionToolParam:
        return to_api_tool(tool, **kwargs)

    @staticmethod
    def to_tool_choice(
        tool_choice: ToolChoice, **kwargs: Any
    ) -> ChatCompletionToolChoiceOptionParam:
        return to_api_tool_choice(tool_choice, **kwargs)

    @staticmethod
    def to_content(
        content: Content, **kwargs: Any
    ) -> Iterable[ChatCompletionContentPartParam]:
        return to_api_content(content, **kwargs)

    @staticmethod
    def from_content(
        raw_content: str | Iterable[ChatCompletionContentPartParam],
        **kwargs: Any,
    ) -> Content:
        return from_api_content(raw_content, **kwargs)

    @staticmethod
    def to_completion(completion: Completion, **kwargs: Any) -> ChatCompletion:
        return to_api_completion(completion, **kwargs)

    @staticmethod
    def from_completion(
        raw_completion: ChatCompletion,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> Completion:
        return from_api_completion(raw_completion, model_id=model_id, **kwargs)

    @staticmethod
    def to_completion_chunk(
        chunk: CompletionChunk, **kwargs: Any
    ) -> ChatCompletionChunk:
        raise NotImplementedError

    @staticmethod
    def from_completion_chunk(
        raw_chunk: ChatCompletionChunk,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> CompletionChunk:
        return from_api_completion_chunk(raw_chunk, model_id=model_id, **kwargs)

    @staticmethod
    def from_completion_chunk_iterator(  # type: ignore[override]
        raw_chunk_iterator: ChatCompletionAsyncStream[ChatCompletionChunk],
        model_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        return from_api_completion_chunk_iterator(
            raw_chunk_iterator, model_id=model_id, **kwargs
        )
