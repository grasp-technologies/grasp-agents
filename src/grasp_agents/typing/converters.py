from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .completion import Completion, Usage
from .completion_chunk import CompletionChunk
from .content import Content
from .message import AssistantMessage, Message, SystemMessage, ToolMessage, UserMessage
from .tool import BaseTool, ToolChoice


class Converters(ABC):
    @staticmethod
    @abstractmethod
    def to_system_message(system_message: SystemMessage, **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def from_system_message(raw_message: Any, **kwargs: Any) -> SystemMessage:
        pass

    @staticmethod
    @abstractmethod
    def to_user_message(user_message: UserMessage, **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def from_user_message(raw_message: Any, **kwargs: Any) -> UserMessage:
        pass

    @staticmethod
    @abstractmethod
    def to_assistant_message(assistant_message: AssistantMessage, **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def from_completion_usage(raw_usage: Any, **kwargs: Any) -> Usage:
        pass

    @staticmethod
    @abstractmethod
    def from_assistant_message(raw_message: Any, **kwargs: Any) -> AssistantMessage:
        pass

    @staticmethod
    @abstractmethod
    def to_tool_message(tool_message: ToolMessage, **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def from_tool_message(raw_message: Any, **kwargs: Any) -> ToolMessage:
        pass

    @classmethod
    def to_message(cls, message: Message, **kwargs: Any) -> Any:
        if isinstance(message, UserMessage):
            return cls.to_user_message(message, **kwargs)
        if isinstance(message, AssistantMessage):
            return cls.to_assistant_message(message, **kwargs)
        if isinstance(message, ToolMessage):
            return cls.to_tool_message(message, **kwargs)

        return cls.to_system_message(message, **kwargs)

    @staticmethod
    @abstractmethod
    def to_tool(tool: BaseTool[BaseModel, Any, Any], **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def to_tool_choice(tool_choice: ToolChoice, **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def to_content(content: Content, **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def from_content(raw_content: Any, **kwargs: Any) -> Content:
        pass

    @staticmethod
    @abstractmethod
    def to_completion(completion: Completion, **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def from_completion(raw_completion: Any, **kwargs: Any) -> Completion:
        pass

    @staticmethod
    @abstractmethod
    def to_completion_chunk(chunk: CompletionChunk, **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def from_completion_chunk(raw_chunk: Any, **kwargs: Any) -> CompletionChunk:
        pass
