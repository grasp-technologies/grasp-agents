import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any, Generic, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel
from typing_extensions import TypedDict

from .memory import MessageHistory
from .typing.completion import Completion, CompletionChunk
from .typing.converters import Converters
from .typing.message import AssistantMessage, Conversation
from .typing.tool import BaseTool, ToolChoice

logger = logging.getLogger(__name__)


class LLMSettings(TypedDict):
    pass


SettingsT = TypeVar("SettingsT", bound=LLMSettings, covariant=True)  # noqa: PLC0105
ConvertT = TypeVar("ConvertT", bound=Converters, covariant=True)  # noqa: PLC0105


class LLM(ABC, Generic[SettingsT, ConvertT]):
    @abstractmethod
    def __init__(
        self,
        converters: ConvertT,
        model_name: str | None = None,
        model_id: str | None = None,
        llm_settings: SettingsT | None = None,
        tools: list[BaseTool[BaseModel, Any, Any]] | None = None,
        response_format: type | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._converters = converters
        self._model_id = model_id or str(uuid4())[:8]
        self._model_name = model_name
        self._tools = {t.name: t for t in tools} if tools else None
        self._response_format = response_format
        self._llm_settings: SettingsT = llm_settings or cast("SettingsT", {})

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def llm_settings(self) -> SettingsT:
        return self._llm_settings

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, Any]] | None:
        return self._tools

    @property
    def response_format(self) -> type | None:
        return self._response_format

    @tools.setter
    def tools(self, tools: list[BaseTool[BaseModel, Any, Any]] | None) -> None:
        self._tools = {t.name: t for t in tools} if tools else None

    @response_format.setter
    def response_format(self, response_format: type | None) -> None:
        self._response_format = response_format

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_id={self.model_id}; "
            f"model_name={self._model_name})"
        )

    @abstractmethod
    async def generate_completion(
        self,
        conversation: Conversation,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> Completion:
        pass

    @abstractmethod
    async def generate_completion_stream(
        self,
        conversation: Conversation,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        pass

    @abstractmethod
    async def generate_message_batch(
        self,
        message_history: MessageHistory,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> Sequence[AssistantMessage]:
        pass
