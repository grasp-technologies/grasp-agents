import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Generic, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypedDict

from grasp_agents.utils import validate_obj_from_json_or_py_string

from .message_history import MessageHistory
from .typing.completion import Completion
from .typing.converters import Converters
from .typing.events import CompletionChunkEvent, CompletionEvent
from .typing.message import Messages
from .typing.tool import BaseTool, ToolChoice

logger = logging.getLogger(__name__)


class LLMSettings(TypedDict, total=False):
    max_completion_tokens: int | None
    temperature: float | None
    top_p: float | None
    seed: int | None


SettingsT_co = TypeVar("SettingsT_co", bound=LLMSettings, covariant=True)
ConvertT_co = TypeVar("ConvertT_co", bound=Converters, covariant=True)


class LLM(ABC, Generic[SettingsT_co, ConvertT_co]):
    @abstractmethod
    def __init__(
        self,
        converters: ConvertT_co,
        model_name: str | None = None,
        model_id: str | None = None,
        llm_settings: SettingsT_co | None = None,
        tools: list[BaseTool[BaseModel, Any, Any]] | None = None,
        response_format: type | Mapping[str, type] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._converters = converters
        self._model_id = model_id or str(uuid4())[:8]
        self._model_name = model_name
        self._tools = {t.name: t for t in tools} if tools else None
        self._llm_settings: SettingsT_co = llm_settings or cast("SettingsT_co", {})

        self._response_format = response_format
        self._response_format_adapter: TypeAdapter[Any] | Mapping[str, TypeAdapter[Any]]
        if isinstance(response_format, type):
            self._response_format_adapter = TypeAdapter(response_format)
        elif isinstance(response_format, Mapping):
            self._response_format_adapter = {
                k: TypeAdapter(v) for k, v in response_format.items()
            }
        else:
            self._response_format_adapter = TypeAdapter(Any)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def llm_settings(self) -> SettingsT_co:
        return self._llm_settings

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, Any]] | None:
        return self._tools

    @property
    def response_format(self) -> type | Mapping[str, type] | None:
        return self._response_format

    @tools.setter
    def tools(self, tools: list[BaseTool[BaseModel, Any, Any]] | None) -> None:
        self._tools = {t.name: t for t in tools} if tools else None

    @response_format.setter
    def response_format(
        self, response_format: type | Mapping[str, type] | None
    ) -> None:
        self._response_format = response_format

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_id={self.model_id}; "
            f"model_name={self._model_name})"
        )

    def _validate_completion(self, completion: Completion) -> None:
        for message in completion.messages:
            if not message.tool_calls:
                validate_obj_from_json_or_py_string(
                    message.content or "",
                    adapter=self._response_format_adapter,
                    from_substring=True,
                )

    def _validate_tool_calls(self, completion: Completion) -> None:
        for message in completion.messages:
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.tool_name
                    tool_arguments = tool_call.tool_arguments

                    available_tool_names = list(self.tools) if self.tools else []
                    if tool_name not in available_tool_names or not self.tools:
                        raise ValueError(
                            f"Tool '{tool_name}' is not available in the LLM tools "
                            f"(available: {available_tool_names}"
                        )
                    tool = self.tools[tool_name]
                    validate_obj_from_json_or_py_string(
                        tool_arguments, adapter=TypeAdapter(tool.in_type)
                    )

    @abstractmethod
    async def generate_completion(
        self,
        conversation: Messages,
        *,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
    ) -> Completion:
        pass

    @abstractmethod
    async def generate_completion_stream(
        self,
        conversation: Messages,
        *,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
    ) -> AsyncIterator[CompletionChunkEvent | CompletionEvent]:
        pass

    @abstractmethod
    async def generate_completion_batch(
        self, message_history: MessageHistory, *, tool_choice: ToolChoice | None = None
    ) -> Sequence[Completion]:
        pass
