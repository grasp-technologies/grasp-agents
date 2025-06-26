import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from typing import Any, Generic, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypedDict

from grasp_agents.utils import validate_obj_from_json_or_py_string

from .errors import ToolValidationError
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
        response_format: Any | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._converters = converters
        self._model_id = model_id or str(uuid4())[:8]
        self._model_name = model_name
        self._tools = {t.name: t for t in tools} if tools else None
        self._llm_settings: SettingsT_co = llm_settings or cast("SettingsT_co", {})

        self._response_format = response_format
        self._response_format_adapter: (
            TypeAdapter[Any] | Mapping[str, TypeAdapter[Any]]
        ) = self._get_response_format_adapter(response_format=response_format)

    @staticmethod
    def _get_response_format_adapter(
        response_format: Any | Mapping[str, Any] | None = None,
    ) -> TypeAdapter[Any] | Mapping[str, TypeAdapter[Any]]:
        if response_format is None:
            return TypeAdapter(Any)
        if isinstance(response_format, Mapping):
            return {k: TypeAdapter(v) for k, v in response_format.items()}  # type: ignore[return-value]
        return TypeAdapter(response_format)

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
    def response_format(self) -> Any | Mapping[str, Any] | None:
        return self._response_format

    @response_format.setter
    def response_format(self, response_format: Any | Mapping[str, Any] | None) -> None:
        self._response_format = response_format
        self._response_format_adapter = self._get_response_format_adapter(
            response_format
        )

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, Any]] | None:
        return self._tools

    @tools.setter
    def tools(self, tools: list[BaseTool[BaseModel, Any, Any]] | None) -> None:
        self._tools = {t.name: t for t in tools} if tools else None

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
                        raise ToolValidationError(
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
