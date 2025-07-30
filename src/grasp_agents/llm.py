import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Generic, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel
from typing_extensions import TypedDict

from grasp_agents.typing.completion_chunk import CompletionChunk
from grasp_agents.utils import (
    validate_obj_from_json_or_py_string,
    validate_tagged_objs_from_json_or_py_string,
)

from .errors import (
    JSONSchemaValidationError,
    LLMResponseValidationError,
    LLMToolCallValidationError,
)
from .typing.completion import Completion
from .typing.converters import Converters
from .typing.events import (
    AnnotationsChunkEvent,
    AnnotationsEndEvent,
    AnnotationsStartEvent,
    CompletionChunkEvent,
    CompletionEndEvent,
    CompletionEvent,
    CompletionStartEvent,
    LLMStateChangeEvent,
    LLMStreamingErrorEvent,
    # RefusalChunkEvent,
    ResponseChunkEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    ThinkingChunkEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from .typing.message import Messages
from .typing.tool import BaseTool, ToolChoice

logger = logging.getLogger(__name__)


LLMStreamGenerator = AsyncIterator[
    CompletionChunkEvent[CompletionChunk]
    | CompletionEvent
    | LLMStateChangeEvent[Any]
    | LLMStreamingErrorEvent
]


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
        tools: Sequence[BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._converters = converters
        self._model_id = model_id or str(uuid4())[:8]
        self._model_name = model_name
        self._tools = {t.name: t for t in tools} if tools else None
        self._llm_settings: SettingsT_co = llm_settings or cast("SettingsT_co", {})

        if response_schema and response_schema_by_xml_tag:
            raise ValueError(
                "Only one of response_schema and response_schema_by_xml_tag can be "
                "provided, but not both."
            )
        self._response_schema = response_schema
        self._response_schema_by_xml_tag = response_schema_by_xml_tag

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
    def response_schema(self) -> Any | None:
        return self._response_schema

    @response_schema.setter
    def response_schema(self, response_schema: Any | None) -> None:
        self._response_schema = response_schema

    @property
    def response_schema_by_xml_tag(self) -> Mapping[str, Any] | None:
        return self._response_schema_by_xml_tag

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, Any]] | None:
        return self._tools

    @tools.setter
    def tools(self, tools: Sequence[BaseTool[BaseModel, Any, Any]] | None) -> None:
        self._tools = {t.name: t for t in tools} if tools else None

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.model_id}]; model_name={self._model_name})"

    def _validate_response(self, completion: Completion) -> None:
        parsing_params = {
            "from_substring": False,
            "strip_language_markdown": True,
        }
        try:
            for message in completion.messages:
                if not message.tool_calls:
                    if self._response_schema:
                        validate_obj_from_json_or_py_string(
                            message.content or "",
                            schema=self._response_schema,
                            **parsing_params,
                        )

                    elif self._response_schema_by_xml_tag:
                        validate_tagged_objs_from_json_or_py_string(
                            message.content or "",
                            schema_by_xml_tag=self._response_schema_by_xml_tag,
                            **parsing_params,
                        )
        except JSONSchemaValidationError as exc:
            raise LLMResponseValidationError(
                exc.s, exc.schema, message=str(exc)
            ) from exc

    def _validate_tool_calls(self, completion: Completion) -> None:
        parsing_params = {
            "from_substring": False,
            "strip_language_markdown": True,
        }
        for message in completion.messages:
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.tool_name
                    tool_arguments = tool_call.tool_arguments

                    available_tool_names = list(self.tools) if self.tools else []
                    if tool_name not in available_tool_names or not self.tools:
                        raise LLMToolCallValidationError(
                            tool_name,
                            tool_arguments,
                            message=f"Tool '{tool_name}' is not available in the LLM "
                            f"tools (available: {available_tool_names})",
                        )
                    tool = self.tools[tool_name]
                    try:
                        validate_obj_from_json_or_py_string(
                            tool_arguments, schema=tool.in_type, **parsing_params
                        )
                    except JSONSchemaValidationError as exc:
                        raise LLMToolCallValidationError(
                            tool_name, tool_arguments
                        ) from exc

    @staticmethod
    async def postprocess_event_stream(
        stream: LLMStreamGenerator,
    ) -> LLMStreamGenerator:
        prev_completion_id: str | None = None
        chunk_op_evt: CompletionChunkEvent[CompletionChunk] | None = None
        response_op_evt: ResponseChunkEvent | None = None
        thinking_op_evt: ThinkingChunkEvent | None = None
        annotations_op_evt: AnnotationsChunkEvent | None = None
        tool_calls_op_evt: ToolCallChunkEvent | None = None

        def _close_open_events() -> list[LLMStateChangeEvent[Any]]:
            nonlocal \
                chunk_op_evt, \
                thinking_op_evt, \
                tool_calls_op_evt, \
                response_op_evt, \
                annotations_op_evt

            events: list[LLMStateChangeEvent[Any]] = []

            if tool_calls_op_evt:
                events.append(ToolCallEndEvent.from_chunk_event(tool_calls_op_evt))

            if response_op_evt:
                events.append(ResponseEndEvent.from_chunk_event(response_op_evt))

            if thinking_op_evt:
                events.append(ThinkingEndEvent.from_chunk_event(thinking_op_evt))

            if annotations_op_evt:
                events.append(AnnotationsEndEvent.from_chunk_event(annotations_op_evt))

            if chunk_op_evt:
                events.append(CompletionEndEvent.from_chunk_event(chunk_op_evt))

            chunk_op_evt = None
            thinking_op_evt = None
            tool_calls_op_evt = None
            response_op_evt = None
            annotations_op_evt = None

            return events

        async for event in stream:
            if isinstance(event, CompletionChunkEvent) and not isinstance(
                event, LLMStateChangeEvent
            ):
                chunk = event.data
                if len(chunk.choices) != 1:
                    raise ValueError(
                        "Expected exactly one choice in completion chunk, "
                        f"got {len(chunk.choices)}"
                    )

                new_completion = chunk.id != prev_completion_id

                if new_completion:
                    for close_event in _close_open_events():
                        yield close_event

                    chunk_op_evt = event
                    yield CompletionStartEvent.from_chunk_event(event)

                sub_events = event.split_into_specialized()

                for sub_event in sub_events:
                    if isinstance(sub_event, ThinkingChunkEvent):
                        if not thinking_op_evt:
                            thinking_op_evt = sub_event
                            yield ThinkingStartEvent.from_chunk_event(sub_event)
                        yield sub_event
                    elif thinking_op_evt:
                        yield ThinkingEndEvent.from_chunk_event(thinking_op_evt)
                        thinking_op_evt = None

                    if isinstance(sub_event, ToolCallChunkEvent):
                        tc = sub_event.data.tool_call
                        if tc.id:
                            # Tool call ID is not None only for the first chunk of a tool call
                            if tool_calls_op_evt:
                                yield ToolCallEndEvent.from_chunk_event(
                                    tool_calls_op_evt
                                )
                                tool_calls_op_evt = None
                            tool_calls_op_evt = sub_event
                            yield ToolCallStartEvent.from_chunk_event(sub_event)
                        yield sub_event
                    elif tool_calls_op_evt:
                        yield ToolCallEndEvent.from_chunk_event(tool_calls_op_evt)
                        tool_calls_op_evt = None

                    if isinstance(sub_event, ResponseChunkEvent):
                        if not response_op_evt:
                            response_op_evt = sub_event
                            yield ResponseStartEvent.from_chunk_event(sub_event)
                        yield sub_event
                    elif response_op_evt:
                        yield ResponseEndEvent.from_chunk_event(response_op_evt)
                        response_op_evt = None

                    if isinstance(sub_event, AnnotationsChunkEvent):
                        if not annotations_op_evt:
                            annotations_op_evt = sub_event
                            yield AnnotationsStartEvent.from_chunk_event(sub_event)
                        yield sub_event
                    elif annotations_op_evt:
                        yield AnnotationsEndEvent.from_chunk_event(annotations_op_evt)
                        annotations_op_evt = None

                prev_completion_id = chunk.id

            else:
                for close_event in _close_open_events():
                    yield close_event

                yield event

    @abstractmethod
    async def generate_completion(
        self,
        conversation: Messages,
        *,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
    ) -> Completion:
        pass

    @abstractmethod
    async def generate_completion_stream(
        self,
        conversation: Messages,
        *,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
    ) -> AsyncIterator[
        CompletionChunkEvent[CompletionChunk] | CompletionEvent | LLMStreamingErrorEvent
    ]:
        pass

    @abstractmethod
    def combine_completion_chunks(self, completion_chunks: list[Any]) -> Any:
        raise NotImplementedError
