import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
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
    # CompletionEndEvent,
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


@dataclass(frozen=True)
class LLM(ABC, Generic[SettingsT_co, ConvertT_co]):
    model_name: str
    converters: ConvertT_co
    llm_settings: SettingsT_co | None = None
    model_id: str = field(default_factory=lambda: str(uuid4())[:8])

    def _validate_response(
        self,
        completion: Completion,
        response_schema: Any | None,
        response_schema_by_xml_tag: Mapping[str, Any] | None,
    ) -> None:
        if response_schema and response_schema_by_xml_tag:
            raise ValueError(
                "Only one of response_schema and response_schema_by_xml_tag can be "
                "provided, but not both."
            )
        parsing_params = {
            "from_substring": False,
            "strip_language_markdown": True,
        }
        try:
            for message in completion.messages:
                if not message.tool_calls:
                    if response_schema:
                        validate_obj_from_json_or_py_string(
                            message.content or "",
                            schema=response_schema,
                            **parsing_params,
                        )

                    elif response_schema_by_xml_tag:
                        validate_tagged_objs_from_json_or_py_string(
                            message.content or "",
                            schema_by_xml_tag=response_schema_by_xml_tag,
                            **parsing_params,
                        )
        except JSONSchemaValidationError as exc:
            raise LLMResponseValidationError(
                exc.s, exc.schema, message=str(exc)
            ) from exc

    def _validate_tool_calls(
        self, completion: Completion, tools: Mapping[str, BaseTool[BaseModel, Any, Any]]
    ) -> None:
        parsing_params = {
            "from_substring": False,
            "strip_language_markdown": True,
        }
        for message in completion.messages:
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.tool_name
                    tool_arguments = tool_call.tool_arguments

                    available_tool_names = list(tools) if tools else []
                    if tool_name not in available_tool_names or not tools:
                        raise LLMToolCallValidationError(
                            tool_name,
                            tool_arguments,
                            message=f"Tool '{tool_name}' is not available in the LLM "
                            f"tools (available: {available_tool_names})",
                        )
                    tool = tools[tool_name]
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

        def _close_open_events(
            _event: CompletionChunkEvent[CompletionChunk] | None = None,
        ) -> list[LLMStateChangeEvent[Any]]:
            nonlocal \
                chunk_op_evt, \
                thinking_op_evt, \
                tool_calls_op_evt, \
                response_op_evt, \
                annotations_op_evt

            events: list[LLMStateChangeEvent[Any]] = []

            if not isinstance(_event, ThinkingChunkEvent) and thinking_op_evt:
                events.append(ThinkingEndEvent.from_chunk_event(thinking_op_evt))
                thinking_op_evt = None

            if not isinstance(_event, ToolCallChunkEvent) and tool_calls_op_evt:
                events.append(ToolCallEndEvent.from_chunk_event(tool_calls_op_evt))
                tool_calls_op_evt = None

            if not isinstance(_event, ResponseChunkEvent) and response_op_evt:
                events.append(ResponseEndEvent.from_chunk_event(response_op_evt))
                response_op_evt = None

            if not isinstance(_event, AnnotationsChunkEvent) and annotations_op_evt:
                events.append(AnnotationsEndEvent.from_chunk_event(annotations_op_evt))
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
                    for close_event in _close_open_events(sub_event):
                        yield close_event

                    if isinstance(sub_event, ThinkingChunkEvent):
                        if not thinking_op_evt:
                            thinking_op_evt = sub_event
                            yield ThinkingStartEvent.from_chunk_event(sub_event)
                        yield sub_event

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

                    if isinstance(sub_event, ResponseChunkEvent):
                        if not response_op_evt:
                            response_op_evt = sub_event
                            yield ResponseStartEvent.from_chunk_event(sub_event)
                        yield sub_event

                    if isinstance(sub_event, AnnotationsChunkEvent):
                        if not annotations_op_evt:
                            annotations_op_evt = sub_event
                            yield AnnotationsStartEvent.from_chunk_event(sub_event)
                        yield sub_event

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
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
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
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
    ) -> AsyncIterator[
        CompletionChunkEvent[CompletionChunk] | CompletionEvent | LLMStreamingErrorEvent
    ]:
        pass

    @abstractmethod
    def combine_completion_chunks(
        self,
        completion_chunks: list[Any],
        response_schema: Any | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
    ) -> Any:
        raise NotImplementedError
