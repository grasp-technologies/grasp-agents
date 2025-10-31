import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, final
from uuid import uuid4

from pydantic import BaseModel
from typing_extensions import TypedDict

from grasp_agents.typing.completion_chunk import CompletionChunk
from grasp_agents.utils.validation import (
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
    LLMStreamingErrorData,
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
from .typing.message import AssistantMessage, Messages
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


def make_refusal_completion(model_name: str, err: BaseException) -> Completion:
    failed_message = AssistantMessage(content=None, refusal=str(err))

    return Completion(model=model_name, message=failed_message, finish_reason=None)


@dataclass(frozen=True)
class LLM(ABC):
    model_name: str
    converters: ClassVar[Converters]
    llm_settings: LLMSettings | None = None
    model_id: str = field(default_factory=lambda: str(uuid4())[:8])
    max_response_retries: int = 0  # try to regenerate to pass validation

    @abstractmethod
    async def _generate_completion_once(
        self,
        messages: Messages,
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tool_choice: ToolChoice | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
        **extra_llm_settings: Any,
    ) -> Completion:
        pass

    @abstractmethod
    async def _generate_completion_stream_once(
        self,
        messages: Messages,
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tool_choice: ToolChoice | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[
        CompletionChunkEvent[CompletionChunk] | CompletionEvent | LLMStreamingErrorEvent
    ]:
        yield NotImplemented

    @abstractmethod
    def combine_completion_chunks(
        self,
        completion_chunks: list[Any],
        response_schema: Any | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
    ) -> Any:
        raise NotImplementedError

    @final
    async def generate_completion(
        self,
        messages: Messages,
        *,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
        **extra_llm_settings: Any,
    ) -> Completion:
        n_attempt = 0
        while n_attempt <= self.max_response_retries:
            try:
                return await self._generate_completion_once(
                    messages,
                    response_schema=response_schema,
                    response_schema_by_xml_tag=response_schema_by_xml_tag,
                    tools=tools,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                )

            except (LLMResponseValidationError, LLMToolCallValidationError) as err:
                n_attempt += 1
                if n_attempt <= self.max_response_retries:
                    logger.warning(
                        f"\nCloudLLM completion failed (retry attempt {n_attempt}):\n{err}"
                    )
                else:
                    raise err
                    # return make_refusal_completion(self._model_name, err)

            except Exception as err:
                raise err

        return make_refusal_completion(
            self.model_name,
            Exception("Unexpected error: retry loop exited without returning"),
        )

    @final
    async def generate_completion_stream(
        self,
        messages: Messages,
        *,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[
        CompletionChunkEvent[CompletionChunk] | CompletionEvent | LLMStreamingErrorEvent
    ]:
        n_attempt = 0
        while n_attempt <= self.max_response_retries:
            try:
                async for event in self._generate_completion_stream_once(
                    messages,
                    response_schema=response_schema,
                    response_schema_by_xml_tag=response_schema_by_xml_tag,
                    tools=tools,
                    tool_choice=tool_choice,
                    proc_name=proc_name,
                    call_id=call_id,
                    **extra_llm_settings,
                ):
                    yield event
                return

            except (LLMResponseValidationError, LLMToolCallValidationError) as err:
                n_attempt += 1
                err_data = LLMStreamingErrorData(
                    error=err, model_name=self.model_name, model_id=self.model_id
                )
                yield LLMStreamingErrorEvent(
                    data=err_data, src_name=proc_name, call_id=call_id
                )
                if n_attempt <= self.max_response_retries:
                    logger.warning(
                        f"\nCloudLLM completion failed (retry attempt {n_attempt}):\n{err}"
                    )
                else:
                    raise err
                    # refusal_completion = make_refusal_completion(
                    #     self.model_name, err
                    # )
                    # yield CompletionEvent(
                    #     data=refusal_completion,
                    #     proc_name=proc_name,
                    #     call_id=call_id,
                    # )

            except Exception as err:
                raise err

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
            message = completion.message
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
            raise LLMResponseValidationError(exc.s, exc.schema) from exc

    def _validate_tool_calls(
        self, completion: Completion, tools: Mapping[str, BaseTool[BaseModel, Any, Any]]
    ) -> None:
        parsing_params = {
            "from_substring": False,
            "strip_language_markdown": True,
        }
        message = completion.message
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
                    raise LLMToolCallValidationError(tool_name, tool_arguments) from exc

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
