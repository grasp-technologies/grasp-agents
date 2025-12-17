import fnmatch
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar

from openai import AsyncOpenAI
from openai._types import omit
from openai.lib.streaming.responses._responses import (
    AsyncResponseStreamManager,
)
from openai.types.responses import (
    ParsedResponse,
    Response,
    ResponseCompletedEvent,
    ResponseInputParam,
    ResponseOutputItemDoneEvent,
    ResponseStreamEvent,
    ResponseTextConfigParam,
)
from openai.types.responses.response_create_params import (
    StreamOptions as ResponsesStreamOptionsParam,
)
from openai.types.responses.response_create_params import (
    ToolChoice as ResponseToolChoice,
)
from openai.types.responses.tool_param import (
    ToolParam as ResponsesToolParam,
)
from openai.types.shared import Reasoning
from pydantic import BaseModel

from grasp_agents.cloud_llm import APIProvider, CloudLLM, CloudLLMSettings
from grasp_agents.typing.events import CompletionChunkEvent, CompletionItemEvent
from grasp_agents.typing.tool import BaseTool

from .converters import OpenAIResponsesConverters, ResponseApiChunk

logger = logging.getLogger(__name__)


class OpenAIResponsesLLMSettings(CloudLLMSettings, total=False):
    # web search should be put as a tool:
    # tools=[{
    #   "type": "web_search_preview",
    #   "search_context_size": "high",  # Options: "low", "medium", "high"
    #    "user_location": {...}
    # }]
    reasoning: Reasoning
    parallel_tool_calls: bool
    max_output_tokens: int
    top_logprobs: int | None

    text: ResponseTextConfigParam
    stream_options: ResponsesStreamOptionsParam | None
    store: bool | None
    user: str


@dataclass(frozen=True)
class OpenAIResponsesLLM(CloudLLM):
    llm_settings: OpenAIResponsesLLMSettings | None = None
    converters: ClassVar[OpenAIResponsesConverters] = OpenAIResponsesConverters()
    openai_client_timeout: float = 120.0
    openai_client_max_retries: int = 2
    extra_openai_client_params: dict[str, Any] | None = None
    client: AsyncOpenAI = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        _model_name = self.model_name
        _openai_client_params = deepcopy(self.extra_openai_client_params or {})
        _openai_client_params["timeout"] = self.openai_client_timeout
        _openai_client_params["max_retries"] = self.openai_client_max_retries
        if self.http_client is not None:
            _openai_client_params["http_client"] = self.http_client

        _client = AsyncOpenAI(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            **_openai_client_params,
        )

        object.__setattr__(self, "model_name", _model_name)
        object.__setattr__(self, "client", _client)

    async def _get_api_completion(
        self,
        api_messages: list[ResponseInputParam],
        *,
        api_tools: list[ResponsesToolParam] | None = None,
        api_tool_choice: ResponseToolChoice | None = None,
        api_response_schema: type[Any] | None = None,
        **api_llm_settings: Any,
    ) -> ParsedResponse[Any] | Response:
        messages = [subitem for item in api_messages for subitem in item]
        tools = api_tools or []
        tool_choice = api_tool_choice if api_tool_choice is not None else omit
        text_format = api_response_schema if api_response_schema is not None else omit
        response_id = api_llm_settings.get("previous_response_id")
        if self.apply_response_schema_via_provider:
            return await self.client.responses.parse(
                model=self.model_name,
                input=[messages[-1]] if response_id else messages,
                tools=tools,
                tool_choice=tool_choice,
                text_format=text_format,
                **api_llm_settings,
            )
        return await self.client.responses.create(
            model=self.model_name,
            input=[messages[-1]] if response_id else messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
            **api_llm_settings,
        )

    async def _get_api_completion_stream(
        self,
        api_messages: list[ResponseInputParam],
        api_tools: Iterable[ResponsesToolParam] | None = None,
        api_tool_choice: ResponseToolChoice | None = None,
        api_response_schema: type[Any] | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[ResponseStreamEvent]:
        messages = [subitem for item in api_messages for subitem in item]
        tools = api_tools if api_tools is not None else omit
        response_id = api_llm_settings.get("previous_response_id")
        tool_choice = api_tool_choice if api_tool_choice is not None else omit
        text_format = api_response_schema if api_response_schema is not None else omit
        _api_llm_settings = dict(api_llm_settings)
        if "stream_options" in _api_llm_settings:
            so = dict(_api_llm_settings.get("stream_options") or {})
            so.pop("include_usage", None)
            _api_llm_settings["stream_options"] = so

        async def iterator() -> AsyncIterator[ResponseStreamEvent]:
            effective_text_format = (
                text_format if self.apply_response_schema_via_provider else omit
            )
            stream_manager: AsyncResponseStreamManager[Any] = (
                self.client.responses.stream(
                    model=self.model_name,
                    input=[messages[-1]] if response_id else messages,
                    tool_choice=tool_choice,
                    tools=tools,
                    text_format=effective_text_format,
                    **_api_llm_settings,
                )
            )

            async with stream_manager as stream:
                async for response_event in stream:
                    yield response_event

        return iterator()

    def combine_completion_chunks(
        self,
        completion_chunks: list[ResponseStreamEvent],
        response_schema: Any | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
    ) -> Response:
        final_resp = None
        if len(completion_chunks) > 0:
            final_resp = completion_chunks[-1]
            if isinstance(final_resp, ResponseCompletedEvent):
                return final_resp.response
        raise RuntimeError("No 'response.completed' event received")

    async def _handle_api_stream_event(  # type: ignore
        self,
        event: ResponseStreamEvent,
        proc_name: str | None = None,
        call_id: str | None = None,
    ) -> AsyncGenerator[CompletionChunkEvent[Any] | CompletionItemEvent, None]:
        if isinstance(event, ResponseApiChunk):
            try:
                completion_chunk = self.converters.from_completion_chunk(
                    event, name=self.model_id
                )
            except TypeError as e:
                logger.debug(
                    "Skipping chunk conversion for event type %s: %s",
                    type(event).__name__,
                    e,
                )
                return
            yield CompletionChunkEvent(
                data=completion_chunk, src_name=proc_name, call_id=call_id
            )
        if isinstance(event, ResponseOutputItemDoneEvent):
            try:
                completion_item = self.converters.from_api_item(
                    event, name=self.model_id
                )
            except TypeError:
                logger.exception(
                    "Failed to convert ResponseOutputItemDoneEvent "
                    "to CompletionItem (item type: %s)",
                    getattr(event.item, "type", "unknown"),
                )
                return
            yield CompletionItemEvent(
                data=completion_item, src_name=proc_name, call_id=call_id
            )
