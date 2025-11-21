import logging
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from openai._types import NOT_GIVEN
from pydantic import BaseModel

from ...typing.message import AssistantMessage, Messages
from ...typing.tool import BaseTool
from .. import (
    OpenAIAsyncResponseStreamManager,
    OpenAILLM,
    OpenAIParsedResponse,
    OpenAIReasoning,
    OpenAIResponse,
    OpenAIResponseCompletedEvent,
    OpenAIResponsesInputParam,
    OpenAIResponsesStreamOptionsParam,
    OpenAIResponsesToolParam,
    OpenAIResponseStreamEvent,
    OpenAIResponseTextConfigParam,
    OpenAIResponseToolChoice,
)
from ..openai_llm import OpenAILLMSettings
from .responses_converters import OpenAIResponsesConverters
from .responses_chunk_converters import is_supported_stream_event

logger = logging.getLogger(__name__)


class OpenAIResponsesLLMSettings(OpenAILLMSettings, total=False):
    # web search should be put as a tool:
    # tools=[{
    #   "type": "web_search_preview",
    #   "search_context_size": "high",  # Options: "low", "medium", "high"
    #    "user_location": {...}
    # }]
    reasoning: OpenAIReasoning
    max_output_tokens: int

    text: OpenAIResponseTextConfigParam
    stream_options: OpenAIResponsesStreamOptionsParam | None


@dataclass(frozen=True)
class OpenAIResponsesLLM(OpenAILLM):
    llm_settings: OpenAIResponsesLLMSettings | None = None
    converters: ClassVar[OpenAIResponsesConverters] = OpenAIResponsesConverters()

    async def _get_api_completion(
        self,
        api_messages: OpenAIResponsesInputParam,
        *,
        response_id: str | None = None,
        api_tools: list[OpenAIResponsesToolParam] | None = None,
        api_tool_choice: OpenAIResponseToolChoice | None = None,
        api_response_schema: type[Any] | None = None,
        **api_llm_settings: Any,
    ) -> OpenAIParsedResponse[Any] | OpenAIResponse:
        tools = api_tools or []
        tool_choice = api_tool_choice or NOT_GIVEN
        text_format = api_response_schema or NOT_GIVEN

        if self.apply_response_schema_via_provider:
            return await self.client.responses.parse(
                model=self.model_name,
                previous_response_id=response_id or NOT_GIVEN,
                input=api_messages,
                tools=tools,
                tool_choice=tool_choice,
                text_format=text_format,
                **api_llm_settings,
            )
        return await self.client.responses.create(
            model=self.model_name,
            previous_response_id=response_id or NOT_GIVEN,
            input=api_messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
            **api_llm_settings,
        )

    async def _get_api_completion_stream(
        self,
        api_messages: OpenAIResponsesInputParam,
        response_id: str | None = None,
        api_tools: Iterable[OpenAIResponsesToolParam] | None = None,
        api_tool_choice: OpenAIResponseToolChoice | None = None,
        api_response_schema: type[Any] | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[OpenAIResponseStreamEvent]:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN
        text_format = api_response_schema or NOT_GIVEN
        _api_llm_settings = dict(api_llm_settings)
        if "stream_options" in _api_llm_settings:
            so = dict(_api_llm_settings.get("stream_options") or {})
            so.pop("include_usage", None)
            _api_llm_settings["stream_options"] = so

        async def iterator() -> AsyncIterator[OpenAIResponseStreamEvent]:
            effective_text_format = (
                text_format if self.apply_response_schema_via_provider else NOT_GIVEN
            )
            stream_manager: OpenAIAsyncResponseStreamManager[Any] = (
                self.client.responses.stream(
                    model=self.model_name,
                    input=api_messages,
                    tool_choice=tool_choice,
                    tools=tools,
                    previous_response_id=response_id or NOT_GIVEN,
                    text_format=effective_text_format,
                    **_api_llm_settings,
                )
            )

            async with stream_manager as stream:
                async for response_event in stream:
                    if isinstance(
                        response_event, OpenAIResponseCompletedEvent
                    ) or is_supported_stream_event(response_event):
                        yield response_event

        return iterator()

    def combine_completion_chunks(
        self,
        completion_chunks: list[OpenAIResponseStreamEvent],
        response_schema: Any | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
    ) -> OpenAIResponse:
        final_resp = None
        if len(completion_chunks) > 0:
            final_resp = completion_chunks[-1]
        if final_resp is not None and isinstance(
            final_resp, OpenAIResponseCompletedEvent
        ):
            return final_resp.response
        raise RuntimeError("No 'response.completed' event received")
