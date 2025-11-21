import fnmatch
import logging
import os
from collections.abc import AsyncIterator, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from openai import AsyncOpenAI, AsyncStream
from openai._types import NOT_GIVEN  # type: ignore[import]
from openai.lib.streaming.chat import ChatCompletionStreamState
from pydantic import BaseModel

from ..typing.tool import BaseTool
from . import (
    OpenAIAsyncChatCompletionStreamManager,
    OpenAIChunkEvent,
    OpenAICompletion,
    OpenAICompletionChunk,
    OpenAILLM,
    OpenAILLMSettings,
    OpenAIMessageParam,
    OpenAIParsedCompletion,
    OpenAIPredictionContentParam,
    OpenAIResponseFormatJSONObject,
    OpenAIResponseFormatText,
    OpenAIStreamOptionsParam,
    OpenAIToolChoiceOptionParam,
    OpenAIToolParam,
    OpenAIWebSearchOptions,
)
from .converters import OpenAIConverters

logger = logging.getLogger(__name__)


class OpenAILLMCompletionSettings(OpenAILLMSettings, total=False):
    reasoning_effort: (
        Literal["none", "disable", "minimal", "low", "medium", "high"] | None
    )

    modalities: list[Literal["text"]] | None
    stream_options: OpenAIStreamOptionsParam | None

    frequency_penalty: float | None
    presence_penalty: float | None
    logit_bias: dict[str, int] | None
    stop: str | list[str] | None
    logprobs: bool | None

    prediction: OpenAIPredictionContentParam | None

    web_search_options: OpenAIWebSearchOptions | None

    # To support the old JSON mode without respose schemas
    response_format: OpenAIResponseFormatJSONObject | OpenAIResponseFormatText

    # TODO: support audio


@dataclass(frozen=True)
class OpenAICompletionsLLM(OpenAILLM):
    llm_settings: OpenAILLMSettings | None = None
    converters: ClassVar[OpenAIConverters] = OpenAIConverters()

    async def _get_api_completion(
        self,
        api_messages: Iterable[OpenAIMessageParam],
        *,
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        api_response_schema: type[Any] | None = None,
        **api_llm_settings: Any,
    ) -> OpenAICompletion | OpenAIParsedCompletion[Any]:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN
        response_format = api_response_schema or NOT_GIVEN

        if self.apply_response_schema_via_provider:
            return await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=api_messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                **api_llm_settings,
            )

        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
            **api_llm_settings,
        )

    async def _get_api_completion_stream(
        self,
        api_messages: Iterable[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        api_response_schema: type[Any] | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[OpenAICompletionChunk]:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN
        response_format = api_response_schema or NOT_GIVEN

        # Ensure usage is included in the streamed responses
        stream_options = dict(api_llm_settings.get("stream_options") or {})
        stream_options["include_usage"] = True
        _api_llm_settings = api_llm_settings | {"stream_options": stream_options}

        # Need to wrap the iterator to make it work with decorators
        async def iterator() -> AsyncIterator[OpenAICompletionChunk]:
            if self.apply_response_schema_via_provider:
                stream_manager: OpenAIAsyncChatCompletionStreamManager[Any] = (
                    self.client.beta.chat.completions.stream(
                        model=self.model_name,
                        messages=api_messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        response_format=response_format,
                        **_api_llm_settings,
                    )
                )
                async with stream_manager as stream:
                    async for chunk_event in stream:
                        if isinstance(chunk_event, OpenAIChunkEvent):
                            yield chunk_event.chunk
            else:
                stream_generator: AsyncStream[
                    OpenAICompletionChunk
                ] = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=api_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=True,
                    **api_llm_settings,
                )
                async with stream_generator as stream:
                    async for completion_chunk in stream:
                        yield completion_chunk

        return iterator()

    def combine_completion_chunks(
        self,
        completion_chunks: list[OpenAICompletionChunk],
        response_schema: Any | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
    ) -> OpenAICompletion:
        response_format = NOT_GIVEN
        input_tools = NOT_GIVEN
        if self.apply_response_schema_via_provider and response_schema:
            response_format = response_schema
        if self.apply_tool_call_schema_via_provider and tools:
            input_tools = [
                self.converters.to_tool(tool, strict=True) for tool in tools.values()
            ]
        state = ChatCompletionStreamState[Any](
            input_tools=input_tools, response_format=response_format
        )
        for chunk in completion_chunks:
            state.handle_chunk(chunk)

        return state.get_final_completion()
