import logging
from collections.abc import AsyncIterator, Iterable, Mapping
from copy import deepcopy
from typing import Any, Literal, NamedTuple

import httpx
from openai import AsyncOpenAI
from openai._types import NOT_GIVEN  # type: ignore[import]
from openai.lib.streaming.chat import (
    AsyncChatCompletionStreamManager as OpenAIAsyncChatCompletionStreamManager,
)
from openai.lib.streaming.chat import ChunkEvent as OpenAIChunkEvent
from pydantic import BaseModel

from ..cloud_llm import APIProvider, CloudLLM, CloudLLMSettings
from ..http_client import AsyncHTTPClientParams
from ..rate_limiting.rate_limiter_chunked import RateLimiterC
from ..typing.message import AssistantMessage, Messages
from ..typing.tool import BaseTool
from . import (
    OpenAICompletion,
    OpenAICompletionChunk,
    OpenAIMessageParam,
    OpenAIParsedCompletion,
    OpenAIPredictionContentParam,
    OpenAIStreamOptionsParam,
    OpenAIToolChoiceOptionParam,
    OpenAIToolParam,
)
from .converters import OpenAIConverters

logger = logging.getLogger(__name__)


class ToolCallSettings(NamedTuple):
    strict: bool | None = None


class OpenAILLMSettings(CloudLLMSettings, total=False):
    reasoning_effort: Literal["low", "medium", "high"] | None

    parallel_tool_calls: bool

    modalities: list[Literal["text", "audio"]] | None

    frequency_penalty: float | None
    presence_penalty: float | None
    logit_bias: dict[str, int] | None
    stop: str | list[str] | None
    logprobs: bool | None
    top_logprobs: int | None

    prediction: OpenAIPredictionContentParam | None

    stream_options: OpenAIStreamOptionsParam | None

    metadata: dict[str, str] | None
    store: bool | None
    user: str

    # response_format: (
    #     OpenAIResponseFormatText
    #     | OpenAIResponseFormatJSONSchema
    #     | OpenAIResponseFormatJSONObject
    # )


class OpenAILLM(CloudLLM[OpenAILLMSettings, OpenAIConverters]):
    def __init__(
        self,
        # Base LLM args
        model_name: str,
        llm_settings: OpenAILLMSettings | None = None,
        tools: list[BaseTool[BaseModel, Any, Any]] | None = None,
        response_format: type | Mapping[str, type] | None = None,
        model_id: str | None = None,
        # Custom LLM provider
        api_provider: APIProvider | None = None,
        # Connection settings
        async_http_client: httpx.AsyncClient | None = None,
        async_http_client_params: (
            dict[str, Any] | AsyncHTTPClientParams | None
        ) = None,
        async_openai_client_params: dict[str, Any] | None = None,
        # Rate limiting
        rate_limiter: (RateLimiterC[Messages, AssistantMessage] | None) = None,
        rate_limiter_rpm: float | None = None,
        rate_limiter_chunk_size: int = 1000,
        rate_limiter_max_concurrency: int = 300,
        # Retries
        num_generation_retries: int = 0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            llm_settings=llm_settings,
            converters=OpenAIConverters(),
            tools=tools,
            response_format=response_format,
            api_provider=api_provider,
            async_http_client=async_http_client,
            async_http_client_params=async_http_client_params,
            rate_limiter=rate_limiter,
            rate_limiter_rpm=rate_limiter_rpm,
            rate_limiter_chunk_size=rate_limiter_chunk_size,
            rate_limiter_max_concurrency=rate_limiter_max_concurrency,
            num_generation_retries=num_generation_retries,
        )

        self._tool_call_settings = {
            "strict": self._llm_settings.get("use_struct_outputs", False),
        }

        _async_openai_client_params = deepcopy(async_openai_client_params or {})
        if self._async_http_client is not None:
            _async_openai_client_params["http_client"] = self._async_http_client

        self._client: AsyncOpenAI = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            **_async_openai_client_params,
        )

    async def _get_completion(
        self,
        api_messages: Iterable[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> OpenAICompletion:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN
        n = n_choices or NOT_GIVEN

        return await self._client.chat.completions.create(
            model=self._api_model_name,
            messages=api_messages,
            tools=tools,
            tool_choice=tool_choice,
            n=n,
            stream=False,
            **api_llm_settings,
        )

    async def _get_parsed_completion(
        self,
        api_messages: Iterable[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        api_response_format: type | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> OpenAIParsedCompletion[Any]:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN
        n = n_choices or NOT_GIVEN
        response_format = api_response_format or NOT_GIVEN

        return await self._client.beta.chat.completions.parse(
            model=self._api_model_name,
            messages=api_messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            n=n,
            **api_llm_settings,
        )

    async def _get_completion_stream(
        self,
        api_messages: Iterable[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[OpenAICompletionChunk]:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN
        n = n_choices or NOT_GIVEN

        stream_generator = await self._client.chat.completions.create(
            model=self._api_model_name,
            messages=api_messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            n=n,
            **api_llm_settings,
        )

        async def iterate() -> AsyncIterator[OpenAICompletionChunk]:
            async with stream_generator as stream:
                async for completion_chunk in stream:
                    yield completion_chunk

        return iterate()

    async def _get_parsed_completion_stream(
        self,
        api_messages: Iterable[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        api_response_format: type | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[OpenAICompletionChunk]:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN
        response_format = api_response_format or NOT_GIVEN
        n = n_choices or NOT_GIVEN

        stream_manager: OpenAIAsyncChatCompletionStreamManager[
            OpenAICompletionChunk
        ] = self._client.beta.chat.completions.stream(
            model=self._api_model_name,
            messages=api_messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            n=n,
            **api_llm_settings,
        )

        async def iterate() -> AsyncIterator[OpenAICompletionChunk]:
            async with stream_manager as stream:
                async for chunk_event in stream:
                    if isinstance(chunk_event, OpenAIChunkEvent):
                        yield chunk_event.chunk

        return iterate()
