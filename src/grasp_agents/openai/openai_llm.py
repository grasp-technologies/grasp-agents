import logging
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Literal

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN  # type: ignore[import]
from pydantic import BaseModel

from ..cloud_llm import APIProvider, CloudLLM, CloudLLMSettings
from ..http_client import AsyncHTTPClientParams
from ..rate_limiting.rate_limiter_chunked import RateLimiterC
from ..typing.message import AssistantMessage, Conversation
from ..typing.tool import BaseTool
from . import (
    ChatCompletion,
    ChatCompletionAsyncStream,  # type: ignore[import]
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ParsedChatCompletion,
    # ResponseFormatJSONObject,
    # ResponseFormatJSONSchema,
    # ResponseFormatText,
)
from .converters import OpenAIConverters

logger = logging.getLogger(__name__)


class OpenAILLMSettings(CloudLLMSettings, total=False):
    reasoning_effort: Literal["low", "medium", "high"] | None

    parallel_tool_calls: bool

    # response_format: (
    #     ResponseFormatText | ResponseFormatJSONSchema | ResponseFormatJSONObject
    # )

    modalities: list[Literal["text", "audio"]] | None

    frequency_penalty: float | None
    presence_penalty: float | None
    logit_bias: dict[str, int] | None
    stop: str | list[str] | None
    logprobs: bool | None
    top_logprobs: int | None
    n: int | None

    prediction: ChatCompletionPredictionContentParam | None

    stream_options: ChatCompletionStreamOptionsParam | None

    metadata: dict[str, str] | None
    store: bool | None
    user: str


class OpenAILLM(CloudLLM[OpenAILLMSettings, OpenAIConverters]):
    def __init__(
        self,
        # Base LLM args
        model_name: str,
        model_id: str | None = None,
        llm_settings: OpenAILLMSettings | None = None,
        tools: list[BaseTool[BaseModel, Any, Any]] | None = None,
        response_format: type | None = None,
        # Connection settings
        api_provider: APIProvider = "openai",
        async_http_client_params: (
            dict[str, Any] | AsyncHTTPClientParams | None
        ) = None,
        async_openai_client_params: dict[str, Any] | None = None,
        # Rate limiting
        rate_limiter: (RateLimiterC[Conversation, AssistantMessage] | None) = None,
        rate_limiter_rpm: float | None = None,
        rate_limiter_chunk_size: int = 1000,
        rate_limiter_max_concurrency: int = 300,
        # Retries
        num_generation_retries: int = 0,
        # Disable tqdm for batch processing
        no_tqdm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            llm_settings=llm_settings,
            converters=OpenAIConverters(),
            tools=tools,
            response_format=response_format,
            api_provider=api_provider,
            async_http_client_params=async_http_client_params,
            rate_limiter=rate_limiter,
            rate_limiter_rpm=rate_limiter_rpm,
            rate_limiter_chunk_size=rate_limiter_chunk_size,
            rate_limiter_max_concurrency=rate_limiter_max_concurrency,
            num_generation_retries=num_generation_retries,
            no_tqdm=no_tqdm,
            **kwargs,
        )

        async_openai_client_params_ = deepcopy(async_openai_client_params or {})
        if self._async_http_client is not None:
            async_openai_client_params_["http_client"] = self._async_http_client

        # TODO: context manager for async client
        self._client: AsyncOpenAI = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            **async_openai_client_params_,
        )

    async def _get_completion(
        self,
        api_messages: Iterable[ChatCompletionMessageParam],
        api_tools: list[ChatCompletionToolParam] | None = None,
        api_tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        **api_llm_settings: Any,
    ) -> ChatCompletion:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN

        return await self._client.chat.completions.create(
            model=self._model_name,
            messages=api_messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
            **api_llm_settings,
        )

    async def _get_parsed_completion(
        self,
        api_messages: Iterable[ChatCompletionMessageParam],
        api_tools: list[ChatCompletionToolParam] | None = None,
        api_tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        api_response_format: type | None = None,
        **api_llm_settings: Any,
    ) -> ParsedChatCompletion[Any]:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN
        response_format = api_response_format or NOT_GIVEN

        return await self._client.beta.chat.completions.parse(
            model=self._model_name,
            messages=api_messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,  # type: ignore[arg-type]
            **api_llm_settings,
        )

    async def _get_completion_stream(
        self,
        api_messages: Iterable[ChatCompletionMessageParam],
        api_tools: list[ChatCompletionToolParam] | None = None,
        api_tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        **api_llm_settings: Any,
    ) -> ChatCompletionAsyncStream[ChatCompletionChunk]:
        assert not api_tools, "Tool use is not supported in streaming mode"

        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN

        return await self._client.chat.completions.create(
            model=self._model_name,
            messages=api_messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            **api_llm_settings,
        )
