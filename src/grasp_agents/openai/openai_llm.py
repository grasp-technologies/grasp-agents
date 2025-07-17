import fnmatch
import logging
import os
from collections.abc import AsyncIterator, Iterable, Mapping
from copy import deepcopy
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI, AsyncStream
from openai._types import NOT_GIVEN  # type: ignore[import]
from openai.lib.streaming.chat import (
    AsyncChatCompletionStreamManager as OpenAIAsyncChatCompletionStreamManager,
)
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.lib.streaming.chat import ChunkEvent as OpenAIChunkEvent
from pydantic import BaseModel

from ..cloud_llm import APIProvider, CloudLLM, CloudLLMSettings, LLMRateLimiter
from ..http_client import AsyncHTTPClientParams
from ..typing.tool import BaseTool
from . import (
    OpenAICompletion,
    OpenAICompletionChunk,
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


def get_openai_compatible_providers() -> list[APIProvider]:
    """Returns a dictionary of available OpenAI-compatible API providers."""
    return [
        APIProvider(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            response_schema_support=("*",),
        ),
        APIProvider(
            name="gemini_openai",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY"),
            response_schema_support=("*",),
        ),
        APIProvider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            response_schema_support=(),
        ),
    ]


class OpenAILLMSettings(CloudLLMSettings, total=False):
    reasoning_effort: Literal["low", "medium", "high"] | None

    parallel_tool_calls: bool

    modalities: list[Literal["text"]] | None

    frequency_penalty: float | None
    presence_penalty: float | None
    logit_bias: dict[str, int] | None
    stop: str | list[str] | None
    logprobs: bool | None
    top_logprobs: int | None

    stream_options: OpenAIStreamOptionsParam | None

    prediction: OpenAIPredictionContentParam | None

    web_search_options: OpenAIWebSearchOptions | None

    metadata: dict[str, str] | None
    store: bool | None
    user: str

    # To support the old JSON mode without respose schemas
    response_format: OpenAIResponseFormatJSONObject | OpenAIResponseFormatText

    # TODO: support audio


class OpenAILLM(CloudLLM[OpenAILLMSettings, OpenAIConverters]):
    def __init__(
        self,
        # Base LLM args
        model_name: str,
        llm_settings: OpenAILLMSettings | None = None,
        tools: list[BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        apply_response_schema_via_provider: bool = False,
        model_id: str | None = None,
        # Custom LLM provider
        api_provider: APIProvider | None = None,
        # Connection settings
        max_client_retries: int = 2,
        async_http_client: httpx.AsyncClient | None = None,
        async_http_client_params: (
            dict[str, Any] | AsyncHTTPClientParams | None
        ) = None,
        async_openai_client_params: dict[str, Any] | None = None,
        # Rate limiting
        rate_limiter: LLMRateLimiter | None = None,
        # LLM response retries: try to regenerate to pass validation
        max_response_retries: int = 1,
    ) -> None:
        openai_compatible_providers = get_openai_compatible_providers()

        model_name_parts = model_name.split("/", 1)
        if api_provider is not None:
            provider_model_name = model_name
        elif len(model_name_parts) == 2:
            compat_providers_map = {
                provider["name"]: provider for provider in openai_compatible_providers
            }
            provider_name, provider_model_name = model_name_parts
            if provider_name not in compat_providers_map:
                raise ValueError(
                    f"OpenAI compatible API provider '{provider_name}' "
                    "is not supported. Supported providers are: "
                    f"{', '.join(compat_providers_map.keys())}"
                )
            api_provider = compat_providers_map[provider_name]
        else:
            raise ValueError(
                "Model name must be in the format 'provider/model_name' or "
                "you must provide an 'api_provider' argument."
            )

        super().__init__(
            model_name=provider_model_name,
            model_id=model_id,
            llm_settings=llm_settings,
            converters=OpenAIConverters(),
            tools=tools,
            response_schema=response_schema,
            response_schema_by_xml_tag=response_schema_by_xml_tag,
            apply_response_schema_via_provider=apply_response_schema_via_provider,
            api_provider=api_provider,
            async_http_client=async_http_client,
            async_http_client_params=async_http_client_params,
            rate_limiter=rate_limiter,
            max_client_retries=max_client_retries,
            max_response_retries=max_response_retries,
        )

        response_schema_support: bool = any(
            fnmatch.fnmatch(self._model_name, pat)
            for pat in api_provider.get("response_schema_support") or []
        )
        if apply_response_schema_via_provider:
            if self._tools:
                for tool in self._tools.values():
                    tool.strict = True
            if not response_schema_support:
                raise ValueError(
                    "Native response schema validation is not supported for model "
                    f"'{self._model_name}' by the API provider. Please set "
                    "apply_response_schema_via_provider=False."
                )

        _async_openai_client_params = deepcopy(async_openai_client_params or {})
        if self._async_http_client is not None:
            _async_openai_client_params["http_client"] = self._async_http_client

        self._client: AsyncOpenAI = AsyncOpenAI(
            base_url=self.api_provider.get("base_url"),
            api_key=self.api_provider.get("api_key"),
            max_retries=max_client_retries,
            **_async_openai_client_params,
        )

    async def _get_completion(
        self,
        api_messages: Iterable[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        api_response_schema: type[Any] | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> OpenAICompletion | OpenAIParsedCompletion[Any]:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN
        response_format = api_response_schema or NOT_GIVEN
        n = n_choices or NOT_GIVEN

        if self._apply_response_schema_via_provider:
            return await self._client.beta.chat.completions.parse(
                model=self._model_name,
                messages=api_messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                n=n,
                **api_llm_settings,
            )

        return await self._client.chat.completions.create(
            model=self._model_name,
            messages=api_messages,
            tools=tools,
            tool_choice=tool_choice,
            n=n,
            stream=False,
            **api_llm_settings,
        )

    async def _get_completion_stream(  # type: ignore[override]
        self,
        api_messages: Iterable[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        api_response_schema: type[Any] | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[OpenAICompletionChunk]:
        tools = api_tools or NOT_GIVEN
        tool_choice = api_tool_choice or NOT_GIVEN
        response_format = api_response_schema or NOT_GIVEN
        n = n_choices or NOT_GIVEN

        if self._apply_response_schema_via_provider:
            stream_manager: OpenAIAsyncChatCompletionStreamManager[Any] = (
                self._client.beta.chat.completions.stream(
                    model=self._model_name,
                    messages=api_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    n=n,
                    **api_llm_settings,
                )
            )
            async with stream_manager as stream:
                async for chunk_event in stream:
                    if isinstance(chunk_event, OpenAIChunkEvent):
                        yield chunk_event.chunk
        else:
            stream_generator: AsyncStream[
                OpenAICompletionChunk
            ] = await self._client.chat.completions.create(
                model=self._model_name,
                messages=api_messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
                n=n,
                **api_llm_settings,
            )
            async with stream_generator as stream:
                async for completion_chunk in stream:
                    yield completion_chunk

    def combine_completion_chunks(
        self, completion_chunks: list[OpenAICompletionChunk]
    ) -> OpenAICompletion:
        response_format = NOT_GIVEN
        input_tools = NOT_GIVEN
        if self._apply_response_schema_via_provider:
            if self._response_schema:
                response_format = self._response_schema
            if self._tools:
                input_tools = [
                    self._converters.to_tool(tool) for tool in self._tools.values()
                ]
        state = ChatCompletionStreamState[Any](
            input_tools=input_tools, response_format=response_format
        )
        for chunk in completion_chunks:
            state.handle_chunk(chunk)

        return state.get_final_completion()
