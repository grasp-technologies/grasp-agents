import fnmatch
import logging
import os
from collections.abc import AsyncIterator, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import AsyncOpenAI, AsyncStream
from openai._types import NOT_GIVEN  # type: ignore[import]
from openai.lib.streaming.chat import (
    AsyncChatCompletionStreamManager as OpenAIAsyncChatCompletionStreamManager,
)
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.lib.streaming.chat import ChunkEvent as OpenAIChunkEvent
from pydantic import BaseModel

from ..cloud_llm import APIProvider, CloudLLM, CloudLLMSettings
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
    reasoning_effort: Literal["disable", "minimal", "low", "medium", "high"] | None

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


@dataclass(frozen=True)
class OpenAILLM(CloudLLM[OpenAILLMSettings, OpenAIConverters]):
    converters: OpenAIConverters = field(default_factory=OpenAIConverters)
    async_openai_client_params: dict[str, Any] | None = None
    client: AsyncOpenAI = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        openai_compatible_providers = get_openai_compatible_providers()

        _api_provider = self.api_provider

        model_name_parts = self.model_name.split("/", 1)
        if _api_provider is not None:
            _model_name = self.model_name
        elif len(model_name_parts) == 2:
            compat_providers_map = {
                provider["name"]: provider for provider in openai_compatible_providers
            }
            provider_name, _model_name = model_name_parts
            if provider_name not in compat_providers_map:
                raise ValueError(
                    f"API provider '{provider_name}' is not a supported OpenAI "
                    f"compatible provider. Supported providers are: "
                    f"{', '.join(compat_providers_map.keys())}"
                )
            _api_provider = compat_providers_map[provider_name]
        else:
            raise ValueError(
                "Model name must be in the format 'provider/model_name' or "
                "you must provide an 'api_provider' argument."
            )

        if self.llm_settings is not None:
            stream_options = self.llm_settings.get("stream_options") or {}
            stream_options["include_usage"] = True
            _llm_settings = deepcopy(self.llm_settings)
            _llm_settings["stream_options"] = stream_options
        else:
            _llm_settings = OpenAILLMSettings(stream_options={"include_usage": True})

        response_schema_support: bool = any(
            fnmatch.fnmatch(_model_name, pat)
            for pat in _api_provider.get("response_schema_support") or []
        )
        if self.apply_response_schema_via_provider and not response_schema_support:
            raise ValueError(
                "Native response schema validation is not supported for model "
                f"'{_model_name}' by the API provider. Please set "
                "apply_response_schema_via_provider=False."
            )

        _async_openai_client_params = deepcopy(self.async_openai_client_params or {})
        if self.async_http_client is not None:
            _async_openai_client_params["http_client"] = self.async_http_client

        _client = AsyncOpenAI(
            base_url=_api_provider.get("base_url"),
            api_key=_api_provider.get("api_key"),
            max_retries=self.max_client_retries,
            **_async_openai_client_params,
        )

        object.__setattr__(self, "model_name", _model_name)
        object.__setattr__(self, "api_provider", _api_provider)
        object.__setattr__(self, "llm_settings", _llm_settings)
        object.__setattr__(self, "client", _client)

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

        if api_llm_settings and api_llm_settings.get("stream_options"):
            api_llm_settings.pop("stream_options")

        if self.apply_response_schema_via_provider:
            return await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=api_messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                n=n,
                **api_llm_settings,
            )

        return await self.client.chat.completions.create(
            model=self.model_name,
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

        if self.apply_response_schema_via_provider:
            stream_manager: OpenAIAsyncChatCompletionStreamManager[Any] = (
                self.client.beta.chat.completions.stream(
                    model=self.model_name,
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
            ] = await self.client.chat.completions.create(
                model=self.model_name,
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
        self,
        completion_chunks: list[OpenAICompletionChunk],
        response_schema: Any | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
    ) -> OpenAICompletion:
        response_format = NOT_GIVEN
        input_tools = NOT_GIVEN
        if self.apply_response_schema_via_provider:
            if response_schema:
                response_format = response_schema
            if tools:
                input_tools = [
                    self.converters.to_tool(tool, strict=True)
                    for tool in tools.values()
                ]
        state = ChatCompletionStreamState[Any](
            input_tools=input_tools, response_format=response_format
        )
        for chunk in completion_chunks:
            state.handle_chunk(chunk)

        return state.get_final_completion()
