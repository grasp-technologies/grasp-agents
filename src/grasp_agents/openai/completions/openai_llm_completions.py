import fnmatch
import logging
import os
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from openai import AsyncOpenAI, AsyncStream
from openai._types import omit  # type: ignore  # noqa: PLC2701
from openai.lib.streaming.chat import (
    AsyncChatCompletionStreamManager as OpenAIAsyncChatCompletionStreamManager,
)
from openai.lib.streaming.chat import ChunkEvent as OpenAIChunkEvent
from pydantic import BaseModel

from ...cloud_llm import APIProvider, CloudLLM, CloudLLMSettings
from ...typing.response import Response
from ...typing.stream_events import StreamEvent
from ...typing.tool import BaseTool, ToolChoice
from . import (
    OpenAICompletion,
    OpenAICompletionChunk,
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
from .response_converters import from_openai_completion
from .stream_event_converters import CompletionsStreamConverter

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
        # Openrouter does not support structured outputs ATM
        APIProvider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            response_schema_support=(),
        ),
    ]


class OpenAILLMSettings(CloudLLMSettings, total=False):
    reasoning_effort: (
        Literal["none", "disable", "minimal", "low", "medium", "high"] | None
    )

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
class OpenAILLM(CloudLLM):
    llm_settings: OpenAILLMSettings | None = None
    converters: ClassVar[OpenAIConverters] = OpenAIConverters()
    openai_client_timeout: float = 60.0
    openai_client_max_retries: int = 2
    extra_openai_client_params: dict[str, Any] | None = None
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
            _provider_name, _model_name = model_name_parts
            if _provider_name not in compat_providers_map:
                raise ValueError(
                    f"API provider '{_provider_name}' is not a supported OpenAI "
                    f"compatible provider. Supported providers are: "
                    f"{', '.join(compat_providers_map.keys())}"
                )
            _api_provider = compat_providers_map[_provider_name]
        else:
            raise ValueError(
                "Model name must be in the format 'provider/model_name' or "
                "you must provide an 'api_provider' argument."
            )

        response_schema_support: bool = any(
            fnmatch.fnmatch(_model_name, pat)
            for pat in _api_provider.get("response_schema_support") or []
        )
        if self.apply_response_schema_via_provider and not response_schema_support:
            raise ValueError(
                "Native response schema validation is not supported for model "
                f"'{_model_name}' by the API provider '{_api_provider['name']}'. "
                "Please set apply_response_schema_via_provider=False."
            )

        _openai_client_params = deepcopy(self.extra_openai_client_params or {})
        _openai_client_params["timeout"] = self.openai_client_timeout
        _openai_client_params["max_retries"] = self.openai_client_max_retries
        if self.http_client is not None:
            _openai_client_params["http_client"] = self.http_client

        _client = AsyncOpenAI(
            base_url=_api_provider.get("base_url"),
            api_key=_api_provider.get("api_key"),
            **_openai_client_params,
        )

        object.__setattr__(self, "model_name", _model_name)
        object.__setattr__(self, "api_provider", _api_provider)
        object.__setattr__(self, "client", _client)

    # --- Input preparation ---

    def _make_api_input(
        self,
        input: list[Any],  # noqa: A002
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        response_schema: Any | None = None,
        **extra_llm_settings: Any,
    ) -> dict[str, Any]:
        api_tools: list[OpenAIToolParam] | None = None
        if tools:
            strict = self.apply_tool_call_schema_via_provider
            api_tools = [
                self.converters.to_tool(tool, strict=strict)
                for tool in tools.values()
            ]

        api_tool_choice: OpenAIToolChoiceOptionParam | None = None
        if tool_choice is not None:
            api_tool_choice = self.converters.to_tool_choice(tool_choice)

        merged: dict[str, Any] = dict(self.llm_settings or {})
        merged.update(extra_llm_settings)

        return {
            "api_input": input,
            "api_tools": api_tools,
            "api_tool_choice": api_tool_choice,
            "api_response_schema": response_schema,
            **merged,
        }

    # --- Provider API layer ---

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> OpenAICompletion | OpenAIParsedCompletion[Any]:
        tools = api_tools or omit
        tool_choice = api_tool_choice or omit
        response_format = api_response_schema or omit

        if self.apply_response_schema_via_provider:
            return await self.client.beta.chat.completions.parse(  # type: ignore[reportUnknownVariableType]
                model=self.model_name,
                messages=api_input,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                **api_llm_settings,
            )

        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=api_input,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
            **api_llm_settings,
        )

    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[OpenAICompletionChunk]:
        tools = api_tools or omit
        tool_choice = api_tool_choice or omit
        response_format = api_response_schema or omit

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
                        messages=api_input,
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
                    messages=api_input,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=True,
                    **api_llm_settings,
                )
                async with stream_generator as stream:
                    async for completion_chunk in stream:
                        yield completion_chunk

        return iterator()

    # --- Conversion layer ---

    def _convert_api_response(self, raw: Any) -> Response:
        return from_openai_completion(raw)

    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[StreamEvent]:
        converter = CompletionsStreamConverter()
        async for event in converter.convert(api_stream):
            yield event
