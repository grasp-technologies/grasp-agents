import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypedDict

from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai._types import omit  # type: ignore  # noqa: PLC2701
from openai.lib._parsing._completions import validate_input_tools  # noqa: PLC2701
from openai.lib._pydantic import to_strict_json_schema  # noqa: PLC2701
from openai.lib.streaming.chat import (
    AsyncChatCompletionStreamManager as OpenAIAsyncChatCompletionStreamManager,
)
from openai.lib.streaming.chat import ChunkEvent as OpenAIChunkEvent
from openai.types.chat.completion_create_params import (
    Moderation as CompletionsModeration,
)
from openai.types.chat.completion_create_params import (
    PromptCacheOptions as CompletionsPromptCacheOptions,
)
from pydantic import BaseModel, ConfigDict, with_config

from grasp_agents.llm.cloud_llm import (
    ApiCallParams,
    APIProvider,
    CloudLLM,
    CloudLLMSettings,
)
from grasp_agents.tools.base import BaseTool, ToolChoice
from grasp_agents.types.items import InputItem
from grasp_agents.types.llm_errors import LlmError
from grasp_agents.types.llm_events import LlmEvent
from grasp_agents.types.response import Response

from . import (
    OpenAICompletion,
    OpenAICompletionChunk,
    OpenAIParsedCompletion,
    OpenAIPredictionContentParam,
    OpenAIResponseFormatJSONObject,
    OpenAIResponseFormatJSONSchema,
    OpenAIResponseFormatText,
    OpenAIStreamOptionsParam,
    OpenAIToolChoiceOptionParam,
    OpenAIToolParam,
)
from .error_mapping import map_api_error
from .llm_event_converters import CompletionsStreamConverter
from .provider_output_to_response import provider_output_to_response
from .response_to_provider_inputs import items_to_provider_inputs
from .tool_converters import to_api_tool, to_api_tool_choice

logger = logging.getLogger(__name__)


OpenAICloudPlatform = Literal["azure"]

CompletionsReasoningEffort = Literal[
    "none", "disable", "minimal", "low", "medium", "high", "xhigh", "max"
]


@with_config(ConfigDict(extra="allow"))
class OpenAILLMSettings(CloudLLMSettings, total=False):
    max_completion_tokens: int | None
    seed: int | None
    frequency_penalty: float | None
    presence_penalty: float | None
    logit_bias: dict[str, int] | None
    stop: str | list[str] | None
    logprobs: bool | None
    top_logprobs: int | None

    reasoning_effort: CompletionsReasoningEffort | None

    parallel_tool_calls: bool

    stream_options: OpenAIStreamOptionsParam | None

    # The SDK's full response_format union (text / json_schema / json_object).
    # Sending a json_schema here is the manual path; the
    # ``apply_output_schema_via_provider`` gate instead routes the
    # ``output_schema`` through ``beta.chat.completions.parse``.
    response_format: (
        OpenAIResponseFormatText
        | OpenAIResponseFormatJSONSchema
        | OpenAIResponseFormatJSONObject
    )

    modalities: list[Literal["text"]] | None
    metadata: dict[str, str] | None
    store: bool | None
    user: str
    prediction: OpenAIPredictionContentParam | None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None
    verbosity: Literal["low", "medium", "high"] | None
    prompt_cache_key: str
    prompt_cache_retention: Literal["in_memory", "24h"] | None
    prompt_cache_options: CompletionsPromptCacheOptions | None
    moderation: CompletionsModeration | None
    safety_identifier: str

    # TODO: support audio


class AzureClientConfig(TypedDict, total=False):
    """
    Client args for ``platform="azure"`` (an ``AsyncAzureOpenAI`` client).

    Unset values fall back to the SDK's env vars: AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY, OPENAI_API_VERSION. ``api_key`` / ``azure_ad_token``
    are secrets, so the whole config is kept out of repr.
    """

    azure_endpoint: str
    api_version: str
    azure_deployment: str
    api_key: str
    azure_ad_token: str
    azure_ad_token_provider: Callable[[], str | Awaitable[str]]
    organization: str
    base_url: str


@dataclass(frozen=True)
class OpenAILLM(CloudLLM):
    _settings_type: ClassVar[Any] = OpenAILLMSettings

    _native_provider_name: ClassVar[str] = "openai"
    _native_api_key_env_vars: ClassVar[tuple[str, ...]] = ("OPENAI_API_KEY",)
    _cloud_platforms: ClassVar[frozenset[str]] = frozenset({"azure"})

    llm_settings: OpenAILLMSettings | None = None

    openai_client_timeout: float = 600.0
    # SDK-level retries default to 0: ``LLM.retry_policy`` is the retry
    # system, and a non-zero value here would multiply with it.
    openai_client_max_retries: int = 0
    extra_openai_client_params: dict[str, Any] | None = None

    # "azure" builds an ``AsyncAzureOpenAI`` client (the Chat Completions
    # surface is identical); ``None`` uses the plain client — pointed by
    # ``api_provider`` at api.openai.com (the default) or at any
    # OpenAI-compatible endpoint.
    platform: OpenAICloudPlatform | None = None
    # Azure client args (see AzureClientConfig). ``model_name`` is the Azure
    # *deployment* name. May carry secrets — kept out of repr.
    platform_config: AzureClientConfig | None = field(default=None, repr=False)

    client: AsyncOpenAI = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        # Client args common to the OpenAI and Azure clients (so nothing the
        # caller configured is dropped); ``extra_openai_client_params`` has the
        # last word on all of them.
        common: dict[str, Any] = {
            "timeout": self.openai_client_timeout,
            "max_retries": self.openai_client_max_retries,
        }
        if self.http_client is not None:
            common["http_client"] = self.http_client
        if self.default_headers is not None:
            common["default_headers"] = self.default_headers

        config: dict[str, Any] = dict(self.platform_config or {})
        extra: dict[str, Any] = dict(self.extra_openai_client_params or {})

        _client: AsyncOpenAI
        if self.platform == "azure":
            # Anything not supplied here is read from the SDK's Azure env vars.
            _client = AsyncAzureOpenAI(**{**common, **config, **extra})
            _api_provider = APIProvider(
                name=self.platform,
                base_url=config.get("azure_endpoint") or config.get("base_url"),
                api_key=config.get("api_key"),
            )
        else:
            # api.openai.com, or any OpenAI-compatible endpoint.
            _api_provider = self.api_provider or self._default_api_provider()
            client_params: dict[str, Any] = {
                "api_key": _api_provider.get("api_key"),
                "base_url": _api_provider.get("base_url"),
                **common,
                **extra,
            }
            _client = AsyncOpenAI(**client_params)

        object.__setattr__(self, "litellm_provider", self._resolve_litellm_provider())
        object.__setattr__(self, "api_provider", _api_provider)
        object.__setattr__(self, "client", _client)

    # --- Input preparation ---

    def _make_api_input(
        self,
        input: Sequence[InputItem],  # noqa: A002
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        output_schema: Any | None = None,
        **extra_llm_settings: Any,
    ) -> ApiCallParams:
        api_tools: list[OpenAIToolParam] | None = None
        if tools:
            strict = self.apply_tool_call_schema_via_provider
            api_tools = [to_api_tool(tool, strict=strict) for tool in tools.values()]

        api_tool_choice: OpenAIToolChoiceOptionParam | None = None
        if tool_choice is not None:
            api_tool_choice = to_api_tool_choice(tool_choice)

        merged: dict[str, Any] = dict(self.llm_settings or {})
        merged.update(extra_llm_settings)

        reasoning_fmt = (
            "openrouter" if self.litellm_provider == "openrouter" else "anthropic"
        )
        api_kwargs: ApiCallParams = {
            "api_input": items_to_provider_inputs(
                input, reasoning_block_format=reasoning_fmt
            ),
            "api_tools": api_tools,
            "api_tool_choice": api_tool_choice,
        }
        if output_schema is not None:
            api_kwargs["api_output_schema"] = output_schema
        if merged:
            api_kwargs["extra_settings"] = merged

        return api_kwargs

    # --- Error mapping ---

    def _map_api_error(self, err: Exception) -> LlmError | None:
        return map_api_error(err)

    # --- Provider API layer ---

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> OpenAICompletion | OpenAIParsedCompletion[Any]:
        tools = api_tools or omit
        tool_choice = api_tool_choice or omit
        response_format = api_output_schema or omit

        if self.apply_output_schema_via_provider:
            if (
                self.tolerate_output_around_json
                and isinstance(api_output_schema, type)
                and issubclass(api_output_schema, BaseModel)
            ):
                # The same strict schema `.parse()` would send, sent through
                # `.create()` instead: `.parse()` also parses the reply and
                # rejects one whose JSON value is followed by extra bytes, which
                # is exactly what `tolerate_output_around_json` exists to
                # forgive — its check runs first, so leniency downstream would
                # never be reached. Leaving the parsing to `_validate_response`
                # keeps provider-side enforcement either way.
                #
                # Streaming is deliberately not covered: `_get_api_stream` still
                # hands the schema to `.stream()`, whose parse is equally strict.
                #
                # `.parse()` also rejects a non-function or non-strict tool up
                # front; `.create()` does not, so run that check explicitly
                # rather than lose it.
                validate_input_tools(tools)
                schema_format: OpenAIResponseFormatJSONSchema = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": api_output_schema.__name__,
                        "schema": to_strict_json_schema(api_output_schema),
                        "strict": True,
                    },
                }
                return await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=api_input,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=False,
                    response_format=schema_format,
                    **api_llm_settings,
                )

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
        api_output_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[OpenAICompletionChunk]:
        tools = api_tools or omit
        tool_choice = api_tool_choice or omit
        response_format = api_output_schema or omit

        # Ensure usage is included in the streamed responses
        stream_options = dict(api_llm_settings.get("stream_options") or {})
        stream_options["include_usage"] = True
        _api_llm_settings: dict[str, Any] = api_llm_settings | {
            "stream_options": stream_options
        }

        # Need to wrap the iterator to make it work with decorators
        async def iterator() -> AsyncIterator[OpenAICompletionChunk]:
            if self.apply_output_schema_via_provider:
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
        return provider_output_to_response(raw)

    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[LlmEvent]:
        converter = CompletionsStreamConverter()
        async for event in converter.convert(api_stream):
            yield event
