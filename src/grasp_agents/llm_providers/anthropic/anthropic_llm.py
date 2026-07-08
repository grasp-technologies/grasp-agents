"""Native Anthropic Messages API provider."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from anthropic import AsyncAnthropic
from anthropic._types import omit  # type: ignore[import]  # noqa: PLC2701
from anthropic.types import (
    WebFetchTool20260209Param,
    WebSearchTool20250305Param,
    WebSearchTool20260209Param,
)

from grasp_agents.llm.cloud_llm import (
    ApiCallParams,
    APIProvider,
    CloudLLM,
    CloudLLMSettings,
)
from grasp_agents.llm.model_info import get_model_capabilities

from .error_mapping import map_api_error
from .llm_event_converters import AnthropicStreamConverter
from .provider_output_to_response import provider_output_to_response
from .response_to_provider_inputs import items_to_provider_inputs
from .tool_converters import to_api_tool, to_api_tool_choice

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from anthropic import (
        AsyncAnthropicBedrock,  # pyright: ignore[reportPrivateImportUsage]
        AsyncAnthropicBedrockMantle,  # pyright: ignore[reportPrivateImportUsage]
        AsyncAnthropicVertex,  # pyright: ignore[reportPrivateImportUsage]
    )
    from anthropic.types import MetadataParam, OutputConfigParam
    from pydantic import BaseModel

    from grasp_agents.tools.base import BaseTool, ToolChoice
    from grasp_agents.types.items import InputItem
    from grasp_agents.types.llm_errors import LlmError
    from grasp_agents.types.llm_events import LlmEvent
    from grasp_agents.types.response import Response

    from . import (
        AnthropicMessage,
        AnthropicStreamEvent,
        ThinkingConfigParam,
        ToolChoiceParam,
        ToolParam,
    )

logger = logging.getLogger(__name__)

# Fallback ``max_tokens`` when the model's output cap is unknown. Known models
# use their own cap from the pricing/capability registry — sending a value
# above the cap is a hard 400 on the Messages API.
DEFAULT_MAX_TOKENS = 32768

WebSearchToolParam = WebSearchTool20260209Param | WebSearchTool20250305Param
WebFetchToolParam = WebFetchTool20260209Param

AnthropicPlatform = Literal["anthropic", "bedrock", "bedrock_mantle", "vertex"]

_BEDROCK_INSTALL_HINT = (
    "AWS Bedrock support is unavailable. Install it with "
    "`pip install 'grasp_agents[bedrock]'` (and use a recent anthropic SDK for "
    "the 'bedrock_mantle' endpoint)."
)
_VERTEX_INSTALL_HINT = (
    "Google Vertex AI support is unavailable. Install it with "
    "`pip install 'grasp_agents[vertex]'`."
)


class AnthropicLLMSettings(CloudLLMSettings, total=False):
    max_tokens: int
    thinking: ThinkingConfigParam | None
    stop_sequences: list[str] | None
    top_k: int | None

    # Structured outputs sent directly (the manual escape hatch). When
    # ``apply_output_schema_via_provider`` is set the schema travels via the
    # separate ``output_schema`` channel (``messages.parse``) instead.
    output_config: OutputConfigParam | None

    web_search: WebSearchToolParam | None
    web_fetch: WebFetchToolParam | None

    metadata: MetadataParam | None
    service_tier: Literal["auto", "standard_only"] | None
    container: str | None
    inference_geo: str | None


class BedrockClientConfig(TypedDict, total=False):
    """
    Client args for ``platform="bedrock"`` / ``"bedrock_mantle"``.

    Unset values fall back to the standard AWS credential/region chain (env
    vars, shared config/credentials files, SSO, IMDS). ``api_key`` is a Bedrock
    bearer token, mutually exclusive with the ``aws_*`` credential fields.
    """

    aws_region: str
    aws_profile: str
    aws_access_key: str
    aws_secret_key: str
    aws_session_token: str
    api_key: str
    base_url: str


class VertexClientConfig(TypedDict, total=False):
    """
    Client args for ``platform="vertex"``.

    Unset values fall back to Application Default Credentials and the SDK env
    vars (``ANTHROPIC_VERTEX_PROJECT_ID`` / ``CLOUD_ML_REGION``).
    """

    project_id: str
    region: str
    access_token: str
    credentials: Any
    base_url: str


@dataclass(frozen=True)
class AnthropicLLM(CloudLLM):
    litellm_provider: str | None = "anthropic"
    llm_settings: AnthropicLLMSettings | None = None
    # Matches the SDK default. A long generation (large max_tokens, extended
    # thinking) easily exceeds one minute; a short client timeout makes it
    # fail deterministically through every retry and fallback.
    anthropic_client_timeout: float = 600.0
    # SDK-level retries default to 0: ``LLM.retry_policy`` is the retry
    # system, and a non-zero value here would multiply with it.
    anthropic_client_max_retries: int = 0
    # Escape hatch: forwarded verbatim to the underlying client constructor
    # (direct / Bedrock / Vertex) for SDK-specific args not in platform_config.
    extra_anthropic_client_params: dict[str, Any] | None = None

    # Which Anthropic-hosting platform to call. The Messages API surface is
    # identical across all four; only client construction differs.
    platform: AnthropicPlatform = "anthropic"
    # Platform-specific client args: a ``BedrockClientConfig`` for
    # "bedrock"/"bedrock_mantle" or a ``VertexClientConfig`` for "vertex"
    # (ignored for the direct API, which uses ``api_provider``). May carry
    # secrets — kept out of repr.
    platform_config: BedrockClientConfig | VertexClientConfig | None = field(
        default=None, repr=False
    )

    client: (
        AsyncAnthropic
        | AsyncAnthropicBedrock
        | AsyncAnthropicVertex
        | AsyncAnthropicBedrockMantle
    ) = field(init=False)
    _default_max_tokens: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        _api_provider = self.api_provider or APIProvider(
            name=self.platform,
            base_url=None,
            api_key=(
                os.getenv("ANTHROPIC_API_KEY") if self.platform == "anthropic" else None
            ),
        )

        _client = self._build_client(_api_provider)

        # On a cloud platform the model IDs and cost tables differ from the
        # direct API; resolve the pricing identity used for cost lookups
        # unless the caller pinned ``litellm_provider`` explicitly.
        litellm_provider = self.litellm_provider
        if self.platform != "anthropic" and litellm_provider == "anthropic":
            litellm_provider = "vertex_ai" if self.platform == "vertex" else "bedrock"
            object.__setattr__(self, "litellm_provider", litellm_provider)

        cap = get_model_capabilities(
            self.model_name, litellm_provider
        ).max_output_tokens

        object.__setattr__(self, "api_provider", _api_provider)
        object.__setattr__(self, "client", _client)
        object.__setattr__(self, "_default_max_tokens", cap or DEFAULT_MAX_TOKENS)

    def _build_client(
        self, api_provider: APIProvider
    ) -> (
        AsyncAnthropic
        | AsyncAnthropicBedrock
        | AsyncAnthropicVertex
        | AsyncAnthropicBedrockMantle
    ):
        # Client args whose role is identical across the direct / Bedrock /
        # Vertex clients (so nothing the caller configured is dropped).
        common: dict[str, Any] = {
            "timeout": self.anthropic_client_timeout,
            "max_retries": self.anthropic_client_max_retries,
        }
        if api_provider.get("base_url"):
            common["base_url"] = api_provider["base_url"]
        if self.http_client is not None:
            common["http_client"] = self.http_client
        if self.default_headers is not None:
            common["default_headers"] = self.default_headers

        config: dict[str, Any] = dict(self.platform_config or {})
        extra: dict[str, Any] = dict(self.extra_anthropic_client_params or {})

        if self.platform == "anthropic":
            anthropic_kwargs: dict[str, Any] = {
                **common,
                "api_key": api_provider.get("api_key"),
                **extra,
            }
            return AsyncAnthropic(**anthropic_kwargs)

        if self.platform in {"bedrock", "bedrock_mantle"}:
            try:
                from anthropic import (  # noqa: PLC0415
                    AsyncAnthropicBedrock,  # pyright: ignore[reportPrivateImportUsage]
                    AsyncAnthropicBedrockMantle,  # pyright: ignore[reportPrivateImportUsage]
                )
            except ImportError as err:
                raise ImportError(_BEDROCK_INSTALL_HINT) from err

            bedrock_cls = (
                AsyncAnthropicBedrockMantle
                if self.platform == "bedrock_mantle"
                else AsyncAnthropicBedrock
            )
            return bedrock_cls(**{**common, **config, **extra})

        try:
            from anthropic import (  # noqa: PLC0415
                AsyncAnthropicVertex,  # pyright: ignore[reportPrivateImportUsage]
            )
        except ImportError as err:
            raise ImportError(_VERTEX_INSTALL_HINT) from err

        return AsyncAnthropicVertex(**{**common, **config, **extra})

    # --- Input preparation ---

    def _make_api_input(  # type: ignore[override]
        self,
        input: Sequence[InputItem],  # noqa: A002
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        output_schema: type | None = None,
        **extra_llm_settings: Any,
    ) -> ApiCallParams:
        system, messages = items_to_provider_inputs(input)

        api_tools: list[ToolParam | Any] | None = None
        if tools:
            strict = self.apply_tool_call_schema_via_provider
            api_tools = [to_api_tool(tool, strict=strict) for tool in tools.values()]

        api_tool_choice: ToolChoiceParam | None = None
        if tool_choice is not None:
            api_tool_choice = to_api_tool_choice(tool_choice)

        merged: dict[str, Any] = dict(self.llm_settings or {})
        merged.update(extra_llm_settings)

        # Server-side tools: go into the tools list
        web_search_tool_param: WebSearchToolParam | None = merged.pop(
            "web_search", None
        )
        if web_search_tool_param is not None:
            api_tools = api_tools or []
            api_tools.append(web_search_tool_param)

        web_fetch_tool_param: WebFetchToolParam | None = merged.pop("web_fetch", None)
        if web_fetch_tool_param is not None:
            api_tools = api_tools or []
            api_tools.append(web_fetch_tool_param)

        extra_settings: dict[str, Any] = {}
        if system is not None:
            extra_settings["system"] = system

        if merged:
            extra_settings.update(merged)

        api_kwargs: ApiCallParams = ApiCallParams(
            api_input=messages,
            api_tools=api_tools,
            api_tool_choice=api_tool_choice,
        )
        if output_schema is not None:
            api_kwargs["api_output_schema"] = output_schema
        if extra_settings:
            api_kwargs["extra_settings"] = extra_settings

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
        api_output_schema: Any | None = None,
        **api_llm_settings: Any,
    ) -> AnthropicMessage:
        max_tokens: int = (
            api_llm_settings.pop("max_tokens", None) or self._default_max_tokens
        )

        if self.apply_output_schema_via_provider:
            return await self.client.messages.parse(
                max_tokens=max_tokens,
                messages=api_input,
                model=self.model_name,
                stream=False,
                output_format=api_output_schema or omit,
                tools=api_tools or omit,
                tool_choice=api_tool_choice or omit,
                **api_llm_settings,
            )

        return await self.client.messages.create(
            max_tokens=max_tokens,
            messages=api_input,
            model=self.model_name,
            stream=False,
            tools=api_tools or omit,
            tool_choice=api_tool_choice or omit,
            **api_llm_settings,
        )

    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: Any | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[AnthropicStreamEvent]:
        max_tokens: int = (
            api_llm_settings.pop("max_tokens", None) or self._default_max_tokens
        )

        async def iterator() -> AsyncIterator[AnthropicStreamEvent]:
            async with self.client.messages.stream(
                max_tokens=max_tokens,
                messages=api_input,
                model=self.model_name,
                output_format=api_output_schema or omit,
                tools=api_tools or omit,
                tool_choice=api_tool_choice or omit,
                **api_llm_settings,
            ) as stream:
                async for event in stream:
                    yield event  # type: ignore[misc]

        return iterator()

    # --- Conversion layer ---

    def _convert_api_response(self, raw: Any) -> Response:
        return provider_output_to_response(raw)

    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[LlmEvent]:
        converter = AnthropicStreamConverter()
        async for event in converter.convert(api_stream):
            yield event
