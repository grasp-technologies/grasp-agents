"""Native Google Gemini / Vertex AI provider."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict

from google.genai import Client
from google.genai.types import (
    AutomaticFunctionCallingConfigDict,
    GenerationConfigRoutingConfigDict,
    MediaResolution,
    ModelArmorConfigDict,
    ModelSelectionConfigDict,
    ServiceTier,
    UrlContext,
)
from google.genai.types import HttpOptions as GeminiHttpOptions
from pydantic import ConfigDict, with_config

from grasp_agents.llm.cloud_llm import (
    ApiCallParams,
    APIProvider,
    CloudLLM,
    CloudLLMSettings,
)

from . import (
    GeminiConfig,
    GeminiGoogleSearch,
    GeminiGoogleSearchDict,
    GeminiSafetySettingDict,
    GeminiThinkingConfigDict,
    GeminiTool,
)
from .error_mapping import map_api_error
from .llm_event_converters import GeminiStreamConverter
from .provider_output_to_response import provider_output_to_response
from .response_to_provider_inputs import items_to_provider_inputs
from .tool_converters import to_api_tool_config, to_api_tools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from pydantic import BaseModel

    from grasp_agents.tools.base import BaseTool, ToolChoice
    from grasp_agents.types.items import InputItem
    from grasp_agents.types.llm_errors import LlmError
    from grasp_agents.types.llm_events import LlmEvent
    from grasp_agents.types.response import Response

    from . import GeminiHttpOptionsDict, GeminiResponse

logger = logging.getLogger(__name__)


@with_config(ConfigDict(extra="allow"))
class GeminiLLMSettings(CloudLLMSettings, total=False):
    max_output_tokens: int
    top_k: float
    stop_sequences: list[str]
    presence_penalty: float
    frequency_penalty: float
    seed: int | None
    response_logprobs: bool
    logprobs: int

    thinking_config: GeminiThinkingConfigDict | dict[str, Any] | None

    google_search: GeminiGoogleSearchDict | None

    url_context: bool | None
    safety_settings: list[GeminiSafetySettingDict] | None
    media_resolution: MediaResolution
    service_tier: ServiceTier
    cached_content: str
    labels: dict[str, str]
    enable_enhanced_civic_answers: bool
    routing_config: GenerationConfigRoutingConfigDict
    model_selection_config: ModelSelectionConfigDict
    model_armor_config: ModelArmorConfigDict
    automatic_function_calling: AutomaticFunctionCallingConfigDict


# Keys with no Gemini equivalent (silently dropped)
_UNSUPPORTED_BASE_KEYS = frozenset({"extra_query"})

# Keys requiring special handling (not forwarded to GeminiConfig directly)
_SPECIAL_KEYS = frozenset(
    {"google_search", "url_context", "extra_headers", "extra_body"}
)

# Keys forwarded directly to GenerateContentConfig
_CONFIG_KEYS = (
    GeminiLLMSettings.__optional_keys__ - _UNSUPPORTED_BASE_KEYS - _SPECIAL_KEYS
)


GeminiPlatform = Literal["gemini", "vertex"]


class GeminiVertexClientConfig(TypedDict, total=False):
    """
    Client args for ``platform="vertex"``.

    Unset values fall back to ADC and the SDK env vars (GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_LOCATION); ``location`` defaults to "us-central1".
    """

    project: str
    location: str
    credentials: Any


@dataclass(frozen=True)
class GeminiLLM(CloudLLM):
    _settings_type: ClassVar[Any] = GeminiLLMSettings

    litellm_provider: str | None = "vertex_ai"
    llm_settings: GeminiLLMSettings | None = None
    # Per-request HTTP timeout in seconds (``None`` disables).
    gemini_client_timeout: float | None = 600.0
    # Escape hatch: forwarded verbatim to the ``google.genai.Client``
    # constructor for SDK-specific args (e.g. ``debug_config``).
    extra_gemini_client_params: dict[str, Any] | None = None

    # "gemini" = Gemini Developer API (api_key); "vertex" = Google Vertex AI.
    platform: GeminiPlatform = "gemini"
    platform_config: GeminiVertexClientConfig | None = field(default=None, repr=False)

    client: Client = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        extra: dict[str, Any] = dict(self.extra_gemini_client_params or {})

        if self.platform == "vertex":
            _api_provider = self.api_provider or APIProvider(
                name="vertex", base_url=None, api_key=None
            )
            config: dict[str, Any] = dict(self.platform_config or {})
            kwargs: dict[str, Any] = {"vertexai": True, **config, **extra}
            kwargs.setdefault("location", "global")
        else:
            _api_provider = self.api_provider or APIProvider(
                name="gemini",
                base_url=None,
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
            kwargs = {"api_key": _api_provider.get("api_key"), **extra}

        kwargs.setdefault("http_options", self._http_options())
        _client = Client(**kwargs)

        object.__setattr__(self, "api_provider", _api_provider)
        object.__setattr__(self, "client", _client)

    def _http_options(self) -> GeminiHttpOptions:
        # Client args whose role is identical across providers (timeout,
        # default headers, a shared httpx client) map onto the genai client's
        # ``http_options``.
        opts: dict[str, Any] = {}
        if self.gemini_client_timeout is not None:
            opts["timeout"] = int(self.gemini_client_timeout * 1000)  # ms
        if self.default_headers is not None:
            opts["headers"] = dict(self.default_headers)
        if self.http_client is not None:
            opts["httpx_async_client"] = self.http_client
        return GeminiHttpOptions(**opts)

    # --- Input preparation ---

    def _make_api_input(  # type: ignore[override]
        self,
        input: Sequence[InputItem],  # noqa: A002
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        output_schema: type | None = None,
        **extra_llm_settings: Any,
    ) -> ApiCallParams:
        system_instruction, contents = items_to_provider_inputs(input)

        # Merge settings: base llm_settings + per-call overrides
        merged: dict[str, Any] = dict(self.llm_settings or {})
        merged.update(extra_llm_settings)

        # Build config kwargs
        config_kwargs: dict[str, Any] = {}

        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction

        strict_tools = self.apply_tool_call_schema_via_provider
        if tools:
            config_kwargs["tools"] = [to_api_tools(tools)]
        if tool_choice is not None:
            config_kwargs["tool_config"] = to_api_tool_config(
                tool_choice, strict=strict_tools
            )
        elif tools and strict_tools:
            # Schema enforcement is a function-calling *mode* on Gemini, so
            # opting in must set a tool_config even without a tool_choice.
            config_kwargs["tool_config"] = to_api_tool_config("auto", strict=True)

        # Forward settings to GenerateContentConfig
        config_kwargs.update({k: merged.pop(k) for k in _CONFIG_KEYS if k in merged})

        # Map extra_headers / extra_body → GenerateContentConfig.http_options
        extra_headers = merged.pop("extra_headers", None)
        extra_body = merged.pop("extra_body", None)
        merged.pop("extra_query", None)  # no Gemini equivalent
        if extra_headers or extra_body:
            # Request-level http_options replace the client-level ones, so the
            # client timeout and default headers must be re-applied here.
            http_options: GeminiHttpOptionsDict = {}
            if self.gemini_client_timeout is not None:
                http_options["timeout"] = int(self.gemini_client_timeout * 1000)
            merged_headers: dict[str, str] = {
                **(self.default_headers or {}),
                **(extra_headers or {}),
            }
            if merged_headers:
                http_options["headers"] = merged_headers
            if extra_body:
                http_options["extra_body"] = extra_body
            config_kwargs["http_options"] = http_options

        # Structured output is NOT baked into the config here: CloudLLM strips
        # ``api_output_schema`` when ``apply_output_schema_via_provider`` is
        # off (the default), and it is applied to the config at call time
        # (see ``_apply_output_schema``).

        # Google Search grounding tool
        google_search_config: GeminiGoogleSearchDict | None = merged.pop(
            "google_search", None
        )
        if google_search_config is not None:
            config_kwargs.setdefault("tools", []).append(
                GeminiTool(
                    google_search=GeminiGoogleSearch.model_validate(
                        google_search_config
                    )
                )
            )

        # URL context tool (no settings — URLs come from message content)
        if merged.pop("url_context", None):
            config_kwargs.setdefault("tools", []).append(
                GeminiTool(url_context=UrlContext())
            )

        config = GeminiConfig(**config_kwargs)

        api_kwargs = ApiCallParams(
            api_input=contents, extra_settings={"config": config}
        )
        if output_schema is not None:
            api_kwargs["api_output_schema"] = output_schema
        return api_kwargs

    # --- Error mapping ---

    def _map_api_error(self, err: Exception) -> LlmError | None:
        return map_api_error(err)

    # --- Provider API layer ---

    @staticmethod
    def _apply_output_schema(
        config: GeminiConfig | None, output_schema: type | None
    ) -> GeminiConfig | None:
        """Bake the (gate-surviving) output schema into the request config."""
        if output_schema is None or config is None:
            return config
        config.response_schema = output_schema
        if config.response_mime_type is None:
            config.response_mime_type = "application/json"
        return config

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_output_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> GeminiResponse:
        config = self._apply_output_schema(
            api_llm_settings.get("config"), api_output_schema
        )
        return await self.client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model=self.model_name, contents=api_input, config=config
        )

    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_output_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[GeminiResponse]:
        config = self._apply_output_schema(
            api_llm_settings.get("config"), api_output_schema
        )

        async def iterator() -> AsyncIterator[GeminiResponse]:
            stream = await self.client.aio.models.generate_content_stream(  # pyright: ignore[reportUnknownMemberType]
                model=self.model_name, contents=api_input, config=config
            )
            async for chunk in stream:
                yield chunk  # type: ignore[misc]

        return iterator()

    # --- Conversion layer ---

    def _convert_api_response(self, raw: Any) -> Response:
        return provider_output_to_response(raw)

    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[LlmEvent]:
        converter = GeminiStreamConverter(model=self.model_name)
        async for event in converter.convert(api_stream):
            yield event
