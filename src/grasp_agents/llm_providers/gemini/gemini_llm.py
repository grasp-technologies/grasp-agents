"""Native Google Gemini / Vertex AI provider."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from google.genai import Client
from google.genai.types import UrlContext

from grasp_agents.llm.cloud_llm import (
    ApiCallParams,
    APIProvider,
    CloudLLM,
    CloudLLMSettings,
)

from . import GeminiConfig, GeminiGoogleSearch, GeminiGoogleSearchDict, GeminiTool
from .error_mapping import map_api_error
from .llm_event_converters import GeminiStreamConverter
from .provider_output_to_response import provider_output_to_response
from .response_to_provider_inputs import items_to_provider_inputs
from .tool_converters import to_api_tool_config, to_api_tools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from pydantic import BaseModel

    from grasp_agents.types.items import InputItem
    from grasp_agents.types.llm_errors import LlmError
    from grasp_agents.types.llm_events import LlmEvent
    from grasp_agents.types.response import Response
    from grasp_agents.types.tool import BaseTool, ToolChoice

    from . import (
        GeminiHttpOptionsDict,
        GeminiResponse,
        GeminiSafetySettingDict,
        GeminiThinkingConfigDict,
    )

logger = logging.getLogger(__name__)


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
    safety_settings: list[GeminiSafetySettingDict] | None
    google_search: GeminiGoogleSearchDict | None
    url_context: bool | None


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


@dataclass(frozen=True)
class GeminiLLM(CloudLLM):
    litellm_provider: str | None = "vertex_ai"
    llm_settings: GeminiLLMSettings | None = None
    vertexai: bool = False
    project: str | None = None
    location: str | None = None
    client: Client = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.vertexai:
            _client = Client(
                vertexai=True,
                project=self.project,
                location=self.location or "us-central1",
            )
        else:
            _api_provider = self.api_provider or APIProvider(
                name="google",
                base_url=None,
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
            _client = Client(
                api_key=_api_provider.get("api_key"),
            )
            object.__setattr__(self, "api_provider", _api_provider)

        object.__setattr__(self, "client", _client)

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

        if tools:
            config_kwargs["tools"] = [to_api_tools(tools)]
        if tool_choice is not None:
            config_kwargs["tool_config"] = to_api_tool_config(tool_choice)

        # Forward settings to GenerateContentConfig
        config_kwargs.update({k: merged.pop(k) for k in _CONFIG_KEYS if k in merged})

        # Map extra_headers / extra_body → GenerateContentConfig.http_options
        extra_headers = merged.pop("extra_headers", None)
        extra_body = merged.pop("extra_body", None)
        merged.pop("extra_query", None)  # no Gemini equivalent
        if extra_headers or extra_body:
            http_options: GeminiHttpOptionsDict = {}
            if extra_headers:
                http_options["headers"] = extra_headers
            if extra_body:
                http_options["extra_body"] = extra_body
            config_kwargs["http_options"] = http_options

        # Structured output via dynamic output_schema parameter
        if output_schema is not None:
            config_kwargs["response_schema"] = output_schema
            config_kwargs.setdefault("response_mime_type", "application/json")

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

        return ApiCallParams(api_input=contents, extra_settings={"config": config})

    # --- Error mapping ---

    def _map_api_error(self, err: Exception) -> LlmError | None:
        return map_api_error(err)

    # --- Provider API layer ---

    async def _get_api_response(
        self, api_input: list[Any], **api_llm_settings: Any
    ) -> GeminiResponse:
        config = api_llm_settings.get("config")
        return await self.client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model=self.model_name, contents=api_input, config=config
        )

    async def _get_api_stream(
        self, api_input: list[Any], **api_llm_settings: Any
    ) -> AsyncIterator[GeminiResponse]:
        config = api_llm_settings.get("config")

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
