"""Native Anthropic Messages API provider."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from anthropic._types import omit  # type: ignore[import]
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

from .error_mapping import map_api_error
from .llm_event_converters import AnthropicStreamConverter
from .provider_output_to_response import provider_output_to_response
from .response_to_provider_inputs import items_to_provider_inputs
from .tool_converters import to_api_tool, to_api_tool_choice

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from pydantic import BaseModel

    from grasp_agents.types.items import InputItem
    from grasp_agents.types.llm_errors import LlmError
    from grasp_agents.types.llm_events import LlmEvent
    from grasp_agents.types.response import Response
    from grasp_agents.types.tool import BaseTool, ToolChoice

    from . import (
        AnthropicMessage,
        AnthropicStreamEvent,
        ThinkingConfigParam,
        ToolChoiceParam,
        ToolParam,
    )

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 65536

WebSearchToolParam = WebSearchTool20260209Param | WebSearchTool20250305Param
WebFetchToolParam = WebFetchTool20260209Param


class AnthropicLLMSettings(CloudLLMSettings, total=False):
    max_tokens: int
    thinking: ThinkingConfigParam | None
    stop_sequences: list[str] | None
    top_k: int | None
    web_search: WebSearchToolParam | None
    web_fetch: WebFetchToolParam | None


@dataclass(frozen=True)
class AnthropicLLM(CloudLLM):
    litellm_provider: str | None = "anthropic"
    llm_settings: AnthropicLLMSettings | None = None
    anthropic_client_timeout: float = 60.0
    anthropic_client_max_retries: int = 2
    client: AsyncAnthropic = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        _api_provider = self.api_provider or APIProvider(
            name="anthropic",
            base_url=None,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

        _client = AsyncAnthropic(
            base_url=_api_provider.get("base_url"),
            api_key=_api_provider.get("api_key"),
            timeout=self.anthropic_client_timeout,
            max_retries=self.anthropic_client_max_retries,
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
        system, messages = items_to_provider_inputs(input)

        api_tools: list[ToolParam | Any] | None = None
        if tools:
            api_tools = [to_api_tool(tool) for tool in tools.values()]

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
        max_tokens: int = api_llm_settings.pop("max_tokens", DEFAULT_MAX_TOKENS)

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
        max_tokens: int = api_llm_settings.pop("max_tokens", DEFAULT_MAX_TOKENS)

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
