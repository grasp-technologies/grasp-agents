import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Required

import httpx
from pydantic import BaseModel
from typing_extensions import TypedDict

from .llm import LLM, LLMSettings
from .rate_limiting.rate_limiter import RateLimiter, limit_rate
from .typing.completion import Completion
from .typing.completion_chunk import CompletionChunk
from .typing.events import (
    CompletionChunkEvent,
    CompletionEvent,
    LLMStreamingErrorEvent,
)
from .typing.message import AssistantMessage, Messages
from .typing.tool import BaseTool, ToolChoice

logger = logging.getLogger(__name__)


class APIProvider(TypedDict, total=False):
    name: Required[str]
    base_url: Required[str | None]
    api_key: Required[str | None]
    # Wildcard patterns for model names that support response schema validation:
    response_schema_support: tuple[str, ...] | None


class CloudLLMSettings(LLMSettings, total=False):
    extra_headers: dict[str, Any] | None
    extra_body: object | None
    extra_query: dict[str, Any] | None


LLMRateLimiter = RateLimiter[
    AssistantMessage
    | AsyncIterator[
        CompletionChunkEvent[CompletionChunk] | CompletionEvent | LLMStreamingErrorEvent
    ],
]


@dataclass(frozen=True)
class CloudLLM(LLM):
    llm_settings: CloudLLMSettings | None = None
    api_provider: APIProvider | None = None
    rate_limiter: LLMRateLimiter | None = None
    apply_response_schema_via_provider: bool = False
    apply_tool_call_schema_via_provider: bool = False
    http_client: httpx.AsyncClient | None = None

    def __post_init__(self) -> None:
        if self.rate_limiter is not None:
            logger.info(
                f"[{self.__class__.__name__}] Set rate limit to "
                f"{self.rate_limiter.rpm} RPM"
            )

        if self.apply_response_schema_via_provider:
            object.__setattr__(self, "apply_tool_call_schema_via_provider", True)

    @abstractmethod
    async def _get_api_completion(
        self,
        api_messages: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        pass

    @abstractmethod
    async def _get_api_completion_stream(
        self,
        api_messages: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[Any]:
        pass

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)

        if "_get_api_completion" in cls.__dict__:
            cls._get_api_completion = limit_rate(cls._get_api_completion)  # type: ignore[method-assign]

        if "_get_api_completion_stream" in cls.__dict__:
            cls._get_api_completion_stream = limit_rate(cls._get_api_completion_stream)  # type: ignore[method-assign]

    def _make_api_completion_kwargs(
        self,
        messages: Messages,
        response_schema: Any | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> dict[str, Any]:
        api_messages = [self.converters.to_message(m) for m in messages]

        api_tools = None
        api_tool_choice = None
        if tools:
            strict = True if self.apply_tool_call_schema_via_provider else None
            api_tools = [
                self.converters.to_tool(t, strict=strict) for t in tools.values()
            ]
            if tool_choice is not None:
                api_tool_choice = self.converters.to_tool_choice(tool_choice)

        api_llm_settings = deepcopy((self.llm_settings or {}) | extra_llm_settings)

        return dict(
            api_messages=api_messages,
            api_tools=api_tools,
            api_tool_choice=api_tool_choice,
            api_response_schema=response_schema,
            **api_llm_settings,
        )

    async def _generate_completion_once(
        self,
        messages: Messages,
        *,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Completion:
        completion_kwargs = self._make_api_completion_kwargs(
            messages=messages,
            response_schema=response_schema,
            tools=tools,
            tool_choice=tool_choice,
            **extra_llm_settings,
        )

        if not self.apply_response_schema_via_provider:
            completion_kwargs.pop("api_response_schema", None)
        api_completion = await self._get_api_completion(**completion_kwargs)

        completion = self.converters.from_completion(api_completion, name=self.model_id)

        # if not self.apply_response_schema_via_provider:
        self._validate_response(
            completion,
            response_schema=response_schema,
            response_schema_by_xml_tag=response_schema_by_xml_tag,
        )
        # if not self.apply_tool_call_schema_via_provider and tools is not None:
        if tools is not None:
            self._validate_tool_calls(completion, tools=tools)

        return completion

    async def _generate_completion_stream_once(
        self,
        messages: Messages,
        *,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[CompletionChunkEvent[CompletionChunk] | CompletionEvent]:
        completion_kwargs = self._make_api_completion_kwargs(
            messages=messages,
            response_schema=response_schema,
            tools=tools,
            tool_choice=tool_choice,
            **extra_llm_settings,
        )
        if not self.apply_response_schema_via_provider:
            completion_kwargs.pop("api_response_schema", None)

        api_stream = await self._get_api_completion_stream(**completion_kwargs)

        api_completion_chunks: list[Any] = []

        async for api_completion_chunk in api_stream:
            api_completion_chunks.append(api_completion_chunk)
            completion_chunk = self.converters.from_completion_chunk(
                api_completion_chunk, name=self.model_id
            )

            yield CompletionChunkEvent(
                data=completion_chunk, src_name=proc_name, call_id=call_id
            )

        api_completion = self.combine_completion_chunks(
            api_completion_chunks, response_schema=response_schema, tools=tools
        )
        completion = self.converters.from_completion(api_completion, name=self.model_id)

        yield CompletionEvent(data=completion, src_name=proc_name, call_id=call_id)

        # if not self.apply_response_schema_via_provider:
        self._validate_response(
            completion,
            response_schema=response_schema,
            response_schema_by_xml_tag=response_schema_by_xml_tag,
        )
        # if not self.apply_tool_call_schema_via_provider and tools is not None:
        if tools is not None:
            self._validate_tool_calls(completion, tools=tools)
