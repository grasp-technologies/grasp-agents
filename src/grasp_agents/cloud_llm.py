import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Required, cast

import httpx
from pydantic import BaseModel
from typing_extensions import TypedDict

from .errors import LLMResponseValidationError, LLMToolCallValidationError
from .http_client import AsyncHTTPClientParams, create_simple_async_httpx_client
from .llm import LLM, ConvertT_co, LLMSettings, SettingsT_co
from .rate_limiting.rate_limiter_chunked import RateLimiterC, limit_rate
from .typing.completion import Completion
from .typing.completion_chunk import CompletionChoice, CompletionChunk
from .typing.events import (
    CompletionChunkEvent,
    CompletionEvent,
    LLMStreamingErrorData,
    LLMStreamingErrorEvent,
)
from .typing.message import AssistantMessage, Messages
from .typing.tool import BaseTool, ToolChoice

logger = logging.getLogger(__name__)


class APIProvider(TypedDict, total=False):
    name: Required[str]
    base_url: str | None
    api_key: str | None
    # Wildcard patterns for model names that support response schema validation:
    response_schema_support: tuple[str, ...] | None


def make_refusal_completion(model_name: str, err: BaseException) -> Completion:
    failed_message = AssistantMessage(content=None, refusal=str(err))

    return Completion(
        model=model_name,
        choices=[CompletionChoice(message=failed_message, finish_reason=None, index=0)],
    )


class CloudLLMSettings(LLMSettings, total=False):
    extra_headers: dict[str, Any] | None
    extra_body: object | None
    extra_query: dict[str, Any] | None


LLMRateLimiter = RateLimiterC[
    Messages,
    AssistantMessage
    | AsyncIterator[
        CompletionChunkEvent[CompletionChunk] | CompletionEvent | LLMStreamingErrorEvent
    ],
]


@dataclass(frozen=True)
class CloudLLM(LLM[SettingsT_co, ConvertT_co], Generic[SettingsT_co, ConvertT_co]):
    api_provider: APIProvider | None = None
    llm_settings: SettingsT_co | None = None
    rate_limiter: LLMRateLimiter | None = None
    max_client_retries: int = 2  # HTTP client retries for network errors
    max_response_retries: int = (
        0  # LLM response retries: try to regenerate to pass validation
    )
    apply_response_schema_via_provider: bool = False
    apply_tool_call_schema_via_provider: bool = False
    async_http_client: httpx.AsyncClient | None = None
    async_http_client_params: dict[str, Any] | AsyncHTTPClientParams | None = None

    def __post_init__(self) -> None:
        if self.rate_limiter is not None:
            logger.info(
                f"[{self.__class__.__name__}] Set rate limit to "
                f"{self.rate_limiter.rpm} RPM"
            )

        if self.apply_response_schema_via_provider:
            object.__setattr__(self, "apply_tool_call_schema_via_provider", True)

        if self.async_http_client is None and self.async_http_client_params is not None:
            object.__setattr__(
                self,
                "async_http_client",
                create_simple_async_httpx_client(self.async_http_client_params),
            )

    def _make_completion_kwargs(
        self,
        conversation: Messages,
        response_schema: Any | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
    ) -> dict[str, Any]:
        api_messages = [self.converters.to_message(m) for m in conversation]

        api_tools = None
        api_tool_choice = None
        if tools:
            strict = True if self.apply_tool_call_schema_via_provider else None
            api_tools = [
                self.converters.to_tool(t, strict=strict) for t in tools.values()
            ]
            if tool_choice is not None:
                api_tool_choice = self.converters.to_tool_choice(tool_choice)

        api_llm_settings = deepcopy(self.llm_settings or {})

        return dict(
            api_messages=api_messages,
            api_tools=api_tools,
            api_tool_choice=api_tool_choice,
            api_response_schema=response_schema,
            n_choices=n_choices,
            **api_llm_settings,
        )

    @abstractmethod
    async def _get_completion(
        self,
        api_messages: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_schema: type | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        pass

    @abstractmethod
    async def _get_completion_stream(
        self,
        api_messages: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_schema: type | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[Any]:
        pass

    @limit_rate
    async def _generate_completion_once(
        self,
        conversation: Messages,
        *,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
    ) -> Completion:
        completion_kwargs = self._make_completion_kwargs(
            conversation=conversation,
            response_schema=response_schema,
            tools=tools,
            tool_choice=tool_choice,
            n_choices=n_choices,
        )

        if not self.apply_response_schema_via_provider:
            completion_kwargs.pop("api_response_schema", None)
        api_completion = await self._get_completion(**completion_kwargs)

        completion = self.converters.from_completion(api_completion, name=self.model_id)

        if not self.apply_response_schema_via_provider:
            self._validate_response(
                completion,
                response_schema=response_schema,
                response_schema_by_xml_tag=response_schema_by_xml_tag,
            )
        if not self.apply_tool_call_schema_via_provider and tools is not None:
            self._validate_tool_calls(completion, tools=tools)

        return completion

    async def generate_completion(
        self,
        conversation: Messages,
        *,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
    ) -> Completion:
        n_attempt = 0
        while n_attempt <= self.max_response_retries:
            try:
                return await self._generate_completion_once(
                    conversation,  # type: ignore[return]
                    response_schema=response_schema,
                    response_schema_by_xml_tag=response_schema_by_xml_tag,
                    tools=tools,
                    tool_choice=tool_choice,
                    n_choices=n_choices,
                )
            except (LLMResponseValidationError, LLMToolCallValidationError) as err:
                n_attempt += 1

                if n_attempt > self.max_response_retries:
                    if n_attempt == 1:
                        logger.warning(f"\nCloudLLM completion failed:\n{err}")
                    if n_attempt > 1:
                        logger.warning(
                            f"\nCloudLLM completion failed after retrying:\n{err}"
                        )
                    raise err
                    # return make_refusal_completion(self._model_name, err)

                logger.warning(
                    f"\nCloudLLM completion failed (retry attempt {n_attempt}):\n{err}"
                )

        return make_refusal_completion(
            self.model_name,
            Exception("Unexpected error: retry loop exited without returning"),
        )

    @limit_rate
    async def _generate_completion_stream_once(
        self,
        conversation: Messages,
        *,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
    ) -> AsyncIterator[CompletionChunkEvent[CompletionChunk] | CompletionEvent]:
        completion_kwargs = self._make_completion_kwargs(
            conversation=conversation,
            response_schema=response_schema,
            tools=tools,
            tool_choice=tool_choice,
            n_choices=n_choices,
        )
        if not self.apply_response_schema_via_provider:
            completion_kwargs.pop("api_response_schema", None)

        api_stream = self._get_completion_stream(**completion_kwargs)
        api_stream = cast("AsyncIterator[Any]", api_stream)

        async def iterator() -> AsyncIterator[
            CompletionChunkEvent[CompletionChunk] | CompletionEvent
        ]:
            api_completion_chunks: list[Any] = []

            async for api_completion_chunk in api_stream:
                api_completion_chunks.append(api_completion_chunk)
                completion_chunk = self.converters.from_completion_chunk(
                    api_completion_chunk, name=self.model_id
                )

                yield CompletionChunkEvent(
                    data=completion_chunk, proc_name=proc_name, call_id=call_id
                )

            api_completion = self.combine_completion_chunks(
                api_completion_chunks, response_schema=response_schema, tools=tools
            )
            completion = self.converters.from_completion(
                api_completion, name=self.model_id
            )

            yield CompletionEvent(data=completion, proc_name=proc_name, call_id=call_id)

            if not self.apply_response_schema_via_provider:
                self._validate_response(
                    completion,
                    response_schema=response_schema,
                    response_schema_by_xml_tag=response_schema_by_xml_tag,
                )
            if not self.apply_tool_call_schema_via_provider and tools is not None:
                self._validate_tool_calls(completion, tools=tools)

        return iterator()

    async def generate_completion_stream(  # type: ignore[override]
        self,
        conversation: Messages,
        *,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
        proc_name: str | None = None,
        call_id: str | None = None,
    ) -> AsyncIterator[
        CompletionChunkEvent[CompletionChunk] | CompletionEvent | LLMStreamingErrorEvent
    ]:
        n_attempt = 0
        while n_attempt <= self.max_response_retries:
            try:
                async for event in await self._generate_completion_stream_once(  # type: ignore[return]
                    conversation,  # type: ignore[arg-type]
                    response_schema=response_schema,
                    response_schema_by_xml_tag=response_schema_by_xml_tag,
                    tools=tools,
                    tool_choice=tool_choice,
                    n_choices=n_choices,
                    proc_name=proc_name,
                    call_id=call_id,
                ):
                    yield event
                return
            except (LLMResponseValidationError, LLMToolCallValidationError) as err:
                err_data = LLMStreamingErrorData(
                    error=err, model_name=self.model_name, model_id=self.model_id
                )
                yield LLMStreamingErrorEvent(
                    data=err_data, proc_name=proc_name, call_id=call_id
                )

                n_attempt += 1
                if n_attempt > self.max_response_retries:
                    if n_attempt == 1:
                        logger.warning(f"\nCloudLLM completion failed:\n{err}")
                    if n_attempt > 1:
                        logger.warning(
                            f"\nCloudLLM completion failed after retrying:\n{err}"
                        )
                        refusal_completion = make_refusal_completion(
                            self.model_name, err
                        )
                        yield CompletionEvent(
                            data=refusal_completion,
                            proc_name=proc_name,
                            call_id=call_id,
                        )
                    raise err
                    # return

                logger.warning(
                    f"\nCloudLLM completion failed (retry attempt {n_attempt}):\n{err}"
                )
