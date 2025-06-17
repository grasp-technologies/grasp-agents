import fnmatch
import logging
import os
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, Generic, Literal

import httpx
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing_extensions import TypedDict

from .http_client import AsyncHTTPClientParams, create_async_http_client
from .llm import LLM, ConvertT_co, LLMSettings, SettingsT_co
from .message_history import MessageHistory
from .rate_limiting.rate_limiter_chunked import RateLimiterC, limit_rate_chunked
from .typing.completion import Completion
from .typing.completion_chunk import CompletionChunk, combine_completion_chunks
from .typing.events import CompletionChunkEvent, CompletionEvent
from .typing.message import AssistantMessage, Messages
from .typing.tool import BaseTool, ToolChoice

logger = logging.getLogger(__name__)


APIProvider = Literal["openai", "openrouter", "google_ai_studio"]


class APIProviderInfo(TypedDict):
    name: APIProvider
    base_url: str
    api_key: str | None
    struct_outputs_support: tuple[str, ...]


PROVIDERS: dict[APIProvider, APIProviderInfo] = {
    "openai": APIProviderInfo(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        struct_outputs_support=("*",),
    ),
    "openrouter": APIProviderInfo(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        struct_outputs_support=(),
    ),
    "google_ai_studio": APIProviderInfo(
        name="google_ai_studio",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        struct_outputs_support=("*",),
    ),
}


def retry_error_callback(retry_state: RetryCallState) -> None:
    assert retry_state.outcome is not None
    exception = retry_state.outcome.exception()
    if exception:
        if retry_state.attempt_number == 1:
            logger.error(
                f"CloudLLM completion request failed:\n{exception}",
                exc_info=exception,
            )
        if retry_state.attempt_number > 1:
            logger.warning(
                f"CloudLLM completion request failed after retrying:\n{exception}",
                exc_info=exception,
            )


def retry_before_callback(retry_state: RetryCallState) -> None:
    if retry_state.attempt_number > 1:
        logger.info(
            "Retrying CloudLLM completion request "
            f"(attempt {retry_state.attempt_number - 1}) ..."
        )


class CloudLLMSettings(LLMSettings, total=False):
    use_struct_outputs: bool


class CloudLLM(LLM[SettingsT_co, ConvertT_co], Generic[SettingsT_co, ConvertT_co]):
    def __init__(
        self,
        # Base LLM args
        model_name: str,
        converters: ConvertT_co,
        llm_settings: SettingsT_co | None = None,
        model_id: str | None = None,
        tools: list[BaseTool[BaseModel, Any, Any]] | None = None,
        response_format: type | Mapping[str, type] | None = None,
        # Connection settings
        async_http_client_params: (
            dict[str, Any] | AsyncHTTPClientParams | None
        ) = None,
        # Rate limiting
        rate_limiter: (RateLimiterC[Messages, AssistantMessage] | None) = None,
        rate_limiter_rpm: float | None = None,
        rate_limiter_chunk_size: int = 1000,
        rate_limiter_max_concurrency: int = 300,
        # Retries
        num_generation_retries: int = 0,
        # Disable tqdm for batch processing
        no_tqdm: bool = True,
        **kwargs: Any,
    ) -> None:
        self.llm_settings: CloudLLMSettings | None

        super().__init__(
            model_name=model_name,
            llm_settings=llm_settings,
            converters=converters,
            model_id=model_id,
            tools=tools,
            response_format=response_format,
            **kwargs,
        )

        self._model_name = model_name

        api_provider = model_name.split(":", 1)[0]
        api_model_name = model_name.split(":", 1)[-1]
        if api_provider not in PROVIDERS:
            raise ValueError(
                f"API provider '{api_provider}' is not supported. "
                f"Supported providers are: {', '.join(PROVIDERS.keys())}"
            )
        self._api_provider: APIProvider = api_provider
        self._api_model_name: str = api_model_name

        self._struct_outputs_support: bool = any(
            fnmatch.fnmatch(self._model_name, pat)
            for pat in PROVIDERS[api_provider]["struct_outputs_support"]
        )
        if (
            self._llm_settings.get("use_struct_outputs")
            and not self._struct_outputs_support
        ):
            raise ValueError(
                f"Model {self._model_name} does not support structured outputs."
            )

        self._tool_call_settings: dict[str, Any] = {}

        self._rate_limiter: RateLimiterC[Messages, AssistantMessage] | None = (
            self._get_rate_limiter(
                rate_limiter=rate_limiter,
                rpm=rate_limiter_rpm,
                chunk_size=rate_limiter_chunk_size,
                max_concurrency=rate_limiter_max_concurrency,
            )
        )
        self.no_tqdm = no_tqdm

        self._base_url: str = PROVIDERS[api_provider]["base_url"]
        self._api_key: str | None = PROVIDERS[api_provider]["api_key"]
        self._client: Any

        self._async_http_client: httpx.AsyncClient | None = None
        if async_http_client_params is not None:
            val_async_http_client_params = AsyncHTTPClientParams.model_validate(
                async_http_client_params
            )
            self._async_http_client = create_async_http_client(
                val_async_http_client_params
            )

        self.num_generation_retries = num_generation_retries

    @property
    def api_provider(self) -> APIProvider:
        return self._api_provider

    @property
    def rate_limiter(
        self,
    ) -> RateLimiterC[Messages, AssistantMessage] | None:
        return self._rate_limiter

    def _make_completion_kwargs(
        self,
        conversation: Messages,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
    ) -> dict[str, Any]:
        api_messages = [self._converters.to_message(m) for m in conversation]

        api_tools = None
        api_tool_choice = None
        if self.tools:
            api_tools = [
                self._converters.to_tool(t, **self._tool_call_settings)
                for t in self.tools.values()
            ]
            if tool_choice is not None:
                api_tool_choice = self._converters.to_tool_choice(tool_choice)

        api_llm_settings = deepcopy(self.llm_settings or {})
        api_llm_settings.pop("use_struct_outputs", None)

        return dict(
            api_messages=api_messages,
            api_tools=api_tools,
            api_tool_choice=api_tool_choice,
            api_response_format=self._response_format,
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
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        pass

    @abstractmethod
    async def _get_parsed_completion(
        self,
        api_messages: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_format: type | None = None,
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
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[Any]:
        pass

    @abstractmethod
    async def _get_parsed_completion_stream(
        self,
        api_messages: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_format: type | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[Any]:
        pass

    async def generate_completion_no_retry(
        self,
        conversation: Messages,
        *,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
    ) -> Completion:
        completion_kwargs = self._make_completion_kwargs(
            conversation=conversation, tool_choice=tool_choice, n_choices=n_choices
        )

        if not self._llm_settings.get("use_struct_outputs"):
            completion_kwargs.pop("api_response_format", None)
            api_completion = await self._get_completion(**completion_kwargs)
        else:
            api_completion = await self._get_parsed_completion(**completion_kwargs)

        completion = self._converters.from_completion(
            api_completion, name=self.model_id
        )

        if not self._llm_settings.get("use_struct_outputs"):
            # If validation is not handled by the structured output functionality
            # of the LLM provider
            self._validate_completion(completion)
            self._validate_tool_calls(completion)

        return completion

    async def generate_completion_stream(
        self,
        conversation: Messages,
        *,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
    ) -> AsyncIterator[CompletionChunkEvent | CompletionEvent]:
        completion_kwargs = self._make_completion_kwargs(
            conversation=conversation, tool_choice=tool_choice, n_choices=n_choices
        )

        if not self._llm_settings.get("use_struct_outputs"):
            completion_kwargs.pop("api_response_format", None)
            api_stream = await self._get_completion_stream(**completion_kwargs)
        else:
            api_stream = await self._get_parsed_completion_stream(**completion_kwargs)

        async def iterate() -> AsyncIterator[CompletionChunkEvent | CompletionEvent]:
            completion_chunks: list[CompletionChunk] = []
            async for api_completion_chunk in api_stream:
                completion_chunk = self._converters.from_completion_chunk(
                    api_completion_chunk, name=self.model_id
                )
                completion_chunks.append(completion_chunk)
                yield CompletionChunkEvent(data=completion_chunk, name=self.model_id)

            # TODO: can be done using the OpenAI final_completion_chunk
            completion = combine_completion_chunks(completion_chunks)

            yield CompletionEvent(data=completion, name=self.model_id)

            if not self._llm_settings.get("use_struct_outputs"):
                # If validation is not handled by the structured outputs functionality
                # of the LLM provider
                self._validate_completion(completion)
                self._validate_tool_calls(completion)

        return iterate()

    async def generate_completion(
        self,
        conversation: Messages,
        *,
        tool_choice: ToolChoice | None = None,
        n_choices: int | None = None,
    ) -> Completion:
        wrapped_func = retry(
            wait=wait_random_exponential(min=1, max=8),
            stop=stop_after_attempt(self.num_generation_retries + 1),
            before=retry_before_callback,
            retry_error_callback=retry_error_callback,
        )(self.__class__.generate_completion_no_retry)

        return await wrapped_func(
            self, conversation, tool_choice=tool_choice, n_choices=n_choices
        )

    @limit_rate_chunked  # type: ignore
    async def _generate_completion_batch(
        self,
        conversation: Messages,
        *,
        tool_choice: ToolChoice | None = None,
    ) -> Completion:
        return await self.generate_completion(conversation, tool_choice=tool_choice)

    async def generate_completion_batch(
        self, message_history: MessageHistory, *, tool_choice: ToolChoice | None = None
    ) -> Sequence[Completion]:
        return await self._generate_completion_batch(
            list(message_history.conversations),  # type: ignore
            tool_choice=tool_choice,
        )

    def _get_rate_limiter(
        self,
        rate_limiter: RateLimiterC[Messages, AssistantMessage] | None = None,
        rpm: float | None = None,
        chunk_size: int = 1000,
        max_concurrency: int = 300,
    ) -> RateLimiterC[Messages, AssistantMessage] | None:
        if rate_limiter is not None:
            logger.info(
                f"[{self.__class__.__name__}] Set rate limit to {rate_limiter.rpm} RPM"
            )
            return rate_limiter
        if rpm is not None:
            logger.info(f"[{self.__class__.__name__}] Set rate limit to {rpm} RPM")
            return RateLimiterC(
                rpm=rpm, chunk_size=chunk_size, max_concurrency=max_concurrency
            )

        return None
