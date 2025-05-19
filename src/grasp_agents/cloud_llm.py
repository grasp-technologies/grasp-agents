import fnmatch
import logging
import os
from abc import abstractmethod
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from typing import Any, Generic, Literal

import httpx
from pydantic import BaseModel, TypeAdapter
from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing_extensions import TypedDict

from .http_client import AsyncHTTPClientParams, create_async_http_client
from .llm import LLM, ConvertT, LLMSettings, SettingsT
from .memory import MessageHistory
from .rate_limiting.rate_limiter_chunked import (  # type: ignore
    RateLimiterC,
    limit_rate_chunked,
)
from .typing.completion import Completion, CompletionChunk
from .typing.message import AssistantMessage, Conversation
from .typing.tool import BaseTool, ToolChoice
from .utils import validate_obj_from_json_or_py_string

logger = logging.getLogger(__name__)


APIProvider = Literal["openai", "openrouter", "google_ai_studio"]


class APIProviderInfo(TypedDict):
    name: APIProvider
    base_url: str
    api_key: str | None
    struct_output_support: tuple[str, ...]


PROVIDERS: dict[APIProvider, APIProviderInfo] = {
    "openai": APIProviderInfo(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        struct_output_support=("*",),
    ),
    "openrouter": APIProviderInfo(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        struct_output_support=(),
    ),
    "google_ai_studio": APIProviderInfo(
        name="google_ai_studio",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        struct_output_support=("*",),
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
    max_completion_tokens: int | None
    temperature: float | None
    top_p: float | None
    seed: int | None
    use_structured_outputs: bool


class CloudLLM(LLM[SettingsT, ConvertT], Generic[SettingsT, ConvertT]):
    def __init__(
        self,
        # Base LLM args
        model_name: str,
        converters: ConvertT,
        llm_settings: SettingsT | None = None,
        model_id: str | None = None,
        tools: list[BaseTool[BaseModel, Any, Any]] | None = None,
        response_format: type | None = None,
        # Connection settings
        api_provider: APIProvider = "openai",
        async_http_client_params: (
            dict[str, Any] | AsyncHTTPClientParams | None
        ) = None,
        # Rate limiting
        rate_limiter: (RateLimiterC[Conversation, AssistantMessage] | None) = None,
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
        self._api_provider: APIProvider = api_provider

        self._struct_output_support: bool = any(
            fnmatch.fnmatch(self._model_name, pat)
            for pat in PROVIDERS[api_provider]["struct_output_support"]
        )
        self._response_format_pyd: TypeAdapter[Any] | None = (
            TypeAdapter(self._response_format) if response_format else None
        )
        if (
            self._llm_settings.get("use_structured_outputs")
            and not self._struct_output_support
        ):
            raise ValueError(
                f"Model {api_provider}:{self._model_name} does "
                "not support structured outputs."
            )

        self._rate_limiter: RateLimiterC[Conversation, AssistantMessage] | None = (
            self._get_rate_limiter(
                rate_limiter=rate_limiter,
                rate_limiter_rpm=rate_limiter_rpm,
                rate_limiter_chunk_size=rate_limiter_chunk_size,
                rate_limiter_max_concurrency=rate_limiter_max_concurrency,
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
    ) -> RateLimiterC[Conversation, AssistantMessage] | None:
        return self._rate_limiter

    def _make_completion_kwargs(
        self, conversation: Conversation, tool_choice: ToolChoice | None = None
    ) -> dict[str, Any]:
        api_messages = [self._converters.to_message(m) for m in conversation]

        api_tools = None
        api_tool_choice = None
        if self.tools:
            api_tools = [self._converters.to_tool(t) for t in self.tools.values()]
            if tool_choice is not None:
                api_tool_choice = self._converters.to_tool_choice(tool_choice)

        api_llm_settings = deepcopy(self.llm_settings or {})
        api_llm_settings.pop("use_structured_outputs", None)

        return dict(
            api_messages=api_messages,
            api_tools=api_tools,
            api_tool_choice=api_tool_choice,
            api_response_format=self._response_format,
            **api_llm_settings,
        )

    @abstractmethod
    async def _get_completion(
        self,
        api_messages: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
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
        **api_llm_settings: Any,
    ) -> AsyncIterator[Any]:
        pass

    async def generate_completion(
        self,
        conversation: Conversation,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> Completion:
        completion_kwargs = self._make_completion_kwargs(
            conversation=conversation, tool_choice=tool_choice
        )

        if (
            self._response_format is None
            or (not self._struct_output_support)
            or (not self._llm_settings.get("use_structured_outputs"))
        ):
            completion_kwargs.pop("api_response_format", None)
            api_completion = await self._get_completion(**completion_kwargs, **kwargs)
        else:
            api_completion = await self._get_parsed_completion(
                **completion_kwargs, **kwargs
            )

        completion = self._converters.from_completion(
            api_completion, model_id=self.model_id
        )

        self._validate_completion(completion)

        return completion

    def _validate_completion(self, completion: Completion) -> None:
        for choice in completion.choices:
            message = choice.message
            if (
                self._response_format_pyd is not None
                and not self._llm_settings.get("use_structured_outputs")
                and not message.tool_calls
            ):
                validate_obj_from_json_or_py_string(
                    message.content,
                    adapter=self._response_format_pyd,
                    from_substring=True,
                )

    async def generate_completion_stream(
        self,
        conversation: Conversation,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        completion_kwargs = self._make_completion_kwargs(
            conversation=conversation, tool_choice=tool_choice
        )
        completion_kwargs.pop("api_response_format", None)
        api_completion_chunk_iterator = await self._get_completion_stream(
            **completion_kwargs, **kwargs
        )

        return self._converters.from_completion_chunk_iterator(
            api_completion_chunk_iterator, model_id=self.model_id
        )

    async def _generate_completion_with_retry(
        self,
        conversation: Conversation,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> Completion:
        wrapped_func = retry(
            wait=wait_random_exponential(min=1, max=8),
            stop=stop_after_attempt(self.num_generation_retries + 1),
            before=retry_before_callback,
            retry_error_callback=retry_error_callback,
        )(self.__class__.generate_completion)

        return await wrapped_func(self, conversation, tool_choice=tool_choice, **kwargs)

    @limit_rate_chunked  # type: ignore
    async def _generate_completion_batch_with_retry_and_rate_lim(
        self,
        conversation: Conversation,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> Completion:
        return await self._generate_completion_with_retry(
            conversation, tool_choice=tool_choice, **kwargs
        )

    async def generate_completion_batch(
        self,
        message_history: MessageHistory,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> Sequence[Completion]:
        return await self._generate_completion_batch_with_retry_and_rate_lim(
            list(message_history.batched_conversations),  # type: ignore
            tool_choice=tool_choice,
            **kwargs,
        )

    async def generate_message(
        self,
        conversation: Conversation,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        completion = await self.generate_completion(
            conversation, tool_choice=tool_choice, **kwargs
        )

        return completion.choices[0].message

    async def generate_message_batch(
        self,
        message_history: MessageHistory,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> Sequence[AssistantMessage]:
        completion_batch = await self.generate_completion_batch(
            message_history, tool_choice=tool_choice, **kwargs
        )

        return [completion.choices[0].message for completion in completion_batch]

    def _get_rate_limiter(
        self,
        rate_limiter: RateLimiterC[Conversation, AssistantMessage] | None = None,
        rate_limiter_rpm: float | None = None,
        rate_limiter_chunk_size: int = 1000,
        rate_limiter_max_concurrency: int = 300,
    ) -> RateLimiterC[Conversation, AssistantMessage] | None:
        if rate_limiter is not None:
            logger.info(
                f"[{self.__class__.__name__}] Set rate limit to {rate_limiter.rpm} RPM"
            )
            return rate_limiter
        if rate_limiter_rpm is not None:
            logger.info(
                f"[{self.__class__.__name__}] Set rate limit to {rate_limiter_rpm} RPM"
            )
            return RateLimiterC(
                rpm=rate_limiter_rpm,
                chunk_size=rate_limiter_chunk_size,
                max_concurrency=rate_limiter_max_concurrency,
            )

        return None
