import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import litellm
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,  # type: ignore[no-redef]
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.llms.anthropic import AnthropicThinkingParam
from litellm.utils import (
    supports_parallel_function_calling,
    supports_prompt_caching,
    supports_reasoning,
    supports_response_schema,
    supports_tool_choice,
)

# from openai.lib.streaming.chat import ChunkEvent as OpenAIChunkEvent
from pydantic import BaseModel

from ..cloud_llm import APIProvider, CloudLLM, LLMRateLimiter
from ..openai.openai_llm import OpenAILLMSettings
from ..typing.tool import BaseTool
from . import (
    LiteLLMCompletion,
    LiteLLMCompletionChunk,
    OpenAIMessageParam,
    OpenAIToolChoiceOptionParam,
    OpenAIToolParam,
)
from .converters import LiteLLMConverters

logger = logging.getLogger(__name__)


class LiteLLMSettings(OpenAILLMSettings, total=False):
    thinking: AnthropicThinkingParam | None


class LiteLLM(CloudLLM[LiteLLMSettings, LiteLLMConverters]):
    def __init__(
        self,
        # Base LLM args
        model_name: str,
        model_id: str | None = None,
        llm_settings: LiteLLMSettings | None = None,
        tools: list[BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        apply_response_schema_via_provider: bool = False,
        # LLM provider
        api_provider: APIProvider | None = None,
        # deployment_id: str | None = None,
        # api_version: str | None = None,
        # Connection settings
        timeout: float | None = None,
        max_client_retries: int = 2,
        # Rate limiting
        rate_limiter: LLMRateLimiter | None = None,
        # Drop unsupported LLM settings
        drop_params: bool = True,
        additional_drop_params: list[str] | None = None,
        allowed_openai_params: list[str] | None = None,
        # Mock LLM response for testing
        mock_response: str | None = None,
        # LLM response retries: try to regenerate to pass validation
        max_response_retries: int = 1,
    ) -> None:
        self._lite_llm_completion_params: dict[str, Any] = {
            "max_retries": max_client_retries,
            "timeout": timeout,
            "drop_params": drop_params,
            "additional_drop_params": additional_drop_params,
            "allowed_openai_params": allowed_openai_params,
            "mock_response": mock_response,
            # "deployment_id": deployment_id,
            # "api_version": api_version,
        }

        if model_name in litellm.get_valid_models():  # type: ignore[no-untyped-call]
            _, provider_name, _, _ = litellm.get_llm_provider(model_name)  # type: ignore[no-untyped-call]
            api_provider = APIProvider(name=provider_name)
        elif api_provider is not None:
            self._lite_llm_completion_params["api_key"] = api_provider.get("api_key")
            self._lite_llm_completion_params["api_base"] = api_provider.get("api_base")
        elif api_provider is None:
            raise ValueError(
                f"Model '{model_name}' is not supported by LiteLLM and no API provider "
                "was specified. Please provide a valid API provider or use a different "
                "model."
            )
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            llm_settings=llm_settings,
            converters=LiteLLMConverters(),
            tools=tools,
            response_schema=response_schema,
            response_schema_by_xml_tag=response_schema_by_xml_tag,
            apply_response_schema_via_provider=apply_response_schema_via_provider,
            api_provider=api_provider,
            rate_limiter=rate_limiter,
            max_client_retries=max_client_retries,
            max_response_retries=max_response_retries,
        )

        if self._apply_response_schema_via_provider:
            if self._tools:
                for tool in self._tools.values():
                    tool.strict = True
            if not self.supports_response_schema:
                raise ValueError(
                    f"Model '{self._model_name}' does not support response schema "
                    "natively. Please set `apply_response_schema_via_provider=False`"
                )

    def get_supported_openai_params(self) -> list[Any] | None:
        return get_supported_openai_params(  # type: ignore[no-untyped-call]
            model=self._model_name, request_type="chat_completion"
        )

    @property
    def supports_reasoning(self) -> bool:
        return supports_reasoning(model=self._model_name)

    @property
    def supports_parallel_function_calling(self) -> bool:
        return supports_parallel_function_calling(model=self._model_name)

    @property
    def supports_prompt_caching(self) -> bool:
        return supports_prompt_caching(model=self._model_name)

    @property
    def supports_response_schema(self) -> bool:
        return supports_response_schema(model=self._model_name)

    @property
    def supports_tool_choice(self) -> bool:
        return supports_tool_choice(model=self._model_name)

    # # client
    # model_list: Optional[list] = (None,)  # pass in a list of api_base,keys, etc.

    async def _get_completion(
        self,
        api_messages: list[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        api_response_schema: type | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> LiteLLMCompletion:
        completion = await litellm.acompletion(  # type: ignore[no-untyped-call]
            model=self._model_name,
            messages=api_messages,
            tools=api_tools,
            tool_choice=api_tool_choice,  # type: ignore[arg-type]
            response_format=api_response_schema,
            n=n_choices,
            stream=False,
            **self._lite_llm_completion_params,
            **api_llm_settings,
        )
        completion = cast("LiteLLMCompletion", completion)

        # Should not be needed in litellm>=1.74
        completion._hidden_params["response_cost"] = litellm.completion_cost(completion)  # type: ignore[no-untyped-call]

        return completion

    async def _get_completion_stream(  # type: ignore[no-untyped-def]
        self,
        api_messages: list[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        api_response_schema: type | None = None,
        n_choices: int | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[LiteLLMCompletionChunk]:
        stream = await litellm.acompletion(  # type: ignore[no-untyped-call]
            model=self._model_name,
            messages=api_messages,
            tools=api_tools,
            tool_choice=api_tool_choice,  # type: ignore[arg-type]
            response_format=api_response_schema,
            stream=True,
            n=n_choices,
            **self._lite_llm_completion_params,
            **api_llm_settings,
        )
        stream = cast("CustomStreamWrapper", stream)

        async for completion_chunk in stream:
            yield completion_chunk

    def combine_completion_chunks(
        self, completion_chunks: list[LiteLLMCompletionChunk]
    ) -> LiteLLMCompletion:
        combined_chunk = cast(
            "LiteLLMCompletion",
            litellm.stream_chunk_builder(completion_chunks),  # type: ignore[no-untyped-call]
        )
        # Should not be needed in litellm>=1.74
        combined_chunk._hidden_params["response_cost"] = litellm.completion_cost(  # type: ignore[no-untyped-call]
            combined_chunk
        )

        return combined_chunk
