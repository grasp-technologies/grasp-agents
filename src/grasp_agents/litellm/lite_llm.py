import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import litellm
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,  # type: ignore[no-redef]
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.router import Router
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

from ..cloud_llm import APIProvider, CloudLLM
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


@dataclass(frozen=True)
class LiteLLM(CloudLLM):
    llm_settings: LiteLLMSettings | None = None
    converters: ClassVar[LiteLLMConverters] = LiteLLMConverters()

    client_timeout: float = 60.0
    max_client_retries: int = 2

    # Drop unsupported OpenAI params
    drop_params: bool = False
    additional_drop_params: list[str] | None = None
    allowed_openai_params: list[str] | None = None
    # Mock LLM response for testing
    mock_response: str | None = None

    router: Router = field(init=False)

    _lite_llm_completion_params: dict[str, Any] = field(
        default_factory=dict[str, Any], init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        self._lite_llm_completion_params.update(
            {
                "drop_params": self.drop_params,
                "additional_drop_params": self.additional_drop_params,
                "allowed_openai_params": self.allowed_openai_params,
                "mock_response": self.mock_response,
                # "max_retries": self.max_client_retries,
                # "timeout": self.client_timeout,
                # "deployment_id": deployment_id,
                # "api_version": api_version,
            }
        )

        _api_provider = self.api_provider
        try:
            _, provider_name, api_key, api_base = litellm.get_llm_provider(  # type: ignore[no-untyped-call]
                self.model_name
            )
            _api_provider = APIProvider(
                name=provider_name, api_key=api_key, base_url=api_base
            )
        except Exception as exc:
            if self.api_provider is not None:
                self._lite_llm_completion_params["api_key"] = self.api_provider.get(
                    "api_key"
                )
                self._lite_llm_completion_params["api_base"] = self.api_provider.get(
                    "api_base"
                )
            else:
                raise ValueError(
                    f"Failed to retrieve a LiteLLM supported API provider for model "
                    f"'{self.model_name}' and no custom API provider was specified."
                ) from exc

        if (
            self.apply_response_schema_via_provider
            and not self.supports_response_schema
        ):
            raise ValueError(
                f"Model '{self.model_name}' does not support response schema "
                "natively. Please set `apply_response_schema_via_provider=False`"
            )

        if self.http_client is not None:
            raise NotImplementedError(
                "Custom HTTP clients are not yet supported when using LiteLLM."
            )

        _router = Router(
            model_list=[
                {
                    "model_name": self.model_name,
                    "litellm_params": {"model": self.model_name},
                }
            ],
            num_retries=self.max_client_retries,
            timeout=self.client_timeout,
        )

        object.__setattr__(self, "router", _router)
        object.__setattr__(self, "api_provider", _api_provider)

    def get_supported_openai_params(self) -> list[Any] | None:
        return get_supported_openai_params(  # type: ignore[no-untyped-call]
            model=self.model_name, request_type="chat_completion"
        )

    @property
    def supports_reasoning(self) -> bool:
        return supports_reasoning(model=self.model_name)

    @property
    def supports_parallel_function_calling(self) -> bool:
        return supports_parallel_function_calling(model=self.model_name)

    @property
    def supports_prompt_caching(self) -> bool:
        return supports_prompt_caching(model=self.model_name)

    @property
    def supports_response_schema(self) -> bool:
        return supports_response_schema(model=self.model_name)

    @property
    def supports_tool_choice(self) -> bool:
        return supports_tool_choice(model=self.model_name)

    async def _get_api_completion(
        self,
        api_messages: list[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        api_response_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> LiteLLMCompletion:
        completion = await self.router.acompletion(  # type: ignore[no-untyped-call]
            model=self.model_name,
            messages=api_messages,  # type: ignore[arg-type]
            tools=api_tools,
            tool_choice=api_tool_choice,  # type: ignore[arg-type]
            response_format=api_response_schema,
            stream=False,
            **self._lite_llm_completion_params,
            **api_llm_settings,
        )
        completion = cast("LiteLLMCompletion", completion)

        # Should not be needed in litellm>=1.74
        completion._hidden_params["response_cost"] = litellm.completion_cost(completion)  # type: ignore[no-untyped-call]

        return completion

    async def _get_api_completion_stream(
        self,
        api_messages: list[OpenAIMessageParam],
        api_tools: list[OpenAIToolParam] | None = None,
        api_tool_choice: OpenAIToolChoiceOptionParam | None = None,
        api_response_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[LiteLLMCompletionChunk]:
        # Ensure usage is included in the streamed responses
        stream_options = dict(api_llm_settings.get("stream_options") or {})
        stream_options["include_usage"] = True
        _api_llm_settings = api_llm_settings | {"stream_options": stream_options}

        stream = await self.router.acompletion(  # type: ignore[no-untyped-call]
            model=self.model_name,
            messages=api_messages,  # type: ignore[arg-type]
            tools=api_tools,
            tool_choice=api_tool_choice,  # type: ignore[arg-type]
            response_format=api_response_schema,
            stream=True,
            **self._lite_llm_completion_params,
            **_api_llm_settings,
        )
        stream = cast("CustomStreamWrapper", stream)

        tc_indices: dict[int, set[int]] = defaultdict(set)

        # Need to wrap the iterator to make it work with decorators
        async def iterator() -> AsyncIterator[LiteLLMCompletionChunk]:
            async for completion_chunk in stream:
                # Fix tool call indices to be unique within each choice
                if completion_chunk is not None:
                    for n, choice in enumerate(completion_chunk.choices):
                        for tc in choice.delta.tool_calls or []:
                            # Tool call ID is not None only when it is a new tool call
                            if tc.id and tc.index in tc_indices[n]:
                                tc.index = max(tc_indices[n]) + 1
                            tc_indices[n].add(tc.index)

                    yield completion_chunk

        return iterator()

    def combine_completion_chunks(
        self,
        completion_chunks: list[LiteLLMCompletionChunk],
        response_schema: Any | None = None,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
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
