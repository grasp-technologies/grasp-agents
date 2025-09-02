import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
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
class LiteLLM(CloudLLM[LiteLLMSettings, LiteLLMConverters]):
    converters: LiteLLMConverters = field(default_factory=LiteLLMConverters)

    timeout: float | None = None
    # Drop unsupported LLM settings
    drop_params: bool = True
    additional_drop_params: list[str] | None = None
    allowed_openai_params: list[str] | None = None
    # Mock LLM response for testing
    mock_response: str | None = None

    _lite_llm_completion_params: dict[str, Any] = field(
        default_factory=dict[str, Any], init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        self._lite_llm_completion_params.update(
            {
                "max_retries": self.max_client_retries,
                "timeout": self.timeout,
                "drop_params": self.drop_params,
                "additional_drop_params": self.additional_drop_params,
                "allowed_openai_params": self.allowed_openai_params,
                "mock_response": self.mock_response,
                # "deployment_id": deployment_id,
                # "api_version": api_version,
            }
        )

        _api_provider = self.api_provider
        try:
            _, provider_name, _, _ = litellm.get_llm_provider(self.model_name)  # type: ignore[no-untyped-call]
            _api_provider = APIProvider(name=provider_name)
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

        if self.llm_settings is not None:
            stream_options = self.llm_settings.get("stream_options") or {}
            stream_options["include_usage"] = True
            _llm_settings = deepcopy(self.llm_settings)
            _llm_settings["stream_options"] = stream_options
        else:
            _llm_settings = LiteLLMSettings(stream_options={"include_usage": True})

        if (
            self.apply_response_schema_via_provider
            and not self.supports_response_schema
        ):
            raise ValueError(
                f"Model '{self.model_name}' does not support response schema "
                "natively. Please set `apply_response_schema_via_provider=False`"
            )

        object.__setattr__(self, "api_provider", _api_provider)
        object.__setattr__(self, "llm_settings", _llm_settings)

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
        if api_llm_settings and api_llm_settings.get("stream_options"):
            api_llm_settings.pop("stream_options")

        completion = await litellm.acompletion(  # type: ignore[no-untyped-call]
            model=self.model_name,
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
            model=self.model_name,
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

        tc_indices: dict[int, set[int]] = defaultdict(set)

        async for completion_chunk in stream:
            # Fix tool call indices to be unique within each choice
            for n, choice in enumerate(completion_chunk.choices):
                for tc in choice.delta.tool_calls or []:
                    # Tool call ID is not None only when it is a new tool call
                    if tc.id and tc.index in tc_indices[n]:
                        tc.index = max(tc_indices[n]) + 1
                    tc_indices[n].add(tc.index)

            yield completion_chunk

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
