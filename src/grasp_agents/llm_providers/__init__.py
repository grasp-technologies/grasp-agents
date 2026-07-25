"""
Concrete LLM providers, one subpackage each. The headline classes are
re-exported here lazily, so::

    from grasp_agents.llm_providers import AnthropicLLM, GeminiLLM, OpenAILLM

works without eagerly importing providers whose optional extras aren't
installed — a provider's dependencies load only when its class is accessed.

* :mod:`.openai_responses` — ``OpenAIResponsesLLM`` (OpenAI Responses API)
* :mod:`.openai_completions` — ``OpenAILLM`` (Chat Completions; also Gemini /
  OpenRouter OpenAI-compatible endpoints)
* :mod:`.anthropic` — ``AnthropicLLM`` (needs the ``anthropic`` extra)
* :mod:`.gemini` — ``GeminiLLM`` (needs the ``gemini`` extra)
* :mod:`.litellm` — ``LiteLLM`` (long-tail providers via ``litellm``)
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .anthropic import (
        AnthropicCloudPlatform,
        AnthropicLLM,
        AnthropicLLMSettings,
        BedrockClientConfig,
        VertexClientConfig,
    )
    from .gemini import (
        GeminiCloudPlatform,
        GeminiLLM,
        GeminiLLMSettings,
        GeminiVertexClientConfig,
    )
    from .litellm import LiteLLM, LiteLLMSettings
    from .openai_completions import (
        AzureClientConfig,
        OpenAICloudPlatform,
        OpenAILLM,
        OpenAILLMSettings,
    )
    from .openai_responses import OpenAIResponsesLLM, OpenAIResponsesLLMSettings

_SUBMODULE_BY_NAME: dict[str, str] = {
    "AnthropicLLM": "anthropic",
    "AnthropicLLMSettings": "anthropic",
    "AnthropicCloudPlatform": "anthropic",
    "BedrockClientConfig": "anthropic",
    "VertexClientConfig": "anthropic",
    "GeminiLLM": "gemini",
    "GeminiLLMSettings": "gemini",
    "GeminiCloudPlatform": "gemini",
    "GeminiVertexClientConfig": "gemini",
    "LiteLLM": "litellm",
    "LiteLLMSettings": "litellm",
    "AzureClientConfig": "openai_completions",
    "OpenAICloudPlatform": "openai_completions",
    "OpenAILLM": "openai_completions",
    "OpenAILLMSettings": "openai_completions",
    "OpenAIResponsesLLM": "openai_responses",
    "OpenAIResponsesLLMSettings": "openai_responses",
}

_EXTRA_BY_SUBMODULE: dict[str, str] = {
    "anthropic": "anthropic",
    "gemini": "gemini",
}


def __getattr__(name: str) -> Any:
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(f".{submodule}", __name__)
    except ModuleNotFoundError as err:
        extra = _EXTRA_BY_SUBMODULE.get(submodule)
        if extra is not None:
            raise ImportError(
                f"{name} requires the '{extra}' extra: "
                f'pip install "grasp_agents[{extra}]"'
            ) from err
        raise
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULE_BY_NAME))


__all__ = [
    "AnthropicCloudPlatform",
    "AnthropicLLM",
    "AnthropicLLMSettings",
    "AzureClientConfig",
    "BedrockClientConfig",
    "GeminiCloudPlatform",
    "GeminiLLM",
    "GeminiLLMSettings",
    "GeminiVertexClientConfig",
    "LiteLLM",
    "LiteLLMSettings",
    "OpenAICloudPlatform",
    "OpenAILLM",
    "OpenAILLMSettings",
    "OpenAIResponsesLLM",
    "OpenAIResponsesLLMSettings",
    "VertexClientConfig",
]
