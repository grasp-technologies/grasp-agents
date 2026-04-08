"""
Facade over LiteLLM's model metadata database.

Keeps the LiteLLM dependency contained — if LiteLLM changes its internals,
only this module needs updating.
"""

from dataclasses import dataclass

from litellm import get_model_info as _get_model_info
from litellm import (
    token_counter as _token_counter,  # pyright: ignore[reportUnknownVariableType]
)


@dataclass(frozen=True)
class ModelCapabilities:
    """Resolved capabilities for a specific model."""

    function_calling: bool
    vision: bool
    response_schema: bool
    prompt_caching: bool
    reasoning: bool
    web_search: bool
    audio_input: bool
    max_input_tokens: int | None
    max_output_tokens: int | None


_PERMISSIVE_DEFAULTS = ModelCapabilities(
    function_calling=True,
    vision=True,
    response_schema=True,
    prompt_caching=True,
    reasoning=True,
    web_search=True,
    audio_input=True,
    max_input_tokens=None,
    max_output_tokens=None,
)


def get_model_capabilities(
    model: str, provider: str | None = None
) -> ModelCapabilities:
    """
    Look up model capabilities from LiteLLM's database.

    Returns a frozen dataclass — cheap to cache, easy to test with.
    Falls back to permissive defaults for unknown models (all True, no limits).
    """
    try:
        info = _get_model_info(model, custom_llm_provider=provider)
    except Exception:
        return _PERMISSIVE_DEFAULTS

    return ModelCapabilities(
        function_calling=info.get("supports_function_calling") or False,
        vision=info.get("supports_vision") or False,
        response_schema=info.get("supports_response_schema") or False,
        prompt_caching=info.get("supports_prompt_caching") or False,
        reasoning=info.get("supports_reasoning") or False,
        web_search=info.get("supports_web_search") or False,
        audio_input=info.get("supports_audio_input") or False,
        max_input_tokens=info.get("max_input_tokens"),
        max_output_tokens=info.get("max_output_tokens"),
    )


def count_tokens(
    model: str, *, text: str | None = None, messages: list[dict[str, str]] | None = None
) -> int:
    """
    Count tokens for a model. Uses tiktoken for OpenAI, falls back for others.

    Returns 0 on failure (unknown model, missing tokenizer).
    """
    try:
        return _token_counter(model=model, text=text, messages=messages)  # type: ignore[no-any-return]
    except Exception:
        return 0


def get_context_window(model: str, provider: str | None = None) -> int | None:
    """Return max input tokens for a model, or None if unknown."""
    try:
        info = _get_model_info(model, custom_llm_provider=provider)
        return info.get("max_input_tokens")  # type: ignore[return-value]
    except Exception:
        return None
