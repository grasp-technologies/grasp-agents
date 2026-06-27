"""
Facade over LiteLLM's model metadata database.

Keeps the LiteLLM dependency contained — if LiteLLM changes its internals,
only this module needs updating.
"""

import logging
from dataclasses import dataclass
from typing import Any

from litellm import get_model_info as _get_model_info
from litellm import (
    token_counter as _token_counter,  # pyright: ignore[reportUnknownVariableType]
)

logger = logging.getLogger(__name__)

_warned_unresolved: set[tuple[str, str]] = set()


def _warn_unresolved(what: str, model: str) -> None:
    """Warn (once per model) that litellm lacks metadata and a fallback is used."""
    key = (what, model)
    if key not in _warned_unresolved:
        _warned_unresolved.add(key)
        logger.warning(
            "litellm could not resolve %s for model %r; falling back", what, model
        )


@dataclass(frozen=True)
class ModelCapabilities:
    """Resolved capabilities for a specific model."""

    function_calling: bool
    vision: bool
    output_schema: bool
    prompt_caching: bool
    reasoning: bool
    web_search: bool
    audio_input: bool
    max_input_tokens: int | None
    max_output_tokens: int | None


_PERMISSIVE_DEFAULTS = ModelCapabilities(
    function_calling=True,
    vision=True,
    output_schema=True,
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
        _warn_unresolved("model capabilities", model)
        return _PERMISSIVE_DEFAULTS

    return ModelCapabilities(
        function_calling=info.get("supports_function_calling") or False,
        vision=info.get("supports_vision") or False,
        output_schema=info.get("supports_response_schema") or False,
        prompt_caching=info.get("supports_prompt_caching") or False,
        reasoning=info.get("supports_reasoning") or False,
        web_search=info.get("supports_web_search") or False,
        audio_input=info.get("supports_audio_input") or False,
        max_input_tokens=info.get("max_input_tokens"),
        max_output_tokens=info.get("max_output_tokens"),
    )


def count_tokens(
    model: str,
    *,
    text: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> int:
    """
    Count tokens for a model. Uses tiktoken for OpenAI, falls back for others.

    ``messages`` may carry multimodal content (image parts); images are counted
    with a default per-image cost, never fetched. Returns 0 on failure (unknown
    model, missing tokenizer).
    """
    try:
        return _token_counter(  # type: ignore[no-any-return]
            model=model,
            text=text,
            messages=messages,
            use_default_image_token_count=True,
        )
    except Exception:
        _warn_unresolved("a tokenizer", model)
        return 0


def get_context_window(model: str, provider: str | None = None) -> int | None:
    """Return max input tokens for a model, or None if unknown."""
    try:
        info = _get_model_info(model, custom_llm_provider=provider)
        return info.get("max_input_tokens")  # type: ignore[return-value]
    except Exception:
        _warn_unresolved("the context window", model)
        return None
