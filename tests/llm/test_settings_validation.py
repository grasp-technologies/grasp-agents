"""
Provider settings TypedDicts are a pydantic-validatable contract: annotations
resolve at runtime, declared keys are type-checked, and undeclared keys are
PRESERVED (pass through to the provider — never silently dropped). ``CloudLLM``
validates at construction.
"""

from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from grasp_agents.llm.cloud_llm import CloudLLMSettings
from grasp_agents.llm.llm import LLMSettings
from grasp_agents.llm_providers.anthropic.anthropic_llm import (
    AnthropicLLM,
    AnthropicLLMSettings,
)
from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLMSettings
from grasp_agents.llm_providers.litellm.lite_llm import LiteLLM, LiteLLMSettings
from grasp_agents.llm_providers.openai_completions.completions_llm import (
    OpenAILLM,
    OpenAILLMSettings,
)
from grasp_agents.llm_providers.openai_responses.responses_llm import (
    OpenAIResponsesLLMSettings,
)

ALL_SETTINGS_TYPES: list[Any] = [
    LLMSettings,
    CloudLLMSettings,
    AnthropicLLMSettings,
    GeminiLLMSettings,
    OpenAILLMSettings,
    OpenAIResponsesLLMSettings,
    LiteLLMSettings,
]

VALID_SETTINGS: list[tuple[Any, dict[str, Any]]] = [
    (LLMSettings, {"temperature": 1.0, "top_p": 0.9}),
    (CloudLLMSettings, {"temperature": 1.0, "extra_headers": {"x-key": "v"}}),
    (
        AnthropicLLMSettings,
        {
            "temperature": 1.0,
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        },
    ),
    (
        GeminiLLMSettings,
        {
            "temperature": 1.0,
            "thinking_config": {"include_thoughts": True, "thinking_budget": 8192},
        },
    ),
    (OpenAILLMSettings, {"temperature": 1.0, "reasoning_effort": "medium"}),
    (
        OpenAIResponsesLLMSettings,
        {"temperature": 1.0, "reasoning": {"effort": "medium"}},
    ),
    (LiteLLMSettings, {"temperature": 1.0, "reasoning_effort": "disable"}),
]


@pytest.mark.parametrize(
    ("settings_type", "settings"),
    VALID_SETTINGS,
    ids=lambda v: v.__name__ if isinstance(v, type) else None,
)
def test_settings_validate_and_preserve_keys(
    settings_type: Any, settings: dict[str, Any]
) -> None:
    validated = TypeAdapter(settings_type).validate_python(settings)
    assert set(validated) == set(settings)


@pytest.mark.parametrize("settings_type", ALL_SETTINGS_TYPES, ids=lambda v: v.__name__)
def test_undeclared_keys_are_preserved_not_dropped(settings_type: Any) -> None:
    settings = {"temperature": 1.0, "brand_new_provider_param": {"x": 1}}
    assert TypeAdapter(settings_type).validate_python(settings) == settings


@pytest.mark.parametrize("settings_type", ALL_SETTINGS_TYPES, ids=lambda v: v.__name__)
def test_all_settings_keys_are_optional(settings_type: Any) -> None:
    assert TypeAdapter(settings_type).validate_python({}) == {}


@pytest.mark.parametrize("settings_type", ALL_SETTINGS_TYPES, ids=lambda v: v.__name__)
def test_declared_key_type_error_is_rejected(settings_type: Any) -> None:
    with pytest.raises(ValidationError, match="temperature"):
        TypeAdapter(settings_type).validate_python({"temperature": "hot"})


def test_settings_enforce_nested_required_keys() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AnthropicLLMSettings).validate_python(
            {"thinking": {"type": "enabled"}}  # missing required budget_tokens
        )


def test_llm_construction_rejects_mistyped_declared_key() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        OpenAILLM(
            model_name="openai/gpt-5.1",
            llm_settings=cast("OpenAILLMSettings", {"temperature": "hot"}),
        )


def test_llm_construction_accepts_valid_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    llm = AnthropicLLM(
        model_name="claude-sonnet-4-5",
        llm_settings=AnthropicLLMSettings(
            temperature=1.0, thinking={"type": "enabled", "budget_tokens": 4096}
        ),
    )
    assert llm.llm_settings is not None


def test_llm_construction_passes_undeclared_settings_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    llm = LiteLLM(
        model_name="openai/gpt-5.1",
        llm_settings=cast("Any", {"provider_specific_param": 1}),
    )
    assert llm.llm_settings == {"provider_specific_param": 1}
