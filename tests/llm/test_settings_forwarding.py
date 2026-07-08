"""
Newly-exposed provider settings actually reach the API call kwargs.

Each provider forwards ``llm_settings`` verbatim (no conversion); these assert a
representative sample of the expanded settings lands in the API request that
``_make_api_input`` builds — i.e. nothing is silently dropped.
"""

from typing import cast

import pytest

from grasp_agents.llm.cloud_llm import APIProvider

_GOOGLE = APIProvider(name="google", base_url=None, api_key="dummy")
_OPENAI = APIProvider(name="openai", base_url=None, api_key="sk-x")


def test_anthropic_forwards_new_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM

    llm = AnthropicLLM(
        model_name="claude-sonnet-4-5",
        llm_settings={
            "service_tier": "auto",
            "metadata": {"user_id": "u1"},
            "inference_geo": "us",
            "output_config": {"effort": "low"},
        },
    )
    extra = llm._make_api_input([]).get("extra_settings", {})
    assert extra["service_tier"] == "auto"
    assert extra["metadata"] == {"user_id": "u1"}
    assert extra["inference_geo"] == "us"
    assert extra["output_config"] == {"effort": "low"}


def test_openai_completions_forwards_new_settings() -> None:
    from grasp_agents.llm_providers.openai_completions.completions_llm import (
        OpenAILLM,
        OpenAILLMSettings,
    )

    # ``not_yet_declared`` is not in the settings TypedDict — undeclared keys
    # must still reach the API untouched.
    llm = OpenAILLM(
        model_name="gpt-4o",
        api_provider=_OPENAI,
        llm_settings=cast(
            "OpenAILLMSettings",
            {"n": 2, "service_tier": "flex", "not_yet_declared": "kept"},
        ),
    )
    extra = llm._make_api_input([]).get("extra_settings", {})
    assert extra["n"] == 2
    assert extra["service_tier"] == "flex"
    assert extra["not_yet_declared"] == "kept"


def test_openai_responses_forwards_new_settings() -> None:
    from grasp_agents.llm_providers.openai_responses.responses_llm import (
        OpenAIResponsesLLM,
    )

    llm = OpenAIResponsesLLM(
        model_name="gpt-4o",
        api_provider=_OPENAI,
        llm_settings={
            "truncation": "auto",
            "max_tool_calls": 3,
            "service_tier": "flex",
        },
    )
    extra = llm._make_api_input([]).get("extra_settings", {})
    assert extra["truncation"] == "auto"
    assert extra["max_tool_calls"] == 3
    assert extra["service_tier"] == "flex"


def test_gemini_forwards_new_settings() -> None:
    from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

    llm = GeminiLLM(
        model_name="gemini-2.5-flash",
        api_provider=_GOOGLE,
        llm_settings={"enable_enhanced_civic_answers": False, "labels": {"team": "x"}},
    )
    config = llm._make_api_input([])["extra_settings"]["config"]  # type: ignore[index]
    assert config.enable_enhanced_civic_answers is False
    assert config.labels == {"team": "x"}
