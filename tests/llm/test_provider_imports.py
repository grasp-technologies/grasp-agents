"""Provider LLM classes are re-exported lazily at grasp_agents.llm_providers."""

import subprocess  # noqa: S404
import sys
from importlib import import_module

import pytest

REEXPORTED_NAMES = [
    "AnthropicLLM",
    "AnthropicLLMSettings",
    "AnthropicPlatform",
    "AzureClientConfig",
    "BedrockClientConfig",
    "GeminiLLM",
    "GeminiLLMSettings",
    "GeminiPlatform",
    "GeminiVertexClientConfig",
    "LiteLLM",
    "LiteLLMSettings",
    "OpenAILLM",
    "OpenAILLMSettings",
    "OpenAIResponsesLLM",
    "OpenAIResponsesLLMSettings",
    "VertexClientConfig",
]


@pytest.mark.parametrize("name", REEXPORTED_NAMES)
def test_llm_providers_reexports(name: str) -> None:
    providers = import_module("grasp_agents.llm_providers")
    assert getattr(providers, name) is not None
    assert name in providers.__all__
    assert name in dir(providers)


def test_reexports_are_the_provider_classes() -> None:
    providers = import_module("grasp_agents.llm_providers")
    from grasp_agents.llm_providers.anthropic import AnthropicLLM
    from grasp_agents.llm_providers.openai_responses import OpenAIResponsesLLM

    assert providers.AnthropicLLM is AnthropicLLM
    assert providers.OpenAIResponsesLLM is OpenAIResponsesLLM


def test_unknown_name_raises_attribute_error() -> None:
    providers = import_module("grasp_agents.llm_providers")
    with pytest.raises(AttributeError):
        _ = providers.DoesNotExist


def test_provider_packages_not_imported_eagerly() -> None:
    # A fresh interpreter, so this test cannot disturb (or be disturbed by)
    # module identity in the running suite.
    script = (
        "import sys\n"
        "import grasp_agents.llm_providers\n"
        "for sub in ('anthropic', 'gemini', 'litellm'):\n"
        "    assert f'grasp_agents.llm_providers.{sub}' not in sys.modules, sub\n"
        "from grasp_agents.llm_providers import AnthropicLLM\n"
        "assert 'grasp_agents.llm_providers.anthropic' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
