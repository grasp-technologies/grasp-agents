"""
Endpoint selection shared by every CloudLLM provider: which client is built
(`platform`), where it points (`api_provider`), and which LiteLLM provider the
result is priced under.
"""

from typing import Any

import pytest

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM
from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM
from grasp_agents.llm_providers.litellm.lite_llm import LiteLLM
from grasp_agents.llm_providers.openai_completions.completions_llm import OpenAILLM
from grasp_agents.llm_providers.openai_responses.responses_llm import OpenAIResponsesLLM

_GEMINI_CLIENT = "grasp_agents.llm_providers.gemini.gemini_llm.Client"

pytestmark = pytest.mark.usefixtures("_dummy_keys")


@pytest.fixture
def _dummy_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-dummy")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _fake_gemini_client(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _Fake:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(_GEMINI_CLIENT, _Fake)
    return captured


# ---------- platform validation ----------


class TestPlatformValidation:
    def test_unsupported_platform_rejected(self) -> None:
        # A platform with no dedicated client must not silently fall through to
        # the vendor's own endpoint — that would route traffic to the wrong
        # vendor while looking configured.
        with pytest.raises(ValueError, match="no client for platform 'deepinfra'"):
            AnthropicLLM(model_name="claude-sonnet-4-5", platform="deepinfra")  # type: ignore[arg-type]

    def test_unsupported_platform_message_lists_supported(self) -> None:
        with pytest.raises(ValueError, match="'vertex'"):
            GeminiLLM(model_name="gemini-2.5-flash", platform="azure")  # type: ignore[arg-type]

    def test_provider_without_cloud_platforms_reports_none(self) -> None:
        with pytest.raises(ValueError, match="supported: none"):
            LiteLLM(model_name="openai/gpt-5.1", platform="azure")

    def test_api_provider_rejected_on_cloud_platform(self) -> None:
        # The Bedrock/Vertex/Azure clients take their own credentials through
        # platform_config and never read an api_provider, so accepting both
        # would silently drop the key.
        with pytest.raises(ValueError, match="never reads"):
            AnthropicLLM(
                model_name="anthropic.claude-sonnet-4-5-20250929-v1:0",
                platform="bedrock",
                api_provider=APIProvider(
                    name="bedrock", base_url="https://example.test", api_key="k"
                ),
            )

    def test_platform_config_without_platform_rejected(self) -> None:
        with pytest.raises(ValueError, match="no platform is selected"):
            OpenAILLM(
                model_name="gpt-5.1",
                platform_config={"azure_endpoint": "https://example.test"},
            )

    def test_api_provider_without_base_url_is_credentials_only(self) -> None:
        # The vendor's own endpoint with an explicitly supplied key.
        llm = AnthropicLLM(
            model_name="claude-sonnet-4-5",
            api_provider=APIProvider(
                name="anthropic", base_url=None, api_key="sk-explicit"
            ),
        )
        assert llm.client.api_key == "sk-explicit"
        assert llm.litellm_provider == "anthropic"


# ---------- LiteLLM pricing identity ----------


class TestNativeLitellmProvider:
    def test_anthropic(self) -> None:
        llm = AnthropicLLM(model_name="claude-sonnet-4-5")
        assert llm.litellm_provider == "anthropic"

    def test_openai_completions(self) -> None:
        assert OpenAILLM(model_name="gpt-5.1").litellm_provider == "openai"

    def test_openai_responses(self) -> None:
        assert OpenAIResponsesLLM(model_name="gpt-5.1").litellm_provider == "openai"

    def test_gemini_developer_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The Developer API prices under "gemini", not "vertex_ai".
        _fake_gemini_client(monkeypatch)
        assert GeminiLLM(model_name="gemini-2.5-flash").litellm_provider == "gemini"


class TestCloudPlatformLitellmProvider:
    @pytest.mark.parametrize(
        ("platform", "expected"),
        [
            ("bedrock", "bedrock"),
            ("bedrock_mantle", "bedrock"),
            ("vertex", "vertex_ai"),
        ],
    )
    def test_anthropic_platforms(
        self, platform: Any, expected: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = {
            "bedrock": "anthropic.AsyncAnthropicBedrock",
            "bedrock_mantle": "anthropic.AsyncAnthropicBedrockMantle",
            "vertex": "anthropic.AsyncAnthropicVertex",
        }[platform]

        class _Fake:
            def __init__(self, **kwargs: Any) -> None:
                del kwargs

        monkeypatch.setattr(target, _Fake)
        llm = AnthropicLLM(
            model_name="claude-sonnet-4-5",
            platform=platform,
            platform_config={"aws_region": "us-east-1", "region": "us-east5"},
        )
        assert llm.litellm_provider == expected

    def test_azure_completions(self) -> None:
        llm = OpenAILLM(
            model_name="my-deployment",
            platform="azure",
            platform_config={
                "azure_endpoint": "https://r.openai.azure.com",
                "api_version": "2024-10-21",
                "api_key": "az-key",
            },
        )
        assert llm.litellm_provider == "azure"

    def test_azure_responses(self) -> None:
        llm = OpenAIResponsesLLM(
            model_name="my-deployment",
            platform="azure",
            platform_config={
                "azure_endpoint": "https://r.openai.azure.com",
                "api_version": "2025-03-01-preview",
                "api_key": "az-key",
            },
        )
        assert llm.litellm_provider == "azure"

    def test_gemini_vertex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_gemini_client(monkeypatch)
        llm = GeminiLLM(
            model_name="gemini-2.5-flash",
            platform="vertex",
            platform_config={"project": "p"},
        )
        assert llm.litellm_provider == "vertex_ai"


class TestCompatibleEndpointLitellmProvider:
    def test_named_endpoint_prices_under_its_own_name(self) -> None:
        # An OpenAI-compatible endpoint has its own cost table: pricing must
        # follow the endpoint, not the wire protocol.
        llm = OpenAILLM(
            model_name="google/gemma-4-31b-it",
            api_provider=APIProvider(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="or-key",
            ),
        )
        assert llm.litellm_provider == "openrouter"

    def test_responses_named_endpoint(self) -> None:
        llm = OpenAIResponsesLLM(
            model_name="some-model",
            api_provider=APIProvider(
                name="deepinfra", base_url="https://api.deepinfra.test/v1", api_key="k"
            ),
        )
        assert llm.litellm_provider == "deepinfra"

    def test_explicit_litellm_provider_wins(self) -> None:
        llm = OpenAILLM(
            model_name="gpt-5.1",
            litellm_provider="openai",
            api_provider=APIProvider(
                name="my-gateway", base_url="https://gw.test/v1", api_key="k"
            ),
        )
        assert llm.litellm_provider == "openai"


# ---------- the vendor's own endpoint ----------


class TestDefaultApiProvider:
    def test_openai_key_from_env(self) -> None:
        llm = OpenAILLM(model_name="gpt-5.1")
        assert llm.api_provider == {
            "name": "openai",
            "base_url": None,
            "api_key": "sk-openai-dummy",
        }
        assert llm.client.api_key == "sk-openai-dummy"

    def test_anthropic_key_from_env(self) -> None:
        llm = AnthropicLLM(model_name="claude-sonnet-4-5")
        assert llm.api_provider is not None
        assert llm.api_provider["api_key"] == "sk-ant-dummy"

    def test_gemini_falls_back_to_second_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-dummy")
        captured = _fake_gemini_client(monkeypatch)
        llm = GeminiLLM(model_name="gemini-2.5-flash")
        assert captured["api_key"] == "gemini-dummy"
        assert llm.api_provider is not None
        assert llm.api_provider["api_key"] == "gemini-dummy"

    def test_credentials_kept_out_of_repr(self) -> None:
        llm = OpenAILLM(
            model_name="gpt-5.1",
            api_provider=APIProvider(
                name="custom", base_url="https://gw.test/v1", api_key="sk-secret-xyz"
            ),
        )
        assert "sk-secret-xyz" not in repr(llm)


class TestCompatibleEndpointClient:
    def test_base_url_and_key_threaded_into_openai_client(self) -> None:
        llm = OpenAILLM(
            model_name="some-model",
            api_provider=APIProvider(
                name="deepinfra", base_url="https://api.deepinfra.test/v1", api_key="di"
            ),
        )
        assert str(llm.client.base_url).startswith("https://api.deepinfra.test/v1")
        assert llm.client.api_key == "di"

    def test_base_url_and_key_threaded_into_anthropic_client(self) -> None:
        llm = AnthropicLLM(
            model_name="some-model",
            api_provider=APIProvider(
                name="deepinfra", base_url="https://api.deepinfra.test/v1", api_key="di"
            ),
        )
        assert str(llm.client.base_url).startswith("https://api.deepinfra.test/v1")
        assert llm.client.api_key == "di"

    def test_base_url_threaded_into_gemini_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _fake_gemini_client(monkeypatch)
        GeminiLLM(
            model_name="some-model",
            api_provider=APIProvider(
                name="deepinfra", base_url="https://api.deepinfra.test/v1", api_key="di"
            ),
        )
        assert captured["api_key"] == "di"
        assert captured["http_options"].base_url == "https://api.deepinfra.test/v1"
