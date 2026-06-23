"""Azure OpenAI client construction for OpenAILLM (Chat Completions)."""

from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm_providers.openai_completions.completions_llm import OpenAILLM

_AZURE = (
    "grasp_agents.llm_providers.openai_completions.completions_llm.AsyncAzureOpenAI"
)


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _Fake:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(_AZURE, _Fake)
    return captured


class TestAzureCompletions:
    def test_builds_azure_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture(monkeypatch)
        llm = OpenAILLM(
            model_name="my-deployment",
            platform="azure",
            platform_config={
                "azure_endpoint": "https://x.openai.azure.com",
                "api_version": "2024-10-21",
            },
        )
        assert captured["azure_endpoint"] == "https://x.openai.azure.com"
        assert captured["api_version"] == "2024-10-21"
        # The Azure deployment name must be preserved (NOT prefix-split).
        assert llm.model_name == "my-deployment"
        assert llm.litellm_provider == "azure"

    def test_ad_token_provider_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture(monkeypatch)

        def provider() -> str:
            return "token"

        OpenAILLM(
            model_name="dep",
            platform="azure",
            platform_config={
                "azure_endpoint": "https://x",
                "api_version": "2024-10-21",
                "azure_ad_token_provider": provider,
            },
        )
        assert captured["azure_ad_token_provider"] is provider

    def test_shared_client_args_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture(monkeypatch)
        http_client = Mock(spec=httpx.AsyncClient)
        OpenAILLM(
            model_name="dep",
            platform="azure",
            platform_config={"azure_endpoint": "https://x", "api_version": "v"},
            http_client=http_client,
            default_headers={"X-T": "1"},
            extra_openai_client_params={"organization": "org"},
        )
        assert captured["http_client"] is http_client
        assert captured["default_headers"] == {"X-T": "1"}
        assert captured["organization"] == "org"

    def test_real_azure_client_type(self) -> None:
        from openai import AsyncAzureOpenAI

        llm = OpenAILLM(
            model_name="dep",
            platform="azure",
            platform_config={
                "azure_endpoint": "https://x.openai.azure.com",
                "api_version": "2024-10-21",
                "api_key": "sk-azure",
            },
        )
        assert isinstance(llm.client, AsyncAzureOpenAI)

    def test_secrets_kept_out_of_repr(self) -> None:
        llm = OpenAILLM(
            model_name="dep",
            platform="azure",
            platform_config={
                "azure_endpoint": "https://x",
                "api_version": "v",
                "api_key": "SECRET",
            },
        )
        assert "SECRET" not in repr(llm)

    def test_non_azure_path_unaffected(self) -> None:
        from openai import AsyncAzureOpenAI, AsyncOpenAI

        llm = OpenAILLM(
            model_name="gpt-4o",
            api_provider=APIProvider(name="openai", base_url=None, api_key="sk-x"),
        )
        assert isinstance(llm.client, AsyncOpenAI)
        assert not isinstance(llm.client, AsyncAzureOpenAI)
