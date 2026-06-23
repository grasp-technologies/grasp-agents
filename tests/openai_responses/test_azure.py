"""Azure OpenAI client construction for OpenAIResponsesLLM (Responses API)."""

from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm_providers.openai_responses.responses_llm import (
    OpenAIResponsesLLM,
)

_AZURE = "grasp_agents.llm_providers.openai_responses.responses_llm.AsyncAzureOpenAI"


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _Fake:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(_AZURE, _Fake)
    return captured


class TestAzureResponses:
    def test_builds_azure_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture(monkeypatch)
        # The Responses API on Azure needs a preview/v1 api_version.
        llm = OpenAIResponsesLLM(
            model_name="my-deployment",
            platform="azure",
            platform_config={
                "azure_endpoint": "https://x.openai.azure.com",
                "api_version": "2025-03-01-preview",
            },
        )
        assert captured["azure_endpoint"] == "https://x.openai.azure.com"
        assert captured["api_version"] == "2025-03-01-preview"
        assert llm.model_name == "my-deployment"
        assert llm.litellm_provider == "azure"

    def test_shared_client_args_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture(monkeypatch)
        http_client = Mock(spec=httpx.AsyncClient)
        OpenAIResponsesLLM(
            model_name="dep",
            platform="azure",
            platform_config={"azure_endpoint": "https://x", "api_version": "v"},
            http_client=http_client,
            default_headers={"X-T": "1"},
        )
        assert captured["http_client"] is http_client
        assert captured["default_headers"] == {"X-T": "1"}

    def test_real_azure_client_type(self) -> None:
        from openai import AsyncAzureOpenAI

        llm = OpenAIResponsesLLM(
            model_name="dep",
            platform="azure",
            platform_config={
                "azure_endpoint": "https://x.openai.azure.com",
                "api_version": "2025-03-01-preview",
                "api_key": "sk-azure",
            },
        )
        assert isinstance(llm.client, AsyncAzureOpenAI)

    def test_non_azure_path_unaffected(self) -> None:
        from openai import AsyncAzureOpenAI, AsyncOpenAI

        llm = OpenAIResponsesLLM(
            model_name="gpt-4o",
            api_provider=APIProvider(name="openai", base_url=None, api_key="sk-x"),
        )
        assert isinstance(llm.client, AsyncOpenAI)
        assert not isinstance(llm.client, AsyncAzureOpenAI)
