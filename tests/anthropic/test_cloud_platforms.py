"""AnthropicLLM cloud-platform client construction (Bedrock / Vertex)."""

from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM

_BEDROCK_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"
_VERTEX_MODEL = "claude-sonnet-4-5@20250929"


def _capture(monkeypatch: pytest.MonkeyPatch, target: str) -> dict[str, Any]:
    """Patch a client class with a kwargs-capturing fake; return the kwargs."""
    captured: dict[str, Any] = {}

    class _Fake:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(target, _Fake)
    return captured


class TestBedrockConstruction:
    def test_uses_bedrock_client_with_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture(monkeypatch, "anthropic.AsyncAnthropicBedrock")
        llm = AnthropicLLM(
            model_name=_BEDROCK_MODEL,
            platform="bedrock",
            platform_config={"aws_region": "us-east-1", "aws_profile": "prod"},
        )
        assert captured["aws_region"] == "us-east-1"
        assert captured["aws_profile"] == "prod"
        # The framework retry layer is the one retry system → SDK retries off.
        assert captured["max_retries"] == 0
        assert captured["timeout"] == llm.anthropic_client_timeout

    def test_litellm_provider_defaults_to_bedrock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _capture(monkeypatch, "anthropic.AsyncAnthropicBedrock")
        llm = AnthropicLLM(model_name=_BEDROCK_MODEL, platform="bedrock")
        assert llm.litellm_provider == "bedrock"

    def test_bearer_token_via_platform_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture(monkeypatch, "anthropic.AsyncAnthropicBedrock")
        AnthropicLLM(
            model_name=_BEDROCK_MODEL,
            platform="bedrock",
            platform_config={"api_key": "bedrock-bearer-xyz"},
        )
        assert captured["api_key"] == "bedrock-bearer-xyz"

    def test_mantle_uses_mantle_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture(monkeypatch, "anthropic.AsyncAnthropicBedrockMantle")
        AnthropicLLM(
            model_name="anthropic.claude-opus-4-8",
            platform="bedrock_mantle",
            platform_config={"aws_region": "us-east-1"},
        )
        assert captured["aws_region"] == "us-east-1"

    def test_real_bedrock_client_type(self) -> None:
        from anthropic import AsyncAnthropicBedrock

        llm = AnthropicLLM(
            model_name=_BEDROCK_MODEL,
            platform="bedrock",
            platform_config={"aws_region": "us-east-1"},
        )
        assert isinstance(llm.client, AsyncAnthropicBedrock)


class TestVertexConstruction:
    def test_uses_vertex_client_with_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture(monkeypatch, "anthropic.AsyncAnthropicVertex")
        llm = AnthropicLLM(
            model_name=_VERTEX_MODEL,
            platform="vertex",
            platform_config={"project_id": "my-proj", "region": "us-east5"},
        )
        assert captured["project_id"] == "my-proj"
        assert captured["region"] == "us-east5"
        assert llm.litellm_provider == "vertex_ai"

    def test_real_vertex_client_type(self) -> None:
        from anthropic import AsyncAnthropicVertex

        llm = AnthropicLLM(
            model_name=_VERTEX_MODEL,
            platform="vertex",
            platform_config={"project_id": "p", "region": "us-east5"},
        )
        assert isinstance(llm.client, AsyncAnthropicVertex)


class TestClientArgForwarding:
    def test_http_client_headers_and_extra_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture(monkeypatch, "anthropic.AsyncAnthropicBedrock")
        http_client = Mock(spec=httpx.AsyncClient)
        AnthropicLLM(
            model_name=_BEDROCK_MODEL,
            platform="bedrock",
            platform_config={"aws_region": "us-east-1"},
            http_client=http_client,
            default_headers={"X-Test": "1"},
            extra_anthropic_client_params={"default_query": {"q": "1"}},
        )
        assert captured["http_client"] is http_client
        assert captured["default_headers"] == {"X-Test": "1"}
        assert captured["default_query"] == {"q": "1"}

    def test_direct_api_forwards_shared_client_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture(
            monkeypatch,
            "grasp_agents.llm_providers.anthropic.anthropic_llm.AsyncAnthropic",
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        http_client = Mock(spec=httpx.AsyncClient)
        AnthropicLLM(
            model_name="claude-sonnet-4-5",
            http_client=http_client,
            default_headers={"X-Test": "1"},
        )
        assert captured["http_client"] is http_client
        assert captured["default_headers"] == {"X-Test": "1"}

    def test_secrets_kept_out_of_repr(self) -> None:
        llm = AnthropicLLM(
            model_name=_BEDROCK_MODEL,
            platform="bedrock",
            platform_config={"aws_secret_key": "SECRET", "aws_region": "us-east-1"},
        )
        assert "SECRET" not in repr(llm)
