"""GeminiLLM client construction: Developer API (default) and Vertex AI."""

from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

_CLIENT = "grasp_agents.llm_providers.gemini.gemini_llm.Client"


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _Fake:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(_CLIENT, _Fake)
    return captured


class TestGeminiDeveloperApi:
    def test_default_platform_is_developer_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture(monkeypatch)
        GeminiLLM(
            model_name="gemini-2.5-flash",
            api_provider=APIProvider(name="google", base_url=None, api_key="dummy"),
        )
        assert captured["api_key"] == "dummy"
        assert "vertexai" not in captured

    def test_shared_client_args_on_http_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture(monkeypatch)
        http_client = Mock(spec=httpx.AsyncClient)
        llm = GeminiLLM(
            model_name="gemini-2.5-flash",
            api_provider=APIProvider(name="google", base_url=None, api_key="d"),
            http_client=http_client,
            default_headers={"X-T": "1"},
        )
        http_options = captured["http_options"]
        assert http_options.httpx_async_client is http_client
        assert http_options.headers == {"X-T": "1"}
        assert http_options.timeout == int((llm.gemini_client_timeout or 0) * 1000)

    def test_extra_client_params_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture(monkeypatch)
        GeminiLLM(
            model_name="gemini-2.5-flash",
            api_provider=APIProvider(name="google", base_url=None, api_key="d"),
            extra_gemini_client_params={"debug_config": "X"},
        )
        assert captured["debug_config"] == "X"


class TestGeminiVertex:
    def test_vertex_config_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture(monkeypatch)
        creds = object()
        GeminiLLM(
            model_name="gemini-2.5-flash",
            platform="vertex",
            platform_config={
                "project": "p",
                "location": "us-east5",
                "credentials": creds,
            },
        )
        assert captured["vertexai"] is True
        assert captured["project"] == "p"
        assert captured["location"] == "us-east5"
        assert captured["credentials"] is creds

    def test_vertex_location_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture(monkeypatch)
        GeminiLLM(
            model_name="gemini-2.5-flash",
            platform="vertex",
            platform_config={"project": "p"},
        )
        assert captured["location"] == "us-central1"


class TestGeminiRequestHeaders:
    def test_request_headers_merge_client_default_headers(self) -> None:
        # A per-request extra_headers must not drop the client-level
        # default_headers — they are merged onto the request http_options.
        llm = GeminiLLM(
            model_name="gemini-2.5-flash",
            api_provider=APIProvider(name="google", base_url=None, api_key="d"),
            default_headers={"X-Client": "c"},
            llm_settings={"extra_headers": {"X-Req": "r"}},
        )
        params = llm._make_api_input([])
        config = params["extra_settings"]["config"]  # type: ignore[index]
        assert config.http_options.headers == {"X-Client": "c", "X-Req": "r"}
