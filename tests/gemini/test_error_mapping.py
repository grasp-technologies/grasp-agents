"""
Tests for Gemini error mapping.

Regression focus: mapping must never crash while constructing the mapped
error. The mapped ``Llm*`` types are ``openai.APIStatusError`` subclasses
whose ``__init__`` reads ``response.request`` — a synthesized httpx.Response
without a request used to raise ``RuntimeError`` inside ``_map_api_error``,
turning every Gemini API error into an unclassified crash that bypassed
retry and fallback. The SDK's preferred aiohttp transport (used whenever
aiohttp is importable) supplies a non-httpx response on essentially every
install, forcing the synthesis path.
"""

from __future__ import annotations

import httpx
from google.genai import errors as genai_errors

from grasp_agents.llm_providers.gemini.error_mapping import map_api_error
from grasp_agents.types.llm_errors import (
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmRateLimitError,
)

_QUOTA_JSON = {"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}}


class _FakeAiohttpResponse:
    """Duck-typed stand-in for aiohttp.ClientResponse (headers only)."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


class TestGeminiErrorMapping:
    def test_client_error_without_response_maps_without_crash(self):
        err = genai_errors.ClientError(429, _QUOTA_JSON)

        mapped = map_api_error(err)

        assert isinstance(mapped, LlmRateLimitError)
        assert mapped.status_code == 429
        assert mapped.response.request is not None

    def test_client_error_with_aiohttp_response_maps_and_parses_retry_after(self):
        err = genai_errors.ClientError(
            429,
            _QUOTA_JSON,
            response=_FakeAiohttpResponse({"retry-after": "7"}),  # type: ignore[arg-type]
        )

        mapped = map_api_error(err)

        assert isinstance(mapped, LlmRateLimitError)
        assert mapped.retry_after == 7.0

    def test_client_error_with_requestless_httpx_response_maps(self):
        err = genai_errors.ClientError(
            401,
            {"error": {"message": "bad key", "status": "UNAUTHENTICATED"}},
            response=httpx.Response(status_code=401),
        )

        mapped = map_api_error(err)

        assert isinstance(mapped, LlmAuthenticationError)
        assert mapped.response.request is not None

    def test_client_error_with_full_httpx_response_preserved(self):
        response = httpx.Response(
            status_code=429,
            headers={"retry-after": "11"},
            request=httpx.Request("POST", "https://real.example"),
        )
        err = genai_errors.ClientError(429, _QUOTA_JSON, response=response)

        mapped = map_api_error(err)

        assert isinstance(mapped, LlmRateLimitError)
        assert mapped.retry_after == 11.0
        assert mapped.response is response

    def test_server_error_maps_to_internal_server_error(self):
        err = genai_errors.ServerError(
            503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}}
        )

        mapped = map_api_error(err)

        assert isinstance(mapped, LlmInternalServerError)

    def test_not_found_and_bad_request_map(self):
        not_found = genai_errors.ClientError(
            404, {"error": {"message": "no such model", "status": "NOT_FOUND"}}
        )
        bad_request = genai_errors.ClientError(
            400, {"error": {"message": "bad arg", "status": "INVALID_ARGUMENT"}}
        )

        assert isinstance(map_api_error(not_found), LlmNotFoundError)
        assert isinstance(map_api_error(bad_request), LlmBadRequestError)

    def test_non_genai_error_returns_none(self):
        assert map_api_error(ValueError("not an API error")) is None
