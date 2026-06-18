"""Map Anthropic SDK exceptions to typed LlmError values."""

from __future__ import annotations

import anthropic
import httpx

from grasp_agents.llm_providers.anthropic.error_mapping import map_api_error
from grasp_agents.types.llm_errors import (
    LlmApiConnectionError,
    LlmApiStatusError,
    LlmApiTimeoutError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmContextWindowError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmRateLimitError,
)

_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(code: int, headers: dict[str, str] | None = None) -> Exception:
    response = httpx.Response(code, request=_REQUEST, headers=headers or {})
    return anthropic.APIStatusError("boom", response=response, body=None)


class TestAnthropicErrorMapping:
    def test_timeout_maps_to_timeout(self) -> None:
        err = anthropic.APITimeoutError(request=_REQUEST)
        assert isinstance(map_api_error(err), LlmApiTimeoutError)

    def test_connection_maps_to_connection(self) -> None:
        err = anthropic.APIConnectionError(message="down", request=_REQUEST)
        assert isinstance(map_api_error(err), LlmApiConnectionError)

    def test_rate_limit_maps_and_parses_retry_after(self) -> None:
        mapped = map_api_error(_status_error(429, {"retry-after": "12"}))
        assert isinstance(mapped, LlmRateLimitError)
        assert mapped.retry_after == 12.0

    def test_auth_codes_map_to_authentication(self) -> None:
        assert isinstance(map_api_error(_status_error(401)), LlmAuthenticationError)
        assert isinstance(map_api_error(_status_error(403)), LlmAuthenticationError)

    def test_not_found_maps(self) -> None:
        assert isinstance(map_api_error(_status_error(404)), LlmNotFoundError)

    def test_413_maps_to_context_window(self) -> None:
        assert isinstance(map_api_error(_status_error(413)), LlmContextWindowError)

    def test_server_error_maps_to_internal(self) -> None:
        assert isinstance(map_api_error(_status_error(503)), LlmInternalServerError)

    def test_bad_request_maps(self) -> None:
        assert isinstance(map_api_error(_status_error(400)), LlmBadRequestError)

    def test_other_status_maps_to_api_status(self) -> None:
        assert isinstance(map_api_error(_status_error(418)), LlmApiStatusError)

    def test_non_anthropic_error_returns_none(self) -> None:
        assert map_api_error(ValueError("nope")) is None
