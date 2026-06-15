"""Map OpenAI SDK exceptions to typed LlmError values."""

from __future__ import annotations

import httpx
import openai

from grasp_agents.llm_providers.openai_completions.error_mapping import map_api_error
from grasp_agents.types.errors import CompletionError
from grasp_agents.types.llm_errors import (
    LlmApiConnectionError,
    LlmApiStatusError,
    LlmApiTimeoutError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmRateLimitError,
)

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(code: int, headers: dict[str, str] | None = None) -> Exception:
    response = httpx.Response(code, request=_REQUEST, headers=headers or {})
    return openai.APIStatusError("boom", response=response, body=None)


class TestOpenAICompletionsErrorMapping:
    def test_in_body_completion_error_maps_to_retryable_internal(self) -> None:
        # A 200 response carrying an error body must reach retry/fallback.
        mapped = map_api_error(CompletionError("upstream failed"))
        assert isinstance(mapped, LlmInternalServerError)

    def test_timeout_maps_to_timeout(self) -> None:
        err = openai.APITimeoutError(request=_REQUEST)
        assert isinstance(map_api_error(err), LlmApiTimeoutError)

    def test_connection_maps_to_connection(self) -> None:
        err = openai.APIConnectionError(message="down", request=_REQUEST)
        assert isinstance(map_api_error(err), LlmApiConnectionError)

    def test_rate_limit_maps_and_parses_retry_after(self) -> None:
        mapped = map_api_error(_status_error(429, {"retry-after": "7"}))
        assert isinstance(mapped, LlmRateLimitError)
        assert mapped.retry_after == 7.0

    def test_auth_codes_map_to_authentication(self) -> None:
        assert isinstance(map_api_error(_status_error(401)), LlmAuthenticationError)
        assert isinstance(map_api_error(_status_error(403)), LlmAuthenticationError)

    def test_not_found_maps(self) -> None:
        assert isinstance(map_api_error(_status_error(404)), LlmNotFoundError)

    def test_server_error_maps_to_internal(self) -> None:
        assert isinstance(map_api_error(_status_error(500)), LlmInternalServerError)

    def test_bad_request_codes_map(self) -> None:
        assert isinstance(map_api_error(_status_error(400)), LlmBadRequestError)
        assert isinstance(map_api_error(_status_error(422)), LlmBadRequestError)

    def test_other_status_maps_to_api_status(self) -> None:
        assert isinstance(map_api_error(_status_error(418)), LlmApiStatusError)

    def test_non_openai_error_returns_none(self) -> None:
        assert map_api_error(ValueError("nope")) is None
