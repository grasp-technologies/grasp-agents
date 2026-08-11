"""Map OpenAI SDK exceptions to typed LlmError values."""

from __future__ import annotations

import httpx
import openai
from openai.types.chat import ChatCompletion

from grasp_agents.llm_providers.openai_completions.error_mapping import map_api_error
from grasp_agents.types.errors import CompletionError
from grasp_agents.types.llm_errors import (
    LlmApiConnectionError,
    LlmApiError,
    LlmApiStatusError,
    LlmApiTimeoutError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmContentFilterError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmQuotaExceededError,
    LlmRateLimitError,
)

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

_QUOTA_BODY = {
    "message": "You exceeded your current quota",
    "type": "insufficient_quota",
    "code": "insufficient_quota",
    "param": None,
}


def _status_error(
    code: int,
    headers: dict[str, str] | None = None,
    body: object | None = None,
) -> Exception:
    response = httpx.Response(code, request=_REQUEST, headers=headers or {})
    return openai.APIStatusError("boom", response=response, body=body)


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


class TestOpenAIErrorsWithoutStatus:
    """
    Errors the SDK raises without an HTTP status must still be typed: the
    retry and fallback layers act only on ``LlmErrorTuple``, and a raw
    ``openai.APIError`` is that tuple's *parent*, so leaving it unmapped
    makes it skip both.
    """

    def test_bare_api_error_maps_to_api_error(self) -> None:
        # How openai/_streaming.py reports an error frame inside a 200 SSE
        # stream — the shape that used to bypass retry and fallback entirely.
        err = openai.APIError(message="stream blew up", request=_REQUEST, body=None)
        assert isinstance(map_api_error(err), LlmApiError)

    def test_response_validation_error_maps_to_api_error(self) -> None:
        err = openai.APIResponseValidationError(
            response=httpx.Response(200, request=_REQUEST), body=None
        )
        assert isinstance(map_api_error(err), LlmApiError)

    def test_content_filter_finish_reason_maps(self) -> None:
        assert isinstance(
            map_api_error(openai.ContentFilterFinishReasonError()),
            LlmContentFilterError,
        )

    def test_length_finish_reason_maps_to_bad_request(self) -> None:
        completion = ChatCompletion(
            id="c1", choices=[], created=0, model="gpt-5", object="chat.completion"
        )
        err = openai.LengthFinishReasonError(completion=completion)
        assert isinstance(map_api_error(err), LlmBadRequestError)


class TestOpenAIQuotaDetection:
    """
    A spent account must be told apart from an ordinary 429: it never
    clears, so it has to fail over rather than burn the retry budget.
    """

    def test_streamed_quota_maps_to_quota_exceeded(self) -> None:
        # Quota exhaustion on a streamed call arrives as a bare APIError,
        # because the transport status is 200.
        err = openai.APIError(
            message="You exceeded your current quota",
            request=_REQUEST,
            body=_QUOTA_BODY,
        )
        assert isinstance(map_api_error(err), LlmQuotaExceededError)

    def test_status_quota_maps_to_quota_exceeded(self) -> None:
        mapped = map_api_error(_status_error(429, body=_QUOTA_BODY))
        assert isinstance(mapped, LlmQuotaExceededError)

    def test_quota_detected_regardless_of_status_code(self) -> None:
        # Gateways disagree on the code they return for a spent account.
        assert isinstance(
            map_api_error(_status_error(403, body=_QUOTA_BODY)), LlmQuotaExceededError
        )

    def test_plain_rate_limit_is_not_quota(self) -> None:
        mapped = map_api_error(
            _status_error(429, body={"type": "requests", "code": "rate_limit_exceeded"})
        )
        assert isinstance(mapped, LlmRateLimitError)
        assert not isinstance(mapped, LlmQuotaExceededError)


class TestOpenAIContentFilterDetection:
    """
    One policy block reaches the SDK in several shapes. All of them must
    map to the same type, or the same block behaves differently depending
    on whether the call was streamed.
    """

    def test_streamed_block_maps_to_content_filter(self) -> None:
        # An error frame inside a 200 SSE stream: no status, and the body
        # carries no code — only the message identifies the block.
        err = openai.APIError(
            message=(
                "This content was flagged for possible cybersecurity risk. "
                "If this seems wrong, try rephrasing your request."
            ),
            request=_REQUEST,
            body=None,
        )
        mapped = map_api_error(err)
        assert isinstance(mapped, LlmContentFilterError)
        assert not isinstance(mapped, LlmApiError)

    def test_invalid_prompt_status_maps_to_content_filter(self) -> None:
        # The non-streamed shape of the same block: a 400 that the generic
        # mapping would bury as a programmer bug.
        mapped = map_api_error(
            _status_error(400, body={"code": "invalid_prompt", "type": None})
        )
        assert isinstance(mapped, LlmContentFilterError)
        assert mapped.code == "invalid_prompt"

    def test_azure_content_management_policy_maps(self) -> None:
        response = httpx.Response(400, request=_REQUEST)
        err = openai.APIStatusError(
            "The response was filtered due to the prompt triggering Azure "
            "OpenAI's content management policy.",
            response=response,
            body=None,
        )
        assert isinstance(map_api_error(err), LlmContentFilterError)

    def test_block_message_carries_the_provider_explanation(self) -> None:
        err = openai.APIError(
            message="This content was flagged. Join the cyber program.",
            request=_REQUEST,
            body=None,
        )
        assert "Join the cyber program." in str(map_api_error(err))

    def test_rate_limit_quoting_a_policy_is_not_a_block(self) -> None:
        # The message is only trusted on codes a block can arrive under.
        mapped = map_api_error(
            openai.APIStatusError(
                "Rate limit reached; see our usage policy.",
                response=httpx.Response(429, request=_REQUEST),
                body=None,
            )
        )
        assert isinstance(mapped, LlmRateLimitError)
        assert not isinstance(mapped, LlmContentFilterError)

    def test_plain_bad_request_is_not_a_block(self) -> None:
        assert not isinstance(map_api_error(_status_error(400)), LlmContentFilterError)
