"""
Tests for the shared ``parse_retry_after`` helper.

The helper backs ``Retry-After`` extraction in every provider's
``error_mapping.py`` (OpenAI Completions, OpenAI Responses, Anthropic,
Gemini, LiteLLM), so these tests lock in the parser contract once.
"""

import httpx
import pytest

from grasp_agents.llm_providers._http_helpers import parse_retry_after  # noqa: PLC2701


def _response_with_header(value: str | None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.example.test/v1/chat")
    headers = {"retry-after": value} if value is not None else {}
    return httpx.Response(status_code=429, request=request, headers=headers)


class TestParseRetryAfter:
    def test_integer_seconds(self) -> None:
        assert parse_retry_after(_response_with_header("30")) == 30.0

    def test_fractional_seconds(self) -> None:
        assert parse_retry_after(_response_with_header("1.5")) == 1.5

    def test_missing_header_returns_none(self) -> None:
        assert parse_retry_after(_response_with_header(None)) is None

    def test_http_date_form_returns_none(self) -> None:
        """RFC 7231 allows an HTTP-date form; helper doesn't support it yet."""
        assert (
            parse_retry_after(_response_with_header("Wed, 21 Oct 2026 07:28:00 GMT"))
            is None
        )

    def test_non_numeric_garbage_returns_none(self) -> None:
        assert parse_retry_after(_response_with_header("not-a-number")) is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_retry_after(_response_with_header("")) is None

    @pytest.mark.parametrize("value", ["0", "0.0"])
    def test_zero_is_returned_not_treated_as_falsy(self, value: str) -> None:
        """parse_retry_after returns the zero — the *caller* decides what to do."""
        assert parse_retry_after(_response_with_header(value)) == 0.0
