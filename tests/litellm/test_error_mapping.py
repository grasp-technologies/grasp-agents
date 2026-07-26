"""Map LiteLLM exceptions to typed LlmError values."""

from __future__ import annotations

import httpx
import litellm
import openai

from grasp_agents.llm_providers.litellm.error_mapping import map_api_error
from grasp_agents.types.llm_errors import (
    LlmApiError,
    LlmApiStatusError,
    LlmQuotaExceededError,
    LlmRateLimitError,
)

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


class TestLiteLLMErrorsWithoutStatus:
    """
    Every litellm exception derives from the OpenAI SDK's, so the generic
    fallthrough must type the long tail the named branches do not cover —
    anything left untyped skips both retry and fallback.
    """

    def test_bare_api_error_maps_to_api_error(self) -> None:
        err = openai.APIError(message="stream blew up", request=_REQUEST, body=None)
        assert isinstance(map_api_error(err), LlmApiError)

    def test_unlisted_status_maps_to_api_status(self) -> None:
        err = openai.APIStatusError(
            "teapot", response=httpx.Response(418, request=_REQUEST), body=None
        )
        assert isinstance(map_api_error(err), LlmApiStatusError)

    def test_non_openai_error_returns_none(self) -> None:
        assert map_api_error(ValueError("nope")) is None


class TestLiteLLMQuotaDetection:
    def test_budget_exceeded_maps_to_quota_exceeded(self) -> None:
        err = litellm.BudgetExceededError(current_cost=10.0, max_budget=5.0)
        assert isinstance(map_api_error(err), LlmQuotaExceededError)

    def test_insufficient_quota_maps_to_quota_exceeded(self) -> None:
        # LiteLLM rewrites every exception with a fixed code/type and drops
        # the upstream body, so only the message carries the signal.
        err = litellm.RateLimitError(
            "You exceeded your current quota, please check your plan",
            llm_provider="openai",
            model="gpt-5",
            response=httpx.Response(429, request=_REQUEST),
        )
        assert isinstance(map_api_error(err), LlmQuotaExceededError)

    def test_plain_rate_limit_stays_retryable(self) -> None:
        err = litellm.RateLimitError(
            "slow down",
            llm_provider="openai",
            model="gpt-5",
            response=httpx.Response(429, request=_REQUEST),
        )
        mapped = map_api_error(err)
        assert isinstance(mapped, LlmRateLimitError)
        assert not isinstance(mapped, LlmQuotaExceededError)
