"""
LLM core / resilience behavior:

* server ``Retry-After`` hints are sanitized at parse and capped at retry time
* Anthropic usage is cache-normalized (no $0 cache writes, no negative cost)
* a streamed terminal ``response.failed`` raises a typed, retryable error
* ``CompletionError`` (200-with-error-body) maps to a retryable LlmError
* the resolved API key is not visible in ``repr()``
* FallbackLLM stamps cost with the serving member's identity
* Anthropic's default ``max_tokens`` respects the model's output cap
* Gemini requests carry an HTTP timeout
* LiteLLM honors an explicit ``api_provider`` (credentials + base URL)
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anthropic.types as anthropic_types
import httpx
import pytest

from grasp_agents.llm.fallback_llm import FallbackLLM
from grasp_agents.llm.model_info import get_model_capabilities
from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.llm_providers._http_helpers import (
    parse_retry_after,
)
from grasp_agents.llm_providers.anthropic.anthropic_llm import (
    DEFAULT_MAX_TOKENS,
    AnthropicLLM,
)
from grasp_agents.llm_providers.anthropic.provider_output_to_response import (
    convert_usage,
)
from grasp_agents.llm_providers.litellm.error_mapping import (
    map_api_error as litellm_map_api_error,
)
from grasp_agents.llm_providers.litellm.lite_llm import LiteLLM
from grasp_agents.llm_providers.openai_completions.error_mapping import (
    map_api_error as completions_map_api_error,
)
from grasp_agents.types.errors import CompletionError
from grasp_agents.types.llm_errors import (
    LlmInternalServerError,
    LlmRateLimitError,
)
from grasp_agents.types.llm_events import (
    LlmEvent,
    ResponseCompleted,
    ResponseFailed,
    ResponseRetrying,
)
from grasp_agents.types.response import Response, ResponseUsage
from grasp_agents.usage_tracker import add_cost_to_usage
from tests.llm.test_resilience import (  # type: ignore[attr-defined]
    _USER_MSG,
    ErrorLLM,
    LazyStreamCloudLLM,
    StubLLM,
    _resp,
    _text_response,
)


def _retry_after_response(value: str) -> httpx.Response:
    return httpx.Response(
        status_code=429,
        headers={"retry-after": value},
        request=httpx.Request("POST", "https://api.test"),
    )


class TestRetryAfterSanity:
    @pytest.mark.parametrize("raw", ["inf", "-inf", "nan", "-5", "bogus", ""])
    def test_pathological_values_rejected(self, raw: str) -> None:
        assert parse_retry_after(_retry_after_response(raw)) is None

    def test_plain_seconds_parsed(self) -> None:
        assert parse_retry_after(_retry_after_response("7.5")) == 7.5

    def test_huge_server_hint_is_capped(self) -> None:
        policy = RetryPolicy(jitter=0.0)
        err = LlmRateLimitError(
            "429", response=_resp(429), body=None, retry_after=7200.0
        )
        delay = policy.api_delay_for(0, err)
        assert delay <= policy.max_retry_after + policy.max_delay
        assert delay == policy.max_retry_after  # capped, not 2 hours

    def test_sane_server_hint_respected_as_floor(self) -> None:
        policy = RetryPolicy(jitter=0.0)
        err = LlmRateLimitError("429", response=_resp(429), body=None, retry_after=7.0)
        assert policy.api_delay_for(0, err) == 7.0


class TestAnthropicCacheUsage:
    def _usage(self) -> anthropic_types.Usage:
        return anthropic_types.Usage(
            input_tokens=100,
            output_tokens=10,
            cache_read_input_tokens=5000,
            cache_creation_input_tokens=200,
        )

    def test_usage_normalized_to_inclusive_input(self) -> None:
        usage = convert_usage(self._usage())
        assert usage.input_tokens == 100 + 5000 + 200
        assert usage.input_tokens_details.cached_tokens == 5000
        assert usage.cache_creation_tokens == 200
        assert usage.total_tokens == usage.input_tokens + 10

    def test_cost_is_positive_and_prices_cache_writes(self) -> None:
        # The audit repro: exclusive input + cache reads used to go negative.
        usage = convert_usage(self._usage())
        add_cost_to_usage(
            usage, model_name="claude-sonnet-4-5", litellm_provider="anthropic"
        )
        assert usage.cost is not None
        assert usage.cost > 0

        # Cache writes are billed at a premium: dropping them must lower cost.
        no_write = convert_usage(
            anthropic_types.Usage(
                input_tokens=100 + 200,  # same total input, none cache-written
                output_tokens=10,
                cache_read_input_tokens=5000,
                cache_creation_input_tokens=0,
            )
        )
        add_cost_to_usage(
            no_write, model_name="claude-sonnet-4-5", litellm_provider="anthropic"
        )
        assert no_write.cost is not None
        assert usage.cost > no_write.cost


@dataclass(frozen=True)
class FailedStreamCloudLLM(LazyStreamCloudLLM):
    """First ``fail_attempts`` streams end in a terminal ``response.failed``."""

    async def _get_api_stream(
        self,
        api_input: list[Any],
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        del api_input, kwargs

        async def iterator() -> AsyncIterator[Any]:
            count: int = self._attempts  # type: ignore[attr-defined]
            object.__setattr__(self, "_attempts", count + 1)
            yield "failed" if count < self.fail_attempts else "done"

        return iterator()

    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[LlmEvent]:
        seq = 0
        async for chunk in api_stream:
            seq += 1
            resp = _text_response("recovered")
            if chunk == "failed":
                resp = resp.model_copy(update={"status": "failed"})
                yield ResponseFailed(response=resp, sequence_number=seq)  # type: ignore[arg-type]
            else:
                yield ResponseCompleted(response=resp, sequence_number=seq)  # type: ignore[arg-type]


class TestStreamedResponseFailed:
    @pytest.mark.asyncio
    async def test_terminal_failed_event_raises_typed_error(self) -> None:
        llm = FailedStreamCloudLLM(model_name="fake", fail_attempts=10)

        async def _collect() -> None:
            async for _ in llm._generate_response_stream_once(_USER_MSG):
                pass

        with pytest.raises(LlmInternalServerError, match="response failed"):
            await _collect()

    @pytest.mark.asyncio
    async def test_retry_recovers_from_failed_stream(self) -> None:
        llm = FailedStreamCloudLLM(
            model_name="fake",
            fail_attempts=1,
            retry_policy=RetryPolicy(api_retries=2, initial_delay=0.01, max_delay=0.01),
        )
        events = [e async for e in llm.generate_response_stream(_USER_MSG)]
        assert any(isinstance(e, ResponseRetrying) for e in events)
        assert llm.attempts == 2


class TestCompletionErrorMapped:
    def test_completions_mapper(self) -> None:
        mapped = completions_map_api_error(CompletionError("upstream error body"))
        assert isinstance(mapped, LlmInternalServerError)

    def test_litellm_mapper(self) -> None:
        mapped = litellm_map_api_error(CompletionError("upstream error body"))
        assert isinstance(mapped, LlmInternalServerError)


class TestApiKeyNotInRepr:
    def test_repr_hides_resolved_key(self) -> None:
        llm = LazyStreamCloudLLM(
            model_name="fake",
            api_provider={
                "name": "test",
                "base_url": None,
                "api_key": "sk-super-secret-key",
            },
        )
        assert "sk-super-secret-key" not in repr(llm)
        assert "sk-super-secret-key" not in str(llm)


@dataclass(frozen=True)
class _ServingCloudLLM(LazyStreamCloudLLM):
    """CloudLLM stub serving a fixed response on the non-stream path."""

    served: Response = field(default_factory=lambda: _text_response("ok"))

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        return self.served

    def _convert_api_response(self, raw: Any) -> Response:
        return raw  # type: ignore[no-any-return]


def _served_with_usage(text: str) -> Response:
    return _text_response(text).model_copy(
        update={
            "usage": ResponseUsage(input_tokens=100, output_tokens=10, total_tokens=110)
        }
    )


class TestFallbackCostAttribution:
    @pytest.mark.asyncio
    async def test_fallback_response_costed_with_serving_model(self) -> None:
        """The serving CloudLLM member stamps at source with its own identity."""
        primary = ErrorLLM(model_name="primary", retry_policy=None)
        fallback = _ServingCloudLLM(
            model_name="gpt-4o-mini",
            litellm_provider="openai",
            served=_served_with_usage("recovered"),
        )
        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))

        result = await llm.generate_response(_USER_MSG)

        usage = result.usage
        assert usage is not None
        # Cost computed from the SERVING member (gpt-4o-mini), not "primary"
        # (an unknown model, which would silently yield no cost at all).
        assert usage.cost is not None
        assert usage.cost > 0

    @pytest.mark.asyncio
    async def test_non_cloud_member_cost_left_unstamped(self) -> None:
        """
        A non-cloud serving member has no pricing identity: cost stays None.
        Neither the composite (reporting the primary's known name) nor the
        base LLM layer may price it.
        """
        primary = ErrorLLM(model_name="gpt-4o", retry_policy=None)
        fallback = StubLLM(
            model_name="my-custom-model", response=_served_with_usage("recovered")
        )
        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))

        result = await llm.generate_response(_USER_MSG)

        assert result.usage is not None
        assert result.usage.cost is None


class TestAnthropicDefaults:
    def test_default_max_tokens_respects_model_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        llm = AnthropicLLM(model_name="claude-sonnet-4-5")
        cap = get_model_capabilities("claude-sonnet-4-5", "anthropic").max_output_tokens
        assert cap is not None
        assert llm._default_max_tokens == cap

    def test_unknown_model_uses_fallback_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        llm = AnthropicLLM(model_name="claude-nonexistent-model")
        assert llm._default_max_tokens == DEFAULT_MAX_TOKENS

    def test_client_timeout_matches_sdk_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        llm = AnthropicLLM(model_name="claude-sonnet-4-5")
        assert llm.anthropic_client_timeout == 600.0


class TestGeminiTimeout:
    def test_request_level_http_options_carry_timeout(self) -> None:
        from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

        llm = GeminiLLM(
            model_name="gemini-2.5-flash",
            api_provider={"name": "google", "base_url": None, "api_key": "dummy"},
            llm_settings={"extra_headers": {"x-test": "1"}},
        )
        params = llm._make_api_input([])
        config = params["extra_settings"]["config"]  # type: ignore[index]
        assert config.http_options is not None
        assert config.http_options.timeout == int(
            (llm.gemini_client_timeout or 0) * 1000
        )
        assert math.isfinite(config.http_options.timeout)


class TestLiteLLMExplicitProvider:
    def test_explicit_provider_credentials_threaded(self) -> None:
        llm = LiteLLM(
            model_name="openai/some-proxy-model",
            api_provider={
                "name": "custom-proxy",
                "base_url": "http://localhost:8000/v1",
                "api_key": "proxy-key",
            },
        )
        params = llm._lite_llm_completion_params
        assert params["api_base"] == "http://localhost:8000/v1"
        assert params["api_key"] == "proxy-key"
        # The explicit provider is kept, not overwritten by derivation.
        assert llm.api_provider is not None
        assert llm.api_provider["name"] == "custom-proxy"
