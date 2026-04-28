"""
Tests for resilience primitives: FallbackLLM, API retry loop, RetryPolicy.

Focuses on real pain points:
- FallbackLLM: mid-stream failure recovery, error propagation boundaries, last-error-wins
- API retries: deterministic vs transient classification, retry_after floor, layer interaction
- RetryPolicy: exponential backoff math, error classification correctness
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import BaseModel

from grasp_agents.llm.fallback_llm import FallbackLLM
from grasp_agents.llm.llm import LLM
from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import InputItem, InputMessageItem, OutputMessageItem
from grasp_agents.types.llm_errors import (
    LlmApiConnectionError,
    LlmApiTimeoutError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmContentFilterError,
    LlmContextWindowError,
    LlmError,
    LlmInternalServerError,
    LlmRateLimitError,
)
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
    ResponseFallback,
    ResponseRetrying,
)
from grasp_agents.types.response import Response
from grasp_agents.types.tool import BaseTool

_REQ = httpx.Request("POST", "https://test")


def _R(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_REQ)


# ---------- Helpers ----------


def _text_response(text: str, model: str = "mock") -> Response:
    return Response(
        model=model,
        output_items=[
            OutputMessageItem(
                content_parts=[OutputMessageText(text=text)],
                status="completed",
            )
        ],
    )


_USER_MSG = [InputMessageItem.from_text("go", role="user")]


# ---------- Mocks ----------


@dataclass(frozen=True)
class StubLLM(LLM):
    """Returns a single fixed response. Tracks call count."""

    response: Response = field(default_factory=lambda: _text_response("stub"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "_call_count", 0)

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> Response:
        count: int = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        return self.response

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        resp = await self._generate_response_once(input)
        seq = 0
        seq += 1
        yield ResponseCreated(response=resp, sequence_number=seq)  # type: ignore[arg-type]
        for idx, item in enumerate(resp.output):
            seq += 1
            yield OutputItemAdded(item=item, output_index=idx, sequence_number=seq)
            seq += 1
            yield OutputItemDone(item=item, output_index=idx, sequence_number=seq)
        seq += 1
        yield ResponseCompleted(response=resp, sequence_number=seq)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ErrorLLM(LLM):
    """Always raises a given error. Tracks call count."""

    error_to_raise: Exception = field(
        default_factory=lambda: LlmInternalServerError(
            "500", response=_R(500), body=None
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_call_count", 0)

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> Response:
        count: int = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        raise self.error_to_raise

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        count: int = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        raise self.error_to_raise
        yield


@dataclass(frozen=True)
class FailThenSucceedLLM(LLM):
    """Fails with given error N times, then returns success."""

    error: LlmError = field(
        default_factory=lambda: LlmInternalServerError(
            "500", response=_R(500), body=None
        )
    )
    fail_count: int = 2
    success_response: Response = field(default_factory=lambda: _text_response("ok"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "_call_count", 0)

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> Response:
        count: int = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        if count < self.fail_count:
            raise self.error
        return self.success_response

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        resp = await self._generate_response_once(
            input,
            tools=tools,
            output_schema=output_schema,
            tool_choice=tool_choice,
        )
        seq = 0
        seq += 1
        yield ResponseCreated(response=resp, sequence_number=seq)  # type: ignore[arg-type]
        for idx, item in enumerate(resp.output):
            seq += 1
            yield OutputItemAdded(item=item, output_index=idx, sequence_number=seq)
            seq += 1
            yield OutputItemDone(item=item, output_index=idx, sequence_number=seq)
        seq += 1
        yield ResponseCompleted(response=resp, sequence_number=seq)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FailMidStreamLLM(LLM):
    """Yields partial events then raises — simulates mid-stream connection drop."""

    error_to_raise: LlmError = field(
        default_factory=lambda: LlmInternalServerError(
            "connection reset", response=_R(502), body=None
        )
    )

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> Response:
        raise self.error_to_raise

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        resp = _text_response("partial", model="failing-model")
        yield ResponseCreated(response=resp, sequence_number=1)  # type: ignore[arg-type]
        yield OutputItemAdded(item=resp.output[0], output_index=0, sequence_number=2)
        raise self.error_to_raise


# ---------- FallbackLLM ----------


class TestFallbackLLM:
    """FallbackLLM: cascade behavior, mid-stream recovery, error boundaries."""

    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback_called(self) -> None:
        """When primary works, fallback LLMs are never touched."""
        primary = StubLLM(model_name="primary", response=_text_response("from primary"))
        fallback = StubLLM(
            model_name="fallback", response=_text_response("from fallback")
        )

        llm = FallbackLLM(model_name="", primary=primary, fallbacks=(fallback,))
        result = await llm._generate_response_once(_USER_MSG)

        assert result.output_text == "from primary"
        assert primary.call_count == 1
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_succeeds(self) -> None:
        """Primary error → first fallback tried and succeeds."""
        primary = ErrorLLM(
            model_name="primary",
            error_to_raise=LlmInternalServerError(
                "overloaded", response=_R(503), body=None
            ),
        )
        fallback = StubLLM(model_name="fallback", response=_text_response("recovered"))

        llm = FallbackLLM(model_name="", primary=primary, fallbacks=(fallback,))
        result = await llm._generate_response_once(_USER_MSG)

        assert result.output_text == "recovered"

    @pytest.mark.asyncio
    async def test_all_candidates_fail_raises_last_error(self) -> None:
        """All fail → last error raised, not first. Matters for debugging."""
        primary = ErrorLLM(
            model_name="primary",
            error_to_raise=LlmRateLimitError(
                "rate limited", response=_R(429), body=None
            ),
        )
        fallback = ErrorLLM(
            model_name="fallback",
            error_to_raise=LlmInternalServerError(
                "server down", response=_R(500), body=None
            ),
        )

        llm = FallbackLLM(model_name="", primary=primary, fallbacks=(fallback,))

        with pytest.raises(LlmInternalServerError, match="server down"):
            await llm._generate_response_once(_USER_MSG)

    @pytest.mark.asyncio
    async def test_non_llm_error_bypasses_fallback(self) -> None:
        """ValueError in primary → NOT caught, fallback NOT tried."""
        primary = ErrorLLM(
            model_name="primary",
            error_to_raise=ValueError("bad config"),
        )
        fallback = StubLLM(
            model_name="fallback", response=_text_response("should not reach")
        )

        llm = FallbackLLM(model_name="", primary=primary, fallbacks=(fallback,))

        with pytest.raises(ValueError, match="bad config"):
            await llm._generate_response_once(_USER_MSG)
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_model_name_defaults_to_primary(self) -> None:
        """FallbackLLM.model_name inherits from primary when empty."""
        primary = StubLLM(model_name="gpt-4o")
        llm = FallbackLLM(model_name="", primary=primary)
        assert llm.model_name == "gpt-4o"

    @pytest.mark.asyncio
    async def test_stream_mid_failure_emits_fallback_marker(self) -> None:
        """
        Primary yields partial events, then crashes mid-stream.
        Consumer sees: partial events → ResponseFallback → complete fallback stream.
        The ResponseFallback marker lets consumers discard partial state.
        """
        primary = FailMidStreamLLM(
            model_name="primary",
            error_to_raise=LlmInternalServerError(
                "mid-stream crash", response=_R(502), body=None
            ),
        )
        fallback = StubLLM(model_name="fallback", response=_text_response("recovered"))

        llm = FallbackLLM(model_name="", primary=primary, fallbacks=(fallback,))

        events: list[LlmEvent] = []
        async for event in llm._generate_response_stream_once(_USER_MSG):
            events.append(event)

        # Partial events from primary were already yielded
        assert isinstance(events[0], ResponseCreated)
        assert isinstance(events[1], OutputItemAdded)

        # ResponseFallback marks the boundary
        fb = next(e for e in events if isinstance(e, ResponseFallback))
        assert fb.failed_model == "primary"
        assert fb.fallback_model == "fallback"
        assert fb.error_type == "LlmInternalServerError"

        # Fallback stream follows and completes
        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        assert completed[0].response.output_text == "recovered"

    @pytest.mark.asyncio
    async def test_stream_all_fail_yields_fallback_events_then_raises(self) -> None:
        """All candidates fail → ResponseFallback events emitted, then last error raised."""
        primary = ErrorLLM(
            model_name="primary",
            error_to_raise=LlmRateLimitError(
                "rate limited", response=_R(429), body=None
            ),
        )
        fallback = ErrorLLM(
            model_name="fallback",
            error_to_raise=LlmInternalServerError("down", response=_R(500), body=None),
        )

        llm = FallbackLLM(model_name="", primary=primary, fallbacks=(fallback,))

        events: list[LlmEvent] = []
        with pytest.raises(LlmInternalServerError, match="down"):
            async for event in llm._generate_response_stream_once(_USER_MSG):
                events.append(event)

        fallbacks = [e for e in events if isinstance(e, ResponseFallback)]
        assert len(fallbacks) == 2
        assert fallbacks[0].fallback_model == "fallback"
        assert fallbacks[1].fallback_model == "none"


# ---------- API Retry Loop ----------


class TestAPIRetries:
    """API retry layer in LLM base: transient vs deterministic errors."""

    @pytest.mark.asyncio
    @patch("grasp_agents.llm.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_transient_error_retried_then_succeeds(
        self, mock_sleep: AsyncMock
    ) -> None:
        """Server error retried with backoff, eventual success."""
        llm = FailThenSucceedLLM(
            model_name="mock",
            error=LlmInternalServerError("overloaded", response=_R(503), body=None),
            fail_count=2,
            success_response=_text_response("ok"),
            retry_policy=RetryPolicy(api_retries=3),
        )

        result = await llm.generate_response(_USER_MSG)

        assert result.output_text == "ok"
        assert llm.call_count == 3  # 2 failures + 1 success
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_auth_error_never_retried(self) -> None:
        """Authentication error propagates immediately regardless of api_retries."""
        llm = ErrorLLM(
            model_name="mock",
            error_to_raise=LlmAuthenticationError(
                "bad key", response=_R(401), body=None
            ),
            retry_policy=RetryPolicy(api_retries=5),
        )

        with pytest.raises(LlmAuthenticationError):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_context_window_error_never_retried(self) -> None:
        """Context window exceeded is deterministic — no retry."""
        llm = ErrorLLM(
            model_name="mock",
            error_to_raise=LlmContextWindowError(
                "too long", response=_R(400), body=None
            ),
            retry_policy=RetryPolicy(api_retries=5),
        )

        with pytest.raises(LlmContextWindowError):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 1

    @pytest.mark.asyncio
    @patch("grasp_agents.llm.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_exhausted_raises(self, mock_sleep: AsyncMock) -> None:
        """All retries fail → error propagated, not swallowed."""
        llm = ErrorLLM(
            model_name="mock",
            error_to_raise=LlmInternalServerError(
                "persistent failure", response=_R(500), body=None
            ),
            retry_policy=RetryPolicy(api_retries=2),
        )

        with pytest.raises(LlmInternalServerError, match="persistent failure"):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    @patch("grasp_agents.llm.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_stream_emits_retrying_on_transient_error(
        self, mock_sleep: AsyncMock
    ) -> None:
        """Stream: transient error → ResponseRetrying emitted before retry attempt."""
        llm = FailThenSucceedLLM(
            model_name="mock",
            error=LlmInternalServerError("blip", response=_R(500), body=None),
            fail_count=1,
            success_response=_text_response("recovered"),
            retry_policy=RetryPolicy(api_retries=2),
        )

        events: list[LlmEvent] = []
        async for event in llm.generate_response_stream(_USER_MSG):
            events.append(event)

        retrying = [e for e in events if isinstance(e, ResponseRetrying)]
        assert len(retrying) == 1
        assert retrying[0].attempt == 1
        assert "blip" in retrying[0].error

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_auth_error_pierces_both_retry_layers(self) -> None:
        """
        RetryPolicy(api_retries=3, validation_retries=3) + AuthError
        → propagates immediately through both layers. Neither catches it.
        """
        llm = ErrorLLM(
            model_name="mock",
            error_to_raise=LlmAuthenticationError(
                "expired", response=_R(401), body=None
            ),
            retry_policy=RetryPolicy(api_retries=3, validation_retries=3),
        )

        with pytest.raises(LlmAuthenticationError):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 1


# ---------- RetryPolicy unit tests ----------


class TestRetryPolicy:
    """RetryPolicy: error classification, delay bounds."""

    def test_transient_errors_are_retryable(self) -> None:
        policy = RetryPolicy()
        assert policy.is_retryable_api_error(
            LlmRateLimitError("429", response=_R(429), body=None)
        )
        assert policy.is_retryable_api_error(
            LlmInternalServerError("500", response=_R(500), body=None)
        )
        assert policy.is_retryable_api_error(LlmApiTimeoutError(request=_REQ))
        assert policy.is_retryable_api_error(LlmApiConnectionError(request=_REQ))

    def test_deterministic_errors_not_retryable(self) -> None:
        policy = RetryPolicy()
        assert not policy.is_retryable_api_error(
            LlmAuthenticationError("bad key", response=_R(401), body=None)
        )
        assert not policy.is_retryable_api_error(
            LlmBadRequestError("malformed", response=_R(400), body=None)
        )
        assert not policy.is_retryable_api_error(
            LlmContextWindowError("too long", response=_R(400), body=None)
        )
        assert not policy.is_retryable_api_error(LlmContentFilterError())

    def test_non_llm_error_not_retryable(self) -> None:
        policy = RetryPolicy()
        assert not policy.is_retryable_api_error(ValueError("bad"))
        assert not policy.is_retryable_api_error(RuntimeError("crash"))

    def test_exponential_backoff(self) -> None:
        """Delay grows exponentially (jitter=0 for determinism)."""
        policy = RetryPolicy(initial_delay=1.0, exponential_base=2.0, jitter=0.0)
        err = LlmInternalServerError("500", response=_R(500), body=None)

        assert policy.api_delay_for(0, err) == pytest.approx(1.0)
        assert policy.api_delay_for(1, err) == pytest.approx(2.0)
        assert policy.api_delay_for(2, err) == pytest.approx(4.0)

    def test_delay_capped_at_max(self) -> None:
        """Delay never exceeds max_delay even at high attempt counts."""
        policy = RetryPolicy(
            initial_delay=10.0, max_delay=15.0, exponential_base=10.0, jitter=0.0
        )
        err = LlmInternalServerError("500", response=_R(500), body=None)
        assert policy.api_delay_for(5, err) == pytest.approx(15.0)

    def test_rate_limit_retry_after_used_as_floor(self) -> None:
        """Rate limit with retry_after=30 → delay >= 30, not just backoff."""
        policy = RetryPolicy(initial_delay=1.0, jitter=0.0)
        err = LlmRateLimitError("429", response=_R(429), body=None, retry_after=30.0)
        delay = policy.api_delay_for(0, err)
        assert delay >= 30.0
