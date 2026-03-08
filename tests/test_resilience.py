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

import pytest
from pydantic import BaseModel

from grasp_agents.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMContextWindowError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from grasp_agents.fallback_llm import FallbackLLM
from grasp_agents.llm import LLM
from grasp_agents.resilience import RetryPolicy
from grasp_agents.types.content import OutputTextContentPart
from grasp_agents.types.items import InputItem, InputMessageItem, OutputMessageItem
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

# ---------- Helpers ----------


def _text_response(text: str, model: str = "mock") -> Response:
    return Response(
        model=model,
        output_items=[
            OutputMessageItem(
                content_parts=[OutputTextContentPart(text=text)],
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
        response_schema: Any | None = None,
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
        response_schema: Any | None = None,
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
        default_factory=lambda: LLMServerError("500", status_code=500)
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
        response_schema: Any | None = None,
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
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        count: int = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        raise self.error_to_raise
        yield  # noqa: RET503


@dataclass(frozen=True)
class FailThenSucceedLLM(LLM):
    """Fails with given error N times, then returns success."""

    error: LLMError = field(
        default_factory=lambda: LLMServerError("500", status_code=500)
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
        response_schema: Any | None = None,
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
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        resp = await self._generate_response_once(
            input,
            tools=tools,
            response_schema=response_schema,
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

    error_to_raise: LLMError = field(
        default_factory=lambda: LLMServerError("connection reset", status_code=502)
    )

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> Response:
        raise self.error_to_raise

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        resp = _text_response("partial", model="failing-model")
        yield ResponseCreated(response=resp, sequence_number=1)  # type: ignore[arg-type]
        yield OutputItemAdded(
            item=resp.output[0], output_index=0, sequence_number=2
        )
        raise self.error_to_raise


# ---------- FallbackLLM ----------


class TestFallbackLLM:
    """FallbackLLM: cascade behavior, mid-stream recovery, error boundaries."""

    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback_called(self) -> None:
        """When primary works, fallback LLMs are never touched."""
        primary = StubLLM(
            model_name="primary", response=_text_response("from primary")
        )
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
            error_to_raise=LLMServerError("overloaded", status_code=503),
        )
        fallback = StubLLM(
            model_name="fallback", response=_text_response("recovered")
        )

        llm = FallbackLLM(model_name="", primary=primary, fallbacks=(fallback,))
        result = await llm._generate_response_once(_USER_MSG)

        assert result.output_text == "recovered"

    @pytest.mark.asyncio
    async def test_all_candidates_fail_raises_last_error(self) -> None:
        """All fail → last error raised, not first. Matters for debugging."""
        primary = ErrorLLM(
            model_name="primary",
            error_to_raise=LLMRateLimitError("rate limited"),
        )
        fallback = ErrorLLM(
            model_name="fallback",
            error_to_raise=LLMServerError("server down", status_code=500),
        )

        llm = FallbackLLM(model_name="", primary=primary, fallbacks=(fallback,))

        with pytest.raises(LLMServerError, match="server down"):
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
            error_to_raise=LLMServerError("mid-stream crash", status_code=502),
        )
        fallback = StubLLM(
            model_name="fallback", response=_text_response("recovered")
        )

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
        assert fb.error_type == "LLMServerError"

        # Fallback stream follows and completes
        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        assert completed[0].response.output_text == "recovered"

    @pytest.mark.asyncio
    async def test_stream_all_fail_yields_fallback_events_then_raises(self) -> None:
        """All candidates fail → ResponseFallback events emitted, then last error raised."""
        primary = ErrorLLM(
            model_name="primary",
            error_to_raise=LLMRateLimitError("rate limited"),
        )
        fallback = ErrorLLM(
            model_name="fallback",
            error_to_raise=LLMServerError("down", status_code=500),
        )

        llm = FallbackLLM(model_name="", primary=primary, fallbacks=(fallback,))

        events: list[LlmEvent] = []
        with pytest.raises(LLMServerError, match="down"):
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
    @patch("grasp_agents.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_transient_error_retried_then_succeeds(
        self, mock_sleep: AsyncMock
    ) -> None:
        """Server error retried with backoff, eventual success."""
        llm = FailThenSucceedLLM(
            model_name="mock",
            error=LLMServerError("overloaded", status_code=503),
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
            error_to_raise=LLMAuthenticationError("bad key", status_code=401),
            retry_policy=RetryPolicy(api_retries=5),
        )

        with pytest.raises(LLMAuthenticationError):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_context_window_error_never_retried(self) -> None:
        """Context window exceeded is deterministic — no retry."""
        llm = ErrorLLM(
            model_name="mock",
            error_to_raise=LLMContextWindowError("too long", status_code=400),
            retry_policy=RetryPolicy(api_retries=5),
        )

        with pytest.raises(LLMContextWindowError):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 1

    @pytest.mark.asyncio
    @patch("grasp_agents.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_exhausted_raises(self, mock_sleep: AsyncMock) -> None:
        """All retries fail → error propagated, not swallowed."""
        llm = ErrorLLM(
            model_name="mock",
            error_to_raise=LLMServerError("persistent failure", status_code=500),
            retry_policy=RetryPolicy(api_retries=2),
        )

        with pytest.raises(LLMServerError, match="persistent failure"):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    @patch("grasp_agents.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_stream_emits_retrying_on_transient_error(
        self, mock_sleep: AsyncMock
    ) -> None:
        """Stream: transient error → ResponseRetrying emitted before retry attempt."""
        llm = FailThenSucceedLLM(
            model_name="mock",
            error=LLMServerError("blip", status_code=500),
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
            error_to_raise=LLMAuthenticationError("expired", status_code=401),
            retry_policy=RetryPolicy(api_retries=3, validation_retries=3),
        )

        with pytest.raises(LLMAuthenticationError):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 1


# ---------- RetryPolicy unit tests ----------


class TestRetryPolicy:
    """RetryPolicy: error classification, delay bounds."""

    def test_transient_errors_are_retryable(self) -> None:
        policy = RetryPolicy()
        assert policy.is_retryable_api_error(LLMRateLimitError("429"))
        assert policy.is_retryable_api_error(
            LLMServerError("500", status_code=500)
        )
        assert policy.is_retryable_api_error(LLMTimeoutError("timeout"))
        assert policy.is_retryable_api_error(LLMConnectionError("reset"))

    def test_deterministic_errors_not_retryable(self) -> None:
        policy = RetryPolicy()
        assert not policy.is_retryable_api_error(
            LLMAuthenticationError("bad key", status_code=401)
        )
        assert not policy.is_retryable_api_error(
            LLMBadRequestError("malformed", status_code=400)
        )
        assert not policy.is_retryable_api_error(
            LLMContextWindowError("too long", status_code=400)
        )
        assert not policy.is_retryable_api_error(
            LLMContentFilterError("blocked")
        )

    def test_non_llm_error_not_retryable(self) -> None:
        policy = RetryPolicy()
        assert not policy.is_retryable_api_error(ValueError("bad"))
        assert not policy.is_retryable_api_error(RuntimeError("crash"))

    def test_exponential_backoff(self) -> None:
        """Delay grows exponentially (jitter=0 for determinism)."""
        policy = RetryPolicy(
            initial_delay=1.0, exponential_base=2.0, jitter=0.0
        )
        err = LLMServerError("500", status_code=500)

        assert policy.api_delay_for(0, err) == pytest.approx(1.0)
        assert policy.api_delay_for(1, err) == pytest.approx(2.0)
        assert policy.api_delay_for(2, err) == pytest.approx(4.0)

    def test_delay_capped_at_max(self) -> None:
        """Delay never exceeds max_delay even at high attempt counts."""
        policy = RetryPolicy(
            initial_delay=10.0, max_delay=15.0, exponential_base=10.0, jitter=0.0
        )
        err = LLMServerError("500", status_code=500)
        assert policy.api_delay_for(5, err) == pytest.approx(15.0)

    def test_rate_limit_retry_after_used_as_floor(self) -> None:
        """Rate limit with retry_after=30 → delay >= 30, not just backoff."""
        policy = RetryPolicy(initial_delay=1.0, jitter=0.0)
        err = LLMRateLimitError("429", retry_after=30.0)
        delay = policy.api_delay_for(0, err)
        assert delay >= 30.0
