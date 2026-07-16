"""
Tests for resilience primitives: FallbackLLM, API retry loop, RetryPolicy.

Focuses on real pain points:
- FallbackLLM: mid-stream failure recovery, cascade error selection
- API retries: deterministic vs transient classification, retry_after floor
- RetryPolicy: exponential backoff math, error classification correctness
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from pydantic import BaseModel

from grasp_agents.llm.cloud_llm import ApiCallParams, CloudLLM
from grasp_agents.llm.fallback_llm import FallbackLLM, _select_cascade_error
from grasp_agents.llm.llm import LLM
from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import InputItem, InputMessageItem, OutputMessageItem
from grasp_agents.types.llm_errors import (
    LlmAuthenticationError,
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

_REQ = httpx.Request("POST", "https://test")


def _resp(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_REQ)


# ---------- Helpers ----------


def _text_response(text: str, model: str = "mock") -> Response:
    return Response(
        model=model,
        output=[
            OutputMessageItem(
                content=[OutputMessageText(text=text)],
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
        input: Sequence[InputItem],
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
        input: Sequence[InputItem],
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
            "500", response=_resp(500), body=None
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_call_count", 0)

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],
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
        input: Sequence[InputItem],
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
            "500", response=_resp(500), body=None
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
        input: Sequence[InputItem],
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
        input: Sequence[InputItem],
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
            "connection reset", response=_resp(502), body=None
        )
    )

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> Response:
        raise self.error_to_raise

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],
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


@dataclass(frozen=True)
class LazyStreamCloudLLM(CloudLLM):
    """
    Mimics real provider adapters: ``_get_api_stream`` returns a lazy
    generator, so the (fake) SDK request only fires on first iteration —
    exactly where raw SDK errors surface in production.
    """

    fail_attempts: int = 1
    events_before_failure: int = 0
    raw_error: Exception = field(
        default_factory=lambda: openai.RateLimitError(
            "raw 429", response=_resp(429), body=None
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "_attempts", 0)

    @property
    def attempts(self) -> int:
        return self._attempts  # type: ignore[attr-defined]

    def _make_api_input(
        self,
        input: Sequence[InputItem],
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: Any | None = None,
        output_schema: Any | None = None,
        **extra_llm_settings: Any,
    ) -> ApiCallParams:
        return {"api_input": list(input)}

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        raise NotImplementedError

    def _convert_api_response(self, raw: Any) -> Response:
        raise NotImplementedError

    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[Any]:
        async def iterator() -> AsyncIterator[Any]:
            count: int = self._attempts  # type: ignore[attr-defined]
            object.__setattr__(self, "_attempts", count + 1)
            if count < self.fail_attempts:
                for idx in range(self.events_before_failure):
                    yield f"chunk-{idx}"
                raise self.raw_error
            yield "done"

        return iterator()

    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[LlmEvent]:
        seq = 0
        async for chunk in api_stream:
            seq += 1
            resp = _text_response("recovered" if chunk == "done" else str(chunk))
            if chunk == "done":
                yield ResponseCompleted(response=resp, sequence_number=seq)  # type: ignore[arg-type]
            else:
                yield OutputItemAdded(
                    item=resp.output[0], output_index=0, sequence_number=seq
                )

    def _map_api_error(self, err: Exception) -> LlmError | None:
        if isinstance(err, openai.RateLimitError):
            return LlmRateLimitError(err.message, response=err.response, body=err.body)
        return None


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

        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))
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
                "overloaded", response=_resp(503), body=None
            ),
        )
        fallback = StubLLM(model_name="fallback", response=_text_response("recovered"))

        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))
        result = await llm._generate_response_once(_USER_MSG)

        assert result.output_text == "recovered"

    @pytest.mark.asyncio
    async def test_all_fail_retryable_error_wins_over_deterministic(self) -> None:
        """
        Primary fails transiently, last fallback fails deterministically →
        the transient error is raised, so the outer retry layer still
        retries the cascade (the primary deserves another shot).
        """
        primary = ErrorLLM(
            model_name="primary",
            error_to_raise=LlmInternalServerError(
                "overloaded", response=_resp(503), body=None
            ),
        )
        fallback = ErrorLLM(
            model_name="fallback",
            error_to_raise=LlmAuthenticationError(
                "bad key", response=_resp(401), body=None
            ),
        )

        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))

        with pytest.raises(LlmInternalServerError, match="overloaded"):
            await llm._generate_response_once(_USER_MSG)

    @pytest.mark.asyncio
    async def test_all_fail_deterministically_raises_last_error(self) -> None:
        """No retryable member error → last error raised, as before."""
        primary = ErrorLLM(
            model_name="primary",
            error_to_raise=LlmAuthenticationError(
                "bad key", response=_resp(401), body=None
            ),
        )
        fallback = ErrorLLM(
            model_name="fallback",
            error_to_raise=LlmContextWindowError(
                "too long", response=_resp(400), body=None
            ),
        )

        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))

        with pytest.raises(LlmContextWindowError, match="too long"):
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

        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))

        with pytest.raises(ValueError, match="bad config"):
            await llm._generate_response_once(_USER_MSG)
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_model_name_defaults_to_primary(self) -> None:
        """FallbackLLM.model_name inherits from primary when empty."""
        primary = StubLLM(model_name="gpt-4o")
        llm = FallbackLLM(primary=primary)
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
                "mid-stream crash", response=_resp(502), body=None
            ),
        )
        fallback = StubLLM(model_name="fallback", response=_text_response("recovered"))

        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))

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
        """
        All candidates fail → ResponseFallback events, then the selected
        error raised (here the primary's transient error wins over the
        fallback's deterministic one).
        """
        primary = ErrorLLM(
            model_name="primary",
            error_to_raise=LlmInternalServerError(
                "overloaded", response=_resp(503), body=None
            ),
        )
        fallback = ErrorLLM(
            model_name="fallback",
            error_to_raise=LlmAuthenticationError(
                "bad key", response=_resp(401), body=None
            ),
        )

        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))

        events: list[LlmEvent] = []

        async def _collect() -> None:
            async for event in llm._generate_response_stream_once(_USER_MSG):
                events.append(event)

        with pytest.raises(LlmInternalServerError, match="overloaded"):
            await _collect()

        fallbacks = [e for e in events if isinstance(e, ResponseFallback)]
        assert len(fallbacks) == 2
        assert fallbacks[0].fallback_model == "fallback"
        assert fallbacks[1].fallback_model == "none"


# ---------- Cascade error selection ----------


def _transient(msg: str = "500") -> LlmInternalServerError:
    return LlmInternalServerError(msg, response=_resp(500), body=None)


def _auth(msg: str = "401") -> LlmAuthenticationError:
    return LlmAuthenticationError(msg, response=_resp(401), body=None)


def _rate_limited(retry_after: float | None) -> LlmRateLimitError:
    return LlmRateLimitError(
        "429", response=_resp(429), body=None, retry_after=retry_after
    )


class TestCascadeErrorSelection:
    """_select_cascade_error: which member error a failed pass raises."""

    def test_retryable_wins_over_deterministic_regardless_of_order(self) -> None:
        transient = _transient()
        assert _select_cascade_error([transient, _auth()]) is transient
        assert _select_cascade_error([_auth(), transient]) is transient

    def test_all_deterministic_returns_last(self) -> None:
        last = _auth("second")
        assert _select_cascade_error([_auth("first"), last]) is last

    def test_transient_preferred_over_floored_rate_limit(self) -> None:
        """A retry pass needs one healthy member — don't park on retry_after."""
        transient = _transient()
        assert _select_cascade_error([_rate_limited(60.0), transient]) is transient

    def test_rate_limit_without_floor_preferred_over_floored(self) -> None:
        no_floor = _rate_limited(None)
        assert _select_cascade_error([_rate_limited(60.0), no_floor]) is no_floor

    def test_smallest_retry_after_among_floored_rate_limits(self) -> None:
        smallest = _rate_limited(5.0)
        errors = [_rate_limited(60.0), smallest, _rate_limited(30.0)]
        assert _select_cascade_error(errors) is smallest


class TestFallbackRetryGate:
    """Cascade + outer retry layer: retry fires if ANY member failed retryably."""

    @pytest.mark.asyncio
    @patch("grasp_agents.llm.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_cascade_retried_when_last_member_fails_deterministically(
        self, mock_sleep: AsyncMock
    ) -> None:
        """
        Primary blips (503), fallback is misconfigured (401). The 401 from
        the last member must not suppress the retry pass: the cascade is
        retried and the primary succeeds on the second pass.
        """
        primary = FailThenSucceedLLM(
            model_name="primary",
            error=LlmInternalServerError("blip", response=_resp(503), body=None),
            fail_count=1,
            success_response=_text_response("recovered"),
        )
        fallback = ErrorLLM(model_name="fallback", error_to_raise=_auth("bad key"))
        llm = FallbackLLM(
            primary=primary,
            fallbacks=(fallback,),
            retry_policy=RetryPolicy(api_retries=1),
        )

        result = await llm.generate_response(_USER_MSG)

        assert result.output_text == "recovered"
        assert primary.call_count == 2
        assert fallback.call_count == 1
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    @patch("grasp_agents.llm.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_stream_cascade_retried_when_last_member_fails_deterministically(
        self, mock_sleep: AsyncMock
    ) -> None:
        primary = FailThenSucceedLLM(
            model_name="primary",
            error=LlmInternalServerError("blip", response=_resp(503), body=None),
            fail_count=1,
            success_response=_text_response("recovered"),
        )
        fallback = ErrorLLM(model_name="fallback", error_to_raise=_auth("bad key"))
        llm = FallbackLLM(
            primary=primary,
            fallbacks=(fallback,),
            retry_policy=RetryPolicy(api_retries=1),
        )

        events: list[LlmEvent] = []
        async for event in llm.generate_response_stream(_USER_MSG):
            events.append(event)

        retrying = [e for e in events if isinstance(e, ResponseRetrying)]
        assert len(retrying) == 1
        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        assert completed[0].response.output_text == "recovered"
        assert fallback.call_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_when_all_members_fail_deterministically(self) -> None:
        """Deterministic failures across the board → no retry passes."""
        primary = ErrorLLM(model_name="primary", error_to_raise=_auth("bad key"))
        fallback = ErrorLLM(
            model_name="fallback",
            error_to_raise=LlmContextWindowError(
                "too long", response=_resp(400), body=None
            ),
        )
        llm = FallbackLLM(
            primary=primary,
            fallbacks=(fallback,),
            retry_policy=RetryPolicy(api_retries=3),
        )

        with pytest.raises(LlmContextWindowError, match="too long"):
            await llm.generate_response(_USER_MSG)
        assert primary.call_count == 1
        assert fallback.call_count == 1


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
            error=LlmInternalServerError("overloaded", response=_resp(503), body=None),
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
                "bad key", response=_resp(401), body=None
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
                "too long", response=_resp(400), body=None
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
                "persistent failure", response=_resp(500), body=None
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
            error=LlmInternalServerError("blip", response=_resp(500), body=None),
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
                "expired", response=_resp(401), body=None
            ),
            retry_policy=RetryPolicy(api_retries=3, validation_retries=3),
        )

        with pytest.raises(LlmAuthenticationError):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 1


# ---------- RetryPolicy unit tests ----------


class TestRetryPolicy:
    """RetryPolicy: error classification, delay bounds."""

    # Error-classification (retryable vs not) is covered exhaustively by
    # ``test_recovery_hints.py::TestRetryPolicyAlignment``; only the delay
    # math is unique here.

    def test_exponential_backoff(self) -> None:
        """Delay grows exponentially (jitter=0 for determinism)."""
        policy = RetryPolicy(initial_delay=1.0, exponential_base=2.0, jitter=0.0)
        err = LlmInternalServerError("500", response=_resp(500), body=None)

        assert policy.api_delay_for(0, err) == pytest.approx(1.0)
        assert policy.api_delay_for(1, err) == pytest.approx(2.0)
        assert policy.api_delay_for(2, err) == pytest.approx(4.0)

    def test_delay_capped_at_max(self) -> None:
        """Delay never exceeds max_delay even at high attempt counts."""
        policy = RetryPolicy(
            initial_delay=10.0, max_delay=15.0, exponential_base=10.0, jitter=0.0
        )
        err = LlmInternalServerError("500", response=_resp(500), body=None)
        assert policy.api_delay_for(5, err) == pytest.approx(15.0)

    def test_rate_limit_retry_after_used_as_floor(self) -> None:
        """Rate limit with retry_after=30 → delay >= 30, not just backoff."""
        policy = RetryPolicy(initial_delay=1.0, jitter=0.0)
        err = LlmRateLimitError("429", response=_resp(429), body=None, retry_after=30.0)
        delay = policy.api_delay_for(0, err)
        assert delay >= 30.0


# ---------- Streaming error mapping (CloudLLM) ----------


class TestStreamErrorMapping:
    """
    Raw SDK errors surfacing during stream *iteration* (providers return lazy
    generators — the request fires on first iteration) must be mapped to Llm*
    types, or the retry and fallback layers — which catch only LlmErrorTuple —
    never engage for streaming requests.
    """

    @pytest.mark.asyncio
    async def test_lazy_acquisition_error_is_mapped(self) -> None:
        """Raw SDK error on first iteration → LlmRateLimitError, not the raw type."""
        llm = LazyStreamCloudLLM(model_name="mock", fail_attempts=99)

        with pytest.raises(LlmRateLimitError):
            async for _ in llm._generate_response_stream_once(_USER_MSG):
                pass

    @pytest.mark.asyncio
    async def test_midstream_error_is_mapped_after_partial_events(self) -> None:
        """Partial events are yielded, then the mid-stream SDK error is mapped."""
        llm = LazyStreamCloudLLM(
            model_name="mock", fail_attempts=99, events_before_failure=2
        )

        events: list[LlmEvent] = []

        async def _collect() -> None:
            async for event in llm._generate_response_stream_once(_USER_MSG):
                events.append(event)

        with pytest.raises(LlmRateLimitError):
            await _collect()

        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_unmapped_error_passes_through_unchanged(self) -> None:
        """Non-SDK errors (e.g. converter bugs) are not swallowed or re-typed."""
        llm = LazyStreamCloudLLM(
            model_name="mock",
            fail_attempts=99,
            raw_error=ValueError("converter bug"),
        )

        with pytest.raises(ValueError, match="converter bug"):
            async for _ in llm._generate_response_stream_once(_USER_MSG):
                pass

    @pytest.mark.asyncio
    @patch("grasp_agents.llm.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_stream_retry_engages_on_lazy_sdk_error(
        self, mock_sleep: AsyncMock
    ) -> None:
        """429 during iteration → ResponseRetrying, then a successful retry."""
        llm = LazyStreamCloudLLM(
            model_name="mock",
            fail_attempts=1,
            retry_policy=RetryPolicy(api_retries=2),
        )

        events: list[LlmEvent] = []
        async for event in llm.generate_response_stream(_USER_MSG):
            events.append(event)

        retrying = [e for e in events if isinstance(e, ResponseRetrying)]
        assert len(retrying) == 1
        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        assert completed[0].response.output_text == "recovered"
        assert llm.attempts == 2

    @pytest.mark.asyncio
    async def test_fallback_engages_on_lazy_sdk_error(self) -> None:
        """429 during primary's stream iteration → cascade to the fallback."""
        primary = LazyStreamCloudLLM(model_name="primary", fail_attempts=99)
        fallback = StubLLM(model_name="fallback", response=_text_response("rescued"))
        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))

        events: list[LlmEvent] = []
        async for event in llm._generate_response_stream_once(_USER_MSG):
            events.append(event)

        fb = next(e for e in events if isinstance(e, ResponseFallback))
        assert fb.failed_model == "primary"
        assert fb.fallback_model == "fallback"
        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        assert completed[0].response.output_text == "rescued"
