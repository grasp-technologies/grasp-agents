"""Map TypeSafe SDK exceptions to typed LlmError values."""

import httpx2
import pytest
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeInternalServerError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.llm_providers.typesafe.error_mapping import map_api_error
from grasp_agents.types.items import InputMessageItem
from grasp_agents.types.llm_errors import (
    LlmApiConnectionError,
    LlmApiError,
    LlmApiStatusError,
    LlmApiTimeoutError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmContextWindowError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmPermissionDeniedError,
    LlmRateLimitError,
    LlmUnprocessableEntityError,
)

from .test_typesafe_llm import (
    TREND,
    FakeClient,
    Learnable,
    learnable_response,
    make_llm,
)

RETRY_AFTER_SECONDS = 12


def _headers(**values: str) -> httpx2.Headers:
    return httpx2.Headers(values)


class TestTypeSafeErrorMapping:
    def test_timeout_maps_to_timeout(self) -> None:
        assert isinstance(
            map_api_error(TypeSafeAPITimeoutError(30.0)), LlmApiTimeoutError
        )

    def test_connection_maps_to_connection(self) -> None:
        err = TypeSafeAPIConnectionError("connection refused")
        assert isinstance(map_api_error(err), LlmApiConnectionError)

    def test_rate_limit_maps_and_keeps_retry_after_in_seconds(self) -> None:
        err = TypeSafeRateLimitError(
            429, None, _headers(**{"retry-after": str(RETRY_AFTER_SECONDS)})
        )
        mapped = map_api_error(err)
        assert isinstance(mapped, LlmRateLimitError)
        assert mapped.retry_after == RETRY_AFTER_SECONDS

    @pytest.mark.parametrize(
        ("error_type", "status", "mapped_type"),
        [
            (TypeSafeAuthenticationError, 401, LlmAuthenticationError),
            (TypeSafePermissionDeniedError, 403, LlmPermissionDeniedError),
            (TypeSafeNotFoundError, 404, LlmNotFoundError),
            (TypeSafeUnprocessableEntityError, 422, LlmUnprocessableEntityError),
            (TypeSafeBadRequestError, 400, LlmBadRequestError),
            (TypeSafeInternalServerError, 503, LlmInternalServerError),
            (TypeSafeAPIError, 418, LlmApiStatusError),
        ],
    )
    def test_status_errors_map_by_type(
        self, error_type: type[TypeSafeAPIError], status: int, mapped_type: type
    ) -> None:
        mapped = map_api_error(error_type(status, None, _headers()))
        assert isinstance(mapped, mapped_type)
        assert mapped.status_code == status  # type: ignore[union-attr]

    def test_state_over_the_token_limit_maps_to_context_window(self) -> None:
        err = TypeSafeBadRequestError(
            400, {"error": {"code": "max_tokens_exceeded"}}, _headers()
        )
        assert isinstance(map_api_error(err), LlmContextWindowError)

    def test_unreadable_response_maps_to_api_error(self) -> None:
        err = TypeSafeAPIResponseValidationError(200, {}, _headers(), "answers.x")
        assert isinstance(map_api_error(err), LlmApiError)

    def test_non_typesafe_error_returns_none(self) -> None:
        assert map_api_error(ValueError("our own bug")) is None


@pytest.mark.asyncio
async def test_rate_limit_is_retried_by_the_framework_retry_policy() -> None:
    client = FakeClient(
        TypeSafeRateLimitError(429, None, _headers(**{"retry-after": "0"})),
        learnable_response(),
    )
    llm = make_llm(client, retry_policy=RetryPolicy(api_retries=1, initial_delay=0))

    response = await llm.generate_response(
        [InputMessageItem.from_text(TREND)], output_schema=Learnable
    )

    assert len(client.calls) == 2
    assert Learnable.model_validate_json(response.output_text).learnable is True
