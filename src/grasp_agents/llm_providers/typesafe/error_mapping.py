"""Map TypeSafe SDK exceptions to LlmError types."""

import httpx
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

from grasp_agents.types.llm_errors import (
    LlmApiConnectionError,
    LlmApiError,
    LlmApiStatusError,
    LlmApiTimeoutError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmContextWindowError,
    LlmError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmPermissionDeniedError,
    LlmRateLimitError,
    LlmUnprocessableEntityError,
)

# Our error types carry httpx objects; the SDK uses httpx2 and keeps only the
# status, body and headers, so a stand-in request is built for them.
_REQUEST = httpx.Request("POST", "https://api.typesafe.ai/v1/systemone")
_TOKEN_LIMIT_CODE = "max_tokens_exceeded"
MILLISECONDS_PER_SECOND = 1000


def map_api_error(err: Exception) -> LlmError | None:
    # Timeout subclasses connection error, so it must be checked first.
    if isinstance(err, TypeSafeAPITimeoutError):
        return LlmApiTimeoutError(request=_REQUEST)
    if isinstance(err, TypeSafeAPIConnectionError):
        return LlmApiConnectionError(message=str(err), request=_REQUEST)
    # A 200 whose body the SDK could not read: no status error to map.
    if isinstance(err, TypeSafeAPIResponseValidationError):
        return LlmApiError(str(err), _REQUEST, body=err.body)
    if not isinstance(err, TypeSafeAPIError):
        return None

    msg = str(err)
    resp = httpx.Response(err.status, headers=dict(err.headers), request=_REQUEST)
    body = err.body

    if isinstance(err, TypeSafeRateLimitError):
        retry_after = (
            err.retry_after_ms / MILLISECONDS_PER_SECOND
            if err.retry_after_ms is not None
            else None
        )
        return LlmRateLimitError(msg, response=resp, body=body, retry_after=retry_after)
    if isinstance(err, TypeSafeAuthenticationError):
        return LlmAuthenticationError(msg, response=resp, body=body)
    if isinstance(err, TypeSafePermissionDeniedError):
        return LlmPermissionDeniedError(msg, response=resp, body=body)
    if isinstance(err, TypeSafeNotFoundError):
        return LlmNotFoundError(msg, response=resp, body=body)
    if isinstance(err, TypeSafeUnprocessableEntityError):
        return LlmUnprocessableEntityError(msg, response=resp, body=body)
    if isinstance(err, TypeSafeBadRequestError):
        # State plus questions over Jev's token limit: resending cannot help.
        if _TOKEN_LIMIT_CODE in f"{msg} {body}":
            return LlmContextWindowError(msg, response=resp, body=body)
        return LlmBadRequestError(msg, response=resp, body=body)
    if isinstance(err, TypeSafeInternalServerError):
        return LlmInternalServerError(msg, response=resp, body=body)
    return LlmApiStatusError(msg, response=resp, body=body)
