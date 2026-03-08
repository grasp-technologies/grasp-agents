"""Map Anthropic SDK exceptions to LLMError types."""

from __future__ import annotations

import anthropic

from grasp_agents.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMContextWindowError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)


def map_api_error(err: Exception) -> LLMError | None:
    if isinstance(err, anthropic.APITimeoutError):
        return LLMTimeoutError(str(err))
    if isinstance(err, anthropic.APIConnectionError):
        return LLMConnectionError(str(err))
    if not isinstance(err, anthropic.APIStatusError):
        return None

    msg = str(err)
    code = err.status_code
    if code == 429:
        retry_after = _parse_retry_after(err)
        return LLMRateLimitError(msg, retry_after=retry_after)
    if code in {401, 403}:
        return LLMAuthenticationError(msg, status_code=code)
    if code == 404:
        return LLMNotFoundError(msg, status_code=code)
    if code == 413:
        return LLMContextWindowError(msg, status_code=code)
    if code >= 500:
        return LLMServerError(msg, status_code=code)
    if code == 400:
        return LLMBadRequestError(msg, status_code=code)
    return LLMError(msg, status_code=code)


def _parse_retry_after(err: anthropic.APIStatusError) -> float | None:
    raw = err.response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
