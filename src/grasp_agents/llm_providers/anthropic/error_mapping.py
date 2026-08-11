"""Map Anthropic SDK exceptions to LlmError types."""

from __future__ import annotations

import anthropic

from grasp_agents.llm_providers._http_helpers import (
    is_content_filter_message,
    is_quota_message,
    parse_retry_after,
)
from grasp_agents.types.llm_errors import (
    LlmApiConnectionError,
    LlmApiError,
    LlmApiStatusError,
    LlmApiTimeoutError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmContentFilterError,
    LlmContextWindowError,
    LlmError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmQuotaExceededError,
    LlmRateLimitError,
)


def map_api_error(err: Exception) -> LlmError | None:
    if isinstance(err, anthropic.APITimeoutError):
        return LlmApiTimeoutError(request=err.request)
    if isinstance(err, anthropic.APIConnectionError):
        return LlmApiConnectionError(message=str(err), request=err.request)

    if isinstance(err, anthropic.APIStatusError):
        msg = str(err)
        code = err.status_code
        resp, body = err.response, err.body
        # A spent account, which Anthropic reports as a 400 rather than a
        # 429 — retrying will not clear it, so it must fail over instead.
        if is_quota_message(msg):
            return LlmQuotaExceededError(msg, response=resp, body=body)
        # A prompt rejected up front comes back as a plain 400, which the
        # BadRequest branch below would bury as a programmer bug. The
        # classifier refusals that arrive as a normal 200 response are
        # handled where the response is validated, not here.
        if code in {400, 403} and is_content_filter_message(msg):
            return LlmContentFilterError(msg)
        if code == 429:
            return LlmRateLimitError(
                msg, response=resp, body=body, retry_after=parse_retry_after(resp)
            )
        if code in {401, 403}:
            return LlmAuthenticationError(msg, response=resp, body=body)
        if code == 404:
            return LlmNotFoundError(msg, response=resp, body=body)
        if code == 413:
            return LlmContextWindowError(msg, response=resp, body=body)
        if code >= 500:
            return LlmInternalServerError(msg, response=resp, body=body)
        if code == 400:
            return LlmBadRequestError(msg, response=resp, body=body)
        return LlmApiStatusError(msg, response=resp, body=body)

    if isinstance(err, anthropic.APIError):
        # No HTTP status — a response body the SDK could not validate, or an
        # error raised outside the status path. Must still reach the cascade.
        msg = str(err)
        if is_content_filter_message(msg):
            return LlmContentFilterError(msg)
        return LlmApiError(msg, err.request, body=err.body)

    return None
