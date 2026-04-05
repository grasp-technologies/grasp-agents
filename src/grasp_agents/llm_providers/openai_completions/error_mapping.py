"""Map OpenAI SDK exceptions to LLMError types."""

from __future__ import annotations

import openai

from grasp_agents.types.llm_errors import (
    LlmApiConnectionError,
    LlmApiStatusError,
    LlmApiTimeoutError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmRateLimitError,
)


def map_api_error(err: Exception) -> LlmError | None:
    if isinstance(err, openai.APITimeoutError):
        return LlmApiTimeoutError(request=err.request)

    if isinstance(err, openai.APIConnectionError):
        return LlmApiConnectionError(message=str(err), request=err.request)

    if not isinstance(err, openai.APIStatusError):
        return None

    msg = str(err)
    code = err.status_code
    resp, body = err.response, err.body
    if code == 429:
        retry_after = _parse_retry_after(err)
        return LlmRateLimitError(msg, response=resp, body=body, retry_after=retry_after)

    if code in {401, 403}:
        return LlmAuthenticationError(msg, response=resp, body=body)

    if code == 404:
        return LlmNotFoundError(msg, response=resp, body=body)

    if code >= 500:
        return LlmInternalServerError(msg, response=resp, body=body)

    if code in {400, 422}:
        return LlmBadRequestError(msg, response=resp, body=body)

    return LlmApiStatusError(msg, response=resp, body=body)


def _parse_retry_after(err: openai.APIStatusError) -> float | None:
    raw = err.response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
