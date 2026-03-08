"""Map Google GenAI SDK exceptions to LLMError types."""

from __future__ import annotations

from google.genai import errors as genai_errors

from grasp_agents.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServerError,
)


def map_api_error(err: Exception) -> LLMError | None:
    if not isinstance(err, genai_errors.APIError):
        return None

    msg = err.message or str(err)
    code = err.code

    if isinstance(err, genai_errors.ServerError):
        return LLMServerError(msg, status_code=code)

    # ClientError — branch on status code
    if code == 429:
        return LLMRateLimitError(msg)
    if code in {401, 403}:
        return LLMAuthenticationError(msg, status_code=code)
    if code == 404:
        return LLMNotFoundError(msg, status_code=code)
    if code == 400:
        return LLMBadRequestError(msg, status_code=code)

    return LLMError(msg, status_code=code)
