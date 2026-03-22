"""Map Google GenAI SDK exceptions to LlmError types."""

from __future__ import annotations

import httpx
from google.genai import errors as genai_errors

from grasp_agents.types.llm_errors import (
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmRateLimitError,
)


def _get_response(err: genai_errors.APIError) -> httpx.Response:
    """Extract or synthesize an httpx.Response from a Gemini error."""
    raw = err.response  # type: ignore[reportUnknownMemberType]
    if isinstance(raw, httpx.Response):
        return raw
    return httpx.Response(status_code=err.code)


def map_api_error(err: Exception) -> LlmError | None:
    if not isinstance(err, genai_errors.APIError):
        return None

    msg = err.message or str(err)
    code = err.code
    resp = _get_response(err)

    if isinstance(err, genai_errors.ServerError):
        return LlmInternalServerError(msg, response=resp, body=None)

    # ClientError — branch on status code
    if code == 429:
        return LlmRateLimitError(msg, response=resp, body=None)
    if code in {401, 403}:
        return LlmAuthenticationError(msg, response=resp, body=None)
    if code == 404:
        return LlmNotFoundError(msg, response=resp, body=None)
    if code == 400:
        return LlmBadRequestError(msg, response=resp, body=None)

    return None
