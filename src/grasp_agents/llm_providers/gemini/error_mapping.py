"""Map Google GenAI SDK exceptions to LlmError types."""

from __future__ import annotations

from typing import Any, cast

import httpx
from google.genai import errors as genai_errors

from grasp_agents.llm_providers._http_helpers import parse_retry_after
from grasp_agents.types.llm_errors import (
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmRateLimitError,
)

_SYNTHETIC_REQUEST = ("POST", "https://generativelanguage.googleapis.com")


def _get_response(err: genai_errors.APIError) -> httpx.Response:
    """
    Extract or synthesize an httpx.Response from a Gemini error.

    The returned response must always carry a request: the mapped ``Llm*``
    errors are ``openai.APIStatusError`` subclasses whose ``__init__`` reads
    ``response.request`` and raises if it is unset. The SDK's aiohttp
    transport (preferred whenever aiohttp is importable) yields an
    ``aiohttp.ClientResponse``, so non-httpx responses are converted,
    preserving headers for ``Retry-After`` parsing.
    """
    raw = cast("object", err.response)  # type: ignore[reportUnknownMemberType]
    if isinstance(raw, httpx.Response):
        try:
            _ = raw.request
        except RuntimeError:
            raw.request = httpx.Request(*_SYNTHETIC_REQUEST)
        return raw

    headers: Any = getattr(raw, "headers", None)
    return httpx.Response(
        status_code=err.code,
        headers=dict(headers) if headers else None,
        request=httpx.Request(*_SYNTHETIC_REQUEST),
    )


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
        return LlmRateLimitError(
            msg, response=resp, body=None, retry_after=parse_retry_after(resp)
        )
    if code in {401, 403}:
        return LlmAuthenticationError(msg, response=resp, body=None)
    if code == 404:
        return LlmNotFoundError(msg, response=resp, body=None)
    if code == 400:
        return LlmBadRequestError(msg, response=resp, body=None)

    return None
