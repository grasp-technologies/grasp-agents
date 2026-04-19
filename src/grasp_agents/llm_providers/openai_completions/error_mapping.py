"""Map OpenAI SDK exceptions to LLMError types."""

from __future__ import annotations

import openai

from grasp_agents.llm_providers._http_helpers import parse_retry_after  # noqa: PLC2701
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
        return LlmRateLimitError(
            msg, response=resp, body=body, retry_after=parse_retry_after(resp)
        )

    if code in {401, 403}:
        return LlmAuthenticationError(msg, response=resp, body=body)

    if code == 404:
        return LlmNotFoundError(msg, response=resp, body=body)

    if code >= 500:
        return LlmInternalServerError(msg, response=resp, body=body)

    if code in {400, 422}:
        return LlmBadRequestError(msg, response=resp, body=body)

    return LlmApiStatusError(msg, response=resp, body=body)
