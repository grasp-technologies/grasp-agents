# pyright: reportPrivateImportUsage=false
"""Map LiteLLM exceptions to LLMError types."""

from __future__ import annotations

import litellm
from grasp_agents.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMContextWindowError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)


def map_api_error(err: Exception) -> LLMError | None:
    msg = str(err)

    if isinstance(err, litellm.Timeout):
        return LLMTimeoutError(msg, status_code=408)
    if isinstance(err, litellm.APIConnectionError):
        return LLMConnectionError(msg)
    if isinstance(err, litellm.RateLimitError):
        return LLMRateLimitError(msg)
    if isinstance(err, litellm.ContextWindowExceededError):
        return LLMContextWindowError(msg, status_code=400)
    if isinstance(err, litellm.ContentPolicyViolationError):
        return LLMContentFilterError(msg, status_code=400)
    if isinstance(err, litellm.AuthenticationError):
        return LLMAuthenticationError(msg, status_code=401)
    if isinstance(err, litellm.NotFoundError):
        return LLMNotFoundError(msg, status_code=404)
    if isinstance(err, litellm.BadRequestError):
        return LLMBadRequestError(msg, status_code=400)
    if isinstance(err, (litellm.InternalServerError, litellm.ServiceUnavailableError)):
        code = getattr(err, "status_code", 500)
        return LLMServerError(msg, status_code=code)

    return None
