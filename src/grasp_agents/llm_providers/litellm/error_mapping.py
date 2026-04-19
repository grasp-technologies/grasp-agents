# pyright: reportPrivateImportUsage=false
"""Map LiteLLM exceptions to LLMError types."""

from __future__ import annotations

import litellm
from grasp_agents.llm_providers._http_helpers import parse_retry_after  # noqa: PLC2701
from grasp_agents.types.llm_errors import (
    LlmApiConnectionError,
    LlmApiTimeoutError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmContentFilterError,
    LlmContextWindowError,
    LlmError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmPermissionDeniedError,
    LlmRateLimitError,
    LlmUnprocessableEntityError,
)


def map_api_error(err: Exception) -> LlmError | None:
    msg = str(err)

    if isinstance(err, litellm.Timeout):
        return LlmApiTimeoutError(request=err.request)

    if isinstance(err, litellm.APIConnectionError):
        return LlmApiConnectionError(message=msg, request=err.request)

    if isinstance(err, litellm.RateLimitError):
        return LlmRateLimitError(
            message=msg,
            response=err.response,
            body=err.body,
            retry_after=parse_retry_after(err.response),
        )

    if isinstance(err, litellm.ContentPolicyViolationError):
        return LlmContentFilterError()

    if isinstance(err, litellm.AuthenticationError):
        return LlmAuthenticationError(msg, response=err.response, body=err.body)

    if isinstance(err, litellm.PermissionDeniedError):
        return LlmPermissionDeniedError(msg, response=err.response, body=err.body)

    if isinstance(err, litellm.NotFoundError):
        return LlmNotFoundError(msg, response=err.response, body=err.body)

    if isinstance(err, litellm.BadRequestError):
        return LlmBadRequestError(msg, response=err.response, body=err.body)

    if isinstance(err, litellm.exceptions.UnprocessableEntityError):
        return LlmUnprocessableEntityError(msg, response=err.response, body=err.body)

    if isinstance(err, (litellm.InternalServerError, litellm.ServiceUnavailableError)):
        return LlmInternalServerError(msg, response=err.response, body=err.body)

    if isinstance(err, litellm.ContextWindowExceededError):
        return LlmContextWindowError(msg, response=err.response, body=err.body)

    return None
