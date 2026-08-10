# pyright: reportPrivateImportUsage=false
"""Map LiteLLM exceptions to LLMError types."""

from __future__ import annotations

import httpx
import litellm
import openai

from grasp_agents.llm_providers._http_helpers import (
    content_filter_code,
    is_content_filter_message,
    is_quota_error,
    parse_retry_after,
)
from grasp_agents.types.errors import CompletionError
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
    LlmPermissionDeniedError,
    LlmQuotaExceededError,
    LlmRateLimitError,
    LlmUnprocessableEntityError,
)

_SYNTHETIC_REQUEST = ("POST", "https://api.openai.com/v1")


def _synthetic_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request(*_SYNTHETIC_REQUEST),
    )


def map_api_error(err: Exception) -> LlmError | None:
    msg = str(err)

    if isinstance(err, CompletionError):
        # A 200 response carrying an error/invalid body (e.g. OpenRouter's
        # in-body upstream failures) — transient, must reach retry/fallback.
        return LlmInternalServerError(msg, response=_synthetic_response(502), body=None)

    # Both mean the account can no longer pay for the call, whatever status
    # the upstream chose — fail over instead of retrying a dead key.
    if isinstance(err, litellm.BudgetExceededError) or (
        isinstance(err, openai.APIError) and is_quota_error(err)
    ):
        return LlmQuotaExceededError(msg, response=_synthetic_response(429), body=None)

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
        # The code, if any, is whatever the upstream body carried: LiteLLM
        # normalizes the exception type across providers but not the code,
        # and stamping one here would attribute an OpenAI code to whichever
        # provider actually blocked.
        return LlmContentFilterError(msg, code=content_filter_code(err))

    if isinstance(err, litellm.AuthenticationError):
        return LlmAuthenticationError(msg, response=err.response, body=err.body)

    if isinstance(err, litellm.PermissionDeniedError):
        return LlmPermissionDeniedError(msg, response=err.response, body=err.body)

    if isinstance(err, litellm.NotFoundError):
        return LlmNotFoundError(msg, response=err.response, body=err.body)

    # Checked before BadRequestError — ContextWindowExceededError subclasses
    # it, and the generic mapping would shadow the NEEDS_COMPACTION signal.
    if isinstance(err, litellm.ContextWindowExceededError):
        return LlmContextWindowError(msg, response=err.response, body=err.body)

    if isinstance(err, litellm.BadRequestError):
        # LiteLLM only raises ContentPolicyViolationError for the providers
        # whose blocks it recognizes; the rest arrive as a plain bad request
        # that the message alone identifies.
        if is_content_filter_message(msg):
            return LlmContentFilterError(msg, code=content_filter_code(err))
        return LlmBadRequestError(msg, response=err.response, body=err.body)

    if isinstance(err, litellm.exceptions.UnprocessableEntityError):
        return LlmUnprocessableEntityError(msg, response=err.response, body=err.body)

    if isinstance(err, (litellm.InternalServerError, litellm.ServiceUnavailableError)):
        return LlmInternalServerError(msg, response=err.response, body=err.body)

    if isinstance(err, openai.APIStatusError):
        return LlmApiStatusError(msg, response=err.response, body=err.body)

    if isinstance(err, openai.APIError):
        # Every litellm exception derives from the OpenAI SDK's, so this
        # catches the long tail (in-stream error frames, bodies the SDK
        # could not validate) that the branches above do not name.
        cf_code = content_filter_code(err)
        if cf_code or is_content_filter_message(msg):
            return LlmContentFilterError(msg, code=cf_code)
        return LlmApiError(msg, err.request, body=err.body)

    return None
