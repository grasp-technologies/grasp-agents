"""Map OpenAI SDK exceptions to LLMError types."""

from __future__ import annotations

import httpx
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
    LlmError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmQuotaExceededError,
    LlmRateLimitError,
)

_SYNTHETIC_REQUEST = ("POST", "https://api.openai.com/v1")


def _synthetic_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request(*_SYNTHETIC_REQUEST),
    )


def map_api_error(err: Exception) -> LlmError | None:
    if isinstance(err, CompletionError):
        # A 200 response carrying an error/invalid body (e.g. OpenRouter's
        # in-body upstream failures) — transient, must reach retry/fallback.
        return LlmInternalServerError(
            str(err), response=_synthetic_response(502), body=None
        )

    # Finish-reason failures from the `.parse()` helper: not transport errors,
    # but they abort the call and so must still reach the fallback cascade.
    if isinstance(err, openai.ContentFilterFinishReasonError):
        return LlmContentFilterError(str(err), code="content_filter")

    if isinstance(err, openai.LengthFinishReasonError):
        # Re-sampling the same model with the same cap repeats the
        # truncation; another model with more headroom may not.
        return LlmBadRequestError(
            str(err), response=_synthetic_response(400), body=None
        )

    if isinstance(err, openai.APITimeoutError):
        return LlmApiTimeoutError(request=err.request)

    if isinstance(err, openai.APIConnectionError):
        return LlmApiConnectionError(message=str(err), request=err.request)

    if isinstance(err, openai.APIStatusError):
        msg = str(err)
        code = err.status_code
        resp, body = err.response, err.body

        # Checked before the status branches: gateways disagree on the code
        # they return for a spent account (429, 402, 403 are all seen).
        if is_quota_error(err):
            return LlmQuotaExceededError(msg, response=resp, body=body)

        # Also before the status branches, and for the same reason: a policy
        # block is rejected as a plain 400 (``invalid_prompt``), so the
        # BadRequest branch below would bury it as a programmer bug. The
        # message is only trusted on the codes a block can arrive under, so
        # a rate limit that happens to quote a policy is not misread.
        cf_code = content_filter_code(err)
        if cf_code or (code in {400, 403, 422} and is_content_filter_message(msg)):
            return LlmContentFilterError(msg, code=cf_code)

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

    if isinstance(err, openai.APIError):
        # No HTTP status: an error frame inside a 200 SSE stream, or a
        # response body the SDK could not validate. Quota exhaustion and
        # content-policy blocks both land here on streamed calls, where the
        # transport status is 200 and only the message identifies them.
        msg = str(err)
        if is_quota_error(err):
            return LlmQuotaExceededError(
                msg, response=_synthetic_response(429), body=err.body
            )
        cf_code = content_filter_code(err)
        if cf_code or is_content_filter_message(msg):
            return LlmContentFilterError(msg, code=cf_code)
        return LlmApiError(msg, err.request, body=err.body)

    return None
