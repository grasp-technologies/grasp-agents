from typing import Literal

import httpx
import openai

CONTENT_FILTER_DEFAULT_MESSAGE = "The provider's content filter blocked this request."


class LlmContentFilterError(openai.ContentFilterFinishReasonError):
    """
    The provider refused the request on policy grounds, or discarded the
    output it had already produced.

    Never retried — the same request on the same model is blocked again —
    but it does advance a ``FallbackLLM``: another model is the recovery
    both OpenAI and Anthropic recommend for a policy block.

    ``code`` is the provider's own marker when it gave one: an error code
    (``"invalid_prompt"``) or a policy category (``"cyber"``).
    """

    message: str
    code: str | None

    def __init__(self, message: str | None = None, *, code: str | None = None) -> None:
        message = message or CONTENT_FILTER_DEFAULT_MESSAGE
        # Bypasses the parent, whose __init__ hardcodes a fixed message, so
        # the provider's own explanation reaches the caller.
        openai.OpenAIError.__init__(self, message)
        self.message = message
        self.code = code


class LlmApiError(openai.APIError):
    message: str
    request: httpx.Request
    body: object | None
    code: str | None = None
    param: str | None = None
    type: str | None = None


# ---- Inherit from openai.APIError ----


class LlmApiConnectionError(openai.APIConnectionError):
    pass


class LlmApiStatusError(openai.APIStatusError):
    response: httpx.Response
    status_code: int
    request_id: str | None


# ---- Inherit from openai.APIConnectionError ----


class LlmApiTimeoutError(openai.APITimeoutError):
    pass


# ---- Inherit from openai.APIStatusError ----


class LlmRateLimitError(openai.RateLimitError):
    status_code: Literal[429] = 429
    retry_after: float | None

    def __init__(
        self,
        message: str,
        *,
        response: httpx.Response,
        body: object | None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, response=response, body=body)
        self.retry_after = retry_after


class LlmQuotaExceededError(LlmRateLimitError):
    """
    Account credits/quota are exhausted.

    Unlike a plain rate limit this does not clear on its own, so it is not
    retried: only a different key or model can serve the request.
    """


class LlmInternalServerError(openai.InternalServerError):
    pass


class LlmAuthenticationError(openai.AuthenticationError):
    status_code: Literal[401] = 401


class LlmPermissionDeniedError(openai.PermissionDeniedError):
    status_code: Literal[403] = 403


class LlmNotFoundError(openai.NotFoundError):
    status_code: Literal[404] = 404


class LlmBadRequestError(openai.BadRequestError):
    status_code: Literal[400] = 400


class LlmContextWindowError(openai.BadRequestError):
    pass


class LlmConflictError(openai.ConflictError):
    status_code: Literal[409] = 409


class LlmUnprocessableEntityError(openai.UnprocessableEntityError):
    status_code: Literal[422] = 422


type LlmError = (
    LlmContentFilterError
    | LlmContextWindowError
    | LlmApiError
    | LlmApiConnectionError
    | LlmApiStatusError
    | LlmApiTimeoutError
    | LlmRateLimitError
    | LlmQuotaExceededError
    | LlmInternalServerError
    | LlmAuthenticationError
    | LlmPermissionDeniedError
    | LlmNotFoundError
    | LlmBadRequestError
    | LlmConflictError
    | LlmUnprocessableEntityError
)

LlmErrorTuple = (
    LlmContentFilterError,
    LlmContextWindowError,
    LlmApiError,
    LlmApiConnectionError,
    LlmApiStatusError,
    LlmApiTimeoutError,
    LlmRateLimitError,
    LlmQuotaExceededError,
    LlmInternalServerError,
    LlmAuthenticationError,
    LlmPermissionDeniedError,
    LlmNotFoundError,
    LlmBadRequestError,
    LlmConflictError,
    LlmUnprocessableEntityError,
)
