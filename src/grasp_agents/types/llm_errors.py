from typing import Literal, TypeAlias

import httpx
import openai


class LlmContentFilterError(openai.ContentFilterFinishReasonError):
    pass


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


LlmError: TypeAlias = (
    LlmContentFilterError
    | LlmContextWindowError
    | LlmApiError
    | LlmApiConnectionError
    | LlmApiStatusError
    | LlmApiTimeoutError
    | LlmRateLimitError
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
    LlmInternalServerError,
    LlmAuthenticationError,
    LlmPermissionDeniedError,
    LlmNotFoundError,
    LlmBadRequestError,
    LlmConflictError,
    LlmUnprocessableEntityError,
)
