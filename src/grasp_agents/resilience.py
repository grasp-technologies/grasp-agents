"""Resilience primitives: retry policy for LLM API and validation errors."""

import random
from dataclasses import dataclass

from .types.llm_errors import (
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmContentFilterError,
    LlmContextWindowError,
    LlmErrorTuple,
    LlmRateLimitError,
)

# Deterministic errors — retrying the same request won't help.
_NON_RETRYABLE = (
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmContextWindowError,
    LlmContentFilterError,
)


@dataclass(frozen=True)
class RetryPolicy:
    """
    Unified retry configuration for API and validation errors.

    API errors (rate limit, server, timeout, connection) use exponential
    backoff with jitter. Rate-limit errors respect the server's
    ``retry_after`` as a floor.

    Validation errors (bad tool call args, response schema mismatch) are
    retried immediately with no delay — just ask the LLM again.

    Deterministic API errors (auth, bad request, context window, content
    filter) always propagate immediately.
    """

    # --- API error retries ---
    api_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.5

    # --- Validation error retries ---
    validation_retries: int = 0

    def is_retryable_api_error(self, error: Exception) -> bool:
        """Return whether this API error is retryable (transient)."""
        return isinstance(error, LlmErrorTuple) and not isinstance(
            error, _NON_RETRYABLE
        )

    def api_delay_for(self, attempt: int, error: Exception) -> float:
        """Compute delay for an API retry attempt (0-indexed)."""
        base = min(
            self.initial_delay * self.exponential_base**attempt,
            self.max_delay,
        )
        jittered = base + random.uniform(0, base * self.jitter)  # noqa: S311

        if isinstance(error, LlmRateLimitError) and error.retry_after:
            return max(error.retry_after, base) + random.uniform(  # noqa: S311
                0, base * self.jitter
            )

        return jittered
