"""
Recovery-hint taxonomy for runtime errors.

An orthogonal axis to the error-type hierarchy: ``RecoveryHint`` describes
*what action recovers from this error*, letting retry/fallback/compaction
logic key off a single stable label instead of re-matching types everywhere.

Usage:

    from grasp_agents.types.recovery import RecoveryHint, classify_error

    try:
        await llm.generate(...)
    except Exception as err:
        hint = classify_error(err)
        if hint is RecoveryHint.NEEDS_COMPACTION:
            await compact_memory()
        elif hint is RecoveryHint.RATE_LIMITED:
            await asyncio.sleep(getattr(err, "retry_after", 1.0))
        ...

Extensibility (two supported ways to associate a hint with an exception):

1. For exception types you own — define a ``recovery_hint`` class attribute:

       class MyValidationError(Exception):
           recovery_hint = RecoveryHint.INVALID_REQUEST

2. For exception types you do *not* own (e.g. third-party library errors)
   — call :func:`register_recovery_hint` at import time.
"""

from enum import StrEnum

from .llm_errors import (
    LlmApiConnectionError,
    LlmApiError,
    LlmApiStatusError,
    LlmApiTimeoutError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmConflictError,
    LlmContentFilterError,
    LlmContextWindowError,
    LlmInternalServerError,
    LlmNotFoundError,
    LlmPermissionDeniedError,
    LlmRateLimitError,
    LlmUnprocessableEntityError,
)


class RecoveryHint(StrEnum):
    """
    What action is expected to recover from a given error.

    Each value describes a recovery *strategy*, not an error category.
    Use :func:`classify_error` to map an exception to the hint the
    framework recommends; callers are free to override per-call.
    """

    TRANSIENT = "transient"
    """Network/server blip. Retry the same request with backoff."""

    RATE_LIMITED = "rate_limited"
    """Server asked us to slow down. Honor ``retry_after`` if present."""

    NEEDS_COMPACTION = "needs_compaction"
    """Input exceeded the model's context window; reduce before retrying."""

    REAUTH_REQUIRED = "reauth_required"
    """Credentials are missing, expired, or unauthorized for this call."""

    CONTENT_REFUSED = "content_refused"
    """Upstream refused to produce a response (content filter / policy)."""

    INVALID_REQUEST = "invalid_request"
    """Request shape is wrong. Treat as a programmer bug; do not retry."""

    UNKNOWN = "unknown"
    """No classification is available. Callers should treat conservatively."""


_HINT_REGISTRY: dict[type[BaseException], RecoveryHint] = {
    # Rate limiting — check before the generic status parent.
    LlmRateLimitError: RecoveryHint.RATE_LIMITED,
    # Context-window overflow — specific subclass of BadRequestError; must be
    # checked before LlmBadRequestError so the more specific hint wins.
    LlmContextWindowError: RecoveryHint.NEEDS_COMPACTION,
    # Authentication / authorization.
    LlmAuthenticationError: RecoveryHint.REAUTH_REQUIRED,
    LlmPermissionDeniedError: RecoveryHint.REAUTH_REQUIRED,
    # Content filtering (model refused to produce output).
    LlmContentFilterError: RecoveryHint.CONTENT_REFUSED,
    # Request-shape bugs — retrying won't help without code change.
    LlmBadRequestError: RecoveryHint.INVALID_REQUEST,
    LlmNotFoundError: RecoveryHint.INVALID_REQUEST,
    LlmUnprocessableEntityError: RecoveryHint.INVALID_REQUEST,
    # Transient server / network issues.
    LlmConflictError: RecoveryHint.TRANSIENT,
    LlmApiTimeoutError: RecoveryHint.TRANSIENT,
    LlmApiConnectionError: RecoveryHint.TRANSIENT,
    LlmInternalServerError: RecoveryHint.TRANSIENT,
    # Generic parents — matched only if no subclass matched first.
    LlmApiStatusError: RecoveryHint.TRANSIENT,
    LlmApiError: RecoveryHint.TRANSIENT,
}


def register_recovery_hint(
    exc_type: type[BaseException], hint: RecoveryHint
) -> None:
    """
    Associate a recovery hint with an exception type.

    Overrides any previously registered hint for ``exc_type``.
    Subclass lookups use Python's MRO, so registering a hint for a parent
    class applies to all subclasses that do not have their own entry.
    """
    _HINT_REGISTRY[exc_type] = hint


def unregister_recovery_hint(exc_type: type[BaseException]) -> None:
    """
    Remove any hint registered for ``exc_type``.

    No-op if ``exc_type`` is not in the registry. Intended for cleanup
    after test fixtures or dynamic registrations — application code
    should generally register once at startup and leave entries in place.
    """
    _HINT_REGISTRY.pop(exc_type, None)


def classify_error(err: BaseException) -> RecoveryHint:
    """
    Return the recovery hint the framework recommends for ``err``.

    Classification order:

    1. If the exception's class defines a ``recovery_hint`` attribute of
       type :class:`RecoveryHint`, use it. Subclasses inherit it via the
       normal Python MRO. This is the intended extensibility path for
       user-defined exception types.
    2. Walk the type's MRO and return the first hint registered in
       :func:`register_recovery_hint` or the built-in registry. This is
       the extensibility path for third-party exception types you don't
       own.
    3. Fall back to :attr:`RecoveryHint.UNKNOWN`.
    """
    explicit = getattr(err, "recovery_hint", None)
    if isinstance(explicit, RecoveryHint):
        return explicit

    for cls in type(err).__mro__:
        hint = _HINT_REGISTRY.get(cls)
        if hint is not None:
            return hint

    return RecoveryHint.UNKNOWN


_RETRYABLE_HINTS = frozenset(
    {RecoveryHint.TRANSIENT, RecoveryHint.RATE_LIMITED}
)


def is_retryable(hint: RecoveryHint) -> bool:
    """Return whether retrying the same request is expected to help."""
    return hint in _RETRYABLE_HINTS
