"""
Tests for the RecoveryHint taxonomy and error classifier.

Verifies that:
- Every LlmError subclass maps to the expected hint
- MRO precedence picks the most specific hint (e.g. RateLimit beats ApiStatus)
- Instance-level ``recovery_hint`` attribute overrides class registry
- ``register_recovery_hint`` works for user-defined exception types
- ``RetryPolicy.is_retryable_api_error`` matches is_retryable(classify)
  for every LlmError subclass
"""

from collections.abc import Callable

import httpx
import pytest

from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.types.llm_errors import (
    LlmApiConnectionError,
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
from grasp_agents.types.recovery import (
    RecoveryHint,
    classify_error,
    is_retryable,
    register_recovery_hint,
    unregister_recovery_hint,
)


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.example.test/v1/responses")


def _fake_response() -> httpx.Response:
    return httpx.Response(status_code=500, request=_fake_request())


class TestRecoveryHintClassification:
    """Each LlmError subclass must classify to a deterministic hint."""

    def test_rate_limit_maps_to_rate_limited(self) -> None:
        err = LlmRateLimitError(
            "slow down", response=_fake_response(), body=None, retry_after=2.0
        )
        assert classify_error(err) is RecoveryHint.RATE_LIMITED

    def test_context_window_beats_bad_request_parent(self) -> None:
        """LlmContextWindowError must map to NEEDS_COMPACTION, not INVALID_REQUEST."""
        err = LlmContextWindowError(
            "too long",
            response=_fake_response(),
            body={"error": "context_length_exceeded"},
        )
        assert classify_error(err) is RecoveryHint.NEEDS_COMPACTION

    def test_authentication_maps_to_reauth(self) -> None:
        err = LlmAuthenticationError("bad key", response=_fake_response(), body=None)
        assert classify_error(err) is RecoveryHint.REAUTH_REQUIRED

    def test_permission_denied_maps_to_reauth(self) -> None:
        err = LlmPermissionDeniedError(
            "forbidden", response=_fake_response(), body=None
        )
        assert classify_error(err) is RecoveryHint.REAUTH_REQUIRED

    def test_bad_request_maps_to_invalid(self) -> None:
        err = LlmBadRequestError("bad input", response=_fake_response(), body=None)
        assert classify_error(err) is RecoveryHint.INVALID_REQUEST

    def test_not_found_maps_to_invalid(self) -> None:
        err = LlmNotFoundError("not there", response=_fake_response(), body=None)
        assert classify_error(err) is RecoveryHint.INVALID_REQUEST

    def test_unprocessable_maps_to_invalid(self) -> None:
        err = LlmUnprocessableEntityError("bad", response=_fake_response(), body=None)
        assert classify_error(err) is RecoveryHint.INVALID_REQUEST

    def test_conflict_maps_to_transient(self) -> None:
        err = LlmConflictError("conflict", response=_fake_response(), body=None)
        assert classify_error(err) is RecoveryHint.TRANSIENT

    def test_timeout_maps_to_transient(self) -> None:
        err = LlmApiTimeoutError(request=_fake_request())
        assert classify_error(err) is RecoveryHint.TRANSIENT

    def test_connection_maps_to_transient(self) -> None:
        err = LlmApiConnectionError(request=_fake_request())
        assert classify_error(err) is RecoveryHint.TRANSIENT

    def test_internal_server_maps_to_transient(self) -> None:
        err = LlmInternalServerError("boom", response=_fake_response(), body=None)
        assert classify_error(err) is RecoveryHint.TRANSIENT

    def test_content_filter_maps_to_refused(self) -> None:
        err = LlmContentFilterError()
        assert classify_error(err) is RecoveryHint.CONTENT_REFUSED

    def test_unknown_exception_maps_to_unknown(self) -> None:
        assert classify_error(RuntimeError("mystery")) is RecoveryHint.UNKNOWN
        assert classify_error(ValueError("also mystery")) is RecoveryHint.UNKNOWN


class TestClassAttributeStyle:
    """A ``recovery_hint`` class attribute on a user-defined exception works."""

    def test_class_attribute_is_picked_up(self) -> None:
        class MyUserError(Exception):
            recovery_hint = RecoveryHint.INVALID_REQUEST

        assert classify_error(MyUserError("x")) is RecoveryHint.INVALID_REQUEST

    def test_subclass_inherits_class_attribute(self) -> None:
        class ParentError(Exception):
            recovery_hint = RecoveryHint.NEEDS_COMPACTION

        class ChildError(ParentError):
            pass

        assert classify_error(ChildError()) is RecoveryHint.NEEDS_COMPACTION

    def test_child_class_attribute_beats_parent(self) -> None:
        class ParentError(Exception):
            recovery_hint = RecoveryHint.TRANSIENT

        class ChildError(ParentError):
            recovery_hint = RecoveryHint.REAUTH_REQUIRED

        assert classify_error(ChildError()) is RecoveryHint.REAUTH_REQUIRED
        assert classify_error(ParentError()) is RecoveryHint.TRANSIENT

    def test_non_hint_attribute_is_ignored(self) -> None:
        """A class attribute that is not a RecoveryHint falls through to registry."""

        class MyUserError(LlmBadRequestError):
            recovery_hint = "not-a-real-hint"  # type: ignore[assignment]

        err = MyUserError("x", response=_fake_response(), body=None)
        # Falls through to built-in registry mapping for LlmBadRequestError.
        assert classify_error(err) is RecoveryHint.INVALID_REQUEST


class TestRegisterRecoveryHint:
    """register_recovery_hint allows user-defined exceptions to carry hints."""

    def test_register_and_classify_custom_exception(self) -> None:
        class MyCustomError(Exception):
            pass

        assert classify_error(MyCustomError()) is RecoveryHint.UNKNOWN
        register_recovery_hint(MyCustomError, RecoveryHint.TRANSIENT)
        try:
            assert classify_error(MyCustomError()) is RecoveryHint.TRANSIENT
        finally:
            unregister_recovery_hint(MyCustomError)

    def test_subclass_inherits_parent_hint_via_mro(self) -> None:
        class ParentError(Exception):
            pass

        class ChildError(ParentError):
            pass

        register_recovery_hint(ParentError, RecoveryHint.CONTENT_REFUSED)
        try:
            assert classify_error(ChildError()) is RecoveryHint.CONTENT_REFUSED
        finally:
            unregister_recovery_hint(ParentError)

    def test_child_registration_beats_parent(self) -> None:
        class ParentError(Exception):
            pass

        class ChildError(ParentError):
            pass

        register_recovery_hint(ParentError, RecoveryHint.TRANSIENT)
        register_recovery_hint(ChildError, RecoveryHint.REAUTH_REQUIRED)
        try:
            assert classify_error(ChildError()) is RecoveryHint.REAUTH_REQUIRED
            assert classify_error(ParentError()) is RecoveryHint.TRANSIENT
        finally:
            unregister_recovery_hint(ChildError)
            unregister_recovery_hint(ParentError)

    def test_unregister_is_idempotent(self) -> None:
        """Unregistering an unknown type is a no-op, not an error."""

        class NotRegistered(Exception):
            pass

        unregister_recovery_hint(NotRegistered)  # should not raise


class TestIsRetryable:
    def test_transient_and_rate_limited_are_retryable(self) -> None:
        assert is_retryable(RecoveryHint.TRANSIENT)
        assert is_retryable(RecoveryHint.RATE_LIMITED)

    def test_non_transient_are_not_retryable(self) -> None:
        assert not is_retryable(RecoveryHint.NEEDS_COMPACTION)
        assert not is_retryable(RecoveryHint.REAUTH_REQUIRED)
        assert not is_retryable(RecoveryHint.CONTENT_REFUSED)
        assert not is_retryable(RecoveryHint.INVALID_REQUEST)
        assert not is_retryable(RecoveryHint.UNKNOWN)


ErrorFactory = Callable[[], Exception]


class TestRetryPolicyAlignment:
    """RetryPolicy.is_retryable_api_error must agree with is_retryable(classify)."""

    @pytest.mark.parametrize(
        ("factory", "expected_retryable"),
        [
            # Retryable
            (lambda: LlmApiTimeoutError(request=_fake_request()), True),
            (lambda: LlmApiConnectionError(request=_fake_request()), True),
            (
                lambda: LlmRateLimitError(
                    "x", response=_fake_response(), body=None
                ),
                True,
            ),
            (
                lambda: LlmInternalServerError(
                    "x", response=_fake_response(), body=None
                ),
                True,
            ),
            (
                lambda: LlmConflictError("x", response=_fake_response(), body=None),
                True,
            ),
            # Non-retryable
            (
                lambda: LlmAuthenticationError(
                    "x", response=_fake_response(), body=None
                ),
                False,
            ),
            (
                lambda: LlmPermissionDeniedError(
                    "x", response=_fake_response(), body=None
                ),
                False,
            ),
            (
                lambda: LlmBadRequestError(
                    "x", response=_fake_response(), body=None
                ),
                False,
            ),
            (
                lambda: LlmNotFoundError("x", response=_fake_response(), body=None),
                False,
            ),
            (
                lambda: LlmContextWindowError(
                    "x", response=_fake_response(), body=None
                ),
                False,
            ),
            (lambda: LlmContentFilterError(), False),
            # Non-LlmError should not be retryable via this API.
            (lambda: RuntimeError("mystery"), False),
        ],
    )
    def test_alignment_with_classifier(
        self, factory: ErrorFactory, expected_retryable: bool
    ) -> None:
        policy = RetryPolicy()
        err = factory()
        assert policy.is_retryable_api_error(err) is expected_retryable

    def test_policy_classify_returns_hint(self) -> None:
        policy = RetryPolicy()
        err = LlmRateLimitError(
            "slow", response=_fake_response(), body=None, retry_after=2.0
        )
        assert policy.classify(err) is RecoveryHint.RATE_LIMITED


class TestPublicExport:
    def test_top_level_import(self) -> None:
        """Ensure public surface is exposed from top-level package."""
        from grasp_agents import (
            RecoveryHint as PkgRecoveryHint,
        )
        from grasp_agents import (
            classify_error as pkg_classify_error,
        )
        from grasp_agents import (
            is_retryable as pkg_is_retryable,
        )
        from grasp_agents import (
            register_recovery_hint as pkg_register_recovery_hint,
        )

        assert PkgRecoveryHint is RecoveryHint
        assert pkg_classify_error is classify_error
        assert pkg_is_retryable is is_retryable
        assert pkg_register_recovery_hint is register_recovery_hint
