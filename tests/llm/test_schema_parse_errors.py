"""
Provider-side structured-output parse failures.

Provider SDKs validate the caller's output schema outside their own error
handling, so a mismatch arrives as a bare ``pydantic.ValidationError``
rather than an SDK error type. Re-sampling the same model is the recovery,
so it has to reach the validation-retry layer — not the API-retry and
fallback layers, which would burn the retry budget and swap models over a
re-rollable generation failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from grasp_agents.llm.cloud_llm import ApiCallParams
from grasp_agents.llm.fallback_llm import FallbackLLM
from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.errors import LLMResponseValidationError
from grasp_agents.types.items import InputItem
from grasp_agents.types.llm_errors import LlmResponseSchemaError
from grasp_agents.types.response import Response
from tests.llm.test_resilience import (
    _USER_MSG,
    RealMapperCloudLLM,
    StubLLM,
    _text_response,
)


class _Schema(BaseModel):
    n: int


def _parse_failure() -> ValidationError:
    try:
        _Schema.model_validate_json('{"n": "not-an-int"}')
    except ValidationError as err:
        return err
    raise AssertionError("expected a ValidationError")


@dataclass(frozen=True)
class BadInputCloudLLM(RealMapperCloudLLM):
    """Fails while *building* the request, before the mapped region."""

    def _make_api_input(
        self,
        input: Sequence[InputItem],
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: Any | None = None,
        output_schema: Any | None = None,
        **extra_llm_settings: Any,
    ) -> ApiCallParams:
        raise _parse_failure()


@dataclass(frozen=True)
class BadConversionCloudLLM(RealMapperCloudLLM):
    """The provider call succeeds; our conversion of its response fails."""

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        object.__setattr__(self, "_attempts", self.attempts + 1)
        return object()

    def _convert_api_response(self, raw: Any) -> Response:
        raise _parse_failure()


_VALIDATION_ONLY = RetryPolicy(api_retries=0, validation_retries=2, initial_delay=0.0)


class TestParseFailureRoutesToValidationRetries:
    @pytest.mark.asyncio
    async def test_parse_failure_becomes_response_validation_error(self) -> None:
        llm = RealMapperCloudLLM(
            model_name="primary",
            retry_policy=_VALIDATION_ONLY,
            raw_error=_parse_failure(),
        )

        with pytest.raises(LLMResponseValidationError) as excinfo:
            await llm.generate_response(_USER_MSG, output_schema=_Schema)

        assert isinstance(excinfo.value.__cause__, ValidationError)
        assert excinfo.value.schema is _Schema

    @pytest.mark.asyncio
    async def test_parse_failure_is_resampled_not_api_retried(self) -> None:
        # api_retries=0, so reaching 3 attempts proves it went through the
        # validation-retry layer rather than the API one.
        llm = RealMapperCloudLLM(
            model_name="primary",
            retry_policy=_VALIDATION_ONLY,
            raw_error=_parse_failure(),
        )

        with pytest.raises(LLMResponseValidationError):
            await llm.generate_response(_USER_MSG, output_schema=_Schema)

        assert llm.attempts == 3  # initial + 2 validation retries

    @pytest.mark.asyncio
    async def test_parse_failure_does_not_swap_models(self) -> None:
        """A re-rollable generation failure must not spend the cascade."""
        primary = RealMapperCloudLLM(
            model_name="primary",
            retry_policy=RetryPolicy(api_retries=0, validation_retries=0),
            raw_error=_parse_failure(),
        )
        fallback = StubLLM(model_name="fallback", response=_text_response("rescued"))
        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))

        with pytest.raises(LLMResponseValidationError):
            await llm.generate_response(_USER_MSG, output_schema=_Schema)


class TestUnrelatedValidationErrorsAreNotConverted:
    """
    Only the provider-call region is converted. A ``ValidationError`` from
    anywhere else is our own bug and must surface immediately rather than
    being re-sampled as if the model produced it.
    """

    @pytest.mark.asyncio
    async def test_request_building_failure_propagates_raw(self) -> None:
        llm = BadInputCloudLLM(
            model_name="primary",
            retry_policy=_VALIDATION_ONLY,
            raw_error=_parse_failure(),
        )

        with pytest.raises(ValidationError):
            await llm.generate_response(_USER_MSG, output_schema=_Schema)

        assert llm.attempts == 0


class TestResponseConversionFailures:
    """
    A response the provider returned but our types reject is not the model's
    structured output: re-sampling or retrying fails the same way, while
    another model can still serve the turn.
    """

    @pytest.mark.asyncio
    async def test_is_a_schema_error_and_not_retried(self) -> None:
        llm = BadConversionCloudLLM(
            model_name="primary",
            retry_policy=RetryPolicy(
                api_retries=2, validation_retries=2, initial_delay=0.0
            ),
        )

        with pytest.raises(LlmResponseSchemaError):
            await llm.generate_response(_USER_MSG, output_schema=_Schema)

        assert llm.attempts == 1

    @pytest.mark.asyncio
    async def test_advances_the_fallback(self) -> None:
        primary = BadConversionCloudLLM(model_name="primary", retry_policy=None)
        fallback = StubLLM(model_name="fallback", response=_text_response("rescued"))
        llm = FallbackLLM(primary=primary, fallbacks=(fallback,))

        response = await llm.generate_response(_USER_MSG)

        assert response.output_text == "rescued"
