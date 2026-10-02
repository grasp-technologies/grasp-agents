"""
LLM base interface using OpenResponses types.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Self, TypedDict, final
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, with_config

from grasp_agents import grasp_logging
from grasp_agents.tools.base import BaseTool, ToolChoice
from grasp_agents.types.errors import (
    JSONSchemaValidationError,
    LLMResponseValidationError,
    LLMToolCallValidationError,
)
from grasp_agents.types.items import InputItem
from grasp_agents.types.llm_errors import LlmContentFilterError, LlmErrorTuple
from grasp_agents.types.llm_events import (
    LlmEvent,
    ResponseCompleted,
    ResponseIncomplete,
    ResponseRetrying,
)
from grasp_agents.types.response import REFUSAL_CATEGORY_KEY, Response, ResponseUsage
from grasp_agents.utils.validation import validate_obj_from_json_or_py_string

from .model_info import ModelCapabilities, get_model_capabilities
from .resilience import RetryPolicy

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (LLMToolCallValidationError, LLMResponseValidationError)


def _superseded_usage(
    previous: ResponseUsage | None, failed: Response | None
) -> ResponseUsage | None:
    if failed is None or failed.usage is None:
        return previous
    return failed.usage if previous is None else previous + failed.usage


async def _aclose(stream: AsyncIterator[Any]) -> None:
    """Close an abandoned attempt's stream now rather than at garbage collection."""
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()


@with_config(ConfigDict(extra="allow"))
class LLMSettings(TypedDict, total=False):
    temperature: float | None
    top_p: float | None


@dataclass(frozen=True)
class LLM(ABC):
    model_name: str
    llm_settings: LLMSettings | None = None
    model_id: str = field(default_factory=lambda: str(uuid4())[:8])
    litellm_provider: str | None = None
    # The framework's retry layer is the ONE retry system: provider SDK
    # client retries default to 0 so the two never multiply. ``None``
    # disables retries entirely.
    retry_policy: RetryPolicy | None = field(default_factory=RetryPolicy)

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        # Frozen + non-copyable SDK clients (AsyncOpenAI, etc.) — share by ref
        return self

    @cached_property
    def capabilities(self) -> ModelCapabilities:
        """Model capabilities from LiteLLM's database."""
        return get_model_capabilities(self.model_name, self.litellm_provider)

    # --- Abstract methods for subclasses ---

    @abstractmethod
    async def _generate_response_once(
        self,
        input: Sequence[InputItem],  # ruff: ignore[builtin-argument-shadowing]
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Response: ...

    @abstractmethod
    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # ruff: ignore[builtin-argument-shadowing]
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        yield NotImplemented

    # --- API retry layer ---

    async def _generate_with_api_retries(
        self,
        input: Sequence[InputItem],  # ruff: ignore[builtin-argument-shadowing]
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        """Inner retry loop for transient API errors."""
        policy = self.retry_policy
        if not policy:
            return await self._generate_response_once(
                input,
                tools=tools,
                output_schema=output_schema,
                tool_choice=tool_choice,
                **extra_llm_settings,
            )

        attempt = 0
        while True:
            try:
                return await self._generate_response_once(
                    input,
                    tools=tools,
                    output_schema=output_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                )
            except LlmErrorTuple as err:
                attempt += 1
                if policy.is_retryable_api_error(err) and attempt <= policy.api_retries:
                    delay = policy.api_delay_for(attempt - 1, err)
                    logger.warning(
                        "Model %s: %s (attempt %d/%d, retrying in %.1fs)",
                        self.model_name,
                        type(err).__name__,
                        attempt,
                        policy.api_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    async def _generate_stream_with_api_retries(
        self,
        input: Sequence[InputItem],  # ruff: ignore[builtin-argument-shadowing]
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        """
        Streaming variant of API retry loop. Yields ResponseRetrying on retry;
        ``attempt`` counts this layer's own retries. An attempt ends at its final
        response: a failure after it neither retries nor raises.
        """
        policy = self.retry_policy
        attempt = 0
        last_seq = 0

        while True:
            stream = self._generate_response_stream_once(
                input,
                tools=tools,
                output_schema=output_schema,
                tool_choice=tool_choice,
                **extra_llm_settings,
            )
            delivered = False
            try:
                async for event in stream:
                    last_seq = event.sequence_number
                    yield event
                    if isinstance(event, (ResponseCompleted, ResponseIncomplete)):
                        delivered = True
                return
            except LlmErrorTuple as err:
                if delivered:
                    # Retrying or falling back would replace a complete response
                    # and bill it again.
                    logger.warning(
                        "Model %s: stream failed after its final response "
                        "(%s: %s); keeping the response",
                        self.model_name,
                        type(err).__name__,
                        err,
                    )
                    return
                attempt += 1
                if (
                    policy is not None
                    and policy.is_retryable_api_error(err)
                    and attempt <= policy.api_retries
                ):
                    delay = policy.api_delay_for(attempt - 1, err)
                    logger.warning(
                        "Model %s: %s (attempt %d/%d, retrying in %.1fs)",
                        self.model_name,
                        type(err).__name__,
                        attempt,
                        policy.api_retries,
                        delay,
                    )
                    yield ResponseRetrying(
                        attempt=attempt, error=str(err), sequence_number=last_seq + 1
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
            finally:
                await _aclose(stream)

    # --- Public interface ---

    @final
    async def generate_response(
        self,
        input: Sequence[InputItem],  # ruff: ignore[builtin-argument-shadowing]
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        max_validation = (
            self.retry_policy.validation_retries if self.retry_policy else 0
        )
        n_attempt = 0
        superseded: ResponseUsage | None = None
        while n_attempt <= max_validation:
            response: Response | None = None
            try:
                response = await self._generate_with_api_retries(
                    input,
                    tools=tools,
                    output_schema=output_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                )
                response.superseded_usage = superseded
                self._validate_response(
                    response, tools=tools, output_schema=output_schema
                )
                return response

            except _RETRYABLE_ERRORS as err:
                superseded = _superseded_usage(superseded, response)
                n_attempt += 1
                if n_attempt <= max_validation:
                    logger.warning(
                        "LLM response failed [%s] (retry %d): %s",
                        self.model_name,
                        n_attempt,
                        err,
                    )
                else:
                    raise

        raise RuntimeError("Unexpected: retry loop exited without return or raise")

    @final
    async def generate_response_stream(
        self,
        input: Sequence[InputItem],  # ruff: ignore[builtin-argument-shadowing]
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        max_validation = (
            self.retry_policy.validation_retries if self.retry_policy else 0
        )
        n_attempt = 0
        last_seq = 0
        superseded: ResponseUsage | None = None
        while n_attempt <= max_validation:
            attempt_response: Response | None = None
            stream = self._generate_stream_with_api_retries(
                input,
                tools=tools,
                output_schema=output_schema,
                tool_choice=tool_choice,
                **extra_llm_settings,
            )
            try:
                async for event in stream:
                    if isinstance(event, (ResponseCompleted, ResponseIncomplete)):
                        attempt_response = event.response
                        attempt_response.superseded_usage = superseded
                        self._validate_response(
                            attempt_response, tools=tools, output_schema=output_schema
                        )
                    yield event
                    last_seq = event.sequence_number
                return

            except _RETRYABLE_ERRORS as err:
                superseded = _superseded_usage(superseded, attempt_response)
                n_attempt += 1
                if n_attempt <= max_validation:
                    logger.warning(
                        "LLM response failed [%s] (retry %d): %s",
                        self.model_name,
                        n_attempt,
                        err,
                    )
                    # ``attempt`` counts validation retries; API retries keep
                    # their own count in the layer below.
                    yield ResponseRetrying(
                        attempt=n_attempt,
                        error=str(err),
                        sequence_number=last_seq + 1,
                    )
                else:
                    raise
            finally:
                # The attempt's stream is abandoned when validation raises
                # above; close it before the next attempt opens another.
                await _aclose(stream)

    # --- Validation ---

    def _check_content_filter(self, response: Response) -> None:
        """
        Raise when the provider blocked this response on policy grounds.

        Every provider normalizes such a block to
        ``incomplete_details.reason == "content_filter"``, and every one
        of them directs callers to discard whatever partial output
        preceded it — so there is nothing for the agent to act on.
        Returning it would stall the loop on a turn that adds nothing to
        the transcript, or fail schema validation with an error that
        blames the model for empty output. Raising skips retries (the
        same request is blocked again) while still advancing a
        ``FallbackLLM`` to the next model, which is the recovery
        providers recommend.

        A refusal the model *wrote* is different, and is not raised on:
        it carries text the agent can read and correct course from. See
        :meth:`_warn_refusal`.
        """
        details = response.incomplete_details
        if details is None or details.reason != "content_filter":
            return

        raw_category = (response.provider_specific_fields or {}).get(
            REFUSAL_CATEGORY_KEY
        )
        category = str(raw_category) if raw_category else None
        named = f" ({category})" if category else ""
        explanation = response.refusal or "no explanation given"
        raise LlmContentFilterError(
            f"llm {self.model_name}: content filter blocked this response"
            f"{named}: {explanation}",
            code=category,
        )

    def _warn_refusal(self, response: Response) -> None:
        """
        Log a warning when the model declined to answer in its own words.

        Deliberately not an error: the refusal text flows into the
        transcript, so the agent — and the caller — can see it and
        correct course. Structured-output calls still fail schema
        validation on the refusal text and re-sample via
        ``validation_retries``.
        """
        refusal = response.refusal
        if refusal:
            logger.warning(
                "llm %s: response is a refusal (status=%s): %s",
                self.model_name,
                response.status,
                grasp_logging.body_for_log(refusal, full=grasp_logging.LOG_LLM_OUTPUT),
            )

    def _validate_response(
        self,
        response: Response,
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
    ) -> None:
        # Before any other check: a blocked response has no output to
        # validate, and a schema failure here would misreport the cause.
        self._check_content_filter(response)
        self._warn_refusal(response)

        if tools is not None:
            self._validate_tool_calls(response, tools)

        if output_schema is not None and not response.tool_call_items:
            try:
                validate_obj_from_json_or_py_string(
                    response.output_text, schema=output_schema
                )
            except JSONSchemaValidationError as exc:
                raise LLMResponseValidationError(
                    response.output_text, output_schema
                ) from exc

    def _validate_tool_calls(
        self,
        response: Response,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]],
    ) -> None:
        available_tool_names = list(tools)
        failed: list[tuple[str, str, str]] = []  # (call_id, name, error)
        for tc in response.tool_call_items:
            if tc.name not in available_tool_names:
                failed.append(
                    (
                        tc.call_id,
                        tc.name,
                        (
                            f"Tool '{tc.name}' is not available "
                            f"(available: {available_tool_names})"
                        ),
                    )
                )
                continue
            tool = tools[tc.name]
            try:
                validate_obj_from_json_or_py_string(
                    tc.arguments, schema=tool.llm_in_type
                )
            except JSONSchemaValidationError as exc:
                failed.append(
                    (
                        tc.call_id,
                        tc.name,
                        f"Tool '{tc.name}' arguments failed validation: {exc}",
                    )
                )

        if not failed:
            return

        # Surface *every* failed call (→ logs + ResponseRetrying), mirroring
        # ``failed_calls``, which the agent loop turns into one tool_result
        # per bad call. Avoids first-error-only feedback.
        names = ", ".join(dict.fromkeys(f"'{name}'" for _, name, _ in failed))
        detail = "\n".join(f"- {msg}" for _, _, msg in failed)
        plural = "s" if len(failed) != 1 else ""
        message = (
            f"{len(failed)} tool call{plural} failed validation ({names}):\n{detail}"
        )
        raise LLMToolCallValidationError(
            message, response=response, failed_calls=failed
        )
