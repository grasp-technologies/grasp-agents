"""
LLM base interface using OpenResponses types.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Self, final
from uuid import uuid4

from pydantic import BaseModel
from typing_extensions import TypedDict

from ..types.errors import (
    JSONSchemaValidationError,
    LLMResponseValidationError,
    LLMToolCallValidationError,
)
from ..types.items import InputItem
from ..types.llm_errors import LlmErrorTuple
from ..types.llm_events import LlmEvent, ResponseCompleted, ResponseRetrying
from ..types.response import Response
from ..types.tool import BaseTool, ToolChoice
from ..utils.validation import validate_obj_from_json_or_py_string
from .model_info import ModelCapabilities, get_model_capabilities
from .resilience import RetryPolicy

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (LLMToolCallValidationError, LLMResponseValidationError)


class LLMSettings(TypedDict, total=False):
    temperature: float | None
    top_p: float | None


@dataclass(frozen=True)
class LLM(ABC):
    model_name: str
    llm_settings: LLMSettings | None = None
    model_id: str = field(default_factory=lambda: str(uuid4())[:8])
    litellm_provider: str | None = None
    retry_policy: RetryPolicy | None = None

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
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Response: ...

    @abstractmethod
    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
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
        input: Sequence[InputItem],  # noqa: A002
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
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        """Streaming variant of API retry loop. Yields ResponseRetrying on retry."""
        policy = self.retry_policy

        if not policy:
            async for event in self._generate_response_stream_once(
                input,
                tools=tools,
                output_schema=output_schema,
                tool_choice=tool_choice,
                **extra_llm_settings,
            ):
                yield event
            return

        attempt = 0
        last_seq = 0

        while True:
            try:
                async for event in self._generate_response_stream_once(
                    input,
                    tools=tools,
                    output_schema=output_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                ):
                    last_seq = event.sequence_number
                    yield event
                return
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
                    yield ResponseRetrying(
                        attempt=attempt, error=str(err), sequence_number=last_seq + 1
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    # --- Public interface ---

    @final
    async def generate_response(
        self,
        input: Sequence[InputItem],  # noqa: A002
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
        while n_attempt <= max_validation:
            try:
                response = await self._generate_with_api_retries(
                    input,
                    tools=tools,
                    output_schema=output_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                )
                self._validate_response(
                    response, tools=tools, output_schema=output_schema
                )
                return response

            except _RETRYABLE_ERRORS as err:
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
        input: Sequence[InputItem],  # noqa: A002
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
        while n_attempt <= max_validation:
            try:
                async for event in self._generate_stream_with_api_retries(
                    input,
                    tools=tools,
                    output_schema=output_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                ):
                    yield event
                    last_seq = event.sequence_number

                    if isinstance(event, ResponseCompleted):
                        self._validate_response(
                            event.response, tools=tools, output_schema=output_schema
                        )
                return

            except _RETRYABLE_ERRORS as err:
                n_attempt += 1
                if n_attempt <= max_validation:
                    logger.warning(
                        "LLM response failed [%s] (retry %d): %s",
                        self.model_name,
                        n_attempt,
                        err,
                    )
                    yield ResponseRetrying(
                        attempt=n_attempt,
                        error=str(err),
                        sequence_number=last_seq + 1,
                    )
                else:
                    raise

    # --- Validation ---

    def _validate_response(
        self,
        response: Response,
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
    ) -> None:
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
        for tc in response.tool_call_items:
            if tc.name not in available_tool_names:
                raise LLMToolCallValidationError(
                    tc.name,
                    tc.arguments,
                    message=f"Tool '{tc.name}' is not available "
                    f"(available: {available_tool_names})",
                )
            tool = tools[tc.name]
            try:
                validate_obj_from_json_or_py_string(
                    tc.arguments, schema=tool.llm_in_type
                )
            except JSONSchemaValidationError as exc:
                raise LLMToolCallValidationError(
                    tc.name,
                    tc.arguments,
                    message=f"Tool '{tc.name}' arguments failed validation: {exc}",
                ) from exc
