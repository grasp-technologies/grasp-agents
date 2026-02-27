"""
LLM base interface using OpenResponses StreamEvent / Response types.

Replaces the Completion/CompletionChunk-based interface in llm.py.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, final
from uuid import uuid4

from pydantic import BaseModel
from typing_extensions import TypedDict

from .errors import LLMToolCallValidationError
from .typing.response import Response
from .typing.stream_events import ResponseCompleted, StreamEvent
from .typing.tool import BaseTool, ToolChoice
from .utils.validation import validate_obj_from_json_or_py_string

logger = logging.getLogger(__name__)


ResponseStreamGenerator = AsyncIterator[StreamEvent]


class LLMSettings(TypedDict, total=False):
    max_completion_tokens: int | None
    temperature: float | None
    top_p: float | None
    seed: int | None


@dataclass(frozen=True)
class LLM(ABC):
    model_name: str
    llm_settings: LLMSettings | None = None
    model_id: str = field(default_factory=lambda: str(uuid4())[:8])
    max_response_retries: int = 0

    # --- Abstract methods for subclasses ---

    @abstractmethod
    async def _generate_response_once(
        self,
        input: list[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        ...

    @abstractmethod
    async def _generate_response_stream_once(
        self,
        input: list[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[StreamEvent]:
        yield  # type: ignore[misc]

    # --- Public interface ---

    @final
    async def generate_response(
        self,
        input: list[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        n_attempt = 0
        last_err: BaseException | None = None
        while n_attempt <= self.max_response_retries:
            try:
                response = await self._generate_response_once(
                    input,
                    tools=tools,
                    response_schema=response_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                )
                self._validate_response(response, tools=tools)
                return response

            except Exception as err:
                last_err = err
                n_attempt += 1
                if n_attempt <= self.max_response_retries:
                    logger.warning(
                        f"\nLLM response failed (retry {n_attempt}): {err}"
                    )
                else:
                    raise

        raise last_err or RuntimeError("Unexpected: retry loop exited")

    @final
    async def generate_response_stream(
        self,
        input: list[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[StreamEvent]:
        n_attempt = 0
        while n_attempt <= self.max_response_retries:
            try:
                async for event in self._generate_response_stream_once(
                    input,
                    tools=tools,
                    response_schema=response_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                ):
                    yield event

                    # Validate the final response
                    if isinstance(event, ResponseCompleted):
                        self._validate_response(
                            Response.model_validate(event.response),
                            tools=tools,
                        )
                return

            except Exception as err:
                n_attempt += 1
                if n_attempt <= self.max_response_retries:
                    logger.warning(
                        f"\nLLM response failed (retry {n_attempt}): {err}"
                    )
                else:
                    raise

    # --- Validation ---

    def _validate_response(
        self,
        response: Response,
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
    ) -> None:
        if tools is not None:
            self._validate_tool_calls(response, tools)

    def _validate_tool_calls(
        self,
        response: Response,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]],
    ) -> None:
        for tc in response.tool_call_items:
            available_tool_names = list(tools)
            if tc.name not in available_tool_names:
                raise LLMToolCallValidationError(
                    tc.name,
                    tc.arguments,
                    message=f"Tool '{tc.name}' is not available "
                    f"(available: {available_tool_names})",
                )
            tool = tools[tc.name]
            validate_obj_from_json_or_py_string(
                tc.arguments, schema=tool.in_type
            )
