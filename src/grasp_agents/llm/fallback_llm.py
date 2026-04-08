"""FallbackLLM — composite LLM that tries models in order."""

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .llm import LLM
from ..types.items import InputItem
from ..types.llm_errors import LlmErrorTuple
from ..types.llm_events import LlmEvent, ResponseFallback
from ..types.response import Response
from ..types.tool import BaseTool, ToolChoice

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FallbackLLM(LLM):
    """
    Tries LLMs in order until one succeeds.

    The primary LLM's model_name is used for logging/cost tracking
    unless overridden.
    """

    primary: LLM = field(default=None)  # type: ignore[assignment]
    fallbacks: tuple[LLM, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_name:
            object.__setattr__(self, "model_name", self.primary.model_name)

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        candidates = [self.primary, *self.fallbacks]
        last_error: Exception | None = None

        for llm in candidates:
            try:
                return await llm._generate_response_once(
                    input,
                    tools=tools,
                    response_schema=response_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                )
            except LlmErrorTuple as e:
                last_error = e
                logger.warning(
                    "Model %s failed (%s), trying next fallback",
                    llm.model_name,
                    type(e).__name__,
                )

        assert last_error is not None
        raise last_error

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        candidates = [self.primary, *self.fallbacks]
        last_error: Exception | None = None
        seq = 0
        attempt = 0

        for llm in candidates:
            try:
                async for event in llm._generate_response_stream_once(
                    input,
                    tools=tools,
                    response_schema=response_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                ):
                    seq = event.sequence_number
                    yield event
                return

            except LlmErrorTuple as e:
                last_error = e
                attempt += 1

                idx = candidates.index(llm)
                next_model = (
                    candidates[idx + 1].model_name
                    if idx + 1 < len(candidates)
                    else "none"
                )
                yield ResponseFallback(
                    sequence_number=seq + 1,
                    failed_model=llm.model_name,
                    fallback_model=next_model,
                    error_type=type(e).__name__,
                    attempt=attempt,
                )
                logger.warning(
                    "Model %s failed (%s), trying next fallback",
                    llm.model_name,
                    type(e).__name__,
                )

        assert last_error is not None
        raise last_error
