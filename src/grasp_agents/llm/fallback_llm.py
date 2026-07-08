"""FallbackLLM — composite LLM that tries models in order."""

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from grasp_agents.tools.base import BaseTool, ToolChoice
from grasp_agents.types.items import InputItem
from grasp_agents.types.llm_errors import LlmErrorTuple
from grasp_agents.types.llm_events import LlmEvent, ResponseCompleted, ResponseFallback
from grasp_agents.types.response import Response

from .llm import LLM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FallbackLLM(LLM):
    """
    Tries LLMs in order until one succeeds.

    The public ``generate_response(_stream)`` entrypoints validate and retry
    (per this instance's ``retry_policy``) around the whole cascade; member
    retry policies are bypassed — the cascade itself is the API-retry layer.
    ``model_name`` defaults to the primary's and is used for logging, cost
    attribution and context-budget resolution. ``llm_settings`` is inert
    here: settings live on the member LLMs.
    """

    model_name: str = ""
    primary: LLM = field(kw_only=True)
    fallbacks: tuple[LLM, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_name:
            object.__setattr__(self, "model_name", self.primary.model_name)

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        candidates = [self.primary, *self.fallbacks]
        last_error: Exception | None = None

        for llm in candidates:
            try:
                # Call the un-retried core on purpose: the cascade is the
                # retry layer here.
                response = await llm._generate_response_once(  # noqa: SLF001
                    input,
                    tools=tools,
                    output_schema=output_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                )
                # CloudLLM members stamp at the source; this covers members
                # that don't (so cost is attributed to the SERVING member,
                # never to the primary's name reported by this wrapper).
                llm._stamp_cost(response)  # noqa: SLF001
                return response
            except LlmErrorTuple as e:
                last_error = e
                logger.warning(
                    "Model %s failed (%s: %s), trying next fallback",
                    llm.model_name,
                    type(e).__name__,
                    e,
                )

        assert last_error is not None
        raise last_error

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        candidates = [self.primary, *self.fallbacks]
        last_error: Exception | None = None
        seq = 0
        attempt = 0

        for llm in candidates:
            try:
                async for event in llm._generate_response_stream_once(  # noqa: SLF001
                    input,
                    tools=tools,
                    output_schema=output_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                ):
                    if isinstance(event, ResponseCompleted):
                        llm._stamp_cost(event.response)  # noqa: SLF001
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
                    "Model %s failed (%s: %s), trying next fallback",
                    llm.model_name,
                    type(e).__name__,
                    e,
                )

        assert last_error is not None
        raise last_error
