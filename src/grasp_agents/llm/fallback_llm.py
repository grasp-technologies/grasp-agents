"""FallbackLLM — composite LLM that tries models in order."""

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from pydantic import BaseModel

from grasp_agents.tools.base import BaseTool, ToolChoice
from grasp_agents.types.items import InputItem
from grasp_agents.types.llm_errors import LlmErrorTuple, LlmRateLimitError
from grasp_agents.types.llm_events import LlmEvent, ResponseFallback
from grasp_agents.types.recovery import classify_error, is_retryable
from grasp_agents.types.response import Response

from .llm import LLM
from .model_info import ModelCapabilities
from .resilience import RetryPolicy

logger = logging.getLogger(__name__)


def _select_cascade_error(errors: Sequence[Exception]) -> Exception:
    """
    Choose the error a fully-failed cascade raises.

    Prefer a retryable member error: it tells the caller the cascade may
    succeed later, and must not be masked by another member's deterministic
    failure (e.g. a misconfigured fallback). Among retryable errors, prefer
    one without a server-imposed backoff floor (shortest implied wait);
    among floored rate limits, the smallest ``retry_after``.
    """
    retryable = [e for e in errors if is_retryable(classify_error(e))]
    if not retryable:
        return errors[-1]
    no_floor = [
        e for e in retryable if not (isinstance(e, LlmRateLimitError) and e.retry_after)
    ]
    if no_floor:
        return no_floor[0]
    return min(
        (e for e in retryable if isinstance(e, LlmRateLimitError)),
        key=lambda e: e.retry_after or 0.0,
    )


@dataclass(frozen=True)
class FallbackLLM(LLM):
    """
    Tries LLMs in order until one succeeds.

    Each member runs its full pipeline — API retries per its own
    ``retry_policy``, response validation, and validation retries that
    re-sample the same member — before the cascade advances. A transient
    blip on the primary is retried on the primary, and a malformed response
    is re-sampled from the member that produced it: models are never
    swapped over a validation failure (validation errors propagate instead
    of cascading). Deterministic member errors (auth, bad request, context
    window) skip that member's retries and advance the cascade immediately.

    ``retry_policy`` must be ``None`` here: all retry behavior (API and
    validation) is member-level, and a composite policy would silently
    multiply member retries. ``llm_settings`` and ``litellm_provider``
    must be ``None`` too: settings live on the member LLMs, and cost /
    capability resolution uses each member's own identity.

    ``capabilities`` is the conservative merge across members — the
    smallest known token windows, and a feature only if every member
    supports it — so context budgeting and feature gating hold for
    whichever member ends up serving. ``model_name`` defaults to the
    primary's and is used for logging and tokenizer selection.
    """

    model_name: str = ""
    retry_policy: RetryPolicy | None = None
    primary: LLM = field(kw_only=True)
    fallbacks: tuple[LLM, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_name:
            object.__setattr__(self, "model_name", self.primary.model_name)
        if self.retry_policy is not None:
            raise ValueError(
                "FallbackLLM takes no retry_policy: API and validation "
                "retries are configured per member (set retry_policy on "
                "each member LLM)."
            )
        if self.llm_settings is not None:
            raise ValueError(
                "FallbackLLM takes no llm_settings: settings live on the member LLMs."
            )
        if self.litellm_provider is not None:
            raise ValueError(
                "FallbackLLM takes no litellm_provider: cost and capability "
                "resolution use each member's own identity (set "
                "litellm_provider on the member LLMs)."
            )

    @cached_property
    def capabilities(self) -> ModelCapabilities:
        """
        Conservative merge across members: the smallest known token
        windows, and a feature only if every member supports it — valid
        for whichever member serves. Members with unknown windows are
        skipped (unknown models also report permissive feature defaults).
        """
        caps = [m.capabilities for m in (self.primary, *self.fallbacks)]
        input_windows = [
            c.max_input_tokens for c in caps if c.max_input_tokens is not None
        ]
        output_windows = [
            c.max_output_tokens for c in caps if c.max_output_tokens is not None
        ]
        return ModelCapabilities(
            function_calling=all(c.function_calling for c in caps),
            vision=all(c.vision for c in caps),
            output_schema=all(c.output_schema for c in caps),
            prompt_caching=all(c.prompt_caching for c in caps),
            reasoning=all(c.reasoning for c in caps),
            web_search=all(c.web_search for c in caps),
            audio_input=all(c.audio_input for c in caps),
            max_input_tokens=min(input_windows) if input_windows else None,
            max_output_tokens=min(output_windows) if output_windows else None,
        )

    def _validate_response(
        self,
        response: Response,
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
    ) -> None:
        """
        No-op: members validate their own responses (and run their own
        validation retries); re-validating here would double the work and
        the refusal warnings.
        """

    def _validate_tool_calls(
        self,
        response: Response,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]],
    ) -> None:
        """No-op: tool calls are validated by the serving member."""

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
        errors: list[Exception] = []

        for llm in candidates:
            try:
                # The member's full pipeline: its own API retries,
                # validation, and validation retries. The cascade advances
                # only on exhausted or deterministic API errors; validation
                # errors are not LlmErrorTuple and propagate.
                return await llm.generate_response(
                    input,
                    tools=tools,
                    output_schema=output_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                )

            except LlmErrorTuple as e:
                errors.append(e)
                logger.warning(
                    "Model %s failed (%s: %s), trying next fallback",
                    llm.model_name,
                    type(e).__name__,
                    e,
                )

        assert errors
        raise _select_cascade_error(errors)

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
        errors: list[Exception] = []
        seq = 0
        attempt = 0

        for llm in candidates:
            try:
                # Member-internal retries (API and validation) surface as
                # ResponseRetrying events within this member's segment.
                async for event in llm.generate_response_stream(
                    input,
                    tools=tools,
                    output_schema=output_schema,
                    tool_choice=tool_choice,
                    **extra_llm_settings,
                ):
                    seq = event.sequence_number
                    yield event
                return

            except LlmErrorTuple as e:
                errors.append(e)
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

        assert errors
        raise _select_cascade_error(errors)
