"""
Streamed OpenAI Responses events: a final event that fails validation ends
the attempt with a typed, non-retryable error rather than being skipped, while
an unrecognized event in the middle of a stream is skipped.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm.fallback_llm import FallbackLLM
from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.llm_providers.openai_responses.responses_llm import (
    OpenAIResponsesLLM,
)
from grasp_agents.types.items import InputMessageItem
from grasp_agents.types.llm_errors import LlmResponseSchemaError
from grasp_agents.types.llm_events import ResponseCompleted
from tests._helpers import MockLLM, _text_response

_USER_MSG = [InputMessageItem.from_text("hi")]


def _completed(status: str = "completed") -> dict[str, Any]:
    return {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {
            "id": "resp_1",
            "created_at": 1.0,
            "model": "gpt-5.4-nano",
            "object": "response",
            "output": [],
            "status": status,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
    }


def _sdk_event(data: dict[str, Any]) -> Any:
    return SimpleNamespace(model_dump=lambda **_: data)


@dataclass(frozen=True)
class _ScriptedResponsesLLM(OpenAIResponsesLLM):
    """Real Responses stream conversion; only the wire stream is scripted."""

    script: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "_attempts", 0)

    @property
    def attempts(self) -> int:
        return self._attempts  # type: ignore[attr-defined]

    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: Any | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[Any]:
        object.__setattr__(self, "_attempts", self.attempts + 1)

        async def stream() -> AsyncIterator[Any]:
            for data in self.script:
                yield _sdk_event(data)

        return stream()


def _llm(*script: dict[str, Any], **kwargs: Any) -> _ScriptedResponsesLLM:
    return _ScriptedResponsesLLM(
        model_name="gpt-5.4-nano",
        api_provider=APIProvider(name="openai", base_url=None, api_key="test-key"),
        script=list(script),
        **kwargs,
    )


class TestFinalEventValidation:
    @pytest.mark.asyncio
    async def test_final_event_failing_validation_is_not_retried(self) -> None:
        llm = _llm(
            _completed(status="queued_for_review"),
            retry_policy=RetryPolicy(api_retries=2, initial_delay=0.0),
        )

        with pytest.raises(LlmResponseSchemaError, match=r"response\.completed"):
            async for _ in llm.generate_response_stream(_USER_MSG):
                pass

        assert llm.attempts == 1

    @pytest.mark.asyncio
    async def test_final_event_failing_validation_advances_the_fallback(
        self,
    ) -> None:
        fallback = MockLLM(responses_queue=[_text_response("rescued")])
        llm = FallbackLLM(
            primary=_llm(_completed(status="queued_for_review"), retry_policy=None),
            fallbacks=(fallback,),
        )

        events = [e async for e in llm.generate_response_stream(_USER_MSG)]

        (completed,) = [e for e in events if isinstance(e, ResponseCompleted)]
        assert completed.response.output_text == "rescued"

    @pytest.mark.asyncio
    async def test_unrecognized_event_mid_stream_is_skipped(self) -> None:
        llm = _llm({"type": "response.some_future_event"}, _completed())

        events = [e async for e in llm.generate_response_stream(_USER_MSG)]

        assert [type(e) for e in events] == [ResponseCompleted]
