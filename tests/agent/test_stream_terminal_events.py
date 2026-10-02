"""
A streamed response that ends with ``response.incomplete`` (e.g. max output
tokens) carries a response the agent can act on, exactly like the
non-streaming path returns it. It must reach the loop, not trip the
"no response" invariant.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from openai.types.responses.response import IncompleteDetails

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.session_context import SessionContext
from grasp_agents.types.llm_events import (
    LlmEvent,
    ResponseCompleted,
    ResponseIncomplete,
)
from tests._helpers import MockLLM, _text_response


@dataclass(frozen=True)
class IncompleteStreamLLM(MockLLM):
    """Streams end with ``response.incomplete`` (``max_output_tokens``)."""

    async def _generate_response_stream_once(
        self, input: Any, **kwargs: Any
    ) -> AsyncIterator[LlmEvent]:
        async for event in super()._generate_response_stream_once(input, **kwargs):
            if isinstance(event, ResponseCompleted):
                response = event.response.model_copy(
                    update={
                        "status": "incomplete",
                        "incomplete_details": IncompleteDetails(
                            reason="max_output_tokens"
                        ),
                    }
                )
                yield ResponseIncomplete(
                    response=response,  # type: ignore[arg-type]
                    sequence_number=event.sequence_number,
                )
            else:
                yield event


@pytest.mark.asyncio
async def test_incomplete_terminal_event_reaches_the_agent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx: SessionContext[None] = SessionContext(checkpoint_store=None)
    agent = LLMAgent[str, str, None](
        name="a",
        ctx=ctx,
        llm=IncompleteStreamLLM(responses_queue=[_text_response("truncated")]),
        stream_llm=True,
    )

    with caplog.at_level(logging.WARNING, logger="grasp_agents.agent.agent_loop"):
        result = await agent.run("hi")

    assert result.payloads[0] == "truncated"
    assert "LLM response is incomplete (max_output_tokens)" in caplog.text
