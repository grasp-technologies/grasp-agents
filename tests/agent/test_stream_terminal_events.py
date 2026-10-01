"""
A streamed response that ends with ``response.incomplete`` (e.g. max output
tokens) carries a response the agent can act on, exactly like the
non-streaming path returns it. It must reach the loop, not trip the
"no response" invariant.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

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
    """Streams end with ``response.incomplete`` instead of ``response.completed``."""

    async def _generate_response_stream_once(
        self, input: Any, **kwargs: Any
    ) -> AsyncIterator[LlmEvent]:
        async for event in super()._generate_response_stream_once(input, **kwargs):
            if isinstance(event, ResponseCompleted):
                yield ResponseIncomplete(
                    response=event.response,  # type: ignore[arg-type]
                    sequence_number=event.sequence_number,
                )
            else:
                yield event


@pytest.mark.asyncio
async def test_incomplete_terminal_event_reaches_the_agent() -> None:
    ctx: SessionContext[None] = SessionContext(checkpoint_store=None)
    agent = LLMAgent[str, str, None](
        name="a",
        ctx=ctx,
        llm=IncompleteStreamLLM(responses_queue=[_text_response("truncated")]),
        stream_llm=True,
    )

    result = await agent.run("hi")

    assert result.payloads[0] == "truncated"
