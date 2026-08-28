"""
Streaming-path behavior of the agent loop: which streamed items reach the
transcript when the LLM layer restarts a turn mid-stream, and terminal-response
capture when a stream ends truncated instead of completed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from openai.types.responses.response import IncompleteDetails
from pydantic import BaseModel

from grasp_agents.agent.agent_loop import AgentLoop
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.llm.llm import LLM
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import Event, GenerationEndEvent
from grasp_agents.types.items import (
    InputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemDone,
    ResponseCompleted,
    ResponseFallback,
    ResponseIncomplete,
)
from grasp_agents.types.response import Response
from tests._helpers import _make_agent_loop, _make_usage

# ---------- Infrastructure ----------


@dataclass(frozen=True)
class ScriptedStreamLLM(LLM):
    """Replays a fixed ``LlmEvent`` script, as a provider adapter would."""

    model_name: str = "scripted"
    events: list[LlmEvent] = field(default_factory=list)

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        raise NotImplementedError

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        for event in self.events:
            yield event


def _message_item(text: str) -> OutputMessageItem:
    return OutputMessageItem(content=[OutputMessageText(text=text)], status="completed")


def _make_loop(
    events: list[LlmEvent],
) -> tuple[AgentLoop[None], LLMAgentTranscript]:
    transcript = LLMAgentTranscript()
    transcript.messages = [InputMessageItem.from_text("sys", role="system")]
    transcript.update([InputMessageItem.from_text("go", role="user")])

    loop = _make_agent_loop(
        agent_name="test",
        llm=ScriptedStreamLLM(events=events),
        transcript=transcript,
        ctx=SessionContext[None](state=None),
        max_turns=10,
        stream_llm=True,
    )
    return loop, transcript


async def _drain(loop: AgentLoop[None]) -> list[Event[Any]]:
    events: list[Event[Any]] = []
    async for event in loop.execute_stream(exec_id="t"):
        events.append(event)
    return events


def _output_items(transcript: LLMAgentTranscript) -> Sequence[InputItem]:
    return [m for m in transcript.messages if not isinstance(m, InputMessageItem)]


# ---------- Mid-stream fallback ----------


class TestFallbackDiscardsPartials:
    @pytest.mark.asyncio
    async def test_only_the_serving_members_items_are_committed(self) -> None:
        """
        A cascade member that dies mid-stream leaves partial items behind. The
        next member re-streams the whole turn, so the dead member's items must
        not be committed alongside it.
        """
        dead_reasoning = ReasoningItem(encrypted_content="dead-enc")
        dead_message = _message_item("partial from the dead member")
        served_message = _message_item("recovered")
        served = Response(
            model="fallback", output=[served_message], usage=_make_usage()
        )

        loop, transcript = _make_loop(
            [
                OutputItemDone(item=dead_reasoning, output_index=0, sequence_number=1),
                OutputItemDone(item=dead_message, output_index=1, sequence_number=2),
                ResponseFallback(
                    failed_model="primary",
                    fallback_model="fallback",
                    error_type="LlmInternalServerError",
                    attempt=1,
                    sequence_number=3,
                ),
                OutputItemDone(item=served_message, output_index=0, sequence_number=4),
                ResponseCompleted(response=served, sequence_number=5),
            ]
        )

        await _drain(loop)

        assert _output_items(transcript) == [served_message]
        assert loop.final_answer == "recovered"


# ---------- Truncated (incomplete) streams ----------


class TestIncompleteStreamIsTerminal:
    @pytest.mark.asyncio
    async def test_response_captured_and_usage_recorded(self) -> None:
        """
        ``response.incomplete`` ends a stream just as ``response.completed``
        does, carrying the final Response — the turn must finish on it and its
        usage must reach the tracker.
        """
        truncated_message = _message_item("truncated answ")
        truncated = Response(
            model="mock",
            output=[truncated_message],
            usage=_make_usage(),
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        )

        loop, transcript = _make_loop(
            [
                OutputItemDone(
                    item=truncated_message, output_index=0, sequence_number=1
                ),
                ResponseIncomplete(response=truncated, sequence_number=2),
            ]
        )
        ctx = loop.ctx

        events = await _drain(loop)

        generation_ends = [e for e in events if isinstance(e, GenerationEndEvent)]
        assert [e.data.status for e in generation_ends] == ["incomplete"]
        assert ctx.usage_tracker.usages["test"].input_tokens == 10
        assert _output_items(transcript) == [truncated_message]
        assert loop.final_answer == "truncated answ"
