"""
Integration tests for AgentTool with real OpenAI Responses API.

Tests subagent spawning, streaming, tool inheritance, and background execution
against a real LLM provider. Skipped by default.

Run with:
    uv run pytest -m integration -k test_agent_tool_integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, Field

from grasp_agents.agent.agent_tool import AgentTool, AgentToolInput
from grasp_agents.agent.function_tool import function_tool
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.run_context import RunContext
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskLaunchedEvent,
    Event,
    LLMStreamEvent,
    ToolOutputEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessageEvent,
)

if TYPE_CHECKING:
    from grasp_agents.llm.cloud_llm import CloudLLM


@pytest.mark.integration
class TestAgentToolIntegration:
    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:  # noqa: ARG002
        from grasp_agents.llm_providers.openai_responses.responses_llm import (
            OpenAIResponsesLLM,
        )

        return OpenAIResponsesLLM(
            model_name="gpt-4.1-nano",
            llm_settings={"max_output_tokens": 200},
        )

    @pytest.mark.asyncio
    async def test_foreground_agent_tool(self, llm: CloudLLM) -> None:
        """Parent agent calls AgentTool in foreground, gets child's answer."""
        agent_tool = AgentTool[None](
            name="summarizer",
            description=(
                "Summarize a topic in one sentence. "
                "Call this tool when you need a summary."
            ),
            llm=llm,
            max_turns=1,
        )

        parent = LLMAgent[str, str, None](
            name="coordinator",
            llm=llm,
            tools=[agent_tool],
            sys_prompt=(
                "You are a coordinator. When asked about a topic, "
                "use the summarizer tool to get a summary, then "
                "respond with that summary. Always use the tool."
            ),
            max_turns=3,
            stream_llm_responses=True,
        )

        result = await parent.run(chat_inputs="What is photosynthesis?")
        answer = result.payloads[0]
        assert isinstance(answer, str)
        assert len(answer) > 10

    @pytest.mark.asyncio
    async def test_foreground_streaming(self, llm: CloudLLM) -> None:
        """Streaming parent.run_stream yields child agent events."""
        agent_tool = AgentTool[None](
            name="explainer",
            description=(
                "Explain a concept briefly. "
                "Call this tool when asked to explain something."
            ),
            llm=llm,
            max_turns=1,
        )

        parent = LLMAgent[str, str, None](
            name="coordinator",
            llm=llm,
            tools=[agent_tool],
            sys_prompt=(
                "You are a coordinator. When asked to explain "
                "something, always use the explainer tool, then "
                "give the explanation back."
            ),
            max_turns=3,
            stream_llm_responses=True,
        )

        events: list[Event[Any]] = []
        async for event in parent.run_stream(
            chat_inputs="Explain gravity in one sentence."
        ):
            events.append(event)

        # Should have events from both parent and child
        turn_starts = [e for e in events if isinstance(e, TurnStartEvent)]
        assert len(turn_starts) >= 2  # At least parent turn + child turn

        # Should have LLM stream events (from child agent)
        llm_events = [e for e in events if isinstance(e, LLMStreamEvent)]
        assert len(llm_events) > 0

    @pytest.mark.asyncio
    async def test_tool_inheritance(self, llm: CloudLLM) -> None:
        """Child agent inherits parent's tools and can use them."""

        @function_tool
        async def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        agent_tool = AgentTool[None](
            name="calculator",
            description=(
                "A calculator agent that can do math. "
                "Call this when you need calculations."
            ),
            llm=llm,
            max_turns=3,
            inherit_tools=True,
        )

        parent = LLMAgent[str, str, None](
            name="coordinator",
            llm=llm,
            tools=[multiply, agent_tool],
            sys_prompt=(
                "You are a coordinator. When asked a math question, "
                "use the calculator tool with a clear prompt. "
                "Always use the calculator tool."
            ),
            max_turns=3,
        )

        result = await parent.run(chat_inputs="What is 7 times 8?")
        answer = result.payloads[0]
        assert "56" in answer

    @pytest.mark.asyncio
    async def test_background_agent_tool(self, llm: CloudLLM) -> None:
        """Background AgentTool: launch event, notification, completion."""
        agent_tool = AgentTool[None](
            name="researcher",
            description=(
                "Research a topic in the background. "
                "Call this to start background research."
            ),
            llm=llm,
            max_turns=1,
            background=True,
        )

        parent = LLMAgent[str, str, None](
            name="coordinator",
            llm=llm,
            tools=[agent_tool],
            sys_prompt=(
                "You are a coordinator. When asked to research "
                "something, use the researcher tool, wait for "
                "the result, then summarize it."
            ),
            max_turns=5,
            stream_llm_responses=True,
        )

        events: list[Event[Any]] = []
        async for event in parent.run_stream(
            chat_inputs="Research what DNA stands for."
        ):
            events.append(event)

        launched = [e for e in events if isinstance(e, BackgroundTaskLaunchedEvent)]
        completed = [e for e in events if isinstance(e, BackgroundTaskCompletedEvent)]
        notifications = [
            e
            for e in events
            if isinstance(e, UserMessageEvent) and "researcher" in str(e.data)
        ]

        assert len(launched) == 1, "Should launch one background task"
        assert len(completed) == 1, "Should complete one background task"
        assert len(notifications) >= 1, "Should deliver notification"
