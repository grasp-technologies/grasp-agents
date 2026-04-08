"""
Tests for AgentTool — dynamic subagent spawning.

Verifies:
- Basic foreground execution: creates child agent, runs with prompt, returns str
- Streaming: child events pass through, final packet → ToolOutputEvent
- Tool inheritance: parent tools inherited (minus AgentTools), dedup by name
- No inheritance when disabled
- Fresh agent per invocation (independent memory)
- Type resolution: in_type=AgentToolInput, out_type=str
- Resumable property
- Background mode integration (via BackgroundTaskManager)
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.agent_tool import AgentPromptBuilder, AgentTool, AgentToolInput
from grasp_agents.agent.function_tool import function_tool
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.llm_agent_memory import LLMAgentMemory
from grasp_agents.llm.llm import LLM
from grasp_agents.run_context import RunContext
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskLaunchedEvent,
    Event,
    ToolOutputEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import OutputMessageItem
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
)
from grasp_agents.types.response import Response, ResponseUsage
from grasp_agents.types.tool import BaseTool

# ruff: noqa: ARG002

# ------------------------------------------------------------------ #
#  Test helpers                                                        #
# ------------------------------------------------------------------ #


def _make_usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=10,
        output_tokens=10,
        total_tokens=20,
    )


def _text_response(text: str) -> Response:
    from grasp_agents.types.content import OutputMessageText

    return Response(
        model="mock",
        output_items=[
            OutputMessageItem(
                content_parts=[OutputMessageText(text=text)],
                status="completed",
            )
        ],
        usage_with_cost=_make_usage(),
    )


def _tool_call_response(name: str, arguments: str, call_id: str) -> Response:
    from grasp_agents.types.items import FunctionToolCallItem

    return Response(
        model="mock",
        output_items=[
            FunctionToolCallItem(
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
        ],
        usage_with_cost=_make_usage(),
    )


@dataclass(frozen=True)
class MockLLM(LLM):
    model_name: str = "mock"
    responses_queue: list[Response] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_call_count", 0)

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        count = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        if count < len(self.responses_queue):
            return self.responses_queue[count]
        return _text_response("default answer")

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        response = await self._generate_response_once(
            input,
            tools=tools,
            response_schema=response_schema,
            tool_choice=tool_choice,
            **extra_llm_settings,
        )
        seq = 0
        seq += 1
        yield ResponseCreated(response=response, sequence_number=seq)  # type: ignore[arg-type]
        for idx, item in enumerate(response.output):
            seq += 1
            yield OutputItemAdded(item=item, output_index=idx, sequence_number=seq)
            seq += 1
            yield OutputItemDone(item=item, output_index=idx, sequence_number=seq)
        seq += 1
        yield ResponseCompleted(response=response, sequence_number=seq)


def _make_child_llm(*texts: str) -> MockLLM:
    """Create a MockLLM that returns the given text responses in sequence."""
    return MockLLM(responses_queue=[_text_response(t) for t in texts])


# ------------------------------------------------------------------ #
#  Tests                                                               #
# ------------------------------------------------------------------ #


class TestAgentToolBasics:
    @pytest.mark.anyio
    async def test_type_resolution(self) -> None:
        tool = AgentTool(
            name="sub",
            description="A subagent",
            llm=_make_child_llm("hi"),
        )
        assert tool.in_type is AgentToolInput
        assert tool.out_type is str

    @pytest.mark.anyio
    async def test_resumable(self) -> None:
        tool = AgentTool(
            name="sub",
            description="A subagent",
            llm=_make_child_llm("hi"),
        )
        assert tool.resumable is True

    @pytest.mark.anyio
    async def test_basic_foreground_execution(self) -> None:
        tool = AgentTool[None](
            name="sub",
            description="A subagent",
            llm=_make_child_llm("hello world"),
        )
        ctx = RunContext[None]()
        result = await tool.run(
            AgentToolInput(prompt="say hello"),
            ctx=ctx,
            exec_id="test",
        )
        assert result == "hello world"

    @pytest.mark.anyio
    async def test_streaming_yields_child_events(self) -> None:
        tool = AgentTool[None](
            name="sub",
            description="A subagent",
            llm=_make_child_llm("streamed result"),
        )
        ctx = RunContext[None]()
        events: list[Event[Any]] = []
        async for event in tool.run_stream(
            AgentToolInput(prompt="stream this"),
            ctx=ctx,
            exec_id="test",
        ):
            events.append(event)

        # Should have at least one ToolOutputEvent with the result
        output_events = [e for e in events if isinstance(e, ToolOutputEvent)]
        assert len(output_events) == 1
        assert output_events[0].data == "streamed result"

    @pytest.mark.anyio
    async def test_fresh_agent_per_invocation(self) -> None:
        """Each call creates a fresh agent — no memory leaking."""
        llm = _make_child_llm("first", "second", "third", "fourth")
        tool = AgentTool[None](
            name="sub",
            description="A subagent",
            llm=llm,
        )
        ctx = RunContext[None]()

        r1 = await tool.run(
            AgentToolInput(prompt="call 1"),
            ctx=ctx,
            exec_id="e1",
        )
        r2 = await tool.run(
            AgentToolInput(prompt="call 2"),
            ctx=ctx,
            exec_id="e2",
        )
        # Both calls get fresh agents — LLM counter increments
        assert r1 == "first"
        assert r2 == "second"


class TestToolInheritance:
    @pytest.mark.anyio
    async def test_inherit_tools_excludes_agent_tools(self) -> None:
        @function_tool
        async def search(query: str) -> str:
            return f"results for {query}"

        other_agent_tool = AgentTool[None](
            name="other",
            description="Another agent tool",
            llm=_make_child_llm("x"),
        )

        tool = AgentTool[None](
            name="sub",
            description="A subagent",
            llm=_make_child_llm("done"),
            inherit_tools=True,
        )
        # Simulate parent wiring
        tool.set_parent_tools([search, other_agent_tool])  # type: ignore[list-item]

        resolved = tool._resolve_tools()
        assert resolved is not None
        names = [t.name for t in resolved]
        assert "search" in names
        assert "other" not in names  # AgentTool excluded

    @pytest.mark.anyio
    async def test_no_inheritance_when_disabled(self) -> None:
        @function_tool
        async def search(query: str) -> str:
            return f"results for {query}"

        tool = AgentTool[None](
            name="sub",
            description="A subagent",
            llm=_make_child_llm("done"),
            inherit_tools=False,
        )
        tool.set_parent_tools([search])  # type: ignore[list-item]

        resolved = tool._resolve_tools()
        assert resolved is None  # No own tools, inheritance disabled

    @pytest.mark.anyio
    async def test_own_tools_take_precedence(self) -> None:
        @function_tool(name="search")
        async def parent_search(query: str) -> str:
            return "parent"

        @function_tool(name="search")
        async def own_search(query: str) -> str:
            return "own"

        tool = AgentTool[None](
            name="sub",
            description="A subagent",
            llm=_make_child_llm("done"),
            tools=[own_search],  # type: ignore[list-item]
            inherit_tools=True,
        )
        tool.set_parent_tools([parent_search])  # type: ignore[list-item]

        resolved = tool._resolve_tools()
        assert resolved is not None
        # Only one "search" — own version wins
        assert len(resolved) == 1

    @pytest.mark.anyio
    async def test_parent_wiring_in_llm_agent(self) -> None:
        """LLMAgent.__init__ wires parent tools into AgentTool."""

        @function_tool
        async def helper(x: int) -> int:
            return x + 1

        agent_tool = AgentTool[None](
            name="sub",
            description="A subagent",
            llm=_make_child_llm("done"),
            inherit_tools=True,
        )

        # Creating an LLMAgent with both tools should wire them
        _parent = LLMAgent[str, str, None](
            name="parent",
            llm=_make_child_llm("parent done"),
            tools=[helper, agent_tool],  # type: ignore[list-item]
        )

        resolved = agent_tool._resolve_tools()
        assert resolved is not None
        names = [t.name for t in resolved]
        assert "helper" in names
        assert "sub" not in names  # Self excluded (AgentTool)


class TestAgentToolWithParentAgent:
    @pytest.mark.anyio
    async def test_agent_calls_agent_tool(self) -> None:
        """Parent agent calls AgentTool, gets child's answer."""
        # Child LLM returns "child answer"
        child_llm = _make_child_llm("child answer")

        agent_tool = AgentTool[None](
            name="research",
            description="Research a topic",
            llm=child_llm,
        )

        # Parent LLM: first call triggers tool, second uses result
        parent_llm = MockLLM(
            responses_queue=[
                _tool_call_response(
                    "research",
                    '{"prompt": "find papers on AI"}',
                    "tc1",
                ),
                _text_response("Based on research: all good"),
            ]
        )

        parent = LLMAgent[str, str, None](
            name="parent",
            llm=parent_llm,
            tools=[agent_tool],
        )

        result = await parent.run(chat_inputs="do research")
        assert result.payloads[0] == "Based on research: all good"

    @pytest.mark.anyio
    async def test_background_agent_tool(self) -> None:
        """Background AgentTool spawns, drains, delivers notification."""
        child_llm = _make_child_llm("bg result")

        agent_tool = AgentTool[None](
            name="bg_research",
            description="Background research",
            llm=child_llm,
            background=True,
        )

        # Parent: call bg tool → "waiting" (suppressed) → "final" after drain
        parent_llm = MockLLM(
            responses_queue=[
                _tool_call_response(
                    "bg_research",
                    '{"prompt": "find stuff"}',
                    "tc1",
                ),
                _text_response("waiting for results"),
                _text_response("final answer with bg results"),
            ]
        )

        parent = LLMAgent[str, str, None](
            name="parent",
            llm=parent_llm,
            tools=[agent_tool],
        )

        events: list[Event[Any]] = []
        async for event in parent.run_stream(chat_inputs="go"):
            events.append(event)

        # Should have launch and completion events
        launched = [e for e in events if isinstance(e, BackgroundTaskLaunchedEvent)]
        completed = [e for e in events if isinstance(e, BackgroundTaskCompletedEvent)]
        assert len(launched) == 1
        assert len(completed) == 1

        # Notification should be in events
        notifications = [
            e
            for e in events
            if isinstance(e, UserMessageEvent) and "bg_research" in str(e.data)
        ]
        assert len(notifications) >= 1


class TestAgentPromptBuilders:
    @pytest.mark.anyio
    async def test_sys_prompt_builder_overrides_static(self) -> None:
        """sys_prompt_builder replaces static sys_prompt."""
        child_llm = _make_child_llm("ok")

        def build_sys(
            prompt: str, memory: LLMAgentMemory, ctx: RunContext[None]
        ) -> str:
            return f"Dynamic: {prompt}"

        agent_tool = AgentTool[None](
            name="t",
            description="d",
            llm=child_llm,
            sys_prompt="static prompt",
            sys_prompt_builder=build_sys,
        )

        ctx: RunContext[None] = RunContext()
        result = await agent_tool._run(
            AgentToolInput(prompt="hello"),
            ctx=ctx,
            exec_id="x",
        )
        assert result == "ok"
        # Verify builder was called — child agent's sys_prompt was set
        # by _build_prompts via the builder, not the static string.
        agent, _ = await agent_tool._prepare_child(
            AgentToolInput(prompt="hello"), ctx=ctx, exec_id="x2", session_id=None
        )
        assert agent._prompt_builder.sys_prompt == "Dynamic: hello"

    @pytest.mark.anyio
    async def test_in_prompt_builder_transforms_user_msg(self) -> None:
        """in_prompt_builder transforms the prompt into the child's user message."""
        captured: list[str] = []

        # Child LLM captures the input it receives
        class CaptureLLM(MockLLM):
            async def _generate_response_once(
                self,
                input: Any,
                **kwargs: Any,
            ) -> Response:
                for item in input:
                    if hasattr(item, "content_parts"):
                        for part in item.content_parts:
                            if hasattr(part, "text"):
                                captured.append(part.text)
                return _text_response("done")

        def build_input(
            prompt: str, memory: LLMAgentMemory, ctx: RunContext[None]
        ) -> str:
            return f"[enriched] {prompt}"

        agent_tool = AgentTool[None](
            name="t",
            description="d",
            llm=CaptureLLM(responses_queue=[_text_response("done")]),
            in_prompt_builder=build_input,
        )

        ctx: RunContext[None] = RunContext()
        await agent_tool._run(
            AgentToolInput(prompt="raw task"),
            ctx=ctx,
            exec_id="x",
        )
        assert any("[enriched] raw task" in c for c in captured)

    @pytest.mark.anyio
    async def test_builder_receives_parent_memory(self) -> None:
        """Builder receives the parent agent's memory."""
        received_memory: list[LLMAgentMemory | None] = []

        def build_sys(
            prompt: str, memory: LLMAgentMemory, ctx: RunContext[None]
        ) -> str:
            received_memory.append(memory)
            return "sys"

        child_llm = _make_child_llm("ok")
        agent_tool = AgentTool[None](
            name="t",
            description="d",
            llm=child_llm,
            sys_prompt_builder=build_sys,
        )

        # Wire parent memory manually (normally done by LLMAgent.__init__)
        parent_mem = LLMAgentMemory()
        from grasp_agents.types.items import InputMessageItem

        parent_mem.update([InputMessageItem.from_text("user said hi", role="user")])
        agent_tool.set_parent_memory(parent_mem)

        ctx: RunContext[None] = RunContext()
        await agent_tool._run(AgentToolInput(prompt="go"), ctx=ctx, exec_id="x")
        assert len(received_memory) == 1
        assert received_memory[0] is parent_mem
        assert len(received_memory[0].messages) == 1

    @pytest.mark.anyio
    async def test_async_builder(self) -> None:
        """Async builders are awaited properly."""
        child_llm = _make_child_llm("ok")

        async def async_build(
            prompt: str, memory: LLMAgentMemory, ctx: RunContext[None]
        ) -> str:
            return f"async: {prompt}"

        agent_tool = AgentTool[None](
            name="t",
            description="d",
            llm=child_llm,
            sys_prompt_builder=async_build,
        )

        ctx: RunContext[None] = RunContext()
        agent, _ = await agent_tool._prepare_child(
            AgentToolInput(prompt="test"), ctx=ctx, exec_id="x", session_id=None
        )
        assert agent._prompt_builder.sys_prompt == "async: test"

    @pytest.mark.anyio
    async def test_no_builders_preserves_defaults(self) -> None:
        """Without builders, static sys_prompt and raw prompt are used."""
        child_llm = _make_child_llm("ok")

        agent_tool = AgentTool[None](
            name="t",
            description="d",
            llm=child_llm,
            sys_prompt="static",
        )

        ctx: RunContext[None] = RunContext()
        agent, user_msg = await agent_tool._prepare_child(
            AgentToolInput(prompt="raw"), ctx=ctx, exec_id="x", session_id=None
        )
        assert agent._prompt_builder.sys_prompt == "static"
        assert user_msg == "raw"

    @pytest.mark.anyio
    async def test_parent_agent_wires_memory(self) -> None:
        """LLMAgent.__init__ automatically wires parent memory to AgentTool."""
        child_llm = _make_child_llm("ok")
        agent_tool = AgentTool[None](name="sub", description="d", llm=child_llm)

        parent_llm = MockLLM(responses_queue=[_text_response("done")])
        parent = LLMAgent[str, str, None](
            name="parent", llm=parent_llm, tools=[agent_tool]
        )

        assert agent_tool._parent_memory is parent._memory


class TestToolCopy:
    def test_deepcopy_isolates_mutable_state(self) -> None:
        """BaseTool.copy() deep-copies, isolating mutable state."""
        child_llm = _make_child_llm("ok")
        original = AgentTool[None](name="t", description="d", llm=child_llm)
        copied = original.copy()

        assert copied is not original
        assert copied.name == original.name
        assert copied._llm is original._llm  # LLM.__deepcopy__ shares
        assert copied._own_tools is not original._own_tools  # mutable: isolated

    def test_class_level_name(self) -> None:
        """Tools can define name/description at class level."""

        class MyTool(BaseTool[BaseModel, str, None]):
            name = "my_tool"
            description = "A test tool"

            async def _run(self, inp: BaseModel | None = None, **kwargs: Any) -> str:  # type: ignore[override]
                return "ok"

        tool = MyTool()
        assert tool.name == "my_tool"
        assert tool.description == "A test tool"

    def test_class_level_name_not_in_dict(self) -> None:
        """Class-level name stays off __dict__ — shared via class."""

        class MyTool(BaseTool[BaseModel, str, None]):
            name = "my_tool"
            description = "desc"

            async def _run(self, inp: BaseModel | None = None, **kwargs: Any) -> str:  # type: ignore[override]
                return "ok"

        tool = MyTool()
        assert "name" not in tool.__dict__
        assert "description" not in tool.__dict__

    def test_instance_override_in_dict(self) -> None:
        """Instance override via __init__ goes in __dict__."""
        child_llm = _make_child_llm("ok")
        tool = AgentTool[None](name="t", description="d", llm=child_llm)
        assert "name" in tool.__dict__

    def test_empty_name_raises(self) -> None:
        """Tool with no name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty name"):

            class BadTool(BaseTool[BaseModel, str, None]):
                description = "no name"

                async def _run(
                    self, inp: BaseModel | None = None, **kwargs: Any
                ) -> str:  # type: ignore[override]
                    return "ok"

            BadTool()
