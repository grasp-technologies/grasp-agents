"""
Tests for session persistence layer.

Verifies that:
- LLMAgent with store/session_id auto-loads and auto-saves conversation state
- Resume applies CC-style cleanup (strips dangling tool calls)
- Checkpoints fire at correct boundaries (after tools, after final answer)
- Multi-turn sessions maintain conversation across run() calls
- Sessions survive simulated restarts (save -> new agent -> load -> continue)
- InMemoryCheckpointStore operations work correctly
- AgentCheckpoint round-trips through JSON serialization
- TaskRecord lifecycle: created on bg spawn, updated on completion/failure
- Pending TaskRecords inject interruption notifications on session resume
"""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import (
    AgentCheckpoint,
    InMemoryCheckpointStore,
    InterruptionType,
    TaskRecord,
    TaskStatus,
    prepare_messages_for_resume,
)
from grasp_agents.llm.llm import LLM
from grasp_agents.packet import Packet
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.events import Event
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
)
from grasp_agents.types.response import Response, ResponseUsage
from grasp_agents.types.tool import BaseTool
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow

# ---------- Infrastructure ----------


def _make_usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _text_response(text: str) -> Response:
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
    return Response(
        model="mock",
        output_items=[
            FunctionToolCallItem(
                call_id=call_id,
                name=name,
                arguments=arguments,
            ),
        ],
        usage_with_cost=_make_usage(),
    )


@dataclass(frozen=True)
class MockLLM(LLM):
    model_name: str = "mock"
    responses_queue: list[Response] = field(default_factory=list)

    def __post_init__(self):
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
        assert self.responses_queue, "MockLLM: no more responses"
        return self.responses_queue.pop(0)

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
        yield ResponseCompleted(response=response, sequence_number=seq)  # type: ignore[arg-type]


# --- Tools ---


class EchoInput(BaseModel):
    text: str


class EchoTool(BaseTool[EchoInput, str, Any]):
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes input")

    async def _run(
        self,
        inp: EchoInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
        session_id: str | None = None,  # noqa: ARG002
    ) -> str:
        return f"echo: {inp.text}"


# --- Helpers ---


def _make_agent(
    responses: list[Response],
    *,
    tools: list[BaseTool[Any, Any, Any]] | None = None,
    session_id: str | None = None,
    store: InMemoryCheckpointStore | None = None,
    reset_memory_on_run: bool = False,
) -> tuple[LLMAgent[str, str, None], RunContext[None]]:
    llm = MockLLM(responses_queue=responses)
    agent = LLMAgent[str, str, None](
        name="test_agent",
        llm=llm,
        tools=tools,
        reset_memory_on_run=reset_memory_on_run,
        stream_llm_responses=True,
        session_id=session_id,
    )
    ctx: RunContext[None] = RunContext(store=store)
    return agent, ctx


async def _drain_stream(
    stream: AsyncIterator[Event[Any]],
) -> list[Event[Any]]:
    return [event async for event in stream]


# ================================================================== #
#  InMemoryCheckpointStore                                             #
# ================================================================== #


class TestInMemoryCheckpointStore:
    @pytest.mark.anyio
    async def test_save_and_load(self):
        store = InMemoryCheckpointStore()
        await store.save("key1", b"data1")
        assert await store.load("key1") == b"data1"

    @pytest.mark.anyio
    async def test_load_missing(self):
        store = InMemoryCheckpointStore()
        assert await store.load("missing") is None

    @pytest.mark.anyio
    async def test_delete(self):
        store = InMemoryCheckpointStore()
        await store.save("key1", b"data1")
        await store.delete("key1")
        assert await store.load("key1") is None

    @pytest.mark.anyio
    async def test_delete_missing_is_noop(self):
        store = InMemoryCheckpointStore()
        await store.delete("missing")  # should not raise

    @pytest.mark.anyio
    async def test_list_keys(self):
        store = InMemoryCheckpointStore()
        await store.save("chat/1", b"a")
        await store.save("chat/2", b"b")
        await store.save("other/1", b"c")

        keys = await store.list_keys("chat/")
        assert sorted(keys) == ["chat/1", "chat/2"]

    @pytest.mark.anyio
    async def test_overwrite(self):
        store = InMemoryCheckpointStore()
        await store.save("key1", b"v1")
        await store.save("key1", b"v2")
        assert await store.load("key1") == b"v2"


# ================================================================== #
#  AgentCheckpoint serialization                                       #
# ================================================================== #


class TestAgentCheckpoint:
    def test_round_trip_empty(self):
        snap = AgentCheckpoint(
            session_id="s1",
            processor_name="agent",
            messages=[],
        )
        json_bytes = snap.model_dump_json().encode()
        restored = AgentCheckpoint.model_validate_json(json_bytes)
        assert restored.session_id == "s1"
        assert restored.messages == []
        assert restored.checkpoint_number == 0

    def test_round_trip_with_messages(self):
        messages: list[Any] = [
            InputMessageItem.from_text("system prompt", role="system"),
            InputMessageItem.from_text("hello", role="user"),
            OutputMessageItem(
                content_parts=[OutputMessageText(text="hi there")],
                status="completed",
            ),
            FunctionToolCallItem(
                call_id="fc_1", name="echo", arguments='{"text":"test"}'
            ),
            FunctionToolOutputItem.from_tool_result(
                call_id="fc_1", output="echo: test"
            ),
        ]
        snap = AgentCheckpoint(
            session_id="s1",
            processor_name="agent",
            messages=messages,
            checkpoint_number=3,
            step=0,
            session_metadata={"parent_id": "p1"},
        )
        json_bytes = snap.model_dump_json().encode()
        restored = AgentCheckpoint.model_validate_json(json_bytes)
        assert restored.checkpoint_number == 3
        assert len(restored.messages) == 5
        assert restored.session_metadata == {"parent_id": "p1"}

        # Verify types survive round-trip
        assert isinstance(restored.messages[0], InputMessageItem)
        assert isinstance(restored.messages[2], OutputMessageItem)
        assert isinstance(restored.messages[3], FunctionToolCallItem)
        assert isinstance(restored.messages[4], FunctionToolOutputItem)


# ================================================================== #
#  prepare_messages_for_resume                                         #
# ================================================================== #


class TestResumeCleanup:
    def test_empty_messages(self):
        state = prepare_messages_for_resume([])
        assert state.messages == []
        assert state.interruption == InterruptionType.NONE
        assert state.removed_count == 0

    def test_clean_conversation(self):
        """No dangling tool calls -> messages unchanged."""
        messages: list[Any] = [
            InputMessageItem.from_text("prompt", role="system"),
            InputMessageItem.from_text("hi", role="user"),
            OutputMessageItem(
                content_parts=[OutputMessageText(text="hello")],
                status="completed",
            ),
        ]
        state = prepare_messages_for_resume(messages)
        assert len(state.messages) == 3
        assert state.interruption == InterruptionType.NONE
        assert state.removed_count == 0

    def test_strips_dangling_tool_calls(self):
        """Unresolved tool calls at the end are stripped."""
        messages: list[Any] = [
            InputMessageItem.from_text("hi", role="user"),
            OutputMessageItem(
                content_parts=[OutputMessageText(text="let me search")],
                status="completed",
            ),
            FunctionToolCallItem(
                call_id="fc_1", name="search", arguments='{"q":"test"}'
            ),
            FunctionToolCallItem(
                call_id="fc_2", name="search", arguments='{"q":"test2"}'
            ),
        ]
        state = prepare_messages_for_resume(messages)
        # Should strip: OutputMessageItem + 2 tool calls (entire assistant turn)
        assert len(state.messages) == 1  # Only user message remains
        assert state.removed_count == 3
        assert state.interruption == InterruptionType.MID_ASSISTANT_TURN

    def test_strips_partial_tool_outputs(self):
        """If some tool outputs exist but not all, strip the whole batch."""
        messages: list[Any] = [
            InputMessageItem.from_text("hi", role="user"),
            OutputMessageItem(
                content_parts=[OutputMessageText(text="searching")],
                status="completed",
            ),
            FunctionToolCallItem(call_id="fc_1", name="search", arguments='{"q":"a"}'),
            FunctionToolCallItem(call_id="fc_2", name="search", arguments='{"q":"b"}'),
            FunctionToolOutputItem.from_tool_result(call_id="fc_1", output="result a"),
            # fc_2 has no output -- crash during tool execution
        ]
        state = prepare_messages_for_resume(messages)
        # Strips: OutputMessageItem + 2 tool calls + 1 partial output
        assert len(state.messages) == 1
        assert state.removed_count == 4
        assert state.interruption == InterruptionType.MID_ASSISTANT_TURN

    def test_fully_resolved_tools_untouched(self):
        """All tool calls have outputs -> nothing stripped."""
        messages: list[Any] = [
            InputMessageItem.from_text("hi", role="user"),
            FunctionToolCallItem(call_id="fc_1", name="echo", arguments='{"text":"a"}'),
            FunctionToolOutputItem.from_tool_result(call_id="fc_1", output="echo: a"),
        ]
        state = prepare_messages_for_resume(messages)
        assert len(state.messages) == 3
        assert state.removed_count == 0
        assert state.interruption == InterruptionType.AFTER_TOOL_RESULT

    def test_strips_reasoning_with_dangling_calls(self):
        """ReasoningItem before tool calls is also stripped."""
        messages: list[Any] = [
            InputMessageItem.from_text("hi", role="user"),
            ReasoningItem(summary=[]),
            OutputMessageItem(
                content_parts=[OutputMessageText(text="thinking...")],
                status="completed",
            ),
            FunctionToolCallItem(
                call_id="fc_1", name="search", arguments='{"q":"test"}'
            ),
        ]
        state = prepare_messages_for_resume(messages)
        assert len(state.messages) == 1  # Only user message
        assert state.removed_count == 3  # reasoning + output + tool call

    def test_pending_user_message(self):
        """Last message is user message waiting for LLM."""
        messages: list[Any] = [
            InputMessageItem.from_text("prompt", role="system"),
            InputMessageItem.from_text("what's 2+2?", role="user"),
        ]
        state = prepare_messages_for_resume(messages)
        assert len(state.messages) == 2
        assert state.interruption == InterruptionType.PENDING_USER_MESSAGE
        assert state.removed_count == 0

    def test_preserves_earlier_resolved_tools(self):
        """Only trailing incomplete turn is stripped, not earlier resolved ones."""
        messages: list[Any] = [
            InputMessageItem.from_text("hi", role="user"),
            # Turn 1: resolved
            FunctionToolCallItem(call_id="fc_1", name="echo", arguments='{"text":"a"}'),
            FunctionToolOutputItem.from_tool_result(call_id="fc_1", output="echo: a"),
            # Turn 2: unresolved
            OutputMessageItem(
                content_parts=[OutputMessageText(text="more work")],
                status="completed",
            ),
            FunctionToolCallItem(call_id="fc_2", name="echo", arguments='{"text":"b"}'),
        ]
        state = prepare_messages_for_resume(messages)
        # Keeps: user + fc_1 call + fc_1 output = 3
        # Strips: OutputMessageItem + fc_2 = 2
        assert len(state.messages) == 3
        assert state.removed_count == 2


# ================================================================== #
#  LLMAgent session integration                                        #
# ================================================================== #


class TestAgentSessionPersistence:
    @pytest.mark.anyio
    async def test_fresh_session_saves_snapshot(self):
        """First run with store saves state via checkpoint callback."""
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent(
            [_text_response("hello")],
            session_id="s1",
            store=store,
        )

        result = await agent.run("hi", ctx=ctx)
        assert result.payloads[0] == "hello"

        # Snapshot should be persisted
        data = await store.load("agent/s1")
        assert data is not None
        snap = AgentCheckpoint.model_validate_json(data)
        assert snap.session_id == "s1"
        assert snap.processor_name == "test_agent"
        assert len(snap.messages) > 0

    @pytest.mark.anyio
    async def test_session_resume_restores_memory(self):
        """Second agent instance picks up where first left off."""
        store = InMemoryCheckpointStore()

        # First run
        agent1, ctx = _make_agent(
            [_text_response("hello")],
            session_id="s1",
            store=store,
        )
        await agent1.run("hi", ctx=ctx)

        # Get saved message count
        data = await store.load("agent/s1")
        assert data is not None
        snap = AgentCheckpoint.model_validate_json(data)
        saved_msg_count = len(snap.messages)
        assert saved_msg_count > 0

        # Second run -- new agent, same session
        agent2, ctx = _make_agent(
            [_text_response("world")],
            session_id="s1",
            store=store,
        )
        await agent2.load_checkpoint(ctx)
        await agent2.run("follow up", ctx=ctx)

        # Snapshot should have more messages now
        data2 = await store.load("agent/s1")
        assert data2 is not None
        snap2 = AgentCheckpoint.model_validate_json(data2)
        assert len(snap2.messages) > saved_msg_count

    @pytest.mark.anyio
    async def test_with_tools_checkpoints_after_tool_turn(self):
        """Checkpoint fires after tool execution, not just on final answer."""
        store = InMemoryCheckpointStore()
        checkpoint_count = 0
        original_save = store.save

        async def counting_save(key: str, data: bytes) -> None:
            nonlocal checkpoint_count
            checkpoint_count += 1
            await original_save(key, data)

        store.save = counting_save  # type: ignore[assignment]

        agent, ctx = _make_agent(
            [
                _tool_call_response("echo", '{"text":"test"}', "fc_1"),
                _text_response("done"),
            ],
            tools=[EchoTool()],
            session_id="s1",
            store=store,
        )

        await agent.run("use echo", ctx=ctx)

        # At least 2 checkpoints: after tool turn + after final answer
        assert checkpoint_count >= 2

    @pytest.mark.anyio
    async def test_reset_memory_on_run_with_store(self):
        """reset_memory_on_run=True wipes memory even with a store."""
        store = InMemoryCheckpointStore()

        # First run -- saves session
        agent1, ctx = _make_agent(
            [_text_response("first")],
            reset_memory_on_run=True,
            session_id="s1",
            store=store,
        )
        await agent1.run("hi", ctx=ctx)

        # Second run -- reset_memory_on_run=True wipes prior messages
        agent2, ctx = _make_agent(
            [_text_response("second")],
            reset_memory_on_run=True,
            session_id="s1",
            store=store,
        )
        await agent2.run("follow up", ctx=ctx)

        # Checkpoint should only have messages from the second run
        data = await store.load("agent/s1")
        assert data is not None
        snap = AgentCheckpoint.model_validate_json(data)
        user_msgs = [
            m
            for m in snap.messages
            if isinstance(m, InputMessageItem) and m.role == "user"
        ]
        assert len(user_msgs) == 1  # only "follow up"

    @pytest.mark.anyio
    async def test_no_store_works_normally(self):
        """Agent without store/session_id works exactly as before."""
        agent, ctx = _make_agent([_text_response("hello")])
        result = await agent.run("hi", ctx=ctx)
        assert result.payloads[0] == "hello"

    @pytest.mark.anyio
    async def test_session_metadata_persisted(self):
        """Session metadata is stored in the snapshot."""
        store = InMemoryCheckpointStore()
        agent = LLMAgent[str, str, None](
            name="test_agent",
            llm=MockLLM(responses_queue=[_text_response("ok")]),
            stream_llm_responses=True,
            session_id="s1",
            session_metadata={"pathway_id": "pw_123"},
        )
        ctx: RunContext[None] = RunContext(store=store)
        await agent.run("hi", ctx=ctx)

        data = await store.load("agent/s1")
        assert data is not None
        snap = AgentCheckpoint.model_validate_json(data)
        assert snap.session_metadata == {"pathway_id": "pw_123"}

    @pytest.mark.anyio
    async def test_checkpoint_number_increments(self):
        """Checkpoint number increases with each save."""
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent(
            [
                _tool_call_response("echo", '{"text":"a"}', "fc_1"),
                _text_response("done"),
            ],
            tools=[EchoTool()],
            session_id="s1",
            store=store,
        )

        await agent.run("go", ctx=ctx)

        data = await store.load("agent/s1")
        assert data is not None
        snap = AgentCheckpoint.model_validate_json(data)
        assert snap.checkpoint_number > 0

    @pytest.mark.anyio
    async def test_stream_interface(self):
        """run_stream with session persistence works."""
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent(
            [_text_response("streamed")],
            session_id="s1",
            store=store,
        )

        events = await _drain_stream(agent.run_stream("hi", ctx=ctx))
        assert len(events) > 0

        # Should have persisted
        data = await store.load("agent/s1")
        assert data is not None


# ================================================================== #
#  Resume input detection                                              #
# ================================================================== #


def _count_user_messages(agent: LLMAgent[Any, Any, Any]) -> int:
    return sum(
        1
        for m in agent.memory.messages
        if isinstance(m, InputMessageItem) and m.role == "user"
    )


def _interrupted_checkpoint(
    messages: list[Any], session_id: str = "s1", step: int | None = 0
) -> AgentCheckpoint:
    """Checkpoint whose last message is a user message → PENDING_USER_MESSAGE."""
    return AgentCheckpoint(
        session_id=session_id,
        processor_name="test_agent",
        messages=messages,
        checkpoint_number=1,
        step=step,
    )


def _completed_checkpoint(
    messages: list[Any], session_id: str = "s1", step: int | None = 0
) -> AgentCheckpoint:
    """Checkpoint whose last message is an assistant response → NONE."""
    return AgentCheckpoint(
        session_id=session_id,
        processor_name="test_agent",
        messages=messages,
        checkpoint_number=1,
        step=step,
    )


class TestResumeInputDetection:
    """
    Verify that resume detection correctly skips or adds inputs
    depending on whether it's a resume (interrupted checkpoint)
    or a chat continuation (clean completion + new input).
    """

    @pytest.mark.anyio
    async def test_standalone_resume_no_inputs(self):
        """Standalone resume (no inputs, interrupted) skips memorization."""
        store = InMemoryCheckpointStore()
        await store.save(
            "agent/s1",
            _interrupted_checkpoint(
                [
                    InputMessageItem.from_text("sys", role="system"),
                    InputMessageItem.from_text("hello", role="user"),
                ]
            )
            .model_dump_json()
            .encode(),
        )

        agent, ctx = _make_agent(
            [_text_response("world")], session_id="s1", store=store
        )
        await agent.run(ctx=ctx, step=0)  # no input — pure resume

        assert (
            _count_user_messages(agent) == 1
        )  # "hello" from checkpoint, not duplicated

    @pytest.mark.anyio
    async def test_workflow_rerun_with_in_args(self):
        """Workflow re-delivers in_args after crash — must not duplicate input."""
        store = InMemoryCheckpointStore()
        await store.save(
            "agent/s1",
            _interrupted_checkpoint(
                [
                    InputMessageItem.from_text("sys", role="system"),
                    InputMessageItem.from_text("hello", role="user"),
                ]
            )
            .model_dump_json()
            .encode(),
        )

        agent, ctx = _make_agent(
            [_text_response("world")], session_id="s1", store=store
        )
        await agent.run(in_args="hello", ctx=ctx, step=0)  # same input re-delivered

        assert _count_user_messages(agent) == 1  # not duplicated

    @pytest.mark.anyio
    async def test_runner_redelivery_with_chat_inputs(self):
        """Runner re-delivers chat_inputs after crash — must not duplicate input."""
        store = InMemoryCheckpointStore()
        await store.save(
            "agent/s1",
            _interrupted_checkpoint(
                [
                    InputMessageItem.from_text("sys", role="system"),
                    InputMessageItem.from_text("start", role="user"),
                ]
            )
            .model_dump_json()
            .encode(),
        )

        agent, ctx = _make_agent([_text_response("done")], session_id="s1", store=store)
        await agent.run("start", ctx=ctx, step=0)  # same chat_inputs re-delivered

        assert _count_user_messages(agent) == 1  # not duplicated

    @pytest.mark.anyio
    async def test_chat_continuation_adds_new_input(self):
        """Clean completion + new chat_inputs = continuation, must memorize."""
        store = InMemoryCheckpointStore()
        await store.save(
            "agent/s1",
            _completed_checkpoint(
                [
                    InputMessageItem.from_text("sys", role="system"),
                    InputMessageItem.from_text("hello", role="user"),
                    OutputMessageItem(
                        content_parts=[OutputMessageText(text="world")],
                        status="completed",
                    ),
                ]
            )
            .model_dump_json()
            .encode(),
        )

        agent, ctx = _make_agent(
            [_text_response("goodbye")], session_id="s1", store=store
        )
        await agent.load_checkpoint(ctx)
        await agent.run("follow up", ctx=ctx)

        assert _count_user_messages(agent) == 2  # "hello" + "follow up"

    @pytest.mark.anyio
    async def test_multi_turn_session_no_duplication(self):
        """Multiple run() calls on same agent — each adds exactly one input."""
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent(
            [
                _text_response("first"),
                _text_response("second"),
                _text_response("third"),
            ],
            session_id="s1",
            store=store,
        )

        await agent.run("turn1", ctx=ctx)
        assert _count_user_messages(agent) == 1

        await agent.run("turn2", ctx=ctx)
        assert _count_user_messages(agent) == 2

        await agent.run("turn3", ctx=ctx)
        assert _count_user_messages(agent) == 3

    @pytest.mark.anyio
    async def test_resume_then_continuation(self):
        """
        Interrupted agent resumes (skip memorization), completes, then
        receives new input (must memorize). Verifies the flags reset properly.
        """
        store = InMemoryCheckpointStore()

        # First agent: run and crash (simulate via pre-seeded interrupted checkpoint)
        await store.save(
            "agent/s1",
            _interrupted_checkpoint(
                [
                    InputMessageItem.from_text("sys", role="system"),
                    InputMessageItem.from_text("hello", role="user"),
                ]
            )
            .model_dump_json()
            .encode(),
        )

        # Resume: no duplication
        agent, ctx = _make_agent(
            [_text_response("world"), _text_response("goodbye")],
            session_id="s1",
            store=store,
        )
        await agent.run(ctx=ctx, step=0)  # resume
        assert _count_user_messages(agent) == 1

        # Continuation: new input added
        await agent.run("follow up", ctx=ctx)
        assert _count_user_messages(agent) == 2


# ================================================================== #
#  Resume integration (workflow / runner)                               #
# ================================================================== #


class _AppendProcessor(Processor[str, str, None]):
    """Appends name to each input."""

    def __init__(self, name: str, *, recipients: list[str] | None = None) -> None:
        super().__init__(name=name, recipients=recipients)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        from grasp_agents.types.events import ProcPayloadOutEvent

        for inp in in_args or []:
            yield ProcPayloadOutEvent(
                data=f"{inp}->{self.name}", source=self.name, exec_id=exec_id
            )


class _CountingAppendProcessor(Processor[str, str, None]):
    """Counts calls and appends name."""

    def __init__(self, name: str, *, recipients: list[str] | None = None) -> None:
        super().__init__(name=name, recipients=recipients)
        self.call_count = 0

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        from grasp_agents.types.events import ProcPayloadOutEvent

        self.call_count += 1
        inputs = in_args or []
        if chat_inputs is not None:
            inputs = [str(chat_inputs)]
        for inp in inputs:
            yield ProcPayloadOutEvent(
                data=f"{inp}->{self.name}", source=self.name, exec_id=exec_id
            )


class _CrashAfterStepWorkflow(SequentialWorkflow[str, str, None]):
    """Crashes before saving the workflow checkpoint for a specific step."""

    def __init__(
        self,
        name: str,
        subprocs: Sequence[Processor[Any, Any, None]],
        crash_after_step: int,
        session_id: str | None = None,
    ) -> None:
        super().__init__(name=name, subprocs=list(subprocs), session_id=session_id)
        self._crash_after_step = crash_after_step

    async def save_checkpoint(
        self,
        ctx: RunContext[None],
        *,
        completed_step: int,
        packet: Packet[Any],
        iteration: int = 0,
    ) -> None:
        if completed_step == self._crash_after_step:
            raise RuntimeError(f"Simulated crash after step {completed_step}")
        await super().save_checkpoint(
            ctx, completed_step=completed_step, packet=packet, iteration=iteration
        )


class TestResumeIntegration:
    """
    Integration tests that run actual Workflow / Runner pipelines
    with LLMAgent sub-processors and verify no input duplication on resume.
    """

    @pytest.mark.anyio
    async def test_workflow_agent_crash_after_agent_step(self):
        """
        Workflow: [Append("A"), LLMAgent].
        Crash after agent step. On resume: A skipped, agent loads
        checkpoint (clean completion), input not duplicated.
        """
        store = InMemoryCheckpointStore()

        # First run — crash after step 1 (agent)
        a1 = _AppendProcessor("A")
        agent1 = LLMAgent[str, str, None](
            name="agent",
            llm=MockLLM(responses_queue=[_text_response("agent_out")]),
            stream_llm_responses=True,
        )
        wf1 = _CrashAfterStepWorkflow(
            name="wf",
            subprocs=[a1, agent1],
            crash_after_step=1,
            session_id="wf1",
        )
        ctx1: RunContext[None] = RunContext(store=store)

        with pytest.raises(ProcRunError):
            async for _ in wf1.run_stream(in_args="start", ctx=ctx1, exec_id="e1"):
                pass

        # Agent checkpoint saved (clean completion)
        agent_data = await store.load("agent/wf1/agent")
        assert agent_data is not None
        agent_cp = AgentCheckpoint.model_validate_json(agent_data)
        user_msgs = [
            m
            for m in agent_cp.messages
            if isinstance(m, InputMessageItem) and m.role == "user"
        ]
        assert len(user_msgs) == 1  # "start->A"

        # Resume
        a2 = _CountingAppendProcessor("A")
        agent2 = LLMAgent[str, str, None](
            name="agent",
            llm=MockLLM(responses_queue=[_text_response("agent_out_2")]),
            stream_llm_responses=True,
        )
        wf2 = SequentialWorkflow[str, str, None](
            name="wf",
            subprocs=[a2, agent2],
            session_id="wf1",
        )
        ctx2: RunContext[None] = RunContext(store=store)

        async for _ in wf2.run_stream(ctx=ctx2, exec_id="e2", step=0):
            pass

        assert a2.call_count == 0  # step 0 skipped

        # Agent checkpoint: still 1 user message (no duplication)
        agent_data2 = await store.load("agent/wf1/agent")
        assert agent_data2 is not None
        agent_cp2 = AgentCheckpoint.model_validate_json(agent_data2)
        user_msgs2 = [
            m
            for m in agent_cp2.messages
            if isinstance(m, InputMessageItem) and m.role == "user"
        ]
        assert len(user_msgs2) == 1

    @pytest.mark.anyio
    async def test_workflow_agent_crash_before_agent_step(self):
        """
        Workflow: [LLMAgent, Append("B")].
        Agent completes, B crashes. On resume: agent loads checkpoint
        (all steps done → emit cached output), B runs fresh.
        """
        store = InMemoryCheckpointStore()

        # First run — crash after step 1 (B)
        agent1 = LLMAgent[str, str, None](
            name="agent",
            llm=MockLLM(responses_queue=[_text_response("agent_out")]),
            stream_llm_responses=True,
        )
        b1 = _AppendProcessor("B")
        wf1 = _CrashAfterStepWorkflow(
            name="wf",
            subprocs=[agent1, b1],
            crash_after_step=1,
            session_id="wf2",
        )
        ctx1: RunContext[None] = RunContext(store=store)

        with pytest.raises(ProcRunError):
            async for _ in wf1.run_stream("start", ctx=ctx1, exec_id="e1"):
                pass

        # Agent checkpoint: clean completion
        assert await store.load("agent/wf2/agent") is not None
        # Workflow checkpoint: step 0 done
        wf_data = await store.load("workflow/wf2")
        assert wf_data is not None

        # Resume
        agent2 = LLMAgent[str, str, None](
            name="agent",
            llm=MockLLM(responses_queue=[]),  # should NOT be called
            stream_llm_responses=True,
        )
        b2 = _CountingAppendProcessor("B")
        wf2 = SequentialWorkflow[str, str, None](
            name="wf",
            subprocs=[agent2, b2],
            session_id="wf2",
        )
        ctx2: RunContext[None] = RunContext(store=store)

        payloads: list[str] = []
        async for event in wf2.run_stream(ctx=ctx2, exec_id="e2", step=0):
            from grasp_agents.types.events import ProcPacketOutEvent

            if isinstance(event, ProcPacketOutEvent) and event.source == "wf":
                payloads = list(event.data.payloads)

        assert b2.call_count == 1
        assert payloads == ["agent_out->B"]

    @pytest.mark.anyio
    async def test_runner_agent_resume_no_duplication(self):
        """
        Runner: Append("A") → LLMAgent → END.
        Agent crashes mid-execution. On resume: runner re-delivers to agent,
        agent loads interrupted checkpoint, input not duplicated.
        """
        from grasp_agents.runner.runner import END_PROC_NAME, Runner
        from grasp_agents.types.events import RunPacketOutEvent

        store = InMemoryCheckpointStore()

        # Pre-seed: A completed, agent's event is pending, agent has checkpoint
        # Simulate: A ran, produced "start->A", runner delivered to agent,
        # agent started (checkpoint saved with user message), then crashed.

        # Seed runner checkpoint with pending event for agent
        from grasp_agents.durability.checkpoints import RunnerCheckpoint
        from grasp_agents.types.events import ProcPacketOutEvent

        agent_input_packet = Packet[str](
            sender="A", payloads=["start->A"], routing=[["agent"]]
        )
        pending_event = ProcPacketOutEvent(
            id=agent_input_packet.id,
            data=agent_input_packet,
            source="A",
            destination="agent",
        )
        runner_cp = RunnerCheckpoint(
            session_id="rs2",
            processor_name="r",
            checkpoint_number=1,
            step=0,
            pending_events=[pending_event],
            active_sessions={"A": "rs2/A", "agent": "rs2/agent"},
            active_steps={"A": 1, "agent": 0},
        )
        await store.save("runner/rs2", runner_cp.model_dump_json().encode())

        # Seed agent's interrupted checkpoint (user message pending)
        agent_cp = AgentCheckpoint(
            session_id="rs2/agent",
            processor_name="agent",
            messages=[
                InputMessageItem.from_text(
                    "You are a helpful assistant.", role="system"
                ),
                InputMessageItem.from_text("start->A", role="user"),
            ],
            checkpoint_number=1,
            step=0,
        )
        await store.save("agent/rs2/agent", agent_cp.model_dump_json().encode())

        # Resume runner
        a2 = _CountingAppendProcessor("A", recipients=["agent"])
        agent2 = LLMAgent[str, str, None](
            name="agent",
            llm=MockLLM(responses_queue=[_text_response("agent_done")]),
            stream_llm_responses=True,
            recipients=[END_PROC_NAME],
        )
        ctx2: RunContext[None] = RunContext(state=None, store=store)
        runner2 = Runner[str, None](
            entry_proc=a2,
            procs=[a2, agent2],
            ctx=ctx2,
            name="r",
        )
        runner2.setup_session("rs2")

        payloads: list[str] = []
        async for event in runner2.run_stream():
            if isinstance(event, RunPacketOutEvent):
                payloads = list(event.data.payloads)

        assert a2.call_count == 0  # A not re-run (not in pending)
        assert payloads == ["agent_done"]

        # Verify: agent checkpoint has exactly 1 user message
        agent_data = await store.load("agent/rs2/agent")
        assert agent_data is not None
        cp = AgentCheckpoint.model_validate_json(agent_data)
        user_msgs = [
            m
            for m in cp.messages
            if isinstance(m, InputMessageItem) and m.role == "user"
        ]
        assert len(user_msgs) == 1


# ================================================================== #
#  TaskRecord model                                                    #
# ================================================================== #


class TestTaskRecord:
    def test_round_trip(self):
        record = TaskRecord(
            task_id="t1",
            parent_session_id="s1",
            tool_call_id="fc_1",
            tool_name="research",
        )
        json_bytes = record.model_dump_json().encode()
        restored = TaskRecord.model_validate_json(json_bytes)
        assert restored.task_id == "t1"
        assert restored.parent_session_id == "s1"
        assert restored.status == TaskStatus.PENDING
        assert restored.result is None

    def test_store_key(self):
        record = TaskRecord(
            task_id="t1",
            parent_session_id="s1",
            tool_call_id="fc_1",
            tool_name="research",
        )
        assert record.store_key == "task/s1/t1"

    def test_model_copy_update(self):
        record = TaskRecord(
            task_id="t1",
            parent_session_id="s1",
            tool_call_id="fc_1",
            tool_name="research",
        )
        updated = record.model_copy(
            update={"status": TaskStatus.COMPLETED, "result": "done"}
        )
        assert updated.status == TaskStatus.COMPLETED
        assert updated.result == "done"
        # Original unchanged
        assert record.status == TaskStatus.PENDING


# ================================================================== #
#  Background tool for persistence tests                               #
# ================================================================== #


class SlowTool(BaseTool[EchoInput, str, Any]):
    """Background tool that simulates a slow operation."""

    def __init__(self, delay: float = 0.05, name: str = "slow") -> None:
        super().__init__(name=name, description="Slow tool", background=True)
        self._delay = delay

    async def _run(
        self,
        inp: EchoInput | None = None,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
        **_kwargs: Any,
    ) -> str:
        assert inp is not None
        await asyncio.sleep(self._delay)
        return f"slow: {inp.text}"


# ================================================================== #
#  TaskRecord persistence in AgentLoop                                 #
# ================================================================== #


class TestTaskRecordPersistence:
    @pytest.mark.anyio
    async def test_bg_task_creates_task_record(self):
        """Spawning a background tool persists a PENDING TaskRecord."""
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent(
            [
                _tool_call_response("slow", '{"text":"data"}', "fc_1"),
                _text_response("waiting"),
                _text_response("done"),
            ],
            tools=[SlowTool(delay=0.05)],
            session_id="s1",
            store=store,
        )

        await agent.run("go", ctx=ctx)

        # Should have a task record under task/s1/
        keys = await store.list_keys("task/s1/")
        assert len(keys) == 1

        data = await store.load(keys[0])
        assert data is not None
        record = TaskRecord.model_validate_json(data)
        assert record.parent_session_id == "s1"
        assert record.tool_name == "slow"
        assert record.tool_call_id == "fc_1"

    @pytest.mark.anyio
    async def test_bg_task_completion_updates_record(self):
        """Completed background task is marked DELIVERED after drain."""
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent(
            [
                _tool_call_response("slow", '{"text":"research"}', "fc_1"),
                _text_response("waiting"),
                _text_response("got it"),
            ],
            tools=[SlowTool(delay=0.05)],
            session_id="s1",
            store=store,
        )

        await agent.run("go", ctx=ctx)

        keys = await store.list_keys("task/s1/")
        assert len(keys) == 1
        data = await store.load(keys[0])
        assert data is not None
        record = TaskRecord.model_validate_json(data)
        assert record.status == TaskStatus.DELIVERED
        assert record.result is not None
        assert "slow: research" in record.result

    @pytest.mark.anyio
    async def test_bg_task_cancellation_marks_failed(self):
        """Background task cancelled on max_turns is marked FAILED."""
        store = InMemoryCheckpointStore()
        very_slow = SlowTool(delay=10.0, name="very_slow")

        llm = MockLLM(
            responses_queue=[
                _tool_call_response("very_slow", '{"text":"x"}', "fc_1"),
                _text_response("still thinking"),
                _text_response("forced final"),
            ]
        )
        agent = LLMAgent[str, str, None](
            name="test_agent",
            llm=llm,
            tools=[very_slow],
            max_turns=1,
            stream_llm_responses=True,
            session_id="s1",
        )
        ctx: RunContext[None] = RunContext(store=store)

        await agent.run("go", ctx=ctx)

        keys = await store.list_keys("task/s1/")
        assert len(keys) == 1
        data = await store.load(keys[0])
        assert data is not None
        record = TaskRecord.model_validate_json(data)
        assert record.status == TaskStatus.CANCELLED
        assert record.error is not None
        assert "Cancelled" in record.error

    @pytest.mark.anyio
    async def test_no_task_records_without_store(self):
        """Without store, no TaskRecords are created."""
        agent, ctx = _make_agent(
            [
                _tool_call_response("slow", '{"text":"data"}', "fc_1"),
                _text_response("waiting"),
                _text_response("done"),
            ],
            tools=[SlowTool(delay=0.05)],
        )

        await agent.run("go", ctx=ctx)
        # No store → nothing to check, just verify no errors


# ================================================================== #
#  Pending task handling on session resume                             #
# ================================================================== #


class TestPendingTaskResume:
    @pytest.mark.anyio
    async def test_pending_record_injects_interruption_notification(self):
        """On resume, a PENDING TaskRecord injects an interruption message."""
        store = InMemoryCheckpointStore()

        # Simulate: agent ran, bg task was spawned, process crashed before completion
        # 1. Save a session snapshot (parent had placeholder in memory)
        snapshot = AgentCheckpoint(
            session_id="s1",
            processor_name="test_agent",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("go", role="user"),
                OutputMessageItem(
                    content_parts=[OutputMessageText(text="launching task")],
                    status="completed",
                ),
                FunctionToolCallItem(
                    call_id="fc_1", name="slow", arguments='{"text":"data"}'
                ),
                FunctionToolOutputItem.from_tool_result(
                    call_id="fc_1",
                    output="Task launched in background (id: abc123)",
                ),
            ],
            checkpoint_number=1,
            step=0,
        )
        await store.save("agent/s1", snapshot.model_dump_json().encode())

        # 2. Save a PENDING task record (simulates crash before completion)
        record = TaskRecord(
            task_id="abc123",
            parent_session_id="s1",
            tool_call_id="fc_1",
            tool_name="slow",
        )
        await store.save(record.store_key, record.model_dump_json().encode())

        # 3. Resume: new agent loads session
        agent, ctx = _make_agent(
            [_text_response("recovered")],
            session_id="s1",
            store=store,
        )
        await agent.run("continue", ctx=ctx, step=0)

        # The interruption notification should be in memory
        memory_texts = [str(m) for m in agent.memory.messages]
        interruption_msgs = [
            t for t in memory_texts if "interrupted" in t and "abc123" in t
        ]
        assert len(interruption_msgs) >= 1

        # TaskRecord should now be FAILED
        data = await store.load(record.store_key)
        assert data is not None
        updated = TaskRecord.model_validate_json(data)
        assert updated.status == TaskStatus.FAILED

    @pytest.mark.anyio
    async def test_completed_record_injects_result(self):
        """On resume, a COMPLETED record whose result isn't in memory gets injected."""
        store = InMemoryCheckpointStore()

        # Snapshot from before the drain notification was checkpointed
        snapshot = AgentCheckpoint(
            session_id="s1",
            processor_name="test_agent",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("go", role="user"),
                FunctionToolCallItem(
                    call_id="fc_1", name="slow", arguments='{"text":"data"}'
                ),
                FunctionToolOutputItem.from_tool_result(
                    call_id="fc_1",
                    output="Task launched in background (id: xyz789)",
                ),
            ],
            checkpoint_number=1,
            step=0,
        )
        await store.save("agent/s1", snapshot.model_dump_json().encode())

        # Task completed between checkpoint and crash
        record = TaskRecord(
            task_id="xyz789",
            parent_session_id="s1",
            tool_call_id="fc_1",
            tool_name="slow",
            status=TaskStatus.COMPLETED,
            result="slow: data",
        )
        await store.save(record.store_key, record.model_dump_json().encode())

        # Resume
        agent, ctx = _make_agent(
            [_text_response("got it")],
            session_id="s1",
            store=store,
        )
        await agent.run("continue", ctx=ctx, step=0)

        # Result notification should be in memory
        memory_texts = [str(m) for m in agent.memory.messages]
        result_msgs = [t for t in memory_texts if "completed" in t and "xyz789" in t]
        assert len(result_msgs) >= 1

        # Record should now be DELIVERED
        data = await store.load(record.store_key)
        assert data is not None
        updated = TaskRecord.model_validate_json(data)
        assert updated.status == TaskStatus.DELIVERED

    @pytest.mark.anyio
    async def test_delivered_record_skipped(self):
        """DELIVERED records are skipped on resume (already injected)."""
        store = InMemoryCheckpointStore()

        # Snapshot has the notification (drained + checkpointed)
        snapshot = AgentCheckpoint(
            session_id="s1",
            processor_name="test_agent",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("go", role="user"),
                FunctionToolCallItem(
                    call_id="fc_1", name="slow", arguments='{"text":"data"}'
                ),
                FunctionToolOutputItem.from_tool_result(
                    call_id="fc_1",
                    output="Task launched in background (id: done1)",
                ),
                InputMessageItem.from_text(
                    "[Background tool 'slow' completed (id: done1)]\n\nslow: data",
                    role="user",
                ),
            ],
            checkpoint_number=2,
            step=0,
        )
        await store.save("agent/s1", snapshot.model_dump_json().encode())

        # Record is DELIVERED — drain already injected + checkpointed
        record = TaskRecord(
            task_id="done1",
            parent_session_id="s1",
            tool_call_id="fc_1",
            tool_name="slow",
            status=TaskStatus.DELIVERED,
            result="slow: data",
        )
        await store.save(record.store_key, record.model_dump_json().encode())

        agent, ctx = _make_agent(
            [_text_response("ok")],
            session_id="s1",
            store=store,
        )
        await agent.run("continue", ctx=ctx, step=0)

        # Should NOT have duplicate notification
        memory_texts = [str(m) for m in agent.memory.messages]
        completed_msgs = [t for t in memory_texts if "completed" in t and "done1" in t]
        assert len(completed_msgs) == 1

    @pytest.mark.anyio
    async def test_failed_record_skipped(self):
        """FAILED records are skipped on resume (already handled)."""
        store = InMemoryCheckpointStore()

        snapshot = AgentCheckpoint(
            session_id="s1",
            processor_name="test_agent",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("go", role="user"),
            ],
            checkpoint_number=1,
            step=0,
        )
        await store.save("agent/s1", snapshot.model_dump_json().encode())

        record = TaskRecord(
            task_id="fail1",
            parent_session_id="s1",
            tool_call_id="fc_1",
            tool_name="slow",
            status=TaskStatus.FAILED,
            error="Already failed",
        )
        await store.save(record.store_key, record.model_dump_json().encode())

        agent, ctx = _make_agent(
            [_text_response("ok")],
            session_id="s1",
            store=store,
        )
        await agent.run("continue", ctx=ctx, step=0)

        # No interruption notification injected for already-failed records
        memory_texts = [str(m) for m in agent.memory.messages]
        fail_msgs = [t for t in memory_texts if "fail1" in t]
        assert len(fail_msgs) == 0

    @pytest.mark.anyio
    async def test_multiple_pending_tasks_all_handled(self):
        """Multiple pending TaskRecords all get interruption notifications."""
        store = InMemoryCheckpointStore()

        snapshot = AgentCheckpoint(
            session_id="s1",
            processor_name="test_agent",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("go", role="user"),
                # Placeholder outputs for both tasks
                FunctionToolCallItem(
                    call_id="fc_1", name="slow_a", arguments='{"text":"a"}'
                ),
                FunctionToolOutputItem.from_tool_result(
                    call_id="fc_1",
                    output="Task launched in background (id: t1)",
                ),
                FunctionToolCallItem(
                    call_id="fc_2", name="slow_b", arguments='{"text":"b"}'
                ),
                FunctionToolOutputItem.from_tool_result(
                    call_id="fc_2",
                    output="Task launched in background (id: t2)",
                ),
            ],
            checkpoint_number=1,
            step=0,
        )
        await store.save("agent/s1", snapshot.model_dump_json().encode())

        for tid, name, cid in [("t1", "slow_a", "fc_1"), ("t2", "slow_b", "fc_2")]:
            record = TaskRecord(
                task_id=tid,
                parent_session_id="s1",
                tool_call_id=cid,
                tool_name=name,
            )
            await store.save(record.store_key, record.model_dump_json().encode())

        agent, ctx = _make_agent(
            [_text_response("recovered")],
            session_id="s1",
            store=store,
        )
        await agent.run("continue", ctx=ctx, step=0)

        # Both should have interruption notifications
        memory_texts = [str(m) for m in agent.memory.messages]
        t1_msgs = [t for t in memory_texts if "t1" in t and "interrupted" in t]
        t2_msgs = [t for t in memory_texts if "t2" in t and "interrupted" in t]
        assert len(t1_msgs) >= 1
        assert len(t2_msgs) >= 1

        # Both records should be FAILED
        for tid in ["t1", "t2"]:
            data = await store.load(f"task/s1/{tid}")
            assert data is not None
            rec = TaskRecord.model_validate_json(data)
            assert rec.status == TaskStatus.FAILED


# ================================================================== #
#  Child agent/tool resume                                             #
# ================================================================== #


def _make_child_tool(
    responses: list[Response],
    *,
    store: InMemoryCheckpointStore,
    tool_name: str = "child_agent",
) -> tuple[LLMAgent[EchoInput, str, None], Any]:
    """Create a child LLMAgent wrapped as a background tool."""
    child = LLMAgent[EchoInput, str, None](
        name="child",
        llm=MockLLM(responses_queue=responses),
        stream_llm_responses=True,
    )
    tool = child.as_tool(
        tool_name=tool_name,
        tool_description="Child agent tool",
        background=True,
    )
    return child, tool


class TestChildTaskResume:
    @pytest.mark.anyio
    async def test_bg_agent_tool_creates_child_session(self):
        """Background agent-as-tool spawns with its own child_session_id."""
        store = InMemoryCheckpointStore()
        _child, tool = _make_child_tool([_text_response("child done")], store=store)

        parent, ctx = _make_agent(
            [
                _tool_call_response("child_agent", '{"text":"work"}', "fc_1"),
                _text_response("waiting"),
                _text_response("parent done"),
            ],
            tools=[tool],
            session_id="parent_s1",
            store=store,
        )
        await parent.run("go", ctx=ctx)

        # TaskRecord should have a child_session_id
        keys = await store.list_keys("task/parent_s1/")
        assert len(keys) == 1
        data = await store.load(keys[0])
        assert data is not None
        record = TaskRecord.model_validate_json(data)
        assert record.child_session_id is not None
        assert record.child_session_id.startswith("child/parent_s1/")

        # Child should have its own session snapshot
        child_snap_data = await store.load(f"agent/{record.child_session_id}")
        assert child_snap_data is not None
        child_snap = AgentCheckpoint.model_validate_json(child_snap_data)
        assert child_snap.processor_name == "child"

    @pytest.mark.anyio
    async def test_child_resumes_from_checkpoint(self):
        """PENDING child with checkpoint is re-spawned, not reported interrupted."""
        store = InMemoryCheckpointStore()

        # 1. Simulate: parent ran, child was spawned but crashed mid-execution
        parent_snapshot = AgentCheckpoint(
            session_id="parent_s1",
            processor_name="test_agent",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("go", role="user"),
                FunctionToolCallItem(
                    call_id="fc_1",
                    name="child_agent",
                    arguments='{"text":"work"}',
                ),
                FunctionToolOutputItem.from_tool_result(
                    call_id="fc_1",
                    output="Task launched in background (id: ch1)",
                ),
            ],
            checkpoint_number=1,
            step=0,
        )
        await store.save("agent/parent_s1", parent_snapshot.model_dump_json().encode())

        # 2. PENDING task record with child_session_id
        task_record = TaskRecord(
            task_id="ch1",
            parent_session_id="parent_s1",
            tool_call_id="fc_1",
            tool_name="child_agent",
            tool_call_arguments='{"text": "hello"}',
            child_session_id="child/parent_s1/ch1",
        )
        await store.save(
            task_record.store_key,
            task_record.model_dump_json().encode(),
        )

        # 3. Child's own session snapshot (checkpointed mid-execution)
        child_snapshot = AgentCheckpoint(
            session_id="child/parent_s1/ch1",
            processor_name="child",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("work", role="user"),
            ],
            checkpoint_number=1,
            step=0,
        )
        await store.save(
            "agent/child/parent_s1/ch1",
            child_snapshot.model_dump_json().encode(),
        )

        # 4. Resume: child needs to complete, parent needs to finish
        _child, tool = _make_child_tool(
            [_text_response("child resumed ok")], store=store
        )
        parent, ctx = _make_agent(
            [
                # Turn 1: waiting (child still running)
                _text_response("waiting for child"),
                # Turn 2: after child completes via drain
                _text_response("parent done"),
            ],
            tools=[tool],
            session_id="parent_s1",
            store=store,
        )
        await parent.run("continue", ctx=ctx, step=0)

        # Should NOT have "interrupted" in memory — child was re-spawned
        memory_texts = [str(m) for m in parent.memory.messages]
        interrupted = [t for t in memory_texts if "interrupted" in t]
        assert len(interrupted) == 0

        # Should have the child's completion notification
        completed = [t for t in memory_texts if "completed" in t and "ch1" in t]
        assert len(completed) >= 1

        # TaskRecord should be DELIVERED (drain completed + notified)
        data = await store.load(task_record.store_key)
        assert data is not None
        updated = TaskRecord.model_validate_json(data)
        assert updated.status == TaskStatus.DELIVERED

    @pytest.mark.anyio
    async def test_non_session_tool_still_reports_interrupted(self):
        """Regular bg tools without child session still get interrupted msg."""
        store = InMemoryCheckpointStore()

        parent_snapshot = AgentCheckpoint(
            session_id="parent_s1",
            processor_name="test_agent",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("go", role="user"),
                FunctionToolCallItem(
                    call_id="fc_1",
                    name="slow",
                    arguments='{"text":"data"}',
                ),
                FunctionToolOutputItem.from_tool_result(
                    call_id="fc_1",
                    output="Task launched in background (id: t1)",
                ),
            ],
            checkpoint_number=1,
            step=0,
        )
        await store.save("agent/parent_s1", parent_snapshot.model_dump_json().encode())

        # PENDING record WITHOUT child_session_id (plain bg tool)
        task_record = TaskRecord(
            task_id="t1",
            parent_session_id="parent_s1",
            tool_call_id="fc_1",
            tool_name="slow",
        )
        await store.save(
            task_record.store_key,
            task_record.model_dump_json().encode(),
        )

        parent, ctx = _make_agent(
            [_text_response("recovered")],
            tools=[SlowTool()],
            session_id="parent_s1",
            store=store,
        )
        await parent.run("continue", ctx=ctx, step=0)

        # Should have "interrupted" notification (not re-spawned)
        memory_texts = [str(m) for m in parent.memory.messages]
        interrupted = [t for t in memory_texts if "interrupted" in t and "t1" in t]
        assert len(interrupted) >= 1

    @pytest.mark.anyio
    async def test_multiple_children_resume_independently(self):
        """Two children from the same tool both resume independently."""
        store = InMemoryCheckpointStore()

        parent_snapshot = AgentCheckpoint(
            session_id="parent_s1",
            processor_name="test_agent",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("go", role="user"),
                FunctionToolCallItem(
                    call_id="fc_1",
                    name="child_agent",
                    arguments='{"text":"task_a"}',
                ),
                FunctionToolOutputItem.from_tool_result(
                    call_id="fc_1",
                    output="Task launched in background (id: ch_a)",
                ),
                FunctionToolCallItem(
                    call_id="fc_2",
                    name="child_agent",
                    arguments='{"text":"task_b"}',
                ),
                FunctionToolOutputItem.from_tool_result(
                    call_id="fc_2",
                    output="Task launched in background (id: ch_b)",
                ),
            ],
            checkpoint_number=1,
            step=0,
        )
        await store.save("agent/parent_s1", parent_snapshot.model_dump_json().encode())

        # Two PENDING children
        for tid, cid, sid in [
            ("ch_a", "fc_1", "child/parent_s1/ch_a"),
            ("ch_b", "fc_2", "child/parent_s1/ch_b"),
        ]:
            rec = TaskRecord(
                task_id=tid,
                parent_session_id="parent_s1",
                tool_call_id=cid,
                tool_name="child_agent",
                tool_call_arguments='{"text": "hello"}',
                child_session_id=sid,
            )
            await store.save(rec.store_key, rec.model_dump_json().encode())
            # Each child has its own snapshot
            snap = AgentCheckpoint(
                session_id=sid,
                processor_name="child",
                messages=[
                    InputMessageItem.from_text("prompt", role="system"),
                    InputMessageItem.from_text(tid, role="user"),
                ],
                checkpoint_number=1,
                step=0,
            )
            await store.save(f"agent/{sid}", snap.model_dump_json().encode())

        # Both children will complete with different results
        _child, tool = _make_child_tool(
            [
                _text_response("result_a"),
                _text_response("result_b"),
            ],
            store=store,
        )
        parent, ctx = _make_agent(
            [
                _text_response("waiting"),
                _text_response("parent done"),
            ],
            tools=[tool],
            session_id="parent_s1",
            store=store,
        )
        await parent.run("continue", ctx=ctx, step=0)

        # Both children should have completed (not interrupted)
        memory_texts = [str(m) for m in parent.memory.messages]
        interrupted = [t for t in memory_texts if "interrupted" in t]
        assert len(interrupted) == 0

        completed = [t for t in memory_texts if "completed" in t]
        assert len(completed) >= 2
