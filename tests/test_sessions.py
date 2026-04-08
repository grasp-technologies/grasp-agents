"""
Tests for session persistence layer.

Verifies that:
- LLMAgent with store/session_id auto-loads and auto-saves conversation state
- Resume applies CC-style cleanup (strips dangling tool calls)
- Checkpoints fire at correct boundaries (after tools, after final answer)
- Multi-turn sessions maintain conversation across run() calls
- Sessions survive simulated restarts (save -> new agent -> load -> continue)
- InMemoryCheckpointStore operations work correctly
- SessionSnapshot round-trips through JSON serialization
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

from grasp_agents.llm import LLM
from grasp_agents.llm_agent import LLMAgent
from grasp_agents.sessions import (
    InMemoryCheckpointStore,
    InterruptionType,
    SessionSnapshot,
    TaskRecord,
    TaskStatus,
    prepare_messages_for_resume,
)
from grasp_agents.types.content import OutputMessageText
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
) -> LLMAgent[str, str, None]:
    llm = MockLLM(responses_queue=responses)
    return LLMAgent[str, str, None](
        name="test_agent",
        llm=llm,
        tools=tools,
        reset_memory_on_run=reset_memory_on_run,
        stream_llm_responses=True,
        session_id=session_id,
        store=store,
    )


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
#  SessionSnapshot serialization                                       #
# ================================================================== #


class TestSessionSnapshot:
    def test_round_trip_empty(self):
        snap = SessionSnapshot(
            session_id="s1",
            agent_name="agent",
            messages=[],
        )
        json_bytes = snap.model_dump_json().encode()
        restored = SessionSnapshot.model_validate_json(json_bytes)
        assert restored.session_id == "s1"
        assert restored.messages == []
        assert restored.turn_number == 0

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
        snap = SessionSnapshot(
            session_id="s1",
            agent_name="agent",
            messages=messages,
            turn_number=3,
            metadata={"parent_id": "p1"},
        )
        json_bytes = snap.model_dump_json().encode()
        restored = SessionSnapshot.model_validate_json(json_bytes)
        assert restored.turn_number == 3
        assert len(restored.messages) == 5
        assert restored.metadata == {"parent_id": "p1"}

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
            FunctionToolCallItem(
                call_id="fc_1", name="search", arguments='{"q":"a"}'
            ),
            FunctionToolCallItem(
                call_id="fc_2", name="search", arguments='{"q":"b"}'
            ),
            FunctionToolOutputItem.from_tool_result(
                call_id="fc_1", output="result a"
            ),
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
            FunctionToolCallItem(
                call_id="fc_1", name="echo", arguments='{"text":"a"}'
            ),
            FunctionToolOutputItem.from_tool_result(
                call_id="fc_1", output="echo: a"
            ),
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
            FunctionToolCallItem(
                call_id="fc_1", name="echo", arguments='{"text":"a"}'
            ),
            FunctionToolOutputItem.from_tool_result(
                call_id="fc_1", output="echo: a"
            ),
            # Turn 2: unresolved
            OutputMessageItem(
                content_parts=[OutputMessageText(text="more work")],
                status="completed",
            ),
            FunctionToolCallItem(
                call_id="fc_2", name="echo", arguments='{"text":"b"}'
            ),
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
        agent = _make_agent(
            [_text_response("hello")],
            session_id="s1",
            store=store,
        )

        result = await agent.run("hi")
        assert result.payloads[0] == "hello"

        # Snapshot should be persisted
        data = await store.load("session/s1")
        assert data is not None
        snap = SessionSnapshot.model_validate_json(data)
        assert snap.session_id == "s1"
        assert snap.agent_name == "test_agent"
        assert len(snap.messages) > 0

    @pytest.mark.anyio
    async def test_session_resume_restores_memory(self):
        """Second agent instance picks up where first left off."""
        store = InMemoryCheckpointStore()

        # First run
        agent1 = _make_agent(
            [_text_response("hello")],
            session_id="s1",
            store=store,
        )
        await agent1.run("hi")

        # Get saved message count
        data = await store.load("session/s1")
        assert data is not None
        snap = SessionSnapshot.model_validate_json(data)
        saved_msg_count = len(snap.messages)
        assert saved_msg_count > 0

        # Second run -- new agent, same session
        agent2 = _make_agent(
            [_text_response("world")],
            session_id="s1",
            store=store,
        )
        await agent2.run("follow up")

        # Snapshot should have more messages now
        data2 = await store.load("session/s1")
        assert data2 is not None
        snap2 = SessionSnapshot.model_validate_json(data2)
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

        agent = _make_agent(
            [
                _tool_call_response("echo", '{"text":"test"}', "fc_1"),
                _text_response("done"),
            ],
            tools=[EchoTool()],
            session_id="s1",
            store=store,
        )

        await agent.run("use echo")

        # At least 2 checkpoints: after tool turn + after final answer
        assert checkpoint_count >= 2

    @pytest.mark.anyio
    async def test_store_suppresses_reset_memory_on_run(self):
        """When store is set, reset_memory_on_run is effectively disabled."""
        store = InMemoryCheckpointStore()

        # First run -- saves session
        agent1 = _make_agent(
            [_text_response("first")],
            reset_memory_on_run=True,
            session_id="s1",
            store=store,
        )
        await agent1.run("hi")

        # Second run -- agent has reset_memory_on_run=True but store overrides
        agent2 = _make_agent(
            [_text_response("second")],
            reset_memory_on_run=True,
            session_id="s1",
            store=store,
        )
        # Before run, memory is empty. Run will load session first.
        await agent2.run("follow up")

        # Check the snapshot has messages from BOTH runs
        data = await store.load("session/s1")
        assert data is not None
        snap = SessionSnapshot.model_validate_json(data)
        # Should have: sys prompt + "hi" + response1 + "follow up" + response2
        # (not just "follow up" + response2, which would happen with reset)
        assert len(snap.messages) > 3

    @pytest.mark.anyio
    async def test_no_store_works_normally(self):
        """Agent without store/session_id works exactly as before."""
        agent = _make_agent([_text_response("hello")])
        result = await agent.run("hi")
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
            store=store,
            session_metadata={"pathway_id": "pw_123"},
        )
        await agent.run("hi")

        data = await store.load("session/s1")
        assert data is not None
        snap = SessionSnapshot.model_validate_json(data)
        assert snap.metadata == {"pathway_id": "pw_123"}

    @pytest.mark.anyio
    async def test_turn_number_increments(self):
        """Turn number increases with each checkpoint."""
        store = InMemoryCheckpointStore()
        agent = _make_agent(
            [
                _tool_call_response("echo", '{"text":"a"}', "fc_1"),
                _text_response("done"),
            ],
            tools=[EchoTool()],
            session_id="s1",
            store=store,
        )

        await agent.run("go")

        data = await store.load("session/s1")
        assert data is not None
        snap = SessionSnapshot.model_validate_json(data)
        assert snap.turn_number > 0

    @pytest.mark.anyio
    async def test_stream_interface(self):
        """run_stream with session persistence works."""
        store = InMemoryCheckpointStore()
        agent = _make_agent(
            [_text_response("streamed")],
            session_id="s1",
            store=store,
        )

        events = await _drain_stream(agent.run_stream("hi"))
        assert len(events) > 0

        # Should have persisted
        data = await store.load("session/s1")
        assert data is not None


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
        inp: EchoInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
    ) -> str:
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
        agent = _make_agent(
            [
                _tool_call_response("slow", '{"text":"data"}', "fc_1"),
                _text_response("waiting"),
                _text_response("done"),
            ],
            tools=[SlowTool(delay=0.05)],
            session_id="s1",
            store=store,
        )

        await agent.run("go")

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
        agent = _make_agent(
            [
                _tool_call_response("slow", '{"text":"research"}', "fc_1"),
                _text_response("waiting"),
                _text_response("got it"),
            ],
            tools=[SlowTool(delay=0.05)],
            session_id="s1",
            store=store,
        )

        await agent.run("go")

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
            store=store,
        )

        await agent.run("go")

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
        agent = _make_agent(
            [
                _tool_call_response("slow", '{"text":"data"}', "fc_1"),
                _text_response("waiting"),
                _text_response("done"),
            ],
            tools=[SlowTool(delay=0.05)],
        )

        await agent.run("go")
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
        snapshot = SessionSnapshot(
            session_id="s1",
            agent_name="test_agent",
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
            turn_number=1,
        )
        await store.save("session/s1", snapshot.model_dump_json().encode())

        # 2. Save a PENDING task record (simulates crash before completion)
        record = TaskRecord(
            task_id="abc123",
            parent_session_id="s1",
            tool_call_id="fc_1",
            tool_name="slow",
        )
        await store.save(record.store_key, record.model_dump_json().encode())

        # 3. Resume: new agent loads session
        agent = _make_agent(
            [_text_response("recovered")],
            session_id="s1",
            store=store,
        )
        await agent.run("continue")

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
        snapshot = SessionSnapshot(
            session_id="s1",
            agent_name="test_agent",
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
            turn_number=1,
        )
        await store.save("session/s1", snapshot.model_dump_json().encode())

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
        agent = _make_agent(
            [_text_response("got it")],
            session_id="s1",
            store=store,
        )
        await agent.run("continue")

        # Result notification should be in memory
        memory_texts = [str(m) for m in agent.memory.messages]
        result_msgs = [
            t for t in memory_texts if "completed" in t and "xyz789" in t
        ]
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
        snapshot = SessionSnapshot(
            session_id="s1",
            agent_name="test_agent",
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
            turn_number=2,
        )
        await store.save("session/s1", snapshot.model_dump_json().encode())

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

        agent = _make_agent(
            [_text_response("ok")],
            session_id="s1",
            store=store,
        )
        await agent.run("continue")

        # Should NOT have duplicate notification
        memory_texts = [str(m) for m in agent.memory.messages]
        completed_msgs = [
            t for t in memory_texts if "completed" in t and "done1" in t
        ]
        assert len(completed_msgs) == 1

    @pytest.mark.anyio
    async def test_failed_record_skipped(self):
        """FAILED records are skipped on resume (already handled)."""
        store = InMemoryCheckpointStore()

        snapshot = SessionSnapshot(
            session_id="s1",
            agent_name="test_agent",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("go", role="user"),
            ],
            turn_number=1,
        )
        await store.save("session/s1", snapshot.model_dump_json().encode())

        record = TaskRecord(
            task_id="fail1",
            parent_session_id="s1",
            tool_call_id="fc_1",
            tool_name="slow",
            status=TaskStatus.FAILED,
            error="Already failed",
        )
        await store.save(record.store_key, record.model_dump_json().encode())

        agent = _make_agent(
            [_text_response("ok")],
            session_id="s1",
            store=store,
        )
        await agent.run("continue")

        # No interruption notification injected for already-failed records
        memory_texts = [str(m) for m in agent.memory.messages]
        fail_msgs = [t for t in memory_texts if "fail1" in t]
        assert len(fail_msgs) == 0

    @pytest.mark.anyio
    async def test_multiple_pending_tasks_all_handled(self):
        """Multiple pending TaskRecords all get interruption notifications."""
        store = InMemoryCheckpointStore()

        snapshot = SessionSnapshot(
            session_id="s1",
            agent_name="test_agent",
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
            turn_number=1,
        )
        await store.save("session/s1", snapshot.model_dump_json().encode())

        for tid, name, cid in [("t1", "slow_a", "fc_1"), ("t2", "slow_b", "fc_2")]:
            record = TaskRecord(
                task_id=tid,
                parent_session_id="s1",
                tool_call_id=cid,
                tool_name=name,
            )
            await store.save(record.store_key, record.model_dump_json().encode())

        agent = _make_agent(
            [_text_response("recovered")],
            session_id="s1",
            store=store,
        )
        await agent.run("continue")

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
        store=store,
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
        _child, tool = _make_child_tool(
            [_text_response("child done")], store=store
        )

        parent = _make_agent(
            [
                _tool_call_response(
                    "child_agent", '{"text":"work"}', "fc_1"
                ),
                _text_response("waiting"),
                _text_response("parent done"),
            ],
            tools=[tool],
            session_id="parent_s1",
            store=store,
        )
        await parent.run("go")

        # TaskRecord should have a child_session_id
        keys = await store.list_keys("task/parent_s1/")
        assert len(keys) == 1
        data = await store.load(keys[0])
        assert data is not None
        record = TaskRecord.model_validate_json(data)
        assert record.child_session_id is not None
        assert record.child_session_id.startswith("child/parent_s1/")

        # Child should have its own session snapshot
        child_snap_data = await store.load(f"session/{record.child_session_id}")
        assert child_snap_data is not None
        child_snap = SessionSnapshot.model_validate_json(child_snap_data)
        assert child_snap.agent_name == "child"

    @pytest.mark.anyio
    async def test_child_resumes_from_checkpoint(self):
        """PENDING child with checkpoint is re-spawned, not reported interrupted."""
        store = InMemoryCheckpointStore()

        # 1. Simulate: parent ran, child was spawned but crashed mid-execution
        parent_snapshot = SessionSnapshot(
            session_id="parent_s1",
            agent_name="test_agent",
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
            turn_number=1,
        )
        await store.save(
            "session/parent_s1", parent_snapshot.model_dump_json().encode()
        )

        # 2. PENDING task record with child_session_id
        task_record = TaskRecord(
            task_id="ch1",
            parent_session_id="parent_s1",
            tool_call_id="fc_1",
            tool_name="child_agent",
            child_session_id="child/parent_s1/ch1",
        )
        await store.save(
            task_record.store_key,
            task_record.model_dump_json().encode(),
        )

        # 3. Child's own session snapshot (checkpointed mid-execution)
        child_snapshot = SessionSnapshot(
            session_id="child/parent_s1/ch1",
            agent_name="child",
            messages=[
                InputMessageItem.from_text("system prompt", role="system"),
                InputMessageItem.from_text("work", role="user"),
            ],
            turn_number=1,
        )
        await store.save(
            "session/child/parent_s1/ch1",
            child_snapshot.model_dump_json().encode(),
        )

        # 4. Resume: child needs to complete, parent needs to finish
        _child, tool = _make_child_tool(
            [_text_response("child resumed ok")], store=store
        )
        parent = _make_agent(
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
        await parent.run("continue")

        # Should NOT have "interrupted" in memory — child was re-spawned
        memory_texts = [str(m) for m in parent.memory.messages]
        interrupted = [t for t in memory_texts if "interrupted" in t]
        assert len(interrupted) == 0

        # Should have the child's completion notification
        completed = [
            t for t in memory_texts if "completed" in t and "ch1" in t
        ]
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

        parent_snapshot = SessionSnapshot(
            session_id="parent_s1",
            agent_name="test_agent",
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
            turn_number=1,
        )
        await store.save(
            "session/parent_s1", parent_snapshot.model_dump_json().encode()
        )

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

        parent = _make_agent(
            [_text_response("recovered")],
            tools=[SlowTool()],
            session_id="parent_s1",
            store=store,
        )
        await parent.run("continue")

        # Should have "interrupted" notification (not re-spawned)
        memory_texts = [str(m) for m in parent.memory.messages]
        interrupted = [
            t for t in memory_texts if "interrupted" in t and "t1" in t
        ]
        assert len(interrupted) >= 1

    @pytest.mark.anyio
    async def test_multiple_children_resume_independently(self):
        """Two children from the same tool both resume independently."""
        store = InMemoryCheckpointStore()

        parent_snapshot = SessionSnapshot(
            session_id="parent_s1",
            agent_name="test_agent",
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
            turn_number=1,
        )
        await store.save(
            "session/parent_s1", parent_snapshot.model_dump_json().encode()
        )

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
                child_session_id=sid,
            )
            await store.save(
                rec.store_key, rec.model_dump_json().encode()
            )
            # Each child has its own snapshot
            snap = SessionSnapshot(
                session_id=sid,
                agent_name="child",
                messages=[
                    InputMessageItem.from_text("prompt", role="system"),
                    InputMessageItem.from_text(tid, role="user"),
                ],
                turn_number=1,
            )
            await store.save(f"session/{sid}", snap.model_dump_json().encode())

        # Both children will complete with different results
        _child, tool = _make_child_tool(
            [
                _text_response("result_a"),
                _text_response("result_b"),
            ],
            store=store,
        )
        parent = _make_agent(
            [
                _text_response("waiting"),
                _text_response("parent done"),
            ],
            tools=[tool],
            session_id="parent_s1",
            store=store,
        )
        await parent.run("continue")

        # Both children should have completed (not interrupted)
        memory_texts = [str(m) for m in parent.memory.messages]
        interrupted = [t for t in memory_texts if "interrupted" in t]
        assert len(interrupted) == 0

        completed = [t for t in memory_texts if "completed" in t]
        assert len(completed) >= 2
