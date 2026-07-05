"""
Durability / resume internals: retry input de-duplication, append-log fsync
and rewrite-crash safety, schema-version floor enforcement, and per-key store
lock eviction.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import (
    AgentCheckpoint,
    CheckpointKind,
    FileCheckpointStore,
    InMemoryCheckpointStore,
)
from grasp_agents.durability.checkpoint_mixin import AgentCheckpointPersistMixin
from grasp_agents.durability.checkpoints import (
    CURRENT_SCHEMA_VERSION,
    MIN_SUPPORTED_SCHEMA_VERSION,
    CheckpointSchemaError,
)
from grasp_agents.durability.context_serialization import (
    ContextKind,
    rehydrate_context,
    serialize_context,
)
from grasp_agents.session_context import SessionContext
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.items import (
    FunctionToolCallItem,
    InputItem,
    InputMessageItem,
)
from tests._helpers import AddTool
from tests.durability.test_sessions import (  # type: ignore[attr-defined]
    MockLLM,
    _text_response,
    _tool_call_response,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from grasp_agents.types.response import Response


def _msgs(*texts: str) -> list[InputItem]:
    return [InputMessageItem.from_text(t, role="user") for t in texts]


def _texts(messages: list[InputItem]) -> list[str]:
    return [m.text or "" for m in messages if isinstance(m, InputMessageItem)]


# ---------- Item 15: with_retry does not duplicate inputs ----------


@dataclass(frozen=True)
class _FlakyLLM(MockLLM):
    fail_times: int = 1

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "_fails_left", self.fail_times)

    async def _generate_response_once(self, *args: Any, **kwargs: Any) -> Any:
        fails_left: int = self._fails_left  # type: ignore[attr-defined]
        if fails_left > 0:
            object.__setattr__(self, "_fails_left", fails_left - 1)
            msg = "transient boom"
            raise RuntimeError(msg)
        return await super()._generate_response_once(*args, **kwargs)


@dataclass(frozen=True)
class _FailsOnceOnCallLLM(MockLLM):
    """Raises once, on its Nth generate call, then defers to the queue."""

    fail_on_call: int = 2

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "_calls_seen", 0)
        object.__setattr__(self, "_failed", False)

    async def _generate_response_once(self, *args: Any, **kwargs: Any) -> Any:
        calls_seen: int = self._calls_seen + 1  # type: ignore[attr-defined]
        object.__setattr__(self, "_calls_seen", calls_seen)
        if not self._failed and calls_seen == self.fail_on_call:  # type: ignore[attr-defined]
            object.__setattr__(self, "_failed", True)
            msg = "transient boom"
            raise RuntimeError(msg)
        return await super()._generate_response_once(*args, **kwargs)


class TestWithRetryTranscript:
    @pytest.mark.asyncio
    async def test_retry_does_not_duplicate_input(self) -> None:
        agent = LLMAgent[str, str, None](
            name="t",
            ctx=SessionContext[None](),
            llm=_FlakyLLM(responses_queue=[_text_response("ok")], fail_times=1),
            max_retries=1,
            env_info=False,
        )
        result = await agent.run("only once")
        assert result.payloads[0] == "ok"

        user_msgs = [
            m
            for m in agent.transcript.messages
            if isinstance(m, InputMessageItem) and m.role == "user"
        ]
        assert len(user_msgs) == 1

    @pytest.mark.asyncio
    async def test_retry_continues_from_settled_state(self) -> None:
        """
        A retry after a mid-run failure continues the settled delivery: the
        completed tool round is kept (not re-issued) and the loop picks up
        where the failed attempt stopped.
        """
        llm = _FailsOnceOnCallLLM(
            responses_queue=[
                _tool_call_response("add", '{"a": 1, "b": 2}', "c1"),
                _text_response("3"),
            ],
            fail_on_call=2,  # the call after the tool round
        )
        agent = LLMAgent[str, str, None](
            name="t",
            ctx=SessionContext[None](),
            llm=llm,
            tools=[AddTool()],
            max_retries=1,
            env_info=False,
        )
        result = await agent.run("sum 1 and 2")
        assert result.payloads[0] == "3"

        user_msgs = [
            m
            for m in agent.transcript.messages
            if isinstance(m, InputMessageItem) and m.role == "user"
        ]
        assert len(user_msgs) == 1  # input not re-memorized
        # The tool ran exactly once — its round survived the failure.
        assert llm.call_count == 2  # tool-call response + final answer
        call_ids = [
            m.call_id
            for m in agent.transcript.messages
            if isinstance(m, FunctionToolCallItem)
        ]
        assert call_ids == ["c1"]


# ---------- Item 16: approval wait bounded by run deadline ----------


class TestRunDeadlineBoundsApprovalWait:
    @pytest.mark.asyncio
    async def test_parked_before_tool_hook_times_out(self) -> None:
        from tests.agent.test_agent_loop import (
            EchoTool,
            _make_loop,
            _tool_outputs_for,
        )

        loop, transcript = _make_loop(
            [
                _tool_call_response("echo", '{"text":"x"}', "tc1"),
                _text_response("done"),
            ],
            tools=[EchoTool()],
        )
        loop.run_timeout = 0.2

        async def park_forever(**kwargs: Any) -> None:
            await asyncio.Event().wait()

        loop.before_tool_hooks = [park_forever]  # type: ignore[assignment]

        async def drain() -> None:
            async for _ in loop.execute_stream(exec_id="t"):
                pass

        await asyncio.wait_for(drain(), timeout=5.0)

        outputs = _tool_outputs_for(transcript, "tc1")
        assert len(outputs) == 1
        assert "run deadline exceeded" in str(outputs[0].output)


# ---------- Item 17: rewrite crash cannot mis-pair head and log ----------


class _AgentHolder(AgentCheckpointPersistMixin):
    _checkpoint_kind = CheckpointKind.AGENT

    def __init__(self) -> None:
        self._path = ["test_agent"]
        self._checkpoint_number = 0


class _CrashBeforeHeadStore(FileCheckpointStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.fail_next_save = False

    async def save(self, key: str, data: bytes) -> None:
        if self.fail_next_save:
            self.fail_next_save = False
            msg = "simulated crash before head save"
            raise RuntimeError(msg)
        await super().save(key, data)


class TestRewriteGenerations:
    @pytest.mark.asyncio
    async def test_crash_between_rewrite_and_head_keeps_old_pair(
        self, tmp_path: Path
    ) -> None:
        store = _CrashBeforeHeadStore(tmp_path)
        ctx = SessionContext[None](checkpoint_store=store, session_key="s1")

        holder = _AgentHolder()
        cp1 = AgentCheckpoint(
            session_key="s1", processor_name="test_agent", messages=_msgs("a", "b")
        )
        await holder._serialize_agent_checkpoint(ctx, cp1)

        # A diverging message set (not a prefix-extension of "a","b") forces a
        # full-history rewrite to a new log version; its head save then crashes.
        cp2 = AgentCheckpoint(
            session_key="s1", processor_name="test_agent", messages=_msgs("x")
        )
        store.fail_next_save = True
        with pytest.raises(RuntimeError, match="simulated crash"):
            await holder._serialize_agent_checkpoint(ctx, cp2)

        # Resume sees the OLD, consistent head + log pair — not the old
        # head's watermark sliced over the rewritten log.
        fresh = _AgentHolder()
        head = await fresh._deserialize_agent_checkpoint(ctx)
        assert head is not None
        assert _texts(head.messages) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_successful_rewrite_supersedes_and_cleans_up(
        self, tmp_path: Path
    ) -> None:
        store = FileCheckpointStore(tmp_path)
        ctx = SessionContext[None](checkpoint_store=store, session_key="s1")
        key = "s1/agent/test_agent"

        holder = _AgentHolder()
        cp1 = AgentCheckpoint(
            session_key="s1", processor_name="test_agent", messages=_msgs("a", "b")
        )
        await holder._serialize_agent_checkpoint(ctx, cp1)

        # Diverging messages (not a prefix-extension) force a full-history
        # rewrite to a new log version.
        cp2 = AgentCheckpoint(
            session_key="s1", processor_name="test_agent", messages=_msgs("x")
        )
        await holder._serialize_agent_checkpoint(ctx, cp2)

        fresh = _AgentHolder()
        head = await fresh._deserialize_agent_checkpoint(ctx)
        assert head is not None
        assert head.current.log_version == 1
        assert _texts(head.messages) == ["x"]
        # The superseded generation-0 file is gone.
        assert await store.read_messages(key, version=0) == []


# ---------- Item 18: log appends are fsynced ----------


class TestAppendFsync:
    @pytest.mark.asyncio
    async def test_append_fsyncs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synced: list[int] = []
        real_fsync = os.fsync

        def spy(fd: int) -> None:
            synced.append(fd)
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", spy)

        store = FileCheckpointStore(tmp_path)
        await store.append_messages("k", _msgs("a"))
        assert synced


# ---------- Item 19: dataclass state rehydration ----------


@dataclass
class _Inner:
    x: int = 0


@dataclass
class _State:
    a: int = 0
    inner: _Inner = field(default_factory=_Inner)
    version: int = field(default=1, init=False)


class TestDataclassRehydration:
    def test_nested_and_init_false_fields(self) -> None:
        state = _State(a=2, inner=_Inner(x=7))
        state.version = 9

        kind, data = serialize_context(state)
        assert kind == ContextKind.DATACLASS

        restored = rehydrate_context(kind, data, _State())
        assert isinstance(restored, _State)
        assert restored.a == 2
        assert isinstance(restored.inner, _Inner)
        assert restored.inner.x == 7
        assert restored.version == 9

    def test_bad_payload_keeps_current_state(self) -> None:
        current = _State(a=1)
        restored = rehydrate_context(
            ContextKind.DATACLASS, {"a": "not-an-int-at-all", "inner": 42}, current
        )
        assert restored is current


# ---------- Item 20: resume log line renders with step=None ----------


class TestResumeLogLine:
    @pytest.mark.asyncio
    async def test_loaded_session_log_renders(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = InMemoryCheckpointStore()
        agent1 = LLMAgent[str, str, None](
            name="t",
            ctx=SessionContext[None](checkpoint_store=store, session_key="s1"),
            llm=MockLLM(responses_queue=[_text_response("one")]),
            env_info=False,
        )
        # A typed-args run stays unstepped (a chat run would auto-mint a step),
        # so the persisted head carries step=None.
        await agent1.run(in_args="hi")

        agent2 = LLMAgent[str, str, None](
            name="t",
            ctx=SessionContext[None](checkpoint_store=store, session_key="s1"),
            llm=MockLLM(responses_queue=[]),
            env_info=False,
        )
        with caplog.at_level(logging.INFO, logger="grasp_agents.agent.llm_agent"):
            await agent2.load_checkpoint()

        loaded = [r for r in caplog.records if "Loaded session" in r.msg]
        assert loaded
        # The original defect: %d with step=None raised inside logging and
        # the diagnostic never rendered.
        assert "step=None" in loaded[0].getMessage()


# ---------- Resume recomposes the ephemeral header (system prompt) ----------


@dataclass(frozen=True)
class _RecordingLLM(MockLLM):
    """MockLLM that records the input items of every generate call."""

    recorded_inputs: list[list[Any]] = field(default_factory=list)

    async def _generate_response_once(
        self, input: Sequence[Any], **kwargs: Any
    ) -> Response:
        self.recorded_inputs.append(list(input))
        return await super()._generate_response_once(input, **kwargs)


class TestResumeRebuildsEphemeralHeader:
    @pytest.mark.asyncio
    async def test_pure_resume_restores_system_prompt(self) -> None:
        """
        The ephemeral header is never persisted, so a resumed step must
        recompose it — otherwise the model continues without its instructions.
        """
        store = InMemoryCheckpointStore()
        sys_prompt = "You are a calculator. After the first sum, add 10 to it."

        # Run 1: the model issues a tool call; the call + result are committed
        # to the log, then the next LLM call crashes (empty response queue).
        agent1 = LLMAgent[str, str, None](
            name="calc",
            ctx=SessionContext[None](checkpoint_store=store, session_key="s1"),
            llm=MockLLM(
                responses_queue=[_tool_call_response("add", '{"a": 1, "b": 2}', "c1")]
            ),
            sys_prompt=sys_prompt,
            tools=[AddTool()],
            env_info=False,
        )
        with pytest.raises(ProcRunError):
            await agent1.run("go")

        # Run 2 (fresh instance, same session): pure resume with no inputs.
        recording_llm = _RecordingLLM(responses_queue=[_text_response("13")])
        agent2 = LLMAgent[str, str, None](
            name="calc",
            ctx=SessionContext[None](checkpoint_store=store, session_key="s1"),
            llm=recording_llm,
            sys_prompt=sys_prompt,
            tools=[AddTool()],
            env_info=False,
        )
        await agent2.run()

        assert recording_llm.recorded_inputs, "resumed run must call the LLM"
        first_call = recording_llm.recorded_inputs[0]
        system_messages = [
            m
            for m in first_call
            if isinstance(m, InputMessageItem) and m.role == "system"
        ]
        assert system_messages, "resumed step must recompose the system prompt"
        assert sys_prompt in system_messages[0].text
        # The restored conversation follows the header.
        assert any(
            isinstance(m, InputMessageItem) and m.role == "user" and m.text == "go"
            for m in first_call
        )


# ---------- Item 21: minimum schema-version floor ----------


class TestSchemaVersionFloor:
    def test_too_old_record_rejected(self) -> None:
        with pytest.raises(CheckpointSchemaError, match="older than the oldest"):
            AgentCheckpoint(
                schema_version=MIN_SUPPORTED_SCHEMA_VERSION - 1,
                session_key="s1",
                processor_name="t",
            )

    def test_floor_and_current_load(self) -> None:
        for version in (MIN_SUPPORTED_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION):
            cp = AgentCheckpoint(
                schema_version=version, session_key="s1", processor_name="t"
            )
            assert cp.schema_version == version


# ---------- Item 22: per-key locks are evicted on delete ----------


class TestLockEviction:
    @pytest.mark.asyncio
    async def test_delete_evicts_lock(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        await store.save("s1/agent/x", b"{}")
        assert "s1/agent/x" in store._locks

        await store.delete("s1/agent/x")
        assert "s1/agent/x" not in store._locks
