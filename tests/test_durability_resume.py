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
from grasp_agents.durability.checkpoint_mixin import CheckpointPersistMixin
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
from grasp_agents.run_context import RunContext
from grasp_agents.types.items import InputItem, InputMessageItem

from .test_sessions import (  # type: ignore[attr-defined]
    MockLLM,
    _text_response,
    _tool_call_response,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


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


class TestWithRetryTranscript:
    async def test_retry_does_not_duplicate_input(self) -> None:
        agent = LLMAgent[str, str, None](
            name="t",
            ctx=RunContext[None](),
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


# ---------- Item 16: approval wait bounded by run deadline ----------


class TestRunDeadlineBoundsApprovalWait:
    async def test_parked_before_tool_hook_times_out(self) -> None:
        from .test_agent_loop import (
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

        loop.before_tool_hook = park_forever  # type: ignore[assignment]

        async def drain() -> None:
            async for _ in loop.execute_stream(exec_id="t"):
                pass

        await asyncio.wait_for(drain(), timeout=5.0)

        outputs = _tool_outputs_for(transcript, "tc1")
        assert len(outputs) == 1
        assert "run deadline exceeded" in str(outputs[0].output_parts)


# ---------- Item 17: rewrite crash cannot mis-pair head and log ----------


class _AgentHolder(CheckpointPersistMixin):
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
    async def test_crash_between_rewrite_and_head_keeps_old_pair(
        self, tmp_path: Path
    ) -> None:
        store = _CrashBeforeHeadStore(tmp_path)
        ctx = RunContext[None](checkpoint_store=store, session_key="s1")

        holder = _AgentHolder()
        cp1 = AgentCheckpoint(
            session_key="s1", processor_name="test_agent", messages=_msgs("a", "b")
        )
        await holder._serialize_agent_checkpoint(ctx, cp1)

        # Full-history rewrite whose head save crashes.
        holder._log_dirty = True
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

    async def test_successful_rewrite_supersedes_and_cleans_up(
        self, tmp_path: Path
    ) -> None:
        store = FileCheckpointStore(tmp_path)
        ctx = RunContext[None](checkpoint_store=store, session_key="s1")
        key = "s1/agent/test_agent"

        holder = _AgentHolder()
        cp1 = AgentCheckpoint(
            session_key="s1", processor_name="test_agent", messages=_msgs("a", "b")
        )
        await holder._serialize_agent_checkpoint(ctx, cp1)

        holder._log_dirty = True
        cp2 = AgentCheckpoint(
            session_key="s1", processor_name="test_agent", messages=_msgs("x")
        )
        await holder._serialize_agent_checkpoint(ctx, cp2)

        fresh = _AgentHolder()
        head = await fresh._deserialize_agent_checkpoint(ctx)
        assert head is not None
        assert head.log_version == 1
        assert _texts(head.messages) == ["x"]
        # The superseded generation-0 file is gone.
        assert await store.read_messages(key, version=0) == []


# ---------- Item 18: log appends are fsynced ----------


class TestAppendFsync:
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
    async def test_loaded_session_log_renders(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = InMemoryCheckpointStore()
        agent1 = LLMAgent[str, str, None](
            name="t",
            ctx=RunContext[None](checkpoint_store=store, session_key="s1"),
            llm=MockLLM(responses_queue=[_text_response("one")]),
            env_info=False,
        )
        await agent1.run("hi")  # step is None (chat-style run)

        agent2 = LLMAgent[str, str, None](
            name="t",
            ctx=RunContext[None](checkpoint_store=store, session_key="s1"),
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
    async def test_delete_evicts_lock(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        await store.save("s1/agent/x", b"{}")
        assert "s1/agent/x" in store._locks

        await store.delete("s1/agent/x")
        assert "s1/agent/x" not in store._locks
