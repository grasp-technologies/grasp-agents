"""
Tests for the filesystem half of agent checkpoints:

- the read-before-write ledger (``read_file_state`` + ``dotfile_overrides``)
  round-trips through ``AgentCheckpoint`` so a resumed agent keeps its
  staleness guard instead of refusing every edit until a re-``Read``;
- ``fs_snapshot`` policy: a ``SnapshotCapable`` environment is snapshotted
  at the configured checkpoint boundaries, only the opaque ref is persisted,
  and resume restores it before anything touches the filesystem;
- failure semantics: a configured-but-incapable environment crashes the
  save; a ref without a capable environment crashes the resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence
    from pathlib import Path

import pytest
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import AgentCheckpoint, InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import CheckpointSchemaError
from grasp_agents.llm.llm import LLM
from grasp_agents.run_context import RunContext
from grasp_agents.sandbox.environment import ExecutionEnvironment, SnapshotCapable
from grasp_agents.sandbox.policy import SandboxPolicy
from grasp_agents.tools.file_backend import LocalFileBackend
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import FunctionToolCallItem, OutputMessageItem
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
)
from grasp_agents.types.response import Response, ResponseUsage
from grasp_agents.types.tool import BaseTool

pytestmark = pytest.mark.asyncio


# ---------- Infrastructure (mirrors tests/test_sessions.py) ----------


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
            FunctionToolCallItem(call_id=call_id, name=name, arguments=arguments)
        ],
        usage_with_cost=_make_usage(),
    )


@dataclass(frozen=True)
class MockLLM(LLM):
    model_name: str = "mock"
    responses_queue: list[Response] = field(default_factory=list[Response])

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        assert self.responses_queue, "MockLLM: no more responses"
        return self.responses_queue.pop(0)

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        response = await self._generate_response_once(input)
        seq = 1
        yield ResponseCreated(response=response, sequence_number=seq)  # type: ignore[arg-type]
        for idx, item in enumerate(response.output):
            seq += 1
            yield OutputItemAdded(item=item, output_index=idx, sequence_number=seq)
            seq += 1
            yield OutputItemDone(item=item, output_index=idx, sequence_number=seq)
        seq += 1
        yield ResponseCompleted(response=response, sequence_number=seq)  # type: ignore[arg-type]


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
        path: Any = None,
        agent_ctx: Any = None,
    ) -> str:
        return inp.text


class _FakeSnapshotEnv(ExecutionEnvironment, SnapshotCapable):
    """SnapshotCapable environment over a local backend, recording calls."""

    def __init__(self, root: Path) -> None:
        self._policy = SandboxPolicy(allowed_roots=(root,))
        self._backend = LocalFileBackend(allowed_roots=[root])
        self.snapshots: list[str] = []
        self.restored: list[str] = []

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    @property
    def file_backend(self) -> LocalFileBackend:
        return self._backend

    @property
    def exec_backend(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def snapshot(self) -> str:
        ref = f"snap-{len(self.snapshots) + 1}"
        self.snapshots.append(ref)
        return ref

    async def restore(self, ref: str) -> None:
        self.restored.append(ref)


class _PlainEnv(ExecutionEnvironment):
    """Environment without the snapshot capability."""

    def __init__(self, root: Path) -> None:
        self._policy = SandboxPolicy(allowed_roots=(root,))
        self._backend = LocalFileBackend(allowed_roots=[root])

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    @property
    def file_backend(self) -> LocalFileBackend:
        return self._backend

    @property
    def exec_backend(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _make_agent(
    responses: list[Response],
    *,
    store: InMemoryCheckpointStore,
    environment: ExecutionEnvironment | None = None,
    fs_snapshot: Literal["off", "final", "turn"] = "off",
    tools: list[BaseTool[Any, Any, Any]] | None = None,
) -> tuple[LLMAgent[str, str, None], RunContext[None]]:
    ctx: RunContext[None] = RunContext(
        checkpoint_store=store,
        session_key="s1",
        environment=environment,
    )
    agent = LLMAgent[str, str, None](
        name="fs_agent",
        ctx=ctx,
        llm=MockLLM(responses_queue=responses),
        tools=tools,
        fs_snapshot=fs_snapshot,
        stream_llm=True,
    )
    return agent, ctx


async def _stored_checkpoint(store: InMemoryCheckpointStore) -> AgentCheckpoint:
    raw = await store.load("s1/agent/fs_agent")
    assert raw is not None
    return AgentCheckpoint.model_validate_json(raw)


# ---------- Read-before-write ledger round-trip ----------


async def test_ledger_round_trips_through_checkpoint(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent([_text_response("done")], store=store)

    target = tmp_path / "notes.md"
    agent._loop.file_edit_state.record_read(target, mtime=123.5)
    agent._loop.file_edit_state.add_dotfile_override(tmp_path / ".env")

    await agent.run("hello")

    checkpoint = await _stored_checkpoint(store)
    assert checkpoint.read_file_state == {str(target): 123.5}
    assert checkpoint.dotfile_overrides == [str(tmp_path / ".env")]

    # Fresh process: same store, new agent — ledger comes back.
    resumed, _ = _make_agent([], store=store)
    loaded = await resumed.load_checkpoint()
    assert loaded is not None
    record = resumed._loop.file_edit_state.get_read_record(target)
    assert record is not None
    assert record.mtime == 123.5
    assert resumed._loop.file_edit_state.dotfile_overrides == {tmp_path / ".env"}


async def test_v1_checkpoint_rejected_by_version_floor() -> None:
    # Pre-v5 records embedded the transcript in the head blob — loading one
    # would silently resume an empty session, so the floor rejects it.
    payload = AgentCheckpoint(
        session_key="s1",
        processor_name="fs_agent",
        messages=[],
    ).model_dump(
        exclude={
            "read_file_state",
            "dotfile_overrides",
            "fs_snapshot_ref",
            "ipy_exec_context_id",
            "nb_exec_context_id",
        }
    )
    payload["schema_version"] = 1
    with pytest.raises(CheckpointSchemaError, match="older than the oldest"):
        AgentCheckpoint.model_validate(payload)


# ---------- fs_snapshot policy ----------


async def test_fs_snapshot_off_by_default(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent([_text_response("done")], store=store, environment=env)

    await agent.run("hello")

    assert env.snapshots == []
    assert (await _stored_checkpoint(store)).fs_snapshot_ref is None


async def test_fs_snapshot_final_snapshots_at_run_end(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [
            _tool_call_response("echo", '{"text": "hi"}', "c1"),
            _text_response("done"),
        ],
        store=store,
        environment=env,
        fs_snapshot="final",
        tools=[EchoTool()],
    )

    await agent.run("hello")

    # Only the run-end boundary snapshots — not AFTER_INPUT / AFTER_TOOL_RESULT.
    assert env.snapshots == ["snap-1"]
    assert (await _stored_checkpoint(store)).fs_snapshot_ref == "snap-1"


async def test_fs_snapshot_turn_snapshots_every_boundary(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [
            _tool_call_response("echo", '{"text": "hi"}', "c1"),
            _text_response("done"),
        ],
        store=store,
        environment=env,
        fs_snapshot="turn",
        tools=[EchoTool()],
    )

    await agent.run("hello")

    # AFTER_INPUT + AFTER_TOOL_RESULT + AFTER_FINAL_ANSWER.
    assert len(env.snapshots) >= 3
    assert (await _stored_checkpoint(store)).fs_snapshot_ref == env.snapshots[-1]


async def test_fs_snapshot_requires_capable_environment(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response("done")],
        store=store,
        environment=_PlainEnv(tmp_path),
        fs_snapshot="final",
    )
    with pytest.raises(TypeError, match="SnapshotCapable"):
        await agent.save_checkpoint()


# ---------- Resume restores the filesystem ----------


async def test_resume_restores_snapshot_ref(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("done")], store=store, environment=env, fs_snapshot="final"
    )
    await agent.run("hello")
    assert env.snapshots == ["snap-1"]

    fresh_env = _FakeSnapshotEnv(tmp_path)
    resumed, _ = _make_agent([], store=store, environment=fresh_env)
    loaded = await resumed.load_checkpoint()
    assert loaded is not None
    assert fresh_env.restored == ["snap-1"]


async def test_resume_with_ref_but_incapable_environment_raises(
    tmp_path: Path,
) -> None:
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("done")], store=store, environment=env, fs_snapshot="final"
    )
    await agent.run("hello")

    resumed, _ = _make_agent([], store=store, environment=_PlainEnv(tmp_path))
    with pytest.raises(RuntimeError, match="not SnapshotCapable"):
        await resumed.load_checkpoint()


# ---------- Resume re-attaches the RunPython kernel ----------


async def test_ipy_exec_context_id_round_trips_through_checkpoint(
    tmp_path: Path,
) -> None:
    """
    The RunPython kernel's context id is captured with the FS snapshot and
    re-seeds the resumed loop's holder. (The actual kernel re-attach is an E2B
    integration test; here we verify the persist + restore wiring offline.)
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("done")], store=store, environment=env, fs_snapshot="final"
    )
    # Stand in for "a RunPython kernel was opened" (no real kernel offline).
    holder = agent._loop.agent_ctx.ipy_kernel_holder
    assert holder is not None
    holder.rebind("ctx-abc")

    await agent.run("hello")

    assert (await _stored_checkpoint(store)).ipy_exec_context_id == "ctx-abc"

    # Fresh process: resume re-seeds the new loop's holder with the same id, so
    # the next RunPython re-attaches instead of opening a fresh context.
    fresh_env = _FakeSnapshotEnv(tmp_path)
    resumed, _ = _make_agent([], store=store, environment=fresh_env)
    loaded = await resumed.load_checkpoint()
    assert loaded is not None
    resumed_holder = resumed._loop.agent_ctx.ipy_kernel_holder
    assert resumed_holder is not None
    assert resumed_holder.context_id == "ctx-abc"


async def test_nb_exec_context_id_round_trips_through_checkpoint(
    tmp_path: Path,
) -> None:
    """Same persist + restore wiring for the RunCell notebook kernel."""
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("done")], store=store, environment=env, fs_snapshot="final"
    )
    # Stand in for "a RunCell kernel was opened" (no real kernel offline).
    agent._loop.agent_ctx.nb_kernel_holder.rebind("nb-ctx-xyz")

    await agent.run("hello")

    assert (await _stored_checkpoint(store)).nb_exec_context_id == "nb-ctx-xyz"

    fresh_env = _FakeSnapshotEnv(tmp_path)
    resumed, _ = _make_agent([], store=store, environment=fresh_env)
    loaded = await resumed.load_checkpoint()
    assert loaded is not None
    assert resumed._loop.agent_ctx.nb_kernel_holder.context_id == "nb-ctx-xyz"


async def test_ipy_exec_context_id_not_captured_without_fs_snapshot(
    tmp_path: Path,
) -> None:
    """
    No FS snapshot -> no context id: the two are a consistent pair (the id is
    only valid inside a restored sandbox).
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("done")], store=store, environment=env, fs_snapshot="off"
    )
    holder = agent._loop.agent_ctx.ipy_kernel_holder
    assert holder is not None
    holder.rebind("ctx-abc")

    await agent.run("hello")

    assert (await _stored_checkpoint(store)).ipy_exec_context_id is None
