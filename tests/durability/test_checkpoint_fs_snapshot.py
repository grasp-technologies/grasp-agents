"""
Tests for the filesystem half of agent checkpoints:

- the read-before-write ledger (``read_file_state`` + ``dotfile_overrides``)
  round-trips through ``AgentCheckpoint`` so a resumed agent keeps its
  staleness guard instead of refusing every edit until a re-``Read``;
- ``SessionContext.fs_snapshot_policy``: a ``SnapshotCapable``
  environment is snapshotted at the configured checkpoint boundaries, only
  the opaque ref is persisted — into the per-session record (current state)
  and the agent's step watermarks (rollback) — and ``ctx.load_checkpoint()``
  restores it before anything touches the filesystem;
- a step rollback rewinds the filesystem through the session
  (``SessionContext.restore_fs_snapshot``), re-pointing the session record
  at the restored ref;
- failure semantics: a configured-but-incapable environment crashes the
  save; a session record with a ref but no capable environment crashes the
  resume and the rollback restore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import (
    AgentCheckpoint,
    InMemoryCheckpointStore,
    SessionCheckpoint,
)
from grasp_agents.durability.checkpoints import CheckpointSchemaError
from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.sandbox.environment import ExecutionEnvironment
from grasp_agents.sandbox.policy import SandboxPolicy
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.response import Response
from tests._helpers import FakeSnapshotEnv as _FakeSnapshotEnv
from tests._helpers import MockLLM, _text_response, _tool_call_response

pytestmark = pytest.mark.asyncio


# ---------- Infrastructure (mirrors tests/test_sessions.py) ----------


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
    fs_snapshot_policy: Literal["off", "final", "turn"] = "off",
    tools: list[BaseTool[Any, Any, Any]] | None = None,
) -> tuple[LLMAgent[str, str, None], SessionContext[None]]:
    ctx: SessionContext[None] = SessionContext(
        checkpoint_store=store,
        session_key="s1",
        environment=environment,
        fs_snapshot_policy=fs_snapshot_policy,
    )
    agent = LLMAgent[str, str, None](
        name="fs_agent",
        ctx=ctx,
        llm=MockLLM(responses_queue=responses),
        tools=tools,
        stream_llm=True,
    )
    return agent, ctx


async def _stored_checkpoint(store: InMemoryCheckpointStore) -> AgentCheckpoint:
    raw = await store.load("s1/agent/fs_agent")
    assert raw is not None
    return AgentCheckpoint.model_validate_json(raw)


async def _session_record(store: InMemoryCheckpointStore) -> SessionCheckpoint | None:
    raw = await store.load("s1/session")
    if raw is None:
        return None
    return SessionCheckpoint.model_validate_json(raw)


# ---------- Read-before-write ledger round-trip ----------


async def test_ledger_round_trips_through_checkpoint(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent([_text_response("done")], store=store)

    target = tmp_path / "notes.md"
    agent._loop.agent_ctx.file_edit_state.record_read(target, mtime=123.5)
    agent._loop.agent_ctx.file_edit_state.add_dotfile_override(tmp_path / ".env")

    await agent.run("hello")

    checkpoint = await _stored_checkpoint(store)
    assert checkpoint.current.agent_ctx_state.read_file_state == {str(target): 123.5}
    assert checkpoint.current.agent_ctx_state.dotfile_overrides == [
        str(tmp_path / ".env")
    ]

    # Fresh process: same store, new agent — ledger comes back.
    resumed, _ = _make_agent([], store=store)
    loaded = await resumed.load_checkpoint()
    assert loaded is not None
    record = resumed._loop.agent_ctx.file_edit_state.get_read_record(target)
    assert record is not None
    assert record.mtime == 123.5
    assert resumed._loop.agent_ctx.file_edit_state.dotfile_overrides == {
        tmp_path / ".env"
    }


async def test_v1_checkpoint_rejected_by_version_floor() -> None:
    # Pre-v5 records embedded the transcript in the head blob — loading one
    # would silently resume an empty session, so the floor rejects it.
    payload = AgentCheckpoint(
        session_key="s1",
        processor_name="fs_agent",
        messages=[],
    ).model_dump(exclude={"current", "step_watermarks"})
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
    assert (await _stored_checkpoint(store)).current.fs_snapshot_ref is None
    # Mode off + no serialized state / metadata: no session record at all.
    assert await _session_record(store) is None


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
        fs_snapshot_policy="final",
        tools=[EchoTool()],
    )

    await agent.run("hello")

    # Only the run-end boundary snapshots — not AFTER_INPUT / AFTER_TOOL_RESULT.
    assert env.snapshots == ["snap-1"]
    assert (await _stored_checkpoint(store)).current.fs_snapshot_ref == "snap-1"
    # The session record pairs the same ref with this boundary.
    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-1"


async def test_session_ref_kept_at_snapshotless_boundary(tmp_path: Path) -> None:
    """
    The session record keeps the *latest* snapshot ref: a later boundary that
    takes no snapshot carries the last ref forward rather than nulling it —
    the record is shared by every agent on the session, so a snapshotless
    save must never erase the ref a cold resume restores from.
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("done")],
        store=store,
        environment=env,
        fs_snapshot_policy="final",
    )
    await agent.run("hello")
    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-1"

    # A non-final boundary (AFTER_INPUT default) takes no snapshot — the
    # record keeps the ref describing the session filesystem.
    await agent.save_checkpoint()
    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-1"


async def test_non_rewinder_checkpoint_keeps_session_ref(tmp_path: Path) -> None:
    """
    A non-rewinder member's checkpoint must not clobber the session ref: its
    ``_fs_snapshot_due`` is gated off, so its saves pass no ref — the shared
    record still has to keep the rewinder's latest snapshot, or a cold resume
    would restore no filesystem at all.
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    ctx: SessionContext[None] = SessionContext(
        checkpoint_store=store,
        session_key="s1",
        environment=env,
        fs_snapshot_policy="final",
        environment_rewinder="lead",
    )
    lead = LLMAgent[str, str, None](
        name="lead",
        ctx=ctx,
        llm=MockLLM(responses_queue=[_text_response("a0")]),
    )
    peer = LLMAgent[str, str, None](
        name="peer",
        ctx=ctx,
        llm=MockLLM(responses_queue=[_text_response("b0")]),
    )

    await lead.run("q0", step=0)
    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-1"

    # The peer's run checkpoints (snapshotless) — the lead's ref survives.
    await peer.run("p0")
    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-1"

    # A cold resume restores the lead's filesystem.
    fresh_env = _FakeSnapshotEnv(tmp_path)
    rctx: SessionContext[None] = SessionContext(
        checkpoint_store=store, session_key="s1", environment=fresh_env
    )
    assert await rctx.load_checkpoint() is not None
    assert fresh_env.restored == ["snap-1"]


async def test_resumed_process_snapshotless_save_keeps_session_ref(
    tmp_path: Path,
) -> None:
    """
    The latest-ref carry-forward survives a process restart: the write path
    reads the ref off the on-disk record, so a resumed process's first
    snapshotless save carries it forward instead of erasing it.
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("done")],
        store=store,
        environment=env,
        fs_snapshot_policy="final",
    )
    await agent.run("hello")

    fresh_env = _FakeSnapshotEnv(tmp_path)
    resumed, rctx = _make_agent(
        [], store=store, environment=fresh_env, fs_snapshot_policy="final"
    )
    await rctx.load_checkpoint()
    await resumed.load_checkpoint()
    await resumed.save_checkpoint()  # AFTER_INPUT: no snapshot due

    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-1"


async def test_concurrent_process_snapshotless_save_preserves_newer_ref(
    tmp_path: Path,
) -> None:
    """
    Two live ctx instances on one session (a multi-process member topology):
    a non-snapshotting save must preserve the NEWER ref another process wrote
    after this one loaded — the write path reads the on-disk record rather
    than trusting anything process-local.
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("a0"), _text_response("a1")],
        store=store,
        environment=env,
        fs_snapshot_policy="final",
    )
    await agent.run("q0")  # record → snap-1

    # "Process B": a second live ctx (shared environment, non-rewinder) that
    # loaded while the record was snap-1.
    ctx_b: SessionContext[None] = SessionContext(
        checkpoint_store=store,
        session_key="s1",
        environment=env,
        environment_rewinder="fs_agent",
        session_metadata={"member": "b"},
    )
    await ctx_b.load_checkpoint()

    await agent.run("q1")  # record → snap-2 ("process A" moved on)

    await ctx_b.save_checkpoint()  # B checkpoints without a snapshot
    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-2"


async def test_midrun_crash_under_final_policy_injects_skew_notice(
    tmp_path: Path,
) -> None:
    """
    Under ``fs_snapshot_policy="final"``, a crash after a mid-run tool-result
    checkpoint resumes a transcript that has advanced past the restored
    snapshot. The resume must say so: a warning plus an injected
    filesystem-restored notice, so the agent re-verifies file claims instead
    of trusting them.
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [
            _text_response("a0"),
            # Run 2: a tool round completes (mid-run AFTER_TOOL_RESULT
            # checkpoint fires), then the next LLM call crashes (empty queue).
            _tool_call_response("echo", '{"text": "hi"}', "c1"),
        ],
        store=store,
        environment=env,
        fs_snapshot_policy="final",
        tools=[EchoTool()],
    )
    await agent.run("q0")  # run 1 ends → snap-1
    with pytest.raises(ProcRunError):
        await agent.run("q1")

    # Cold resume: the session rewinds the filesystem to snap-1 while the
    # agent's head carries run 2's completed tool round.
    fresh_env = _FakeSnapshotEnv(tmp_path)
    resumed, rctx = _make_agent(
        [], store=store, environment=fresh_env, fs_snapshot_policy="final"
    )
    await rctx.load_checkpoint()
    assert fresh_env.restored == ["snap-1"]
    loaded = await resumed.load_checkpoint()
    assert loaded is not None

    blob = str(resumed.transcript.messages)
    assert "<filesystem_restored>" in blob
    assert resumed._resume_notifications  # streamed at the next run

    # A run-end head (paired with its snapshot) resumes without the notice.
    store2 = InMemoryCheckpointStore()
    env2 = _FakeSnapshotEnv(tmp_path)
    agent2, _ = _make_agent(
        [_text_response("done")],
        store=store2,
        environment=env2,
        fs_snapshot_policy="final",
    )
    await agent2.run("q0")
    fresh_env2 = _FakeSnapshotEnv(tmp_path)
    resumed2, rctx2 = _make_agent(
        [], store=store2, environment=fresh_env2, fs_snapshot_policy="final"
    )
    await rctx2.load_checkpoint()
    assert await resumed2.load_checkpoint() is not None
    assert "<filesystem_restored>" not in str(resumed2.transcript.messages)


async def test_restore_fs_snapshot_verifies_the_claimant(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    ctx: SessionContext[None] = SessionContext(
        checkpoint_store=store,
        session_key="s1",
        environment=env,
        fs_snapshot_policy="final",
        environment_rewinder="lead",
    )

    with pytest.raises(RuntimeError, match="rewind right"):
        await ctx.restore_fs_snapshot("snap-1", claimant="peer")
    assert env.restored == []

    await ctx.restore_fs_snapshot("snap-1", claimant="lead")
    assert env.restored == ["snap-1"]


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
        fs_snapshot_policy="turn",
        tools=[EchoTool()],
    )

    await agent.run("hello")

    # AFTER_INPUT + AFTER_TOOL_RESULT + AFTER_FINAL_ANSWER.
    assert len(env.snapshots) >= 3
    stored = await _stored_checkpoint(store)
    assert stored.current.fs_snapshot_ref == env.snapshots[-1]


async def test_fs_snapshot_requires_capable_environment(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response("done")],
        store=store,
        environment=_PlainEnv(tmp_path),
        fs_snapshot_policy="final",
    )
    with pytest.raises(TypeError, match="SnapshotCapable"):
        await agent.save_checkpoint()


# ---------- Resume restores the filesystem (ctx.load_checkpoint) ----------


async def test_session_load_restores_snapshot_ref(tmp_path: Path) -> None:
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("done")],
        store=store,
        environment=env,
        fs_snapshot_policy="final",
    )
    await agent.run("hello")
    assert env.snapshots == ["snap-1"]

    # Fresh process: the ctx (not any one agent) restores the shared fs.
    fresh_env = _FakeSnapshotEnv(tmp_path)
    _, rctx = _make_agent([], store=store, environment=fresh_env)
    record = await rctx.load_checkpoint()
    assert record is not None
    assert fresh_env.restored == ["snap-1"]
    assert rctx.session_fs_restored

    # Idempotent: a second participant's run start restores nothing again.
    assert await rctx.load_checkpoint() is None
    assert fresh_env.restored == ["snap-1"]


async def test_session_load_with_ref_but_incapable_environment_raises(
    tmp_path: Path,
) -> None:
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("done")],
        store=store,
        environment=env,
        fs_snapshot_policy="final",
    )
    await agent.run("hello")

    _, rctx = _make_agent([], store=store, environment=_PlainEnv(tmp_path))
    with pytest.raises(RuntimeError, match="not SnapshotCapable"):
        await rctx.load_checkpoint()


# ---------- Rollback restores the filesystem through the session ----------


async def test_rollback_restores_fs_and_repoints_session_record(
    tmp_path: Path,
) -> None:
    """
    A step rollback rewinds the shared filesystem to the boundary's snapshot
    AND rewrites the session record with that ref, so a crash right after the
    rollback cold-resumes into the rewound filesystem — never the pre-rollback
    one paired with a rolled-back transcript.
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, ctx = _make_agent(
        [_text_response("a0"), _text_response("a1")],
        store=store,
        environment=env,
        fs_snapshot_policy="final",
    )

    await agent.run("q0", step=0)
    await agent.run("q1", step=1)
    assert env.snapshots == ["snap-1", "snap-2"]
    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-2"

    # Step 1's boundary captured the filesystem as of the end of step 0.
    await agent.rollback_to_step(1)

    assert env.restored == ["snap-1"]
    assert ctx.session_fs_restored
    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-1"
    # The agent's rolled-back head pairs the same ref.
    assert (await _stored_checkpoint(store)).current.fs_snapshot_ref == "snap-1"

    # A later snapshotless save keeps the re-pointed ref (the rollback's
    # re-point replaced the latch, even though the ref is older).
    await agent.save_checkpoint()
    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-1"


async def test_restore_fs_snapshot_requires_capable_environment(
    tmp_path: Path,
) -> None:
    ctx: SessionContext[None] = SessionContext(environment=_PlainEnv(tmp_path))
    with pytest.raises(RuntimeError, match="SnapshotCapable"):
        await ctx.restore_fs_snapshot("snap-1")


async def test_rollback_from_cold_instance_restores_fs(tmp_path: Path) -> None:
    """
    A fresh process rolls back from persisted boundaries: the filesystem is
    rewound to the boundary's snapshot, the rewind right is claimed, and the
    session record is re-pointed even though the resumed ctx never snapshots
    itself (the re-point is unconditional).
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    agent, _ = _make_agent(
        [_text_response("a0"), _text_response("a1")],
        store=store,
        environment=env,
        fs_snapshot_policy="final",
    )
    await agent.run("q0", step=0)
    await agent.run("q1", step=1)

    fresh_env = _FakeSnapshotEnv(tmp_path)
    agent2, rctx = _make_agent([], store=store, environment=fresh_env)
    assert await agent2.load_checkpoint() is not None
    await agent2.rollback_to_step(1)

    assert fresh_env.restored == ["snap-1"]
    assert rctx.environment_rewinder == "fs_agent"
    record = await _session_record(store)
    assert record is not None
    assert record.fs_snapshot_ref == "snap-1"


# ---------- Environment-rewind right (one rewinder per session) ----------


async def test_first_stepped_delivery_claims_rewind_right(tmp_path: Path) -> None:
    """
    With snapshots on, the FIRST stepped delivery claims the session's
    unclaimed rewind right at delivery start — steps are the rewind points,
    so the stepper is the rewinder. Every later agent (stepped or not) never
    snapshots: its boundaries carry no refs, so its rollbacks are
    transcript-only and never conflict.
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    ctx: SessionContext[None] = SessionContext(
        checkpoint_store=store,
        session_key="s1",
        environment=env,
        fs_snapshot_policy="final",
    )
    coordinator = LLMAgent[str, str, None](
        name="coordinator",
        ctx=ctx,
        llm=MockLLM(responses_queue=[_text_response("a0"), _text_response("a1")]),
    )
    other = LLMAgent[str, str, None](
        name="other",
        ctx=ctx,
        llm=MockLLM(responses_queue=[_text_response("b0"), _text_response("b1")]),
    )

    # Claimed at step 0's delivery start — not at a later snapshot-carrying
    # boundary — so nothing else in the session ever snapshots.
    await coordinator.run("q0", step=0)
    assert ctx.environment_rewinder == "coordinator"
    assert env.snapshots == ["snap-1"]
    await coordinator.run("q1", step=1)
    assert env.snapshots == ["snap-1", "snap-2"]

    # A second stepped agent never snapshots, so its boundaries carry no
    # refs and stepped delivery keeps working (transcript-only rollback).
    await other.run("p0", step=0)
    await other.run("p1", step=1)
    assert env.snapshots == ["snap-1", "snap-2"]
    assert all(wm.fs_snapshot_ref is None for wm in other._step_watermarks)
    assert ctx.environment_rewinder == "coordinator"

    # An explicit rewind attempt by the non-holder still fails loudly.
    with pytest.raises(RuntimeError, match="rewind right"):
        ctx.claim_environment_rewind("other")


async def test_subagents_never_snapshot_under_stepped_coordinator(
    tmp_path: Path,
) -> None:
    """
    Coordinator+subagents with NO declared rewinder: the coordinator's step-0
    claim happens at delivery start, before any turn runs, so a subagent's
    unstepped ``.as_tool()`` run inside the step is already gated — the only
    snapshot is the coordinator's own run-end one.
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    ctx: SessionContext[None] = SessionContext(
        checkpoint_store=store,
        session_key="s1",
        environment=env,
        fs_snapshot_policy="final",
    )
    sub = LLMAgent[EchoInput, str, None](
        name="sub",
        ctx=ctx,
        llm=MockLLM(responses_queue=[_text_response("sub done")]),
        stream_llm=True,
    )
    coordinator = LLMAgent[str, str, None](
        name="coordinator",
        ctx=ctx,
        llm=MockLLM(
            responses_queue=[
                _tool_call_response("sub_agent", '{"text": "hi"}', "c1"),
                _text_response("done"),
            ]
        ),
        tools=[sub.as_tool(tool_name="sub_agent", tool_description="Sub agent")],
        stream_llm=True,
    )

    await coordinator.run("q0", step=0)

    assert ctx.environment_rewinder == "coordinator"
    # The subagent's run completed mid-step (its own run-end boundary would
    # have snapshotted under an unclaimed right) — only the coordinator's
    # run-end snapshot exists.
    assert env.snapshots == ["snap-1"]


async def test_declared_rewinder_gates_snapshots_from_construction(
    tmp_path: Path,
) -> None:
    """
    ``SessionContext(environment_rewinder=...)`` fixes the lead up front:
    even an agent that runs *first* (which under lazy claiming would have
    snapshotted and claimed) takes no snapshots; only the declared lead does.
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    ctx: SessionContext[None] = SessionContext(
        checkpoint_store=store,
        session_key="s1",
        environment=env,
        fs_snapshot_policy="final",
        environment_rewinder="coordinator",
    )
    other = LLMAgent[str, str, None](
        name="other",
        ctx=ctx,
        llm=MockLLM(responses_queue=[_text_response("b0"), _text_response("b1")]),
    )
    coordinator = LLMAgent[str, str, None](
        name="coordinator",
        ctx=ctx,
        llm=MockLLM(responses_queue=[_text_response("a0")]),
    )

    await other.run("p0", step=0)
    await other.run("p1", step=1)
    assert env.snapshots == []

    await coordinator.run("q0", step=0)
    assert env.snapshots == ["snap-1"]
    assert ctx.environment_rewinder == "coordinator"


async def test_failed_step_settles_and_rollback_restores_boundary_fs(
    tmp_path: Path,
) -> None:
    """
    A failed step settles to its last closed round — completed tool work
    stays — and the step boundary archived at delivery start is untouched, so
    rolling the step back still restores the filesystem snapshot from the end
    of the previous step.
    """
    store = InMemoryCheckpointStore()
    env = _FakeSnapshotEnv(tmp_path)
    responses = [
        _text_response("a0"),
        # Step 1's failed attempt: a tool round completes (mid-run
        # checkpoint fires), then the next LLM call crashes (empty queue).
        _tool_call_response("echo", '{"text": "hi"}', "c1"),
    ]
    agent, _ = _make_agent(
        responses,
        store=store,
        environment=env,
        fs_snapshot_policy="final",
        tools=[EchoTool()],
    )

    await agent.run("q0", step=0)
    assert env.snapshots == ["snap-1"]
    messages_after_step_0 = len(agent.transcript.messages)

    with pytest.raises(ProcRunError):
        await agent.run("q1", step=1)
    # Settled, not reverted: the input and the completed round survive.
    assert len(agent.transcript.messages) > messages_after_step_0

    await agent.rollback_to_step(1)
    assert env.restored == ["snap-1"]
    assert len(agent.transcript.messages) == messages_after_step_0


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
        [_text_response("done")],
        store=store,
        environment=env,
        fs_snapshot_policy="final",
    )
    # Stand in for "a RunPython kernel was opened" (no real kernel offline).
    holder = agent._loop.agent_ctx.ipy_kernel_holder
    assert holder is not None
    holder.rebind("ctx-abc")

    await agent.run("hello")

    assert (
        await _stored_checkpoint(store)
    ).current.agent_ctx_state.ipy_exec_context_id == "ctx-abc"

    # Fresh process: resume re-seeds the new loop's holder with the same id, so
    # the next RunPython re-attaches instead of opening a fresh context. The
    # kernel re-attach is gated on the ctx having actually restored the fs
    # (run order: ctx.load_checkpoint, then the agent's own load).
    fresh_env = _FakeSnapshotEnv(tmp_path)
    resumed, rctx = _make_agent([], store=store, environment=fresh_env)
    await rctx.load_checkpoint()
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
        [_text_response("done")],
        store=store,
        environment=env,
        fs_snapshot_policy="final",
    )
    # Stand in for "a RunCell kernel was opened" (no real kernel offline).
    agent._loop.agent_ctx.nb_kernel_holder.rebind("nb-ctx-xyz")

    await agent.run("hello")

    assert (
        await _stored_checkpoint(store)
    ).current.agent_ctx_state.nb_exec_context_id == "nb-ctx-xyz"

    fresh_env = _FakeSnapshotEnv(tmp_path)
    resumed, rctx = _make_agent([], store=store, environment=fresh_env)
    await rctx.load_checkpoint()
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
        [_text_response("done")], store=store, environment=env, fs_snapshot_policy="off"
    )
    holder = agent._loop.agent_ctx.ipy_kernel_holder
    assert holder is not None
    holder.rebind("ctx-abc")

    await agent.run("hello")

    assert (
        await _stored_checkpoint(store)
    ).current.agent_ctx_state.ipy_exec_context_id is None
