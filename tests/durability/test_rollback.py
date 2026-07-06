"""
Step rollback (``LLMAgent.rollback_to_step``) and its primitives.

Verifies:
- ``LLMAgentTranscript.truncate`` and ``CheckpointStore.truncate_messages``
- a live session rewinds to a step boundary: transcript, durable log, turn,
  cached output, and the boundary map are all cut back
- after rollback the discarded step is a *fresh* delivery, not a cached one
- a cold instance (new process) can roll back via the persisted boundaries
- unknown step raises; view-layer compaction does NOT block rollback (E0)
- unstepped (typed-args) deliveries record no boundaries; chat deliveries
  auto-mint steps (covered in tests/agent/test_human_turn_anchors.py)
"""

import asyncio
import contextlib
from unittest.mock import patch

import pytest

from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.durability import AgentContextState, InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import AgentCheckpointLocation
from grasp_agents.inbox import AgentInbox
from grasp_agents.mailbox import CheckpointMailboxTransport
from grasp_agents.types.items import InputMessageItem
from grasp_agents.types.message import TeamMessage
from tests._helpers import _text_response
from tests.durability.test_sessions import _make_agent, load_agent_checkpoint

_KEY = "s1/agent/test_agent"


# --- Primitives ---


def test_transcript_truncate() -> None:
    transcript = LLMAgentTranscript()
    transcript.update(
        [InputMessageItem.from_text(f"m{i}", role="user") for i in range(5)]
    )
    transcript.truncate(2)
    assert len(transcript.messages) == 2
    transcript.truncate(10)  # count >= len → no-op
    assert len(transcript.messages) == 2
    transcript.truncate(0)
    assert transcript.messages == []


@pytest.mark.asyncio
async def test_store_truncate_messages() -> None:
    store = InMemoryCheckpointStore()
    msgs = [InputMessageItem.from_text(f"m{i}", role="user") for i in range(5)]
    await store.append_messages("k", msgs)

    await store.truncate_messages("k", message_count=2)
    assert len(await store.read_messages("k")) == 2

    await store.truncate_messages("k", message_count=10)  # >= len → no-op
    assert len(await store.read_messages("k")) == 2

    await store.truncate_messages("k", message_count=0)  # → delete
    assert await store.read_messages("k") == []


# --- rollback_to_step ---


@pytest.mark.asyncio
async def test_rollback_rewinds_live_session() -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response(a) for a in ("a0", "a1", "a2", "a1_new")],
        session_key="s1",
        store=store,
    )

    r0 = await agent.run("q0", step=0)
    len_after_0 = len(agent.transcript.messages)
    r1 = await agent.run("q1", step=1)
    r2 = await agent.run("q2", step=2)
    assert (r0.payloads[0], r1.payloads[0], r2.payloads[0]) == ("a0", "a1", "a2")
    assert agent.step == 2
    assert len(agent._step_watermarks) == 3

    await agent.rollback_to_step(1)

    # In-memory: parked at the start of step 1, transcript cut to end of step 0.
    assert agent.step == 1
    assert len(agent.transcript.messages) == len_after_0
    assert len(agent._step_watermarks) == 1

    # Durable: head is a ROLLED_BACK marker for step 1, log truncated, pruned.
    head = await load_agent_checkpoint(store, _KEY)
    assert head is not None
    assert head.current.step == 1
    assert head.location is AgentCheckpointLocation.ROLLED_BACK
    assert head.output is None  # rollback clears the current step's cached answer
    assert head.current.message_count == len_after_0
    assert len(head.messages) == len_after_0
    assert len(head.step_watermarks) == 1  # just the step-0 rollback point

    # Re-delivering step 1 now re-executes (fresh response), not the cached "a1".
    r1b = await agent.run("q1", step=1)
    assert r1b.payloads[0] == "a1_new"
    assert agent.step == 1


@pytest.mark.asyncio
async def test_rollback_from_cold_instance() -> None:
    store = InMemoryCheckpointStore()
    agent1, _ = _make_agent(
        [_text_response("a0"), _text_response("a1")], session_key="s1", store=store
    )
    await agent1.run("q0", step=0)
    len_after_0 = len(agent1.transcript.messages)
    await agent1.run("q1", step=1)

    # A fresh instance (new process / cache eviction) rolls back from the
    # persisted boundaries — grasp-core's between-steps fork-from-step path.
    agent2, _ = _make_agent([], session_key="s1", store=store)
    assert await agent2.load_checkpoint() is not None
    assert len(agent2._step_watermarks) == 2
    # The embedded AgentContextState survived the JSON round-trip through the store.
    assert isinstance(agent2._step_watermarks[1].agent_ctx_state, AgentContextState)

    await agent2.rollback_to_step(1)

    head = await load_agent_checkpoint(store, _KEY)
    assert head is not None
    assert head.current.step == 1
    assert head.location is AgentCheckpointLocation.ROLLED_BACK
    assert head.current.message_count == len_after_0
    assert len(head.step_watermarks) == 1  # just the step-0 rollback point


@pytest.mark.asyncio
async def test_rollback_with_sparse_steps() -> None:
    """Steps need not be dense: boundaries are keyed by their own step value."""
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response(a) for a in ("a0", "a5", "a10", "a5_new")],
        session_key="s1",
        store=store,
    )
    await agent.run("q0", step=0)
    len_after_0 = len(agent.transcript.messages)
    await agent.run("q5", step=5)  # gap: 1..4 skipped
    await agent.run("q10", step=10)
    assert agent.step == 10
    assert sorted(wm.step for wm in agent._step_watermarks) == [0, 5, 10]

    await agent.rollback_to_step(5)

    # self.step is the step rolled back to (5), and only step 0 remains a
    # rollback point — the sparse gap (1..4) is irrelevant.
    assert agent.step == 5
    assert len(agent.transcript.messages) == len_after_0
    assert sorted(wm.step for wm in agent._step_watermarks) == [0]

    head = await load_agent_checkpoint(store, _KEY)
    assert head is not None
    assert head.current.step == 5
    assert head.location is AgentCheckpointLocation.ROLLED_BACK
    assert head.current.message_count == len_after_0

    # Re-delivering step 5 re-executes (fresh response), not the cached "a5".
    r5b = await agent.run("q5", step=5)
    assert r5b.payloads[0] == "a5_new"
    assert agent.step == 5


@pytest.mark.asyncio
async def test_rollback_to_first_step_empties_log_keeps_ephemeral_prompt() -> None:
    """
    Rolling back to the first step empties the conversation log; the system
    prompt is ephemeral (the view header), so it is never lost.
    """
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response(a) for a in ("a0", "a1", "a0_new")],
        session_key="s1",
        store=store,
        sys_prompt="You are a helpful assistant.",
    )
    await agent.run("q0", step=0)
    await agent.run("q1", step=1)
    # The log is pure conversation — the system prompt is not stored in it.
    assert not any(
        isinstance(m, InputMessageItem) and m.role == "system"
        for m in agent.transcript.messages
    )

    await agent.rollback_to_step(0)

    # Back to an empty conversation log — ready to (re)deliver the first step.
    assert agent.transcript.messages == []
    assert agent.step == 0  # parked at the start of step 0

    # The durable head is empty too (the system prompt is never persisted).
    head = await load_agent_checkpoint(store, _KEY)
    assert head is not None
    assert head.messages == []

    # Re-deliver an edited first message; it re-executes fresh, and the system
    # prompt is composed back into the ephemeral header.
    r0b = await agent.run("q0_edited", step=0)
    assert r0b.payloads[0] == "a0_new"
    assert agent.step == 0
    assert isinstance(agent._cw.initial_context[0], InputMessageItem)
    assert agent._cw.initial_context[0].role == "system"


@pytest.mark.asyncio
async def test_rollback_unknown_step_raises() -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent([_text_response("a0")], session_key="s1", store=store)
    await agent.run("q0", step=0)
    with pytest.raises(KeyError):
        await agent.rollback_to_step(5)


@pytest.mark.asyncio
async def test_rollback_survives_compaction() -> None:
    # E0: a view projector compacts what the LLM sees without touching the log,
    # so rollback boundaries (log offsets) stay valid across compaction — the
    # case the retired log_version guard used to reject.
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response(a) for a in ("a0", "a1", "a2", "a1_new")],
        session_key="s1",
        store=store,
    )

    view_lengths: list[int] = []

    @agent.add_view_projector
    async def _compact(messages, *, exec_id, input_tokens):
        view_lengths.append(len(messages))
        return messages[-1:]  # show the LLM only the latest message

    await agent.run("q0", step=0)
    len_after_0 = len(agent.transcript.messages)
    await agent.run("q1", step=1)
    await agent.run("q2", step=2)
    assert agent.step == 2
    # The projector saw the full, growing log each turn — the log was not compacted.
    assert max(view_lengths) > 1
    assert len(agent.transcript.messages) > len_after_0
    assert agent._log_version == 0  # compaction never rewrote the log

    await agent.rollback_to_step(1)  # would have raised under the old log_version guard

    assert agent.step == 1
    assert len(agent.transcript.messages) == len_after_0
    assert agent._log_version == 0

    # The discarded step re-executes fresh, proving the rollback took effect.
    r1b = await agent.run("q1", step=1)
    assert r1b.payloads[0] == "a1_new"


@pytest.mark.asyncio
async def test_unstepped_run_records_no_boundary() -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent([_text_response("a")], session_key="s1", store=store)
    # A typed-args delivery with no step= stays unstepped (a chat delivery
    # would auto-mint one — see tests/agent/test_human_turn_anchors.py).
    await agent.run(in_args="hi")
    assert agent._step_watermarks == []


@pytest.mark.asyncio
async def test_rollback_returns_consumed_inbox_messages_to_pending() -> None:
    # A resident's rollback rewinds its mailbox with its transcript: messages
    # absorbed after the boundary go back to pending (their turns left the
    # history), in consumption order, ready to be re-processed.
    store = InMemoryCheckpointStore()
    transport = CheckpointMailboxTransport(store, session_key="s1")
    agent, _ = _make_agent(
        [
            _text_response("kickoff done"),
            _text_response("reply one"),
            _text_response("reply two"),
        ],
        session_key="s1",
        store=store,
    )
    agent.agent_ctx.inbox = AgentInbox(transport=transport, recipient="test_agent")

    async def drain() -> None:
        async for _ in agent.run_stream("kick off", step=1):
            pass

    run = asyncio.create_task(drain())
    m1 = TeamMessage.from_text(sender="user", to="test_agent", text="task one")
    m2 = TeamMessage.from_text(sender="peer", to="test_agent", text="task two")
    await transport.post(m1)
    await transport.post(m2)
    try:
        # Both messages absorbed, answered, and released at their turn
        # checkpoints; their processed records carry the consumption seqs.
        for _ in range(300):
            if await transport.was_processed(
                "test_agent", m2.message_id
            ) and not await transport.has_pending("test_agent"):
                break
            await asyncio.sleep(0.01)
        assert await transport.was_processed("test_agent", m1.message_id)
        assert await transport.was_processed("test_agent", m2.message_id)
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run

    blob = str(agent.transcript.messages)
    assert "task one" in blob
    assert "task two" in blob

    await agent.rollback_to_step(1)

    # The step-1 boundary predates both absorptions (high-water 0): both
    # messages return to pending, no longer deduped, transcript rewound.
    assert not await transport.was_processed("test_agent", m1.message_id)
    assert not await transport.was_processed("test_agent", m2.message_id)
    assert await transport.has_pending("test_agent")
    inbox = agent.agent_ctx.inbox
    assert inbox is not None
    first, second = await inbox.poll(), await inbox.poll()
    assert first is not None
    assert second is not None
    assert (first.text, second.text) == ("task one", "task two")
    assert "task one" not in str(agent.transcript.messages)


@pytest.mark.asyncio
async def test_interrupted_rollback_completes_on_resume() -> None:
    # A rollback re-marks the head ROLLING_BACK before its durable side
    # effects; a crash before the ROLLED_BACK head commits is therefore
    # visible to resume, which completes the rollback instead of resuming the
    # pre-rollback transcript over already-rewound side channels. Committing
    # the rolled-back head clears the mark by overwriting it.
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response(t) for t in ("a1", "a2")], session_key="s1", store=store
    )
    await agent.run("q1", step=1)
    await agent.run("q2", step=2)

    from grasp_agents.agent.llm_agent import LLMAgent

    with (
        patch.object(LLMAgent, "_persist_rollback", side_effect=RuntimeError("crash")),
        pytest.raises(RuntimeError, match="crash"),
    ):
        await agent.rollback_to_step(2)
    head = await load_agent_checkpoint(store, _KEY)
    assert head.location == AgentCheckpointLocation.ROLLING_BACK
    assert head.current.step == 2  # the target rides on the marked head

    # The FIRST post-crash call is a direct stepped run — resume must both
    # complete the rollback and serve the run a fresh (post-rollback) head,
    # or step 2's pre-rollback cached output would replay here.
    agent2, _ = _make_agent([_text_response("a2_new")], session_key="s1", store=store)
    out = await agent2.run("q2-new", step=2)
    assert out.payloads[0] == "a2_new"

    head = await load_agent_checkpoint(store, _KEY)
    assert head.location != AgentCheckpointLocation.ROLLING_BACK  # mark cleared
    blob = str(agent2.transcript.messages)
    assert "q2-new" in blob
    assert "q2'" not in blob  # the rolled-back turn's input is gone
