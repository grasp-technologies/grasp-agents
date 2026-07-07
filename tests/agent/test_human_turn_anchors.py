"""
Human-turn rollback anchors.

Chat deliveries auto-mint their ``step`` (every human turn is a rollback
boundary); a resident anchors a boundary at each human-message take — before
the message's consumption seq is minted, so a rollback re-pends the message
itself. Typed-args deliveries and peer messages are not anchors.
"""

import asyncio
import contextlib
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.mailbox import CheckpointMailboxTransport
from grasp_agents.session_context import SessionContext
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.message import TeamMessage
from tests._helpers import FailFirstLLM, MockLLM, _text_response
from tests.durability.test_sessions import _make_agent

# --- chat auto-minting ---


@pytest.mark.asyncio
async def test_chat_delivery_mints_step_and_records_boundary() -> None:
    agent, _ = _make_agent([_text_response("a"), _text_response("b")])

    await agent.run("hi")
    assert agent.step == 1
    assert agent.rollback_steps == [1]

    await agent.run("again")
    assert agent.step == 2
    assert agent.rollback_steps == [1, 2]


@pytest.mark.asyncio
async def test_mint_continues_past_an_explicit_step() -> None:
    agent, _ = _make_agent([_text_response("a"), _text_response("b")])

    await agent.run("q", step=5)
    assert agent.step == 5

    await agent.run("chat")
    assert agent.step == 6
    assert agent.rollback_steps == [5, 6]


@pytest.mark.asyncio
async def test_typed_args_delivery_stays_unstepped() -> None:
    agent, _ = _make_agent([_text_response("a")])

    await agent.run(in_args="typed input")
    assert agent.step is None
    assert agent.rollback_steps == []


@pytest.mark.asyncio
async def test_pure_resume_does_not_mint() -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response("a"), _text_response("b")], session_key="s1", store=store
    )

    await agent.run("hi")
    assert agent.step == 1

    await agent.run()  # no inputs → resumes the session, no new anchor
    assert agent.step is None
    assert agent.rollback_steps == [1]


@pytest.mark.asyncio
async def test_chat_after_rollback_re_mints_the_parked_step() -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response(t) for t in ("alpha", "bravo", "charlie")],
        session_key="s1",
        store=store,
    )
    await agent.run("q1")
    await agent.run("q2")

    await agent.rollback_to_step(2)
    assert agent.step == 2  # parked at the discarded turn's start

    # The next chat turn takes the discarded step's place: same step number,
    # fresh delivery (the head was ROLLED_BACK, so no cached replay).
    result = await agent.run("q2-replacement")
    assert result.payloads[0] == "charlie"
    assert agent.step == 2
    assert agent.rollback_steps == [1, 2]
    blob = str(agent.transcript.messages)
    assert "q2-replacement" in blob
    assert "bravo" not in blob


# --- resident human-take anchors ---


def _human(text: str, to: str = "test_agent") -> TeamMessage:
    return TeamMessage.from_text(sender="user", to=to, text=text)


def _peer(text: str, to: str = "test_agent") -> TeamMessage:
    return TeamMessage.from_text(sender="peer", to=to, text=text)


async def _run_resident_until(
    agent: LLMAgent[Any, Any, None],
    transport: CheckpointMailboxTransport,
    messages: list[TeamMessage],
) -> None:
    """Run a resident until every posted message is processed, then cancel."""

    async def drain() -> None:
        async for _ in agent.run_stream():
            pass

    run = asyncio.create_task(drain())
    for message in messages:
        await transport.post(message)
    try:
        for _ in range(500):
            done = all(
                [
                    await transport.was_processed("test_agent", m.message_id)
                    for m in messages
                ]
            )
            if done and not await transport.has_pending("test_agent"):
                break
            await asyncio.sleep(0.01)
        for message in messages:
            assert await transport.was_processed("test_agent", message.message_id)
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run


@pytest.mark.asyncio
async def test_resident_anchors_human_turns_not_peer_turns() -> None:
    store = InMemoryCheckpointStore()
    agent, ctx = _make_agent(
        [_text_response("re: human"), _text_response("re: peer")],
        session_key="s1",
        store=store,
    )
    transport = CheckpointMailboxTransport(store, session_key="s1")
    ctx.transport = transport
    agent.attach_inbox()

    human, peer = _human("human task"), _peer("peer task")
    await _run_resident_until(agent, transport, [human, peer])

    # One anchor: the human turn. The peer message is not a rollback point.
    assert agent.rollback_steps == [1]
    boundary = agent._step_watermarks[0]
    # Archived before the human message's seq was minted: its high-water
    # excludes the message, so a rollback voids it too.
    assert boundary.agent_ctx_state.mail_consumption_seq == 0

    await agent.rollback_to_step(1)
    # Both messages are voided: still deduped, never re-delivered.
    assert await transport.was_processed("test_agent", human.message_id)
    assert await transport.was_processed("test_agent", peer.message_id)
    assert not await transport.has_pending("test_agent")
    assert "human task" not in str(agent.transcript.messages)
    # The peer is notified; the human is not (they are rewriting history).
    nack = await transport.consume("peer")
    assert isinstance(nack, TeamMessage)
    assert "<message_dropped>" in nack.text
    assert not await transport.has_pending("user")


@pytest.mark.asyncio
async def test_resident_anchor_memo_reuses_step_for_redelivery() -> None:
    agent, _ = _make_agent([])
    human = _human("hi")

    agent._anchor_human_turn(human)
    assert (agent.step, agent.rollback_steps) == (1, [1])

    # A re-delivery of the same message (settled turn / rollback re-pend)
    # keeps its step — one anchor per human message.
    agent._anchor_human_turn(human)
    assert (agent.step, agent.rollback_steps) == (1, [1])

    agent._anchor_human_turn(_human("another"))
    assert (agent.step, agent.rollback_steps) == (2, [1, 2])

    agent._anchor_human_turn(_peer("not an anchor"))
    assert agent.rollback_steps == [1, 2]


@pytest.mark.asyncio
async def test_mint_after_rollback_floors_at_parked_step_with_sparse_steps() -> None:
    # Sparse explicit steps: after a rollback drops the parked step's own
    # boundary, the map alone would mint below it — the floor keeps the next
    # chat delivery on the parked step.
    agent, _ = _make_agent([_text_response(t) for t in ("a", "b", "c")])

    await agent.run("q1", step=1)
    await agent.run("q5", step=5)
    await agent.rollback_to_step(5)
    assert agent.step == 5

    await agent.run("q5-replacement")
    assert agent.step == 5
    assert agent.rollback_steps == [1, 5]


@pytest.mark.asyncio
async def test_resident_rerun_after_rollback_re_anchors_a_new_human_message() -> None:
    # Same process, same agent: a rollback voids the absorbed human message
    # and destroys its anchor boundary; the human's NEXT (replacement)
    # message re-mints the parked step and re-archives the boundary, so a
    # second rollback still has a target.
    store = InMemoryCheckpointStore()
    agent, transport = _resident(
        MockLLM(responses_queue=[_text_response("first"), _text_response("second")]),
        store,
    )
    human = _human("human task")
    await transport.post(human)
    await _run_until_processed(agent, transport, human)
    assert agent.rollback_steps == [1]

    await agent.rollback_to_step(1)
    assert agent.rollback_steps == []
    # The voided message is never re-delivered.
    assert not await transport.has_pending("test_agent")

    replacement = _human("replacement task")
    await transport.post(replacement)
    agent.attach_inbox()
    await _run_until_processed(agent, transport, replacement)

    assert agent.step == 1
    assert agent.rollback_steps == [1]
    blob = str(agent.transcript.messages)
    assert "replacement task" in blob
    assert "human task" not in blob

    # The restored boundary is fully usable: roll back again.
    await agent.rollback_to_step(1)
    assert "replacement task" not in str(agent.transcript.messages)


@pytest.mark.asyncio
async def test_rollback_to_step_zero_re_mints_step_zero() -> None:
    # Step 0 is a legitimate explicit step; the parked-step re-mint must not
    # treat it as "no step" (falsy) and drift to 1.
    agent, _ = _make_agent([_text_response(t) for t in ("a", "b", "c")])

    await agent.run("q0", step=0)
    await agent.run("q1", step=1)
    await agent.rollback_to_step(0)
    assert agent.step == 0

    await agent.run("q0-replacement")
    assert agent.step == 0
    assert agent.rollback_steps == [0]


@pytest.mark.asyncio
async def test_read_rollback_anchors_is_side_effect_free() -> None:
    # The read-only picker surface: a cold instance enumerates the persisted
    # anchors without rehydrating the session (no transcript load).
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response("a"), _text_response("b")], session_key="s1", store=store
    )
    await agent.run("q1")
    await agent.run("q2")

    cold, _ = _make_agent([], session_key="s1", store=store)
    anchors = await cold.read_rollback_anchors()
    assert [wm.step for wm in anchors] == [1, 2]
    assert cold.transcript.is_empty  # nothing was rehydrated

    # A live instance answers from memory.
    live_anchors = await agent.read_rollback_anchors()
    assert [wm.step for wm in live_anchors] == [1, 2]


@pytest.mark.asyncio
async def test_cold_rolled_back_resume_restores_the_parked_step() -> None:
    # A ROLLED_BACK head parks the agent at the discarded step's start; a cold
    # resume must restore that parked step so the next chat delivery re-mints
    # it (the mint floor) even when earlier steps are sparse.
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response("a"), _text_response("b")], session_key="s1", store=store
    )
    await agent.run("q1", step=1)
    await agent.run("q5", step=5)
    await agent.rollback_to_step(5)

    resumed, _ = _make_agent([_text_response("c")], session_key="s1", store=store)
    assert await resumed.load_checkpoint() is not None
    assert resumed.step == 5

    await resumed.run("q5-replacement")
    assert resumed.step == 5
    assert resumed.rollback_steps == [1, 5]


@pytest.mark.asyncio
async def test_rollback_mid_run_is_refused() -> None:
    # ``rollback_to_step`` is only valid between runs: rewinding under a live
    # loop would race the turn cycle, so a mid-run call raises instead of
    # silently corrupting the transcript.
    store = InMemoryCheckpointStore()
    agent, transport = _resident(
        MockLLM(responses_queue=[_text_response("handled")]), store
    )
    human = _human("human task")
    await transport.post(human)

    async def drain() -> None:
        async for _ in agent.run_stream():
            pass

    run = asyncio.create_task(drain())
    try:
        for _ in range(500):
            if await transport.was_processed("test_agent", human.message_id):
                break
            await asyncio.sleep(0.01)
        assert agent.rollback_steps == [1]

        with pytest.raises(RuntimeError, match="mid-run"):
            await agent.rollback_to_step(1)
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run

    # Between runs (the stream is closed) the same call succeeds.
    await agent.rollback_to_step(1)
    assert agent.rollback_steps == []


@pytest.mark.asyncio
async def test_resident_retry_keeps_the_take_anchor() -> None:
    # A retry continues an already-anchored delivery: re-archiving on the
    # retry re-entry would move the boundary past the settled human message,
    # so a later rollback would keep the message's turn instead of voiding it.
    store = InMemoryCheckpointStore()
    ctx = SessionContext[None](checkpoint_store=store, session_key="s1")
    transport = CheckpointMailboxTransport(store, session_key="s1")
    ctx.transport = transport
    agent = LLMAgent[str, str, None](
        name="test_agent",
        ctx=ctx,
        llm=FailFirstLLM(responses_queue=[_text_response("handled")]),
        stream_llm=True,
        max_retries=1,
    )
    agent.attach_inbox()

    human = _human("human task")
    await transport.post(human)
    await _run_until_processed(agent, transport, human)

    # The take-time anchor survived the retry: its high-waters exclude the
    # message, so the rollback voids it.
    boundary = agent._step_watermarks[0]
    assert (boundary.step, boundary.message_count) == (1, 0)
    assert boundary.agent_ctx_state.mail_consumption_seq == 0
    assert str(agent.transcript.messages).count("human task") == 1

    await agent.rollback_to_step(1)
    assert "human task" not in str(agent.transcript.messages)
    # Voided: deduped, never re-delivered.
    assert await transport.was_processed("test_agent", human.message_id)
    assert not await transport.has_pending("test_agent")


@pytest.mark.asyncio
async def test_fresh_instance_rollback_voids_mailbox() -> None:
    # A fresh agent instance (same session, no inbox attached — a
    # cross-process rollback) still runs the mailbox half of a rollback,
    # resolved from the session's transport.
    store = InMemoryCheckpointStore()
    agent, ctx = _make_agent(
        [_text_response("re: human"), _text_response("re: peer")],
        session_key="s1",
        store=store,
    )
    transport = CheckpointMailboxTransport(store, session_key="s1")
    ctx.transport = transport
    agent.attach_inbox()

    human, peer = _human("human task"), _peer("peer task")
    await _run_resident_until(agent, transport, [human, peer])

    ctx2 = SessionContext[None](checkpoint_store=store, session_key="s1")
    ctx2.transport = transport
    agent2 = LLMAgent[str, str, None](
        name="test_agent", ctx=ctx2, llm=MockLLM(responses_queue=[])
    )
    await agent2.load_checkpoint()

    await agent2.rollback_to_step(1)
    # Both voided: still deduped, never re-delivered; the peer is notified.
    assert await transport.was_processed("test_agent", human.message_id)
    assert await transport.was_processed("test_agent", peer.message_id)
    assert not await transport.has_pending("test_agent")
    nack = await transport.consume("peer")
    assert isinstance(nack, TeamMessage)
    assert "<message_dropped>" in nack.text


# --- settled human turns are not re-delivered ---


def _resident(
    llm: MockLLM, store: InMemoryCheckpointStore
) -> tuple[LLMAgent[str, str, None], CheckpointMailboxTransport]:
    ctx = SessionContext[None](checkpoint_store=store, session_key="s1")
    transport = CheckpointMailboxTransport(store, session_key="s1")
    ctx.transport = transport
    agent = LLMAgent[str, str, None](
        name="test_agent", ctx=ctx, llm=llm, stream_llm=True
    )
    agent.attach_inbox()
    return agent, transport


async def _run_until_processed(
    agent: LLMAgent[str, str, None],
    transport: CheckpointMailboxTransport,
    message: TeamMessage,
) -> None:
    async def drain() -> None:
        async for _ in agent.run_stream():
            pass

    run = asyncio.create_task(drain())
    try:
        for _ in range(500):
            if await transport.was_processed("test_agent", message.message_id):
                break
            await asyncio.sleep(0.01)
        assert await transport.was_processed("test_agent", message.message_id)
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run


@pytest.mark.asyncio
async def test_settled_human_turn_not_duplicated_on_reentry() -> None:
    # A failed first turn settles keeping the absorbed message; its lease
    # survives the settle, so the next run answers the kept copy instead of
    # re-taking (duplicating) it — and the anchor boundary stays uncorrupted.
    store = InMemoryCheckpointStore()
    agent, transport = _resident(
        FailFirstLLM(responses_queue=[_text_response("handled")]), store
    )
    human = _human("human task")
    await transport.post(human)

    with pytest.raises(ProcRunError):
        async for _ in agent.run_stream():
            pass

    await _run_until_processed(agent, transport, human)

    assert str(agent.transcript.messages).count("human task") == 1
    boundary = agent._step_watermarks[0]
    assert (boundary.step, boundary.message_count) == (1, 0)
    # Rollback fully removes the turn and voids the message.
    await agent.rollback_to_step(1)
    assert "human task" not in str(agent.transcript.messages)
    assert await transport.was_processed("test_agent", human.message_id)
    assert not await transport.has_pending("test_agent")


@pytest.mark.asyncio
async def test_lease_survives_detach_reattach() -> None:
    # A host tearing the run down and re-attaching (same agent) must not
    # re-deliver the settled, still-owed message: the inbox (and its lease)
    # outlives the attach cycle.
    store = InMemoryCheckpointStore()
    agent, transport = _resident(
        FailFirstLLM(responses_queue=[_text_response("handled")]), store
    )
    human = _human("human task")
    await transport.post(human)

    with pytest.raises(ProcRunError):
        async for _ in agent.run_stream():
            pass

    agent.attach_inbox()

    await _run_until_processed(agent, transport, human)
    assert str(agent.transcript.messages).count("human task") == 1


@pytest.mark.asyncio
async def test_rebuilt_agent_neither_wedges_nor_duplicates() -> None:
    # A fresh agent instance (same session) starts from the persisted log —
    # which never absorbed the message — and a fresh inbox: the message is
    # re-delivered once, not blocked by the dead instance's lease.
    store = InMemoryCheckpointStore()
    agent1, transport = _resident(FailFirstLLM(responses_queue=[]), store)
    human = _human("human task")
    await transport.post(human)

    with pytest.raises(ProcRunError):
        async for _ in agent1.run_stream():
            pass

    ctx2 = SessionContext[None](checkpoint_store=store, session_key="s1")
    ctx2.transport = transport
    agent2 = LLMAgent[str, str, None](
        name="test_agent",
        ctx=ctx2,
        llm=MockLLM(responses_queue=[_text_response("handled")]),
        stream_llm=True,
    )
    agent2.attach_inbox()

    await _run_until_processed(agent2, transport, human)
    assert str(agent2.transcript.messages).count("human task") == 1
