"""
End-to-end ``AgentTeam`` behavior with mocked LLMs: communicators run resident
(one loop, consuming their inbox between turns) and message each other to quiescence;
a processor / agent-with-recipients runs triggered and hands off; a hop budget stops
a team with mail pending; a daemon keeps serving past quiescence.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent_team.agent_card import MemberCard
from grasp_agents.agent_team.agent_team import AgentTeam
from grasp_agents.agent_team.events import MessageDeliveredEvent, TeamEndedEvent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import TeamCheckpoint
from grasp_agents.file_backend.local import LocalFileBackend
from grasp_agents.mailbox import CheckpointMailboxTransport, InMemoryMailboxTransport
from grasp_agents.processors.processor import Processor
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.types.content import InputRenderableModel
from grasp_agents.types.message import CONTROL_PRIORITY, TeamMessage
from tests._helpers import (
    FailFirstLLM,
    FakeSnapshotEnv,
    MockLLM,
    _agent,
    _send,
    _text_response,
    _tool_call_response,
    _until,
)


class ForwardProcessor(Processor[Any, Any, None]):
    """A non-agent processor member: forwards its input to a fixed recipient."""

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[Any] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> list[Any]:
        del chat_inputs, exec_id, step
        return list(in_args or [])


class RecordingTransport(InMemoryMailboxTransport):
    """An explicit transport that records that the team routed through it."""

    def __init__(self) -> None:
        super().__init__()
        self.posted = False

    async def post(self, envelope: TeamMessage) -> None:
        self.posted = True
        await super().post(envelope)


class FailingLLM(MockLLM):
    """An LLM whose generation always raises, to exercise member failure."""

    async def _generate_response_once(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")


def _ctx(tmp_path: Path) -> SessionContext[None]:
    return SessionContext[None](
        state=None, file_backend=LocalFileBackend(allowed_roots=[tmp_path])
    )


@pytest.mark.asyncio
async def test_two_member_ping_pong_runs_to_quiescence(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    alice = _agent(
        "alice",
        [
            _send("bob", "ping", "c1"),
            _text_response("alice: sent ping"),
            _text_response("alice: got the reply, done"),
        ],
    )
    bob = _agent(
        "bob",
        [
            _send("alice", "pong", "c2"),
            _text_response("bob: sent pong"),
        ],
    )
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx)

    result = await team.run("kick off")
    await team.aclose()

    assert result.stop_reason == "quiesced"
    # Three deliveries consumed: kickoff→alice, ping→bob, pong→alice.
    assert result.activations == 3
    delivered = {(m.sender, m.recipient, m.text) for m in result.messages}
    assert ("user", "alice", "kick off") in delivered
    assert ("alice", "bob", "ping") in delivered
    assert ("bob", "alice", "pong") in delivered
    # alice generated 3 turns (ping, sent-ping, done), bob 2 (pong, sent-pong).
    assert alice.llm.call_count == 3
    assert bob.llm.call_count == 2


@pytest.mark.asyncio
async def test_single_member_answers_and_quiesces(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("the answer")])
    team = AgentTeam([solo], ctx=ctx)

    result = await team.run("question?")
    await team.aclose()

    assert result.stop_reason == "quiesced"
    assert result.activations == 1
    assert solo.llm.call_count == 1
    assert [(m.sender, m.recipient) for m in result.messages] == [("user", "solo")]


@pytest.mark.asyncio
async def test_hop_budget_stops_with_pending_mail(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    alice = _agent("alice", [_send("bob", "ping", "c1"), _text_response("alice done")])
    # bob would reply, but the budget refuses alice's send before bob is activated.
    bob = _agent("bob", [_send("alice", "pong", "c2"), _text_response("bob done")])
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx, max_hops=1)

    result = await team.run("go")
    await team.aclose()

    assert result.stop_reason == "hop_budget_exhausted"
    assert result.activations == 1
    assert bob.llm.call_count == 0


@pytest.mark.asyncio
async def test_hop_budget_is_per_run_on_a_durable_session() -> None:
    # ``max_hops`` bounds each run's own deliveries. The lifetime activation
    # count restored from the team checkpoint must not eat later runs'
    # budgets — that would permanently wedge a durable team (every new run's
    # entry seed refused) once the session's history crosses the cap.
    store = InMemoryCheckpointStore()
    ctx = SessionContext[None](state=None, checkpoint_store=store)
    solo = _agent("solo", [_text_response("one"), _text_response("two")])
    team = AgentTeam([solo], ctx=ctx, max_hops=1)

    result1 = await team.run("first question")
    assert result1.stop_reason == "quiesced"
    assert result1.activations == 1

    result2 = await team.run("second question")
    await team.aclose()

    assert result2.stop_reason == "quiesced"
    assert result2.activations == 2  # lifetime count keeps accumulating
    assert solo.llm.call_count == 2  # the second seed was delivered


@pytest.mark.asyncio
async def test_token_budget_stops_with_pending_mail(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    alice = _agent("alice", [_send("bob", "ping", "c1"), _text_response("alice done")])
    bob = _agent("bob", [_send("alice", "pong", "c2"), _text_response("bob done")])
    # Each MockLLM turn reports 15 tokens; a 10-token budget is spent after alice's
    # first turn, so her send to bob is refused before bob is ever activated.
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx, max_tokens=10)

    result = await team.run("go")
    await team.aclose()

    assert result.stop_reason == "token_budget_exhausted"
    assert result.activations == 1
    assert bob.llm.call_count == 0


@pytest.mark.asyncio
async def test_token_budget_generous_does_not_trip(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("the answer")])
    team = AgentTeam([solo], ctx=ctx, max_tokens=10_000)

    result = await team.run("question?")
    await team.aclose()

    assert result.stop_reason == "quiesced"
    assert solo.llm.call_count == 1


@pytest.mark.asyncio
async def test_duplicate_member_names_rejected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    a1 = _agent("dup", [_text_response("x")])
    a2 = _agent("dup", [_text_response("y")])
    with pytest.raises(ValueError, match="Duplicate member names"):
        AgentTeam([a1, a2], ctx=ctx)


@pytest.mark.asyncio
async def test_multiple_leads_rejected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    a = _agent("a", [_text_response("x")])
    b = _agent("b", [_text_response("y")])
    cards = [MemberCard(name="a", lead=True), MemberCard(name="b", lead=True)]
    with pytest.raises(ValueError, match="more than one lead"):
        AgentTeam([a, b], cards=cards, ctx=ctx)


@pytest.mark.asyncio
async def test_triggered_lead_rejected(tmp_path: Path) -> None:
    # The lead's role (priority mail, rewind right + announcements) presumes a
    # persistent loop; a triggered member — a processor, an agent with static
    # recipients, or one carded resident=False — cannot be the lead.
    ctx = _ctx(tmp_path)
    router = ForwardProcessor(name="router", recipients=["writer"])
    writer = _agent("writer", [_text_response("x")])
    with pytest.raises(ValueError, match="must run resident"):
        AgentTeam(
            [router, writer],
            entry="router",
            cards=[MemberCard(name="router", lead=True)],
            ctx=ctx,
        )

    with pytest.raises(ValueError, match="must run resident"):
        AgentTeam(
            [_agent("a", []), _agent("b", [])],
            cards=[MemberCard(name="a", lead=True, resident=False)],
            ctx=_ctx(tmp_path),
        )


@pytest.mark.asyncio
async def test_lead_claims_environment_rewind_at_construction(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    a = _agent("a", [_text_response("x")])
    b = _agent("b", [_text_response("y")])
    AgentTeam([a, b], cards=[MemberCard(name="a", lead=True)], ctx=ctx)
    assert ctx.session_writer == "a"

    # A ctx that already declares a different rewinder conflicts immediately.
    other_ctx = _ctx(tmp_path)
    other_ctx.session_writer = "someone-else"
    with pytest.raises(RuntimeError, match="already owns session persistence"):
        AgentTeam([a, b], cards=[MemberCard(name="a", lead=True)], ctx=other_ctx)


@pytest.mark.asyncio
async def test_session_transport_is_used_by_members() -> None:
    # The team always routes through the session's mailbox (``ctx.transport``),
    # so the recording flag being set proves the sends reached it.
    transport = RecordingTransport()
    ctx = SessionContext[None](state=None)
    ctx.transport = transport
    alice = _agent("alice", [_send("bob", "ping", "c1"), _text_response("alice done")])
    bob = _agent("bob", [_text_response("bob got it")])
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx)

    result = await team.run("kick off")
    await team.aclose()

    assert result.stop_reason == "quiesced"
    assert result.activations == 2
    assert bob.llm.call_count == 1
    assert transport.posted
    delivered = {(m.sender, m.recipient, m.text) for m in result.messages}
    assert ("alice", "bob", "ping") in delivered


@pytest.mark.asyncio
async def test_member_failure_reports_member_error(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    solo = LLMAgent[Any, Any, None](name="solo", llm=FailingLLM())
    team = AgentTeam([solo], ctx=ctx)

    result = await team.run("do it")
    await team.aclose()

    assert result.stop_reason == "member_error"
    assert result.activations == 1


@pytest.mark.asyncio
async def test_team_without_file_backend_uses_in_memory() -> None:
    # No file_backend and no explicit transport: the team provisions one shared
    # in-memory transport, so a single-process team needs zero file wiring.
    ctx = SessionContext[None](state=None)
    alice = _agent("alice", [_send("bob", "hi", "c1"), _text_response("alice done")])
    bob = _agent("bob", [_text_response("bob got it")])
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx)

    result = await team.run("go")
    await team.aclose()

    assert result.stop_reason == "quiesced"
    assert result.activations == 2
    assert bob.llm.call_count == 1


@pytest.mark.asyncio
async def test_team_over_checkpoint_transport() -> None:
    # A team running on the durable CheckpointStore-backed mailbox (the same
    # substrate background tasks persist through) — resolved automatically
    # from the session's checkpoint store.
    ctx = SessionContext[None](
        state=None, checkpoint_store=InMemoryCheckpointStore(), session_key="team"
    )
    alice = _agent("alice", [_send("bob", "ping", "c1"), _text_response("alice done")])
    bob = _agent("bob", [_text_response("bob got it")])
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx)
    assert isinstance(ctx.transport, CheckpointMailboxTransport)

    result = await team.run("kick off")
    await team.aclose()

    assert result.stop_reason == "quiesced"
    assert result.activations == 2
    assert bob.llm.call_count == 1
    delivered = {(m.sender, m.recipient, m.text) for m in result.messages}
    assert ("alice", "bob", "ping") in delivered


@pytest.mark.asyncio
async def test_interrupted_run_resumes_without_reseeding() -> None:
    # A cancelled/crashed run never advances the run ordinal, so re-running
    # the SAME input is a resume, not a new turn: the already-processed seed
    # is deduped and the persisted hop count continues instead of resetting.
    # (A COMPLETED run advances the ordinal — a re-run then re-delivers; see
    # test_repeat_identical_input_delivers_on_a_new_run.)
    store = InMemoryCheckpointStore()

    def build() -> tuple[
        AgentTeam[None], LLMAgent[Any, Any, None], LLMAgent[Any, Any, None]
    ]:
        alice = _agent(
            "alice", [_send("bob", "ping", "c1"), _text_response("alice done")]
        )
        bob = _agent("bob", [_text_response("bob got it")])
        ctx = SessionContext[None](state=None, checkpoint_store=store)
        return AgentTeam([alice, bob], entry="alice", ctx=ctx), alice, bob

    team1, _alice1, bob1 = build()

    async def consume() -> None:
        # Daemon: never self-terminates, so cancellation is the only exit —
        # a deterministic stand-in for a crash (the ordinal never advances).
        async for _ in team1.run_stream("kick off", daemon=True):
            pass

    run = asyncio.create_task(consume())
    try:
        await _until(lambda: bob1.llm.call_count >= 1)
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
    await team1.aclose()

    # Reconstruct on the same store (a fresh process) and re-run the same
    # input: same ordinal → the processed seed is deduped, budget continues.
    team2, alice2, _bob2 = build()
    result2 = await team2.run("kick off")
    await team2.aclose()
    assert alice2.llm.call_count == 0  # no duplicate kickoff
    assert result2.activations == 2  # restored, not reset


@pytest.mark.asyncio
async def test_member_error_keeps_the_run_ordinal_for_retry() -> None:
    # A member error stops the run but must NOT advance the seed ordinal:
    # its documented recovery is re-running the SAME input as an idempotent
    # resume — the entry's seed is deduped, not double-processed, and the
    # failed member retries its still-owed message.
    store = InMemoryCheckpointStore()
    ctx = SessionContext[None](state=None, checkpoint_store=store)
    alice = _agent("alice", [_send("bob", "ping", "c1"), _text_response("sent")])
    bob = LLMAgent[Any, Any, None](
        name="bob",
        llm=FailFirstLLM(responses_queue=[_text_response("bob answered")]),
    )
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx)

    result1 = await team.run("kickoff-task")
    assert result1.stop_reason == "member_error"
    assert team._runs_ended == 0  # ordinal kept — the retry is a resume
    alice_calls_after_run1 = alice.llm.call_count

    result2 = await team.run("kickoff-task")
    await team.aclose()

    assert result2.stop_reason == "quiesced"
    assert team._runs_ended == 1
    # The seed was deduped: alice never re-processed the same input.
    assert alice.llm.call_count == alice_calls_after_run1
    assert str(alice.transcript.messages).count("kickoff-task") == 1
    # Bob's owed message was retried and answered.
    assert "bob answered" in str(bob.transcript.messages)


@pytest.mark.asyncio
async def test_slow_seed_dedup_probe_does_not_quiesce_the_run() -> None:
    # The seed's dedup probe is a store round-trip; the monitor polls for
    # quiescence the whole time. A slow store must not let the run read as
    # idle and tear down before the input is even deposited.
    class _SlowDedupTransport(InMemoryMailboxTransport):
        async def was_processed(self, recipient: str, envelope_id: str) -> bool:
            await asyncio.sleep(0.3)  # >> two 0.05s quiescence polls
            return await super().was_processed(recipient, envelope_id)

    ctx = SessionContext[None](state=None)
    ctx.transport = _SlowDedupTransport()
    solo = _agent("solo", [_text_response("the answer")])
    team = AgentTeam([solo], ctx=ctx)

    result = await team.run("question?")
    await team.aclose()

    assert result.stop_reason == "quiesced"
    assert result.activations == 1
    assert solo.llm.call_count == 1  # the input was delivered, not dropped


@pytest.mark.asyncio
async def test_run_ordinal_is_durable_before_the_stream_ends() -> None:
    # The ordinal bump is persisted at quiescence, BEFORE teardown: a crash
    # between the run's last useful work and its end then reads as "ended"
    # (the next identical input is a fresh turn) instead of silently
    # swallowing it.
    store = InMemoryCheckpointStore()
    ctx = SessionContext[None](state=None, checkpoint_store=store)
    solo = _agent("solo", [_text_response("one")])
    team = AgentTeam([solo], name="t1", ctx=ctx)

    ended_seen = False
    async for event in team.run_stream("first question"):
        if isinstance(event, TeamEndedEvent):
            ended_seen = True
            keys = [k for k in await store.list_keys("") if "/team" in k]
            assert keys, "team checkpoint must exist by stream end"
            raw = await store.load(keys[0])
            assert raw is not None
            checkpoint = TeamCheckpoint.model_validate_json(raw)
            assert checkpoint.runs_ended == 1
    await team.aclose()
    assert ended_seen


async def _seed_records(store: InMemoryCheckpointStore) -> list[str]:
    return [
        k for k in await store.list_keys("") if "-seed-" in k and "/processed/" in k
    ]


@pytest.mark.asyncio
async def test_gc_reclaims_stale_ordinal_seeds() -> None:
    # Entry seeds are pinned against GC only at the CURRENT run ordinal —
    # older ordinals can never be re-posted, so their records must not
    # accumulate forever. A triggered entry acks without a consumption seq,
    # so no rollback horizon pins them either.
    store = InMemoryCheckpointStore()
    ctx = SessionContext[None](state=None, checkpoint_store=store)
    solo = _agent(
        "solo",
        [_text_response("one"), _text_response("two"), _text_response("three")],
    )
    team = AgentTeam(
        [solo],
        ctx=ctx,
        cards=[MemberCard(name="solo", resident=False)],
        mailbox_processed_retention=timedelta(seconds=0),
    )

    await team.run("first question")
    await team.run("second question")
    assert len(await _seed_records(store)) == 2

    await team._gc_mailbox()
    assert len(await _seed_records(store)) == 0

    # A fresh input still runs (its new-ordinal seed is unaffected).
    result = await team.run("third question")
    await team.aclose()
    assert result.stop_reason == "quiesced"


@pytest.mark.asyncio
async def test_gc_reclaims_ended_run_seed_and_rollback_still_voids() -> None:
    # Once the run ends (the seed's ordinal goes stale — it can never be
    # re-posted), zero retention reclaims its record: nothing pins consumed
    # mail anymore, since a rollback voids rather than re-delivers. The
    # rollback still rewinds the transcript; the reclaimed seed simply has
    # nothing left to void.
    store = InMemoryCheckpointStore()
    ctx = SessionContext[None](state=None, checkpoint_store=store)
    solo = _agent("solo", [_text_response("one")])
    team = AgentTeam([solo], ctx=ctx, mailbox_processed_retention=timedelta(seconds=0))

    await team.run("kick off")
    assert solo.rollback_steps == [1]
    assert len(await _seed_records(store)) == 1  # consumed by the resident

    await team._gc_mailbox()
    assert len(await _seed_records(store)) == 0  # stale ordinal, reclaimed

    await solo.rollback_to_step(1)
    transport = ctx.transport
    assert transport is not None
    assert not await transport.has_pending("solo")
    assert "kick off" not in str(solo.transcript.messages)
    await team.aclose()
    # The team saves its hop checkpoint BEFORE depositing on the transport. A
    # crash in that window for the entry seed must not strand the kickoff: the
    # deterministic seed id makes the resume re-post idempotent, so the seed is
    # delivered (exactly once) — not suppressed by a "seeded" flag that the
    # checkpoint's mere existence would otherwise imply.
    store = InMemoryCheckpointStore()

    class _DropFirstPost(CheckpointMailboxTransport):
        def __init__(self) -> None:
            super().__init__(store, session_key="s")
            self.drop_next = True

        async def post(self, envelope: TeamMessage) -> None:
            if self.drop_next:
                self.drop_next = False
                raise RuntimeError("crash before deposit")
            await super().post(envelope)

    # RUN 1: the seed's checkpoint is saved, then its deposit crashes.
    entry1 = _agent("entry", [_text_response("hi")])
    ctx1 = SessionContext[None](state=None, checkpoint_store=store)
    ctx1.transport = _DropFirstPost()
    team1 = AgentTeam([entry1], ctx=ctx1)
    result1 = await team1.run("kick off")
    await team1.aclose()
    assert result1.stop_reason == "error"
    assert entry1.llm.call_count == 0  # the seed never reached the entry

    # RUN 2 (resume) over the same store with a working transport: the seed was
    # never processed, so it is re-delivered and the entry finally runs.
    entry2 = _agent("entry", [_text_response("hi")])
    ctx2 = SessionContext[None](state=None, checkpoint_store=store)
    ctx2.transport = CheckpointMailboxTransport(store, session_key="s")
    team2 = AgentTeam([entry2], ctx=ctx2)
    result2 = await team2.run("kick off")
    await team2.aclose()
    assert entry2.llm.call_count == 1  # seed re-delivered, not stranded
    delivered = {(m.sender, m.recipient, m.text) for m in result2.messages}
    assert ("user", "entry", "kick off") in delivered


@pytest.mark.asyncio
async def test_processor_member_routes_to_agent(tmp_path: Path) -> None:
    # A non-agent Processor member (triggered) consumes a message and hands its
    # output off to a resident agent member by name.
    ctx = _ctx(tmp_path)
    router = ForwardProcessor(name="router", recipients=["writer"])
    writer = _agent("writer", [_text_response("writer done")])
    team = AgentTeam([router, writer], entry="router", ctx=ctx)

    result = await team.run("hello")
    await team.aclose()

    assert result.stop_reason == "quiesced"
    assert result.activations == 2  # hello→router, then router→writer
    assert writer.llm.call_count == 1
    # The processor must forward the real content, not a dropped [None] payload.
    assert any(
        m.sender == "router" and m.recipient == "writer" and m.text == "hello"
        for m in result.messages
    )


@pytest.mark.asyncio
async def test_agent_worker_with_recipients_hands_off(tmp_path: Path) -> None:
    # An LLMAgent given static recipients is a worker (triggered), not a
    # communicator: it gets no SendMessage tool and hands its answer off by name.
    ctx = _ctx(tmp_path)
    worker = LLMAgent[Any, Any, None](
        name="worker", llm=MockLLM(responses_queue=[_text_response("did the work")])
    )
    worker.recipients = ["sink"]
    sink = _agent("sink", [_text_response("sink done")])
    team = AgentTeam([worker, sink], entry="worker", ctx=ctx)

    result = await team.run("go")
    await team.aclose()

    assert result.stop_reason == "quiesced"
    assert result.activations == 2
    assert sink.llm.call_count == 1
    assert "SendMessage" not in worker.tools  # worker is hand-off, not messaging
    assert any(m.sender == "worker" and m.recipient == "sink" for m in result.messages)


@pytest.mark.asyncio
async def test_nonblocking_bg_task_does_not_block_quiescence(tmp_path: Path) -> None:
    # A resident that backgrounds a non-answer-blocking task that never finishes must
    # not wedge team quiescence — the loop never blocks its final answer on such a
    # task, so neither should the team (else run() would hang forever).
    release = asyncio.Event()

    @function_tool(auto_background_at=0, blocks_final_answer=False)
    async def slow_job(text: str) -> str:
        """A backgrounded job that does not finish on its own."""
        await release.wait()
        return f"done: {text}"

    ctx = _ctx(tmp_path)
    worker = LLMAgent[Any, Any, None](
        name="worker",
        llm=MockLLM(
            responses_queue=[
                _tool_call_response("slow_job", '{"text": "x"}', "c1"),
                _text_response("launched; idling"),
            ]
        ),
        tools=[slow_job],
    )
    team = AgentTeam([worker], ctx=ctx)
    try:
        result = await asyncio.wait_for(team.run("go"), timeout=5.0)
    finally:
        release.set()  # let the bg task finish so aclose can drain/cancel it cleanly
        await team.aclose()

    assert result.stop_reason == "quiesced"
    assert worker.llm.call_count == 2  # launched the task, then idled


def test_member_unknown_recipient_rejected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    router = ForwardProcessor(name="router", recipients=["ghost"])
    writer = _agent("writer", [_text_response("x")])
    with pytest.raises(ValueError, match="unknown recipient"):
        AgentTeam([router, writer], entry="router", ctx=ctx)


async def _drain(team: AgentTeam[None], seed: str, sink: list[TeamMessage]) -> None:
    async for ev in team.run_stream(seed, daemon=True, poll_interval=0.01):
        if isinstance(ev, MessageDeliveredEvent):
            sink.append(ev.data)


@pytest.mark.asyncio
async def test_daemon_keeps_running_past_quiescence(tmp_path: Path) -> None:
    # A daemon run does not stop at quiescence: after the seeded message is handled
    # (a bounded run would end here), a later-posted message is still picked up — and
    # the two arrive as separate turns (one group per turn), never merged.
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("a1"), _text_response("a2")])
    team = AgentTeam([solo], ctx=ctx)
    delivered: list[TeamMessage] = []

    consumer = asyncio.create_task(_drain(team, "first", delivered))
    try:
        await _until(lambda: solo.llm.call_count == 1)  # seed handled; team idle
        await team.post(TeamMessage.from_text(sender="user", to="solo", text="second"))
        await _until(lambda: solo.llm.call_count == 2)  # daemon picked up the second
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
    await team.aclose()

    assert [m.text for m in delivered] == ["first", "second"]


class _Ticket(InputRenderableModel):
    title: str
    points: int

    def to_input_parts(self) -> str:
        return f"TICKET[{self.title} / {self.points}pts]"


@pytest.mark.asyncio
async def test_typed_structured_handoff_renders_via_input_renderable(
    tmp_path: Path,
) -> None:
    # A typed peer hand-off to a triggered worker: a structured body addressed to a
    # member whose ``InT`` is a model is validated to that model on receipt (so it
    # survives the dict round-trip a durable transport produces) and rendered through
    # the worker's input pipeline by the model's own ``InputRenderable.to_input_parts``
    # — not flattened to JSON text.
    ctx = _ctx(tmp_path)
    planner = LLMAgent[_Ticket, Any, None](
        name="planner", llm=MockLLM(responses_queue=[_text_response("planned")])
    )
    team = AgentTeam(
        [planner],
        # ``resident=False`` runs it triggered: each peer message activates a fresh
        # run whose ``in_packet`` flows through the worker's input pipeline.
        cards=[MemberCard(name="planner", input_type=_Ticket, resident=False)],
        ctx=ctx,
    )

    async def _run_daemon() -> None:
        async for _ in team.run_stream(daemon=True, poll_interval=0.01):
            pass

    consumer = asyncio.create_task(_run_daemon())
    try:
        await _until(lambda: team._driver is not None)  # pyright: ignore[reportPrivateUsage]
        # A *peer* sends a *dict* payload (what a durable transport delivers back),
        # not a _Ticket instance.
        await team.post(
            TeamMessage(
                sender="scout",
                routing=[["planner"]],
                payloads=[{"title": "Fix the bug", "points": 3}],
            )
        )
        await _until(lambda: planner.llm.call_count == 1)
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
    await team.aclose()

    blob = str(planner.transcript.messages)
    # The dict was validated to _Ticket and rendered by its to_input_parts —
    # proving typed (not JSON) delivery.
    assert "TICKET[Fix the bug / 3pts]" in blob
    # The triggered path has no sender fence, so the attribution attachment names
    # the teammate the typed body came from.
    assert "scout" in blob


@pytest.mark.asyncio
async def test_submit_message_delivers_human_input_to_member(tmp_path: Path) -> None:
    # Single-process human input: submit_message routes to a resident member's inbox
    # as control-plane mail and is handled like any peer message — the same-process
    # counterpart to MemberHost.submit_message.
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("a1"), _text_response("a2")])
    team = AgentTeam([solo], ctx=ctx)
    delivered: list[TeamMessage] = []

    consumer = asyncio.create_task(_drain(team, "first", delivered))
    try:
        await _until(lambda: solo.llm.call_count == 1)  # seed handled
        await team.submit_message("solo", "hello from a human")
        await _until(lambda: solo.llm.call_count == 2)  # human input handled
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
    await team.aclose()

    human = next((m for m in delivered if m.text == "hello from a human"), None)
    assert human is not None
    # Stamped as control-plane: from the human, priority-preempting peer mail.
    assert human.sender == "user"
    assert human.priority == CONTROL_PRIORITY


@pytest.mark.asyncio
async def test_environment_rewind_notifies_other_residents(tmp_path: Path) -> None:
    # When the lead rewinds the shared environment mid-run, every OTHER resident
    # gets a control-plane <environment_rewind> notice (so it re-verifies state
    # instead of panicking); the rewinder itself is not notified.
    env = FakeSnapshotEnv(tmp_path)
    ctx = SessionContext[None](state=None, environment=env)
    planner = _agent("planner", [_text_response("planner: kicked off")])
    scout = _agent(
        "scout", [_text_response("scout: saw the rewind, re-checking state")]
    )
    team = AgentTeam(
        [planner, scout],
        entry="planner",
        cards=[MemberCard(name="planner", lead=True)],
        ctx=ctx,
    )
    delivered: list[TeamMessage] = []

    consumer = asyncio.create_task(_drain(team, "kick off", delivered))
    try:
        await _until(lambda: planner.llm.call_count == 1)  # seed handled
        await ctx.restore_fs_snapshot("snap-1")
        await _until(lambda: scout.llm.call_count == 1)  # notice reactivated scout
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
    await team.aclose()

    assert env.restored == ["snap-1"]
    notices = [m for m in delivered if "<environment_rewind>" in m.text]
    assert [(m.sender, m.recipient) for m in notices] == [("planner", "scout")]
    assert notices[0].priority == CONTROL_PRIORITY
    # The notice names the rewinder for the recipient.
    assert "planner" in notices[0].text


@pytest.mark.asyncio
async def test_hosts_share_the_session_transport(tmp_path: Path) -> None:
    # The mailbox is session infrastructure, owned by ``ctx.transport``: every
    # host on the same session uses that instance — the mailbox (and its live
    # consumption counters) survives host rebuilds instead of being silently
    # replaced by a fresh, empty one.
    ctx = _ctx(tmp_path)
    team = AgentTeam([_agent("solo", [_text_response("a")])], ctx=ctx)
    team2 = AgentTeam([_agent("solo2", [_text_response("b")])], ctx=ctx)
    assert team._transport is ctx.transport  # pyright: ignore[reportPrivateUsage]
    assert team2._transport is ctx.transport  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_attach_inbox_uses_session_transport(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.transport = InMemoryMailboxTransport()
    agent = _agent("solo", [])
    agent.on_adopted(ctx=ctx)
    agent.attach_inbox()
    inbox = agent.agent_ctx.inbox
    assert inbox is not None
    assert inbox.transport is ctx.transport


@pytest.mark.asyncio
async def test_lead_member_rolls_back_directly_after_a_team_run(
    tmp_path: Path,
) -> None:
    # The human seed reaches the lead over the mailbox and anchors a rollback
    # boundary; between runs the lead rolls back like any stepping agent
    # (which member may — the lead here — is the app's policy, not the
    # team's), and the consumed human seed is voided — not re-delivered.
    ctx = _ctx(tmp_path)
    alice = _agent("alice", [_text_response("done")])
    bob = _agent("bob", [])
    team = AgentTeam(
        [alice, bob],
        entry="alice",
        cards=[MemberCard(name="alice", lead=True), MemberCard(name="bob")],
        ctx=ctx,
    )
    await team.run("do the thing")

    assert alice.rollback_steps == [1]
    await alice.rollback_to_step(1)
    assert alice.step == 1
    assert "do the thing" not in str(alice.transcript.messages)
    transport = ctx.transport
    assert transport is not None
    assert not await transport.has_pending("alice")  # voided, not re-pended


@pytest.mark.asyncio
async def test_team_runs_twice_on_the_same_session(tmp_path: Path) -> None:
    # A run's driver shutdown must not close the session mailbox
    # (``ctx.transport``), and a new input gets a fresh seed id — so a second
    # run on the same team + ctx actually delivers.
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("one"), _text_response("two")])
    team = AgentTeam([solo], ctx=ctx)

    await team.run("first question")
    await team.run("second question")

    assert solo.llm.call_count == 2
    blob = str(solo.transcript.messages)
    assert "first question" in blob
    assert "second question" in blob


@pytest.mark.asyncio
async def test_rebuilt_team_reuses_open_session_mailbox(tmp_path: Path) -> None:
    # A team rebuilt on the same session must find the mailbox usable, not
    # latched shut by the previous team's run.
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("one"), _text_response("two")])
    team1 = AgentTeam([solo], ctx=ctx)
    await team1.run("first question")

    team2 = AgentTeam([solo], ctx=ctx)
    await team2.run("second question")

    assert solo.llm.call_count == 2


@pytest.mark.asyncio
async def test_repeat_identical_input_delivers_on_a_new_run(tmp_path: Path) -> None:
    # Identical content across completed runs is two legitimate turns: seed
    # identity keys on the run ordinal, never on content.
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("one"), _text_response("two")])
    team = AgentTeam([solo], ctx=ctx)

    await team.run("same question")
    await team.run("same question")

    assert solo.llm.call_count == 2


@pytest.mark.asyncio
async def test_rewind_notice_reaches_peers_between_runs(tmp_path: Path) -> None:
    # A lead rollback restores a snapshot BETWEEN runs (no live driver); the
    # notice must land on the session mailbox for the peers' next take rather
    # than being dropped with the run-scoped routing.
    env = FakeSnapshotEnv(tmp_path)
    ctx = SessionContext[None](state=None, environment=env)
    alice = _agent("alice", [])
    bob = _agent("bob", [])
    AgentTeam(
        [alice, bob],
        cards=[MemberCard(name="alice", lead=True), MemberCard(name="bob")],
        ctx=ctx,
    )

    await ctx.restore_fs_snapshot("snap-1")

    transport = ctx.transport
    assert transport is not None
    assert await transport.has_pending("bob")
    assert not await transport.has_pending("alice")


@pytest.mark.asyncio
async def test_closed_team_stops_announcing_rewinds(tmp_path: Path) -> None:
    # aclose() deregisters the rewind announcer: after a rebuild on the same
    # session, one rewind produces exactly one notice — not one per build.
    env = FakeSnapshotEnv(tmp_path)
    ctx = SessionContext[None](state=None, environment=env)
    cards = [MemberCard(name="alice", lead=True), MemberCard(name="bob")]
    team1 = AgentTeam([_agent("alice", []), _agent("bob", [])], cards=cards, ctx=ctx)
    await team1.aclose()
    AgentTeam([_agent("alice", []), _agent("bob", [])], cards=cards, ctx=ctx)

    await ctx.restore_fs_snapshot("snap-1")

    transport = ctx.transport
    assert isinstance(transport, InMemoryMailboxTransport)
    assert len(transport._boxes.get("bob", [])) == 1  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_unowned_session_persistence_warns_once() -> None:
    # A lead-less team with session persistence on: every member is contained
    # and none claims the writer role, so the record is never written — the
    # session warns exactly once instead of staying silently inert.
    import logging

    store = InMemoryCheckpointStore()
    ctx = SessionContext[None](state=None, checkpoint_store=store, serialize_state=True)
    solo = _agent("solo", [_text_response("one")])
    team = AgentTeam([solo], ctx=ctx)

    logger = logging.getLogger("grasp_agents.session_context")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        await team.run("go")
    finally:
        logger.removeHandler(handler)
        await team.aclose()

    warnings = [r for r in records if "no session writer" in r.getMessage()]
    assert len(warnings) == 1
