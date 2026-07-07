"""
MemberHost: one member as a single serial inbox (human input + mailbox). Hosts
two members sequentially over one in-memory transport (same process) to validate
the activation/send/ack logic the separate-process UI relies on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent_team.agent_card import MemberCard
from grasp_agents.agent_team.member import MemberHost
from grasp_agents.agent_team.message import CONTROL_PRIORITY, LEAD_PRIORITY, TeamMessage
from grasp_agents.mailbox import InMemoryMailboxTransport
from grasp_agents.session_context import SessionContext
from tests._helpers import FakeSnapshotEnv, MockLLM, _agent, _send, _text_response

CARDS = [MemberCard(name="alice"), MemberCard(name="bob")]


def _session() -> tuple[SessionContext[None], InMemoryMailboxTransport]:
    """A per-test session with its mailbox pre-installed on ``ctx.transport``."""
    ctx = SessionContext[None](state=None)
    transport = InMemoryMailboxTransport()
    ctx.transport = transport
    return ctx, transport


async def _drain(host: MemberHost) -> list[Any]:
    return [ev async for ev in host.run_stream(stop_when_idle=True)]


@pytest.mark.asyncio
async def test_host_activates_sends_and_acks() -> None:
    ctx, transport = _session()
    alice = _agent(
        "alice", [_send("bob", "ping", "c1"), _text_response("alice done")], ctx=ctx
    )
    host = MemberHost(alice, cards=CARDS)

    await transport.post(TeamMessage.from_text(sender="user", to="alice", text="go"))
    events = await _drain(host)

    assert events  # alice ran and produced events
    assert alice.llm.call_count == 2
    to_bob = await transport.consume("bob")
    assert isinstance(to_bob, TeamMessage)
    assert (to_bob.sender, to_bob.recipient, to_bob.text) == ("alice", "bob", "ping")
    assert await transport.has_pending("alice") is False


@pytest.mark.asyncio
async def test_human_input_runs_a_turn() -> None:
    ctx, _transport = _session()
    solo = _agent("solo", [_text_response("the answer")], ctx=ctx)
    host = MemberHost(solo, cards=[MemberCard(name="solo")])

    await host.submit_message("hello")
    events = await _drain(host)

    assert events
    assert solo.llm.call_count == 1


@pytest.mark.asyncio
async def test_two_hosts_converse_over_shared_session() -> None:
    # Two hosts on one session share its mailbox (``ctx.transport``) — nothing
    # is passed explicitly.
    ctx, transport = _session()
    alice = _agent(
        "alice",
        [
            _send("bob", "ping", "c1"),
            _text_response("alice sent ping"),
            _text_response("alice got pong"),
        ],
        ctx=ctx,
    )
    bob = _agent(
        "bob", [_send("alice", "pong", "c2"), _text_response("bob done")], ctx=ctx
    )
    host_a = MemberHost(alice, cards=CARDS)
    host_b = MemberHost(bob, cards=CARDS)

    await transport.post(TeamMessage.from_text(sender="user", to="alice", text="go"))
    # Drive the causal chain: alice (→ping), bob (→pong), alice (consumes pong).
    await _drain(host_a)
    await _drain(host_b)
    await _drain(host_a)

    assert alice.llm.call_count == 3
    assert bob.llm.call_count == 2
    assert await transport.has_pending("alice") is False
    assert await transport.has_pending("bob") is False


@pytest.mark.asyncio
async def test_lead_host_claims_rewind_right_and_announces_rewind(
    tmp_path: Path,
) -> None:
    # A lead-carded member's host claims the environment-rewind right at
    # construction; a rewind is announced over the shared mailbox to every peer
    # except itself and peers explicitly carded triggered (fresh per activation,
    # no cross-turn view to invalidate).
    env = FakeSnapshotEnv(tmp_path)
    ctx = SessionContext[None](state=None, environment=env)
    transport = InMemoryMailboxTransport()
    ctx.transport = transport
    alice = LLMAgent[Any, Any, None](
        name="alice", llm=MockLLM(responses_queue=[]), ctx=ctx
    )
    cards = [
        MemberCard(name="alice", lead=True),
        MemberCard(name="bob"),
        MemberCard(name="filer", resident=False),
    ]
    MemberHost(alice, cards=cards)

    assert ctx.session_writer == "alice"

    await ctx.restore_fs_snapshot("snap-1")

    assert env.restored == ["snap-1"]
    notice = await transport.consume("bob")
    assert isinstance(notice, TeamMessage)
    assert notice.sender == "alice"
    assert notice.priority == CONTROL_PRIORITY
    assert "<environment_rewind>" in notice.text
    # Not the rewinder itself, not the triggered-carded peer.
    assert await transport.has_pending("alice") is False
    assert await transport.has_pending("filer") is False


@pytest.mark.asyncio
async def test_triggered_lead_rejected_by_host(tmp_path: Path) -> None:
    # Same construction guard as AgentTeam: a lead carded triggered is refused.
    env = FakeSnapshotEnv(tmp_path)
    ctx = SessionContext[None](state=None, environment=env)
    alice = LLMAgent[Any, Any, None](
        name="alice", llm=MockLLM(responses_queue=[]), ctx=ctx
    )
    cards = [MemberCard(name="alice", lead=True, resident=False)]
    with pytest.raises(ValueError, match="must run resident"):
        MemberHost(alice, cards=cards)


@pytest.mark.asyncio
async def test_lead_sends_carry_lead_priority_via_host(tmp_path: Path) -> None:
    # A resident lead's SendMessage mail is stamped LEAD_PRIORITY through the
    # per-process host too (same cards, same tool).
    env = FakeSnapshotEnv(tmp_path)
    ctx = SessionContext[None](state=None, environment=env)
    transport = InMemoryMailboxTransport()
    ctx.transport = transport
    alice = LLMAgent[Any, Any, None](
        name="alice",
        llm=MockLLM(
            responses_queue=[_send("bob", "ping", "c1"), _text_response("done")]
        ),
        ctx=ctx,
    )
    cards = [MemberCard(name="alice", lead=True), MemberCard(name="bob")]
    host = MemberHost(alice, cards=cards)

    await transport.post(TeamMessage.from_text(sender="user", to="alice", text="go"))
    await _drain(host)

    to_bob = await transport.consume("bob")
    assert isinstance(to_bob, TeamMessage)
    assert to_bob.priority == LEAD_PRIORITY


@pytest.mark.asyncio
async def test_host_resolves_transport_from_session_ctx() -> None:
    # The mailbox is always the session's: the first host installs one on
    # ``ctx.transport`` (in-memory here — no store on the session) and every
    # later host reuses that same instance.
    ctx = SessionContext[None](state=None)
    alice = _agent("alice", [_text_response("hi")], ctx=ctx)
    host = MemberHost(alice, cards=CARDS)
    assert ctx.transport is not None
    assert host._mailbox is ctx.transport  # pyright: ignore[reportPrivateUsage]

    host2 = MemberHost(alice, cards=CARDS)
    assert host2._mailbox is ctx.transport  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_hosted_member_rolls_back_directly_between_runs() -> None:
    # The human turn anchored a boundary; between runs the member rolls back
    # like any stepping agent — its transcript rewinds and the consumed human
    # message is voided (discarded silently, never re-delivered).
    ctx, transport = _session()
    alice = _agent("alice", [_text_response("the answer")], ctx=ctx)
    host = MemberHost(alice, cards=CARDS)

    await host.submit_message("hello")
    await _drain(host)

    assert alice.rollback_steps == [1]
    await alice.rollback_to_step(1)
    assert alice.step == 1
    assert "hello" not in str(alice.transcript.messages)
    assert not await transport.has_pending("alice")  # voided, not re-pended


@pytest.mark.asyncio
async def test_closed_host_stops_announcing_rewinds(tmp_path: Path) -> None:
    # host.aclose() deregisters the lead's rewind announcer, so a host rebuilt
    # on the same session posts exactly one notice per rewind.
    env = FakeSnapshotEnv(tmp_path)
    ctx = SessionContext[None](state=None, environment=env)
    transport = InMemoryMailboxTransport()
    ctx.transport = transport
    cards = [MemberCard(name="alice", lead=True), MemberCard(name="bob")]

    host1 = MemberHost(_agent("alice", [], ctx=ctx), cards=cards)
    await host1.aclose()
    MemberHost(_agent("alice", [], ctx=ctx), cards=cards)

    await ctx.restore_fs_snapshot("snap-1")

    assert len(transport._boxes.get("bob", [])) == 1  # pyright: ignore[reportPrivateUsage]
