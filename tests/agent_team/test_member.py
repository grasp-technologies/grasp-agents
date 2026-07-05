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
from grasp_agents.types.response import Response
from tests._helpers import FakeSnapshotEnv, MockLLM, _text_response, _tool_call_response

CARDS = [MemberCard(name="alice"), MemberCard(name="bob")]


def _agent(name: str, responses: list[Response]) -> LLMAgent[Any, Any, None]:
    return LLMAgent[Any, Any, None](name=name, llm=MockLLM(responses_queue=responses))


def _send(to: str, message: str, call_id: str) -> Response:
    return _tool_call_response(
        "SendMessage", f'{{"to": "{to}", "message": "{message}"}}', call_id
    )


async def _drain(host: MemberHost) -> list[Any]:
    return [ev async for ev in host.run_stream(stop_when_idle=True)]


@pytest.mark.asyncio
async def test_host_activates_sends_and_acks() -> None:
    transport = InMemoryMailboxTransport()
    alice = _agent("alice", [_send("bob", "ping", "c1"), _text_response("alice done")])
    host = MemberHost(alice, cards=CARDS, transport=transport)

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
    transport = InMemoryMailboxTransport()
    solo = _agent("solo", [_text_response("the answer")])
    host = MemberHost(solo, cards=[MemberCard(name="solo")], transport=transport)

    await host.submit_message("hello")
    events = await _drain(host)

    assert events
    assert solo.llm.call_count == 1


@pytest.mark.asyncio
async def test_two_hosts_converse_over_shared_transport() -> None:
    transport = InMemoryMailboxTransport()
    alice = _agent(
        "alice",
        [
            _send("bob", "ping", "c1"),
            _text_response("alice sent ping"),
            _text_response("alice got pong"),
        ],
    )
    bob = _agent("bob", [_send("alice", "pong", "c2"), _text_response("bob done")])
    host_a = MemberHost(alice, cards=CARDS, transport=transport)
    host_b = MemberHost(bob, cards=CARDS, transport=transport)

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
    alice = LLMAgent[Any, Any, None](
        name="alice", llm=MockLLM(responses_queue=[]), ctx=ctx
    )
    cards = [
        MemberCard(name="alice", lead=True),
        MemberCard(name="bob"),
        MemberCard(name="filer", resident=False),
    ]
    MemberHost(alice, cards=cards, transport=transport)

    assert ctx.environment_rewinder == "alice"

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
        MemberHost(alice, cards=cards, transport=InMemoryMailboxTransport())


@pytest.mark.asyncio
async def test_lead_sends_carry_lead_priority_via_host(tmp_path: Path) -> None:
    # A resident lead's SendMessage mail is stamped LEAD_PRIORITY through the
    # per-process host too (same cards, same tool).
    env = FakeSnapshotEnv(tmp_path)
    ctx = SessionContext[None](state=None, environment=env)
    transport = InMemoryMailboxTransport()
    alice = LLMAgent[Any, Any, None](
        name="alice",
        llm=MockLLM(
            responses_queue=[_send("bob", "ping", "c1"), _text_response("done")]
        ),
        ctx=ctx,
    )
    cards = [MemberCard(name="alice", lead=True), MemberCard(name="bob")]
    host = MemberHost(alice, cards=cards, transport=transport)

    await transport.post(TeamMessage.from_text(sender="user", to="alice", text="go"))
    await _drain(host)

    to_bob = await transport.consume("bob")
    assert isinstance(to_bob, TeamMessage)
    assert to_bob.priority == LEAD_PRIORITY
