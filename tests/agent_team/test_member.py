"""
MemberDriver: one member as a single serial inbox (human input + mailbox). Drives
two members sequentially over one in-memory transport (same process) to validate
the activation/send/ack logic the separate-process UI relies on.
"""

from __future__ import annotations

from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent_team.agent_card import MemberCard
from grasp_agents.agent_team.member import MemberDriver
from grasp_agents.agent_team.message import TeamMessage
from grasp_agents.agent_team.transport import InMemoryMailboxTransport
from grasp_agents.types.response import Response
from tests._helpers import MockLLM, _text_response, _tool_call_response

CARDS = [MemberCard(name="alice"), MemberCard(name="bob")]


def _agent(name: str, responses: list[Response]) -> LLMAgent[Any, Any, None]:
    return LLMAgent[Any, Any, None](name=name, llm=MockLLM(responses_queue=responses))


def _send(to: str, message: str, call_id: str) -> Response:
    return _tool_call_response(
        "SendMessage", f'{{"to": "{to}", "message": "{message}"}}', call_id
    )


async def _drain(driver: MemberDriver) -> list[Any]:
    return [ev async for ev in driver.events(stop_when_idle=True)]


@pytest.mark.asyncio
async def test_driver_activates_sends_and_acks() -> None:
    transport = InMemoryMailboxTransport()
    alice = _agent("alice", [_send("bob", "ping", "c1"), _text_response("alice done")])
    driver = MemberDriver(alice, cards=CARDS, transport=transport)

    await transport.send(
        TeamMessage.of_text(sender="user", to="alice", text="go")
    )
    events = await _drain(driver)

    assert events  # alice ran and produced events
    assert alice.llm.call_count == 2
    to_bob = await transport.fetch_next("bob")
    assert to_bob is not None
    assert (to_bob.sender, to_bob.recipient, to_bob.text) == ("alice", "bob", "ping")
    assert await transport.has_mail("alice") is False


@pytest.mark.asyncio
async def test_human_input_runs_a_turn() -> None:
    transport = InMemoryMailboxTransport()
    solo = _agent("solo", [_text_response("the answer")])
    driver = MemberDriver(solo, cards=[MemberCard(name="solo")], transport=transport)

    driver.submit_human("hello")
    events = await _drain(driver)

    assert events
    assert solo.llm.call_count == 1


@pytest.mark.asyncio
async def test_two_drivers_converse_over_shared_transport() -> None:
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
    a = MemberDriver(alice, cards=CARDS, transport=transport)
    b = MemberDriver(bob, cards=CARDS, transport=transport)

    await transport.send(
        TeamMessage.of_text(sender="user", to="alice", text="go")
    )
    # Drive the causal chain: alice (→ping), bob (→pong), alice (consumes pong).
    await _drain(a)
    await _drain(b)
    await _drain(a)

    assert alice.llm.call_count == 3
    assert bob.llm.call_count == 2
    assert await transport.has_mail("alice") is False
    assert await transport.has_mail("bob") is False
