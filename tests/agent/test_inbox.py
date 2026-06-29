"""
The per-agent :class:`AgentInbox` as a view over a pluggable
:class:`MessageTransport`: in-memory FIFO, the parked-idle signal, and durable
delivery (the same message survives a restart over a shared checkpoint store, and a
consumed one is not re-delivered).
"""

from __future__ import annotations

import pytest

from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.inbox import AgentInbox
from grasp_agents.mailbox import CheckpointMailboxTransport, InMemoryMailboxTransport
from grasp_agents.types.message import TeamMessage

pytestmark = pytest.mark.asyncio


def _msg(text: str, *, to: str = "curator") -> TeamMessage:
    return TeamMessage.of_text(sender="user", to=to, text=text)


async def test_in_memory_fifo_and_waiting_signal() -> None:
    inbox = AgentInbox(recipient="curator")
    assert not await inbox.has_pending()
    assert not inbox.is_waiting

    await inbox.post(_msg("a"))
    await inbox.post(_msg("b"))
    assert await inbox.has_pending()

    first = await inbox.poll()
    second = await inbox.poll()
    assert first is not None
    assert first.text == "a"
    assert second is not None
    assert second.text == "b"
    assert await inbox.poll() is None
    assert not await inbox.has_pending()

    with inbox.waiting():
        assert inbox.is_waiting
    assert not inbox.is_waiting


async def test_shared_transport_routes_to_the_right_recipient() -> None:
    # One transport, two members' views — a message addressed to alice lands only
    # in alice's view (the same seam serves every member).
    transport = InMemoryMailboxTransport()
    alice = AgentInbox(transport=transport, recipient="alice")
    bob = AgentInbox(transport=transport, recipient="bob")

    await alice.post(_msg("for bob", to="bob"))
    assert not await alice.has_pending()
    assert await bob.has_pending()
    got = await bob.poll()
    assert got is not None
    assert got.text == "for bob"


async def test_durable_transport_survives_restart() -> None:
    store = InMemoryCheckpointStore()

    producer = AgentInbox(
        transport=CheckpointMailboxTransport(store, session_key="s"),
        recipient="curator",
    )
    await producer.post(_msg("first"))
    await producer.post(_msg("second"))

    # A fresh inbox over a fresh transport on the same store (a restart) sees both.
    resumed = AgentInbox(
        transport=CheckpointMailboxTransport(store, session_key="s"),
        recipient="curator",
    )
    texts = []
    while (msg := await resumed.poll()) is not None:
        texts.append(msg.text)
    assert texts == ["first", "second"]


async def test_durable_consume_is_not_redelivered_after_restart() -> None:
    store = InMemoryCheckpointStore()
    inbox = AgentInbox(
        transport=CheckpointMailboxTransport(store, session_key="s"),
        recipient="curator",
    )
    await inbox.post(_msg("once"))
    await inbox.post(_msg("twice"))
    consumed = await inbox.poll()
    assert consumed is not None
    assert consumed.text == "once"

    resumed = AgentInbox(
        transport=CheckpointMailboxTransport(store, session_key="s"),
        recipient="curator",
    )
    remaining = []
    while (msg := await resumed.poll()) is not None:
        remaining.append(msg.text)
    assert remaining == ["twice"]
