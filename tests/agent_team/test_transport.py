"""Mailbox transports: round-trip, ordering, isolation, parts, default resolution."""

from __future__ import annotations

import pytest

from grasp_agents.agent_team.message import TeamMessage
from grasp_agents.agent_team.transport import (
    CheckpointMailboxTransport,
    InMemoryMailboxTransport,
    MessageTransport,
    default_transport,
)
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.run_context import RunContext
from grasp_agents.types.content import InputText


@pytest.fixture(params=["in_memory", "checkpoint"])
def transport(request: pytest.FixtureRequest) -> MessageTransport:
    if request.param == "checkpoint":
        return CheckpointMailboxTransport(InMemoryCheckpointStore(), session_key="s")
    return InMemoryMailboxTransport()


@pytest.mark.asyncio
async def test_send_fetch_ack_roundtrip(transport: MessageTransport) -> None:
    assert await transport.has_mail("bob") is False

    await transport.send(
        TeamMessage.of_text(sender="alice", to="bob", text="hi")
    )
    assert await transport.has_mail("bob") is True

    msg = await transport.fetch_next("bob")
    assert msg is not None
    assert msg.sender == "alice"
    assert msg.text == "hi"
    assert await transport.has_mail("bob") is True  # fetch_next must not consume

    await transport.ack("bob", [msg.message_id])
    assert await transport.has_mail("bob") is False
    assert await transport.fetch_next("bob") is None


@pytest.mark.asyncio
async def test_fetch_next_orders_oldest_first(transport: MessageTransport) -> None:
    for i in range(3):
        await transport.send(
            TeamMessage.of_text(
                sender="a", to="bob", text=f"m{i}", message_id=f"{i:04d}-x"
            )
        )
    # One group per call, oldest first; ack to advance to the next.
    got: list[str] = []
    for _ in range(3):
        msg = await transport.fetch_next("bob")
        assert msg is not None
        got.append(msg.text)
        await transport.ack("bob", [msg.message_id])
    assert got == ["m0", "m1", "m2"]
    assert await transport.fetch_next("bob") is None


@pytest.mark.asyncio
async def test_mailboxes_are_isolated_per_recipient(
    transport: MessageTransport,
) -> None:
    await transport.send(
        TeamMessage.of_text(sender="a", to="bob", text="for bob")
    )
    assert await transport.has_mail("bob") is True
    assert await transport.has_mail("carol") is False


@pytest.mark.asyncio
async def test_message_payloads_roundtrip(transport: MessageTransport) -> None:
    await transport.send(
        TeamMessage(sender="a", routing=[["bob"]], payloads=[InputText(text="hi")])
    )
    msg = await transport.fetch_next("bob")
    assert msg is not None
    assert isinstance(msg.payloads[0], InputText)
    assert msg.text == "hi"


def test_default_transport_requires_checkpoint_store() -> None:
    with pytest.raises(ValueError, match="checkpoint_store"):
        default_transport(RunContext[None](state=None))


def test_default_transport_uses_checkpoint_store() -> None:
    ctx = RunContext[None](state=None, checkpoint_store=InMemoryCheckpointStore())
    assert isinstance(default_transport(ctx), CheckpointMailboxTransport)
