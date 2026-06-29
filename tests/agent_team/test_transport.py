"""
Mailbox transports as runtime ``Transport[TeamMessage]`` implementations: post /
consume / ack round-trip, ordering, isolation, payloads, shutdown, and the default
resolution. One transport seam — no separate mailbox type, no adapter.
"""

from __future__ import annotations

import pytest

from grasp_agents.agent_team.transport import default_transport
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.mailbox import (
    CheckpointMailboxTransport,
    InMemoryMailboxTransport,
)
from grasp_agents.run_context import RunContext
from grasp_agents.runtime import CLOSED, Transport
from grasp_agents.types.content import InputText
from grasp_agents.types.message import TeamMessage


@pytest.fixture(params=["in_memory", "checkpoint"])
def transport(request: pytest.FixtureRequest) -> Transport[TeamMessage]:
    if request.param == "checkpoint":
        return CheckpointMailboxTransport(InMemoryCheckpointStore(), session_key="s")
    return InMemoryMailboxTransport()


@pytest.mark.asyncio
async def test_post_consume_ack_roundtrip(transport: Transport[TeamMessage]) -> None:
    assert await transport.has_pending("bob") is False

    await transport.post(TeamMessage.of_text(sender="alice", to="bob", text="hi"))
    assert await transport.has_pending("bob") is True

    msg = await transport.consume("bob")
    assert isinstance(msg, TeamMessage)
    assert msg.sender == "alice"
    assert msg.text == "hi"
    assert await transport.has_pending("bob") is True  # consume must not remove it

    await transport.ack("bob", msg)
    assert await transport.has_pending("bob") is False


@pytest.mark.asyncio
async def test_consume_orders_oldest_first(transport: Transport[TeamMessage]) -> None:
    for i in range(3):
        await transport.post(
            TeamMessage.of_text(
                sender="a", to="bob", text=f"m{i}", message_id=f"{i:04d}-x"
            )
        )
    # One group per call, oldest first; ack to advance to the next.
    got: list[str] = []
    for _ in range(3):
        msg = await transport.consume("bob")
        assert isinstance(msg, TeamMessage)
        got.append(msg.text)
        await transport.ack("bob", msg)
    assert got == ["m0", "m1", "m2"]
    assert await transport.has_pending("bob") is False


@pytest.mark.asyncio
async def test_mailboxes_are_isolated_per_recipient(
    transport: Transport[TeamMessage],
) -> None:
    await transport.post(TeamMessage.of_text(sender="a", to="bob", text="for bob"))
    assert await transport.has_pending("bob") is True
    assert await transport.has_pending("carol") is False


@pytest.mark.asyncio
async def test_message_payloads_roundtrip(transport: Transport[TeamMessage]) -> None:
    await transport.post(
        TeamMessage(sender="a", routing=[["bob"]], payloads=[InputText(text="hi")])
    )
    msg = await transport.consume("bob")
    assert isinstance(msg, TeamMessage)
    assert isinstance(msg.payloads[0], InputText)
    assert msg.text == "hi"


@pytest.mark.asyncio
async def test_shutdown_unblocks_consume(transport: Transport[TeamMessage]) -> None:
    # An empty mailbox blocks consume; shutdown must release it with CLOSED.
    await transport.shutdown()
    assert await transport.consume("bob") is CLOSED


def test_default_transport_requires_checkpoint_store() -> None:
    with pytest.raises(ValueError, match="checkpoint_store"):
        default_transport(RunContext[None](state=None))


def test_default_transport_uses_checkpoint_store() -> None:
    ctx = RunContext[None](state=None, checkpoint_store=InMemoryCheckpointStore())
    assert isinstance(default_transport(ctx), CheckpointMailboxTransport)
