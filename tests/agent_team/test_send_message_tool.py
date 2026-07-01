"""The stateless SendMessage tool: sender resolution, delivery, roster guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from grasp_agents.agent_team.agent_card import MemberCard
from grasp_agents.agent_team.tools import SendMessageInput, SendMessageTool
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.mailbox import CheckpointMailboxTransport
from grasp_agents.run_context import RunContext
from grasp_agents.types.message import TeamMessage


class _Ticket(BaseModel):
    title: str
    points: int


def _ctx() -> RunContext[None]:
    return RunContext[None](state=None, checkpoint_store=InMemoryCheckpointStore())


@pytest.mark.asyncio
async def test_send_delivers_with_sender_identity() -> None:
    ctx = _ctx()
    tool = SendMessageTool([MemberCard(name="alice"), MemberCard(name="bob")])

    out = await tool._run(
        SendMessageInput(to="bob", message="hello"),
        ctx=ctx,
        agent_ctx=SimpleNamespace(agent_name="alice"),  # type: ignore[arg-type]
    )
    assert "delivered to bob" in out

    transport = CheckpointMailboxTransport(ctx.checkpoint_store)  # type: ignore[arg-type]
    msg = await transport.consume("bob")
    assert isinstance(msg, TeamMessage)
    assert msg.sender == "alice"
    assert msg.text == "hello"


@pytest.mark.asyncio
async def test_unknown_recipient_is_not_sent() -> None:
    ctx = _ctx()
    tool = SendMessageTool([MemberCard(name="bob")])

    # A rejected send raises (→ BaseTool surfaces an is_error ToolErrorInfo to the
    # model), rather than returning a success-looking string.
    with pytest.raises(ValueError, match="No teammate named 'charlie'"):
        await tool._run(
            SendMessageInput(to="charlie", message="x"), ctx=ctx, agent_ctx=None
        )

    transport = CheckpointMailboxTransport(ctx.checkpoint_store)  # type: ignore[arg-type]
    assert await transport.has_pending("charlie") is False


def test_description_does_not_inline_roster() -> None:
    # The roster (who + structured input shapes) lives in the team system-prompt
    # section, not the tool schema — the description stays lean and just points there.
    tool = SendMessageTool(
        [MemberCard(name="researcher", description="finds sources", input_type=_Ticket)]
    )
    assert "researcher" not in tool.description
    assert "finds sources" not in tool.description
    assert "Input message schema" not in tool.description
    assert "roster" in tool.description.lower()


@pytest.mark.asyncio
async def test_structured_body_validated_and_delivered_typed() -> None:
    ctx = _ctx()
    tool = SendMessageTool([MemberCard(name="planner", input_type=_Ticket)])

    out = await tool._run(
        SendMessageInput(to="planner", message={"title": "Fix", "points": 3}),
        ctx=ctx,
        agent_ctx=SimpleNamespace(agent_name="alice"),  # type: ignore[arg-type]
    )
    assert "delivered to planner" in out

    transport = CheckpointMailboxTransport(ctx.checkpoint_store)  # type: ignore[arg-type]
    msg = await transport.consume("planner")
    assert isinstance(msg, TeamMessage)
    # Carried as a structured payload (not flattened to text); the values round-trip.
    assert not msg.is_content
    assert msg.payloads[0]["title"] == "Fix"  # type: ignore[index]


@pytest.mark.asyncio
async def test_structured_body_mismatch_is_rejected() -> None:
    ctx = _ctx()
    tool = SendMessageTool([MemberCard(name="planner", input_type=_Ticket)])

    with pytest.raises(ValueError, match="does not match its expected input"):
        await tool._run(
            SendMessageInput(to="planner", message={"title": "Fix"}),  # missing points
            ctx=ctx,
            agent_ctx=SimpleNamespace(agent_name="alice"),  # type: ignore[arg-type]
        )

    transport = CheckpointMailboxTransport(ctx.checkpoint_store)  # type: ignore[arg-type]
    assert await transport.has_pending("planner") is False
