"""The stateless SendMessage tool: sender resolution, delivery, roster guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from grasp_agents.agent_team.agent_card import MemberCard
from grasp_agents.agent_team.tools import SendMessageInput, SendMessageTool
from grasp_agents.agent_team.transport import CheckpointMailboxTransport
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.run_context import RunContext


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
    msg = await transport.fetch_next("bob")
    assert msg is not None
    assert msg.sender == "alice"
    assert msg.text == "hello"


@pytest.mark.asyncio
async def test_unknown_recipient_is_not_sent() -> None:
    ctx = _ctx()
    tool = SendMessageTool([MemberCard(name="bob")])

    out = await tool._run(
        SendMessageInput(to="charlie", message="x"), ctx=ctx, agent_ctx=None
    )
    assert "No teammate named 'charlie'" in out

    transport = CheckpointMailboxTransport(ctx.checkpoint_store)  # type: ignore[arg-type]
    assert await transport.has_mail("charlie") is False


def test_roster_appears_in_tool_description() -> None:
    tool = SendMessageTool([MemberCard(name="researcher", description="finds sources")])
    assert "researcher" in tool.description
    assert "finds sources" in tool.description
