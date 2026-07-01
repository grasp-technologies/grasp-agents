"""The team system-prompt section (roster) + the sender-attribution attachment."""

from __future__ import annotations

from pydantic import BaseModel

from grasp_agents.agent_team.agent_card import MemberCard
from grasp_agents.agent_team.message import USER_SENDER
from grasp_agents.agent_team.prompt import (
    make_sender_attribution_attachment,
    make_team_section,
)


class _Ticket(BaseModel):
    title: str
    points: int


def test_team_section_carries_roster_and_schemas() -> None:
    # The roster — names, descriptions, skills, and structured input schemas — lives
    # in the team system-prompt section (not the SendMessage tool description).
    section = make_team_section(
        [
            MemberCard(name="researcher", description="finds sources", skills=["web"]),
            MemberCard(name="planner", input_type=_Ticket),
        ]
    )
    text = section.compute()

    assert "team" in text.lower()  # framing
    assert "researcher" in text
    assert "finds sources" in text
    assert "web" in text  # skill
    # planner's structured input shape is advertised here, in the prompt.
    assert "Input message schema" in text
    assert "title" in text
    assert "points" in text


def test_sender_attribution_reminds_of_peer() -> None:
    out = make_sender_attribution_attachment().compute(source="scout")
    assert out is not None
    assert "scout" in out


def test_sender_attribution_inert_for_human_or_none() -> None:
    attach = make_sender_attribution_attachment()
    # A human turn (USER_SENDER), an absent source, or a no-arg call → nothing added.
    assert attach.compute(source=USER_SENDER) is None
    assert attach.compute(source=None) is None
    assert attach.compute() is None
