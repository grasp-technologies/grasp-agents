"""TeamMessage rendering: how inbound mail becomes a recipient's user turn."""

from __future__ import annotations

from grasp_agents.types.message import TeamMessage


def test_peer_mail_is_fenced_with_sender_attribution() -> None:
    peer = TeamMessage.from_text(sender="researcher", to="lead", text="the facts")
    rendered = str(peer.to_chat_inputs()[0])
    assert rendered.startswith("<team_member_message from=researcher>")
    assert rendered.endswith("</team_member_message>")
    assert "the facts" in rendered


def test_reply_to_is_carried_in_the_fence() -> None:
    peer = TeamMessage.from_text(
        sender="writer", to="lead", text="draft", reply_to="m-1"
    )
    assert "<team_member_message from=writer reply_to=m-1>" in str(
        peer.to_chat_inputs()[0]
    )


def test_human_mail_renders_bare() -> None:
    # Human input reads as a plain user turn — no attribution fence.
    human = TeamMessage.from_text(sender="user", to="lead", text="hi team")
    assert human.to_chat_inputs() == ["hi team"]
