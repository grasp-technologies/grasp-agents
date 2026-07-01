"""System-prompt + input-attachment helpers that frame a member as part of a team."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from grasp_agents.context.prompt_builder import InputAttachment, SystemPromptSection

from .message import USER_SENDER

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .agent_card import MemberCard

TEAM_SECTION_NAME = "team"
SENDER_ATTRIBUTION_ATTACHMENT_NAME = "team_sender"

_TEAM_INSTRUCTIONS = (
    "You are one member of a team that collaborates by SENDING MESSAGES — there is no "
    "shared chat and no one sees your reasoning. A message from a teammate (or the "
    "user) arrives as a user turn fenced in <team_member_message from=...>...</"
    "team_member_message>. To reply, delegate, or hand work off, call the SendMessage "
    "tool addressed to a teammate by name; the roster below lists who you can message "
    "and what structured input each accepts. Delivery is asynchronous, so you will not "
    "block for a reply (it arrives later as another message). Use ScheduleWakeup to "
    "revisit something later on your own initiative instead of waiting to be messaged. "
    "When you have nothing left to do, stop and wait for the next message rather than "
    "inventing work — you will be reactivated when one arrives."
)


def make_team_section(
    cards: Sequence[MemberCard], *, section_name: str = TEAM_SECTION_NAME
) -> SystemPromptSection:
    """
    Build the team :class:`SystemPromptSection` attached to every resident member.

    It frames the member as part of a message-passing team — how to send
    (``SendMessage``), receive (the ``<team_member_message>`` fence), schedule its own
    wakeups (``ScheduleWakeup``), and park when idle — and carries the roster: who is
    on the team and what structured input each accepts. The roster lives here, not in
    the ``SendMessage`` tool description, so the tool schema stays lean and the team
    context sits where the model reads its instructions.
    """
    roster = "\n".join(f"- {card.render()}" for card in cards)
    text = f"{_TEAM_INSTRUCTIONS}\n\nTeammates you can message:\n{roster}"

    def compute(**_: Any) -> str:
        return text

    return SystemPromptSection(name=section_name, compute=compute)


def make_sender_attribution_attachment(
    *, name: str = SENDER_ATTRIBUTION_ATTACHMENT_NAME
) -> InputAttachment:
    """
    Attribute a triggered member's input to the teammate that sent it.

    A triggered member receives a peer message as a fresh run whose input renders
    through its normal pipeline (so a typed body keeps its ``InputRenderable``); this
    attachment appends a ``<system-reminder>`` naming the sender — the attribution a
    resident gets from the ``<team_member_message from=...>`` fence. Inert when the
    input has no peer source (a human turn, or a direct non-team run).
    """

    def compute(*, source: str | None = None, **_: Any) -> str | None:
        if source is None or source == USER_SENDER:
            return None
        return f"This message was sent to you by your teammate {source!r}."

    return InputAttachment(name=name, compute=compute)
