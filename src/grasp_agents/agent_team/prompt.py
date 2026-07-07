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
    "shared chat and no one sees your reasoning. A message from a teammate arrives as "
    "a user turn fenced in <team_member_message from=...>...</team_member_message>; "
    "the user's own messages arrive as plain user turns. To reply, delegate, or hand "
    "work off, call the SendMessage "
    "tool addressed to a teammate by name; the roster below lists who you can message "
    "and what structured input each accepts. Delivery is asynchronous, so you will not "
    "block for a reply (it arrives later as another message). "
)

# Appended only for a member carded with ``wakeups=True`` (which is what puts
# the ScheduleWakeup tool on it).
_WAKEUP_INSTRUCTIONS = (
    "Use ScheduleWakeup to "
    "revisit something later on your own initiative instead of waiting to be messaged. "
)

_PARK_INSTRUCTIONS = (
    "When you have nothing left to do, stop and wait for the next message rather than "
    "inventing work — you will be reactivated when one arrives."
)


def make_team_section(
    cards: Sequence[MemberCard],
    *,
    section_name: str = TEAM_SECTION_NAME,
    wakeups: bool = False,
) -> SystemPromptSection:
    """
    Build the team :class:`SystemPromptSection` attached to every resident member.

    It frames the member as part of a message-passing team — how to send
    (``SendMessage``), receive (the ``<team_member_message>`` fence), and park when
    idle — and carries the roster: who is on the team and what structured input each
    accepts. The roster lives here, not in the ``SendMessage`` tool description, so
    the tool schema stays lean and the team context sits where the model reads its
    instructions. Pass ``wakeups=True`` for a member that carries the
    ``ScheduleWakeup`` tool, so the framing mentions it.
    """
    instructions = _TEAM_INSTRUCTIONS
    if wakeups:
        instructions += _WAKEUP_INSTRUCTIONS
    instructions += _PARK_INSTRUCTIONS
    roster = "\n".join(f"- {card.render()}" for card in cards)
    text = f"{instructions}\n\nTeammates you can message:\n{roster}"

    lead = next((c.name for c in cards if c.lead), None)
    if lead is not None:
        text += (
            f"\n\nThe team lead is {lead!r}: messages from the lead arrive ahead "
            "of other peers' in your inbox, and the lead may rewind the team's "
            "shared workspace to an earlier snapshot (you will receive an "
            "<environment_rewind> notice when that happens)."
        )
    text += (
        "\n\nIf a peer's conversation is rolled back, messages you sent it may "
        "be discarded unanswered — you will receive a <message_dropped> notice "
        "for each; resend what is still relevant."
    )

    def compute(**_: Any) -> str:
        return text

    return SystemPromptSection(name=section_name, compute=compute)


def make_rewind_notice(rewinder: str) -> str:
    """
    The message body announcing an environment rewind to the other members —
    posted control-plane by the host when ``rewinder`` restores a filesystem
    snapshot, so each member learns the workspace changed under it before acting
    on queued mail.
    """
    return (
        "<environment_rewind>\n"
        f"The team lead ({rewinder}) has rewound the shared environment to an "
        "earlier snapshot. Files may have changed or reverted, and any running "
        "kernels or shell sessions were reset along with the filesystem. The "
        "lead's own conversation was rewound to the same point: it will not "
        "remember exchanges from after it, and messages you sent it since then "
        "may be re-delivered and answered again. This was deliberate — do not "
        "treat it as corruption. Re-read any files you rely on and "
        "re-establish kernel/shell state before continuing.\n"
        "</environment_rewind>"
    )


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
