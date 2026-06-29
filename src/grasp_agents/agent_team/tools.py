"""The stateless ``SendMessage`` tool team members call to message a peer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from grasp_agents.tools.base import BaseTool, ToolProgressCallback

from .message import USER_SENDER, TeamMessage
from .sources import MessageSink
from .transport import default_transport

if TYPE_CHECKING:
    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.run_context import RunContext

    from .agent_card import MemberCard
    from .sources import WakeupScheduler

SEND_MESSAGE_TOOL_NAME = "SendMessage"
SCHEDULE_WAKEUP_TOOL_NAME = "ScheduleWakeup"

# Resolves the destination a SendMessage call delivers into for the active run —
# the team itself (which routes per recipient), or a bare transport.
type TransportResolver = Callable[[RunContext[Any]], MessageSink]


class SendMessageInput(BaseModel):
    to: str = Field(
        description="Name of the teammate to message, or '*' to broadcast to all."
    )
    message: str = Field(description="The message body.")
    reply_to: str | None = Field(
        default=None, description="Optional id of a message this is a reply to."
    )


class SendMessageTool(BaseTool[SendMessageInput, str, Any]):
    """
    Send a message to a teammate's mailbox and return immediately.

    Asynchronous: the recipient receives the message at the start of its next
    turn and the sender does not block for a reply (any reply arrives as a later
    message). Stateless — the sender identity comes from the calling agent's
    context and the transport from the run context — so one instance is shared
    across every team member.
    """

    def __init__(
        self,
        cards: Sequence[MemberCard],
        *,
        transport_resolver: TransportResolver = default_transport,
    ) -> None:
        roster = "\n".join(f"- {c.render()}" for c in cards)
        super().__init__(
            name=SEND_MESSAGE_TOOL_NAME,
            description=(
                "Send a message to a teammate and keep working — delivery is "
                "asynchronous, you will not block for a reply (any reply arrives "
                "as a later message). Teammates you can message:\n" + roster
            ),
        )
        self._recipients = {c.name for c in cards}
        self._resolve_transport = transport_resolver

    async def _run(
        self,
        inp: SendMessageInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> str:
        del exec_id, progress_callback, path
        if ctx is None:
            raise ValueError("SendMessage requires a RunContext.")
        sender = agent_ctx.agent_name if agent_ctx is not None else USER_SENDER
        if inp.to == "*":
            recipients = sorted(self._recipients - {sender})
            if not recipients:
                return "No other teammates to broadcast to; message not sent."
        elif inp.to in self._recipients:
            recipients = [inp.to]
        else:
            valid = ", ".join(sorted(self._recipients)) or "(none)"
            return (
                f"No teammate named {inp.to!r}; message not sent. "
                f"Valid teammates: {valid}."
            )
        message = TeamMessage.of_text(
            sender=sender,
            to=recipients,
            text=inp.message,
            reply_to=inp.reply_to,
        )
        await self._resolve_transport(ctx).post(message)
        delivered = ", ".join(recipients)
        return f"Message delivered to {delivered} (id={message.message_id})."


class ScheduleWakeupInput(BaseModel):
    delay_seconds: float = Field(
        gt=0, description="How many seconds from now to wake yourself up."
    )
    note: str = Field(
        description=(
            "A note to your future self, delivered when the wake-up fires — why you "
            "scheduled it and what to do then."
        )
    )


class ScheduleWakeupTool(BaseTool[ScheduleWakeupInput, str, Any]):
    """
    Schedule a future, self-addressed wakeup for the calling member and return
    immediately.

    After ``delay_seconds`` a message carrying the member's ``note`` is delivered to
    that member's own mailbox, reactivating it — the initiative lever a triggered
    member pulls to act unprompted (revisit a goal, poll for a change, follow up)
    without holding a live loop open between turns. The call does not block.
    Meaningful on a daemon team; on a bounded team the wakeup fires only if it lands
    before quiescence. Stateless — the member identity comes from the calling agent's
    context — so one instance is shared across the team.
    """

    def __init__(self, scheduler: WakeupScheduler) -> None:
        super().__init__(
            name=SCHEDULE_WAKEUP_TOOL_NAME,
            description=(
                "Schedule a message to your future self after a delay and keep "
                "working — you will be reactivated with your note when it fires. Use "
                "it to act on your own initiative later (revisit a goal, poll for a "
                "change, follow up) instead of waiting for someone to message you."
            ),
        )
        self._scheduler = scheduler

    async def _run(
        self,
        inp: ScheduleWakeupInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> str:
        del ctx, exec_id, progress_callback, path
        if agent_ctx is None:
            raise ValueError(
                "ScheduleWakeup must be called by a team member (no agent context)."
            )
        me = agent_ctx.agent_name
        body = (
            "<scheduled_wakeup>\n"
            "A wake-up you scheduled earlier has fired. Act on it if there is work to "
            "do; otherwise go back to idle without calling any tools.\n"
            f"{inp.note}\n"
            "</scheduled_wakeup>"
        )
        # Self-addressed: sender == recipient == this member. In a daemon team the
        # message lands in the member's own mailbox and reactivates it on the delay.
        message = TeamMessage.of_text(sender=me, to=me, text=body)
        self._scheduler.schedule(message, delay=inp.delay_seconds)
        return f"Wake-up scheduled in {inp.delay_seconds:g}s (id={message.message_id})."
