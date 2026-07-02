"""The stateless ``SendMessage`` tool team members call to message a peer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from grasp_agents.mailbox import CheckpointMailboxTransport
from grasp_agents.tools.base import BaseTool, ToolProgressCallback

from .message import USER_SENDER, TeamMessage

if TYPE_CHECKING:
    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.runtime import Transport
    from grasp_agents.session_context import SessionContext

    from .agent_card import MemberCard

SEND_MESSAGE_TOOL_NAME = "SendMessage"
SCHEDULE_WAKEUP_TOOL_NAME = "ScheduleWakeup"


@runtime_checkable
class MessageSink(Protocol):
    """
    Anything a message can be deposited into — a mailbox
    :class:`~grasp_agents.runtime.Transport`, or a team that routes + counts. The
    minimal write side of a transport; the parameter name matches
    :meth:`Transport.post` so a transport satisfies it structurally.
    """

    async def post(self, envelope: TeamMessage) -> None: ...


# Resolves the destination a SendMessage call delivers into for the active run —
# the team itself (which routes per recipient), or a bare transport.
type TransportResolver = Callable[[SessionContext[Any]], MessageSink]


def default_transport(ctx: SessionContext[Any]) -> Transport[TeamMessage]:
    """
    A durable transport over ``ctx.checkpoint_store``. Raises if no checkpoint
    store is wired — a single-process team uses
    :class:`~grasp_agents.mailbox.InMemoryMailboxTransport` instead (the team
    falls back to it automatically).
    """
    store = ctx.checkpoint_store
    if store is None:
        raise ValueError(
            "Durable AgentTeam messaging requires ctx.checkpoint_store. Wire a "
            "CheckpointStore (e.g. FileCheckpointStore(root=...)), or rely on the "
            "in-memory transport for a single-process team."
        )
    return CheckpointMailboxTransport(store, session_key=ctx.session_key)


class SendMessageInput(BaseModel):
    to: str = Field(
        description="Name of the teammate to message, or '*' to broadcast to all."
    )
    message: str | dict[str, Any] = Field(
        description=(
            "The message body: free text, or — for a teammate that advertises a "
            "structured body — an object matching that teammate's accepted shape."
        )
    )
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
        super().__init__(
            name=SEND_MESSAGE_TOOL_NAME,
            description=(
                "Send a message to a teammate and keep working — delivery is "
                "asynchronous, you will not block for a reply (any reply arrives as "
                "a later message). Address it to a teammate by name (or '*' to "
                "broadcast); the team roster in your instructions lists who you can "
                "message and the structured input each accepts."
            ),
        )
        self._recipients = {c.name for c in cards}
        self._cards = {c.name: c for c in cards}
        self._resolve_transport = transport_resolver

    async def _run(
        self,
        inp: SendMessageInput,
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> str:
        del exec_id, progress_callback, path
        if ctx is None:
            raise ValueError("SendMessage requires a SessionContext.")
        sender = agent_ctx.agent_name if agent_ctx is not None else USER_SENDER
        if inp.to == "*":
            recipients = sorted(self._recipients - {sender})
            if not recipients:
                return "No other teammates to broadcast to; message not sent."
        elif inp.to in self._recipients:
            recipients = [inp.to]
        else:
            # A rejected send (bad recipient) is raised, not returned: BaseTool wraps
            # it into an is_error ToolErrorInfo, so the model still sees this
            # actionable message AND the failure is flagged (not a silent success).
            valid = ", ".join(sorted(self._recipients)) or "(none)"
            raise ValueError(
                f"No teammate named {inp.to!r}; message not sent. "
                f"Valid teammates: {valid}."
            )
        if isinstance(inp.message, str):
            message = TeamMessage.from_text(
                sender=sender, to=recipients, text=inp.message, reply_to=inp.reply_to
            )
        else:
            # A structured body. For a single recipient that declares an input_type,
            # validate now so the sender gets immediate feedback and the payload is
            # delivered typed; otherwise carry the object as-is (the recipient
            # revalidates / renders it).
            payload: Any = inp.message
            if len(recipients) == 1:
                card = self._cards.get(recipients[0])
                if card is not None and issubclass(card.input_type, BaseModel):
                    try:
                        payload = card.input_type.model_validate(inp.message)
                    except ValidationError as err:
                        # Rejected send (schema mismatch) → is_error tool result, with
                        # the pydantic detail carried in the actionable message.
                        raise ValueError(
                            f"Message to {recipients[0]!r} does not match its "
                            f"expected input ({card.input_type.__name__}): {err}. "
                            "Not sent."
                        ) from err
            message = TeamMessage(
                sender=sender,
                routing=[recipients],
                payloads=[payload],
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

    A **background tool** (``auto_background_at=0``): it sleeps ``delay_seconds`` off
    the member's turn, then its completion is drained as the member's next activation
    — reactivating it with the note. It does not block the member's final answer
    (``blocks_final_answer=False``), so the member parks (or keeps working) while the
    wakeup is pending; the completion wakes it even with no peer traffic. This is the
    initiative lever an idle member pulls to act unprompted (revisit a goal, poll for
    a change, follow up) without holding a live loop open. Durable — the wait runs as
    a background task that resumes with the session (a crash re-runs the sleep).
    Stateless, so one instance is shared across the team.
    """

    def __init__(self) -> None:
        super().__init__(
            name=SCHEDULE_WAKEUP_TOOL_NAME,
            description=(
                "Schedule a message to your future self after a delay and keep "
                "working — you will be reactivated with your note when it fires. Use "
                "it to act on your own initiative later (revisit a goal, poll for a "
                "change, follow up) instead of waiting for someone to message you."
            ),
            auto_background_at=0.0,
            blocks_final_answer=False,
        )

    async def _run(
        self,
        inp: ScheduleWakeupInput,
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> str:
        del ctx, exec_id, progress_callback, path, agent_ctx
        # Backgrounded: sleep the delay, then return the note. The completion is
        # delivered to the calling member as a task notification (no self-addressed
        # mailbox message needed — the bg-task substrate owns delivery + durability).
        await asyncio.sleep(inp.delay_seconds)
        return (
            "<scheduled_wakeup>\n"
            "A wake-up you scheduled earlier has fired. Act on it if there is work to "
            "do; otherwise go back to idle without calling any tools.\n"
            f"{inp.note}\n"
            "</scheduled_wakeup>"
        )
