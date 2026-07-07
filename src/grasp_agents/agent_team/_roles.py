"""
How a host runs each team member.

Two concerns shared by both hosts (the in-process :class:`AgentTeam` and the
per-process ``MemberHost``): *classification* — agent vs plain processor,
resident vs triggered, and whether a resident is currently idle (a host reads
these to partition members and detect quiescence) — and the *triggered activation*
itself (running one member for one inbound message), so that path is identical
across hosts and only each host's event sink / output routing / failure policy
differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.types.events import ProcPacketOutEvent
from grasp_agents.types.message import USER_SENDER, TeamMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from grasp_agents.processors.processor import Processor
    from grasp_agents.runtime import Transport
    from grasp_agents.types.events import Event
    from grasp_agents.types.packet import Packet

    from .agent_card import MemberCard


def is_llm_agent(
    member: Processor[Any, Any, Any],
) -> TypeGuard[LLMAgent[Any, Any, Any]]:
    """
    An LLM agent (vs a plain processor). An agent can be fed a rendered user turn
    (``chat_inputs``); a processor only an input packet — the axis the input-routing
    decision turns on.
    """
    return isinstance(member, LLMAgent)


def is_resident(member: Processor[Any, Any, Any], card: MemberCard | None) -> bool:
    """
    Whether a member runs **resident** — a persistent loop consuming its inbox
    between turns — vs **triggered** (one activation per message). Explicit when its
    card sets ``resident``; otherwise inferred: an LLM agent with no static recipients
    runs resident. (Only an LLM agent can be resident; a processor has no loop.)
    """
    if card is not None and card.resident is not None:
        return card.resident
    return is_llm_agent(member) and not member.recipients


async def activate_member(
    member: Processor[Any, Any, Any],
    message: TeamMessage,
    *,
    transport: Transport[TeamMessage],
    run_kwargs: dict[str, Any],
    push: Callable[[Event[Any]], Awaitable[None]],
    post: Callable[[TeamMessage], Awaitable[None]],
) -> None:
    """
    Run a triggered member for one inbound message and route its output.

    Skips a message already processed (a crash inside ``ack`` left it redelivered
    in the inbox — its effects are durable, so re-running would duplicate them).
    Otherwise renders the message into the member's ``run_stream`` — human content
    as a rendered user turn (``chat_inputs``), anything else as a typed input packet
    through the member's own input pipeline — streams its events through ``push``,
    and routes a produced output packet onward through ``post`` (at default
    priority: a triggered member is never the lead, validated at host
    construction). Raises on a member failure; the caller decides whether that
    stops the run or the message is dropped.
    """
    if await transport.was_processed(member.name, message.message_id):
        return

    if message.sender == USER_SENDER and is_llm_agent(member):
        stream = member.run_stream(chat_inputs=message.to_chat_inputs(), **run_kwargs)
    else:
        stream = member.run_stream(in_packet=message.to_packet(), **run_kwargs)

    out_packet: Packet[Any] | None = None
    async for event in stream:
        await push(event)
        if isinstance(event, ProcPacketOutEvent) and event.source == member.name:
            out_packet = event.data

    if out_packet is not None and out_packet.routing:
        await post(TeamMessage.from_packet(out_packet))


async def resident_idle(member: LLMAgent[Any, Any, Any]) -> bool:
    """
    Whether a resident member is idle: parked on an empty inbox with no answer-
    blocking background work outstanding. The per-actor quiescence signal a host
    folds into its stop decision. A non-answer-blocking task that is merely still
    running (e.g. a backgrounded shell command) does NOT hold it open — same as it
    never blocks a lone agent's final answer.
    """
    inbox = member.agent_ctx.inbox
    if inbox is None or not inbox.is_waiting or await inbox.has_pending():
        return False
    bg = member.agent_ctx.bg_tasks
    return not (bg.has_blocking_tasks or bg.has_undelivered_completions)
