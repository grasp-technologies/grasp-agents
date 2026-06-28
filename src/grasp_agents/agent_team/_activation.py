"""Shared member-activation logic for the in-process team and the per-process driver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

from grasp_agents.types.events import ProcPacketOutEvent

from .message import format_inbound

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from grasp_agents.agent.llm_agent import LLMAgent
    from grasp_agents.processors.processor import Processor
    from grasp_agents.types.events import Event
    from grasp_agents.types.packet import Packet

    from .message import TeamMessage


def is_llm_agent(member: Any) -> TypeGuard[LLMAgent[Any, Any, Any]]:
    """
    An LLM agent (exposes a ``tools`` dict, fed a rendered user turn) vs a plain
    processor (fed an input packet). This is the *activation* axis, orthogonal to
    *routing*: an agent with no static recipients gets the ``SendMessage`` tool and
    routes by messaging (a communicator), while a processor — or an agent that
    declares recipients — hands its output off by name (a worker).
    """
    return isinstance(getattr(member, "tools", None), dict)


async def stream_member(
    member: Processor[Any, Any, Any],
    message: TeamMessage,
    *,
    push: Callable[[Event[Any]], Awaitable[None]],
    run_kwargs: dict[str, Any],
) -> Packet[Any] | None:
    """
    Run one member over one inbound message, pushing its events via ``push``, and
    return its output packet. An agent takes the message as a rendered user turn, a
    processor as an input packet — both are Processors, so both yield a final packet.
    """
    if is_llm_agent(member):
        stream = member.run_stream(chat_inputs=format_inbound(message), **run_kwargs)
    else:
        stream = member.run_stream(in_packet=message.to_packet(), **run_kwargs)
    out_packet: Packet[Any] | None = None
    async for event in stream:
        await push(event)
        if isinstance(event, ProcPacketOutEvent) and event.source == member.name:
            out_packet = event.data
    return out_packet
