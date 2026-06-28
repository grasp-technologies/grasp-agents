"""
``AgentTeam`` — a standalone host for a team of peer agents that communicate
asynchronously by sending each other messages.

A sibling to :class:`~grasp_agents.runner.Runner`: where ``Runner`` drives an
orchestrated graph to a single result packet, ``AgentTeam`` lets members each run
their own loop and message one another via a :class:`~.transport.MessageTransport`.
The session ends at **quiescence** (no member running, no mailbox with mail) or
when the ``max_hops`` budget is exhausted.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, TypeGuard, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from grasp_agents.run_context import RunContext, current_run_context
from grasp_agents.types.events import ProcPacketOutEvent

from .agent_card import MemberCard
from .events import (
    MessageDeliveredEvent,
    TeamEndedEvent,
    TeamRunInfo,
    TeamStartedEvent,
    TeamStopReason,
)
from .message import USER_SENDER, TeamMessage, format_inbound
from .tools import SEND_MESSAGE_TOOL_NAME, SendMessageTool
from .transport import (
    CheckpointMailboxTransport,
    InMemoryMailboxTransport,
    MessageTransport,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from grasp_agents.agent.llm_agent import LLMAgent
    from grasp_agents.processors.processor import Processor
    from grasp_agents.tools.base import BaseTool
    from grasp_agents.types.events import Event
    from grasp_agents.types.packet import Packet

logger = logging.getLogger(__name__)

# Sentinel pushed onto the event queue to close the stream.
_STREAM_END = object()


def _is_llm_agent(member: Any) -> TypeGuard[LLMAgent[Any, Any, Any]]:
    """
    An LLM agent (exposes a ``tools`` dict, fed a rendered user turn) vs a plain
    processor (fed an input packet). This is the *activation* axis, orthogonal to
    *routing*: an agent with no static recipients gets the ``SendMessage`` tool and
    routes by messaging (a communicator), while a processor — or an agent that
    declares recipients — hands its output off by name (a worker).
    """
    return isinstance(getattr(member, "tools", None), dict)


class TeamRunResult(BaseModel):
    """Summary of a completed team session (returned by :meth:`AgentTeam.run`)."""

    messages: list[TeamMessage] = Field(default_factory=list[TeamMessage])
    activations: int = 0
    stop_reason: TeamStopReason = TeamStopReason.QUIESCED


class AgentTeam[CtxT]:
    def __init__(
        self,
        members: Sequence[Processor[Any, Any, CtxT]],
        *,
        entry: Processor[Any, Any, CtxT] | str | None = None,
        cards: Sequence[MemberCard] | None = None,
        ctx: RunContext[CtxT] | None = None,
        transport: MessageTransport | None = None,
        name: str | None = None,
        path: list[str] | None = None,
        max_hops: int = 50,
    ) -> None:
        if not members:
            raise ValueError("AgentTeam requires at least one member.")
        names = [m.name for m in members]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise ValueError(f"Duplicate member names {dups}; names must be unique.")

        self._members = list(members)
        self._members_by_name = {m.name: m for m in self._members}
        self._name = name or f"team-{uuid4().hex[:6]}"
        self._path = path or []
        self._max_hops = max_hops

        # A member with static recipients hands its output off by name; those names
        # must be team members. (A communicator routes dynamically via SendMessage
        # and declares no static recipients, so there is nothing to validate.)
        member_names = set(self._members_by_name)
        for member in self._members:
            for r in member.recipients or []:
                if r not in member_names:
                    raise ValueError(
                        f"Member {member.name!r} routes to unknown recipient "
                        f"{r!r}; recipients must be team members: "
                        f"{', '.join(sorted(member_names))}."
                    )

        if entry is None:
            self._entry_name = names[0]
        elif isinstance(entry, str):
            if entry not in self._members_by_name:
                raise ValueError(f"Entry {entry!r} is not a team member.")
            self._entry_name = entry
        else:
            self._entry_name = entry.name

        self._cards = self._resolve_cards(cards)

        # Bind the session: explicit ctx, else the ambient / process-default one.
        self._ctx: RunContext[CtxT] = (
            ctx if ctx is not None else current_run_context()  # type: ignore[assignment]
        )

        # Resolve the mailbox transport once, shared by the coordinator and every
        # member's SendMessage tool so sends and reads never diverge: an explicit
        # one, else a durable transport over the session checkpoint store, else an
        # in-memory one (single process). A separate-process team passes a
        # CheckpointMailboxTransport over a shared store.
        if transport is not None:
            self._transport: MessageTransport = transport
        else:
            store = self._ctx.checkpoint_store
            self._transport = (
                CheckpointMailboxTransport(store, session_key=self._ctx.session_key)
                if store is not None
                else InMemoryMailboxTransport()
            )

        # Give every member the shared, roster-aware SendMessage tool, then adopt
        # them so ctx / path / tracing cascade down (as Runner adopts its procs).
        # Inject before adoption so the tool is adopted too.
        send_tool = cast(
            "BaseTool[BaseModel, Any, CtxT]",
            SendMessageTool(self._cards, transport_resolver=self._transport_for),
        )
        for member in self._members:
            # A communicator — an agent with no static recipients — gets the
            # SendMessage tool and routes by messaging; a worker (a processor, or
            # an agent that declares recipients) hands its output off by name and
            # gets no tool.
            if _is_llm_agent(member) and not member.recipients:
                member.tools[SEND_MESSAGE_TOOL_NAME] = send_tool
            member.on_adopted(self)

    # -- properties read by Processor.on_adopted's duck-typing (mirror Runner) --

    @property
    def name(self) -> str:
        return self._name

    @property
    def ctx(self) -> RunContext[CtxT]:
        return self._ctx

    @property
    def path(self) -> list[str]:
        return self._path

    def _transport_for(self, _ctx: RunContext[Any]) -> MessageTransport:
        # The team resolves one shared transport in __init__; the tool's resolver
        # signature takes a ctx but the team ignores it.
        return self._transport

    def _resolve_cards(self, cards: Sequence[MemberCard] | None) -> list[MemberCard]:
        if cards is None:
            return [MemberCard(name=n) for n in self._members_by_name]
        unknown = sorted({c.name for c in cards} - set(self._members_by_name))
        if unknown:
            raise ValueError(f"Cards reference non-members {unknown}.")
        provided = {c.name: c for c in cards}
        # Fill any unspecified member with a name-only card so the roster is whole.
        return [provided.get(n, MemberCard(name=n)) for n in self._members_by_name]

    # -- lifecycle (session-scoped; the embedder closes the team) --

    async def aclose(self) -> None:
        for member in self._members:
            try:
                await member.aclose()
            except Exception:
                logger.warning(
                    "Failed to close team member %r during teardown",
                    member.name,
                    exc_info=True,
                )

    async def __aenter__(self) -> AgentTeam[CtxT]:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- running --

    async def run(
        self, chat_inputs: Any = None, *, to: str | None = None, **run_kwargs: Any
    ) -> TeamRunResult:
        """Drive the team to quiescence and summarize the session."""
        messages: list[TeamMessage] = []
        activations = 0
        stop_reason = TeamStopReason.QUIESCED
        async for event in self.run_stream(chat_inputs, to=to, **run_kwargs):
            if isinstance(event, MessageDeliveredEvent):
                messages.append(event.data)
            elif isinstance(event, TeamEndedEvent):
                activations = event.data.activations
                stop_reason = event.data.stop_reason or TeamStopReason.QUIESCED
        return TeamRunResult(
            messages=messages, activations=activations, stop_reason=stop_reason
        )

    async def run_stream(
        self,
        chat_inputs: Any = None,
        *,
        to: str | None = None,
        daemon: bool = False,
        poll_interval: float = 0.5,
        **run_kwargs: Any,
    ) -> AsyncIterator[Event[Any]]:
        """
        Seed the entry member with ``chat_inputs`` and stream every member's events.

        By default the run ends at **quiescence** (no member running, no mail). With
        ``daemon=True`` it never self-terminates: on quiescence it idle-polls every
        ``poll_interval`` seconds for new mail (e.g. from an external source), and a
        member failure is dead-lettered and logged rather than stopping the team —
        so one member's crash never takes the team down. Stop a daemon by cancelling
        the stream (break out of the iteration). Events carry
        ``source = <member name>``.
        """
        transport = self._transport_for(self._ctx)
        entry = to or self._entry_name
        if entry not in self._members_by_name:
            raise ValueError(f"Recipient {entry!r} is not a team member.")

        if chat_inputs is not None:
            await transport.send(
                TeamMessage.of_text(
                    sender=USER_SENDER, to=entry, text=str(chat_inputs)
                )
            )

        out: asyncio.Queue[Event[Any] | object] = asyncio.Queue()
        coordinator = asyncio.create_task(
            self._coordinate(transport, out, run_kwargs, daemon, poll_interval)
        )
        try:
            while True:
                item = await out.get()
                if item is _STREAM_END:
                    break
                yield cast("Event[Any]", item)
        finally:
            if not coordinator.done():
                coordinator.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await coordinator

    async def _coordinate(
        self,
        transport: MessageTransport,
        out: asyncio.Queue[Event[Any] | object],
        run_kwargs: dict[str, Any],
        daemon: bool = False,
        poll_interval: float = 0.5,
    ) -> None:
        running: dict[str, asyncio.Task[str | None]] = {}
        activations = 0
        stop_reason = TeamStopReason.QUIESCED
        await out.put(
            TeamStartedEvent(source=self._name, data=TeamRunInfo(team=self._name))
        )
        try:
            activations, stop_reason = await self._drive(
                transport, out, running, run_kwargs, daemon, poll_interval
            )
        except asyncio.CancelledError:
            stop_reason = TeamStopReason.CANCELLED
            raise
        except Exception:
            stop_reason = TeamStopReason.ERROR
            logger.exception("AgentTeam %s coordinator failed", self._name)
        finally:
            await self._shutdown(out, running, activations, stop_reason)

    async def _drive(
        self,
        transport: MessageTransport,
        out: asyncio.Queue[Event[Any] | object],
        running: dict[str, asyncio.Task[str | None]],
        run_kwargs: dict[str, Any],
        daemon: bool = False,
        poll_interval: float = 0.5,
    ) -> tuple[int, TeamStopReason]:
        activations = 0
        failed: list[str] = []
        while True:
            if daemon or activations < self._max_hops:
                for member_name in self._members_by_name:
                    if member_name in running:
                        continue
                    message = await transport.fetch_next(member_name)
                    if message is None:
                        continue
                    activations += 1
                    await self._announce(out, member_name, message)
                    running[member_name] = asyncio.create_task(
                        self._activate(
                            member_name, message, transport, out, run_kwargs
                        )
                    )
                    if not daemon and activations >= self._max_hops:
                        break
            if not running:
                if daemon:
                    # Idle: wait for new mail (e.g. from an external source) rather
                    # than quiescing; cancelled when the consumer stops the stream.
                    await asyncio.sleep(poll_interval)
                    continue
                break
            done, _ = await asyncio.wait(
                running.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                finished = next(n for n, t in running.items() if t is task)
                del running[finished]
                # A daemon dead-letters a failed member (its message was acked in
                # _activate) and keeps serving; a bounded run records it to stop
                # the run with MEMBER_ERROR.
                if task.result() is not None and not daemon:
                    failed.append(finished)

        if failed:
            logger.warning(
                "AgentTeam %s: members failed: %s", self._name, ", ".join(failed)
            )
            return activations, TeamStopReason.MEMBER_ERROR
        if activations >= self._max_hops and await self._any_pending(transport):
            logger.warning(
                "AgentTeam %s reached max_hops=%d with mail still pending",
                self._name,
                self._max_hops,
            )
            return activations, TeamStopReason.HOP_BUDGET_EXHAUSTED
        return activations, TeamStopReason.QUIESCED

    @staticmethod
    async def _announce(
        out: asyncio.Queue[Event[Any] | object],
        member_name: str,
        message: TeamMessage,
    ) -> None:
        await out.put(
            MessageDeliveredEvent(
                source=message.sender, destination=member_name, data=message
            )
        )

    async def _shutdown(
        self,
        out: asyncio.Queue[Event[Any] | object],
        running: dict[str, asyncio.Task[str | None]],
        activations: int,
        stop_reason: TeamStopReason,
    ) -> None:
        for task in running.values():
            task.cancel()
        for task in running.values():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        with contextlib.suppress(Exception):
            await out.put(
                TeamEndedEvent(
                    source=self._name,
                    data=TeamRunInfo(
                        team=self._name,
                        activations=activations,
                        stop_reason=stop_reason,
                    ),
                )
            )
            await out.put(_STREAM_END)

    async def _activate(
        self,
        member_name: str,
        message: TeamMessage,
        transport: MessageTransport,
        out: asyncio.Queue[Event[Any] | object],
        run_kwargs: dict[str, Any],
    ) -> str | None:
        """Run one member over one inbound message; return its name on failure."""
        failed: str | None = None
        try:
            out_packet = await self._run_member(member_name, message, out, run_kwargs)
            # Hand the output off by name — the same mailbox path a SendMessage
            # call takes, performed for a member that has no tool to call it. A
            # communicator routes by sending mid-run and carries no routing here
            # (no static recipients); a worker's output carries its routing.
            if out_packet is not None and out_packet.routing:
                await transport.send(TeamMessage.from_packet(out_packet))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Team member %r failed handling a message", member_name, exc_info=True
            )
            failed = member_name
        finally:
            # Ack regardless of outcome: a failed message is dead-lettered to
            # ``processed/`` rather than retried forever. The failure surfaces
            # in the run's ``stop_reason``.
            await transport.ack(member_name, [message.message_id])
        return failed

    async def _run_member(
        self,
        member_name: str,
        message: TeamMessage,
        out: asyncio.Queue[Event[Any] | object],
        run_kwargs: dict[str, Any],
    ) -> Packet[Any] | None:
        """
        Activate one member over one message, bubbling events; return its output
        packet. An agent takes the message as a rendered user turn, a processor as
        an input packet; both are Processors, so both yield a final output packet.
        """
        member = self._members_by_name[member_name]
        if _is_llm_agent(member):
            stream = member.run_stream(
                chat_inputs=format_inbound(message), **run_kwargs
            )
        else:
            stream = member.run_stream(in_packet=message.to_packet(), **run_kwargs)
        out_packet: Packet[Any] | None = None
        async for event in stream:
            await out.put(event)
            if isinstance(event, ProcPacketOutEvent) and event.source == member_name:
                out_packet = event.data
        return out_packet

    async def _any_pending(self, transport: MessageTransport) -> bool:
        for member_name in self._members_by_name:
            if await transport.has_mail(member_name):
                return True
        return False
