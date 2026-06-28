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
import logging
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from grasp_agents.run_context import RunContext, current_run_context
from grasp_agents.runtime import ActorDriver

from ._activation import is_llm_agent, stream_member
from .agent_card import MemberCard
from .events import (
    MessageDeliveredEvent,
    TeamEndedEvent,
    TeamRunInfo,
    TeamStartedEvent,
    TeamStopReason,
)
from .message import USER_SENDER, TeamMessage
from .tools import SEND_MESSAGE_TOOL_NAME, SendMessageTool
from .transport import (
    CheckpointMailboxTransport,
    InMemoryMailboxTransport,
    MailboxChannel,
    MessageTransport,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from grasp_agents.processors.processor import Processor
    from grasp_agents.tools.base import BaseTool
    from grasp_agents.types.events import Event

logger = logging.getLogger(__name__)


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
            if is_llm_agent(member) and not member.recipients:
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
        poll_interval: float = 0.05,
        **run_kwargs: Any,
    ) -> AsyncIterator[Event[Any]]:
        """
        Seed the entry member with ``chat_inputs`` and stream every member's events.

        By default the run ends at **quiescence** (no member running, no mail) — or at
        the ``max_hops`` budget, with mail still pending. With ``daemon=True`` it never
        self-terminates: it keeps serving mail (e.g. from an external source), polling
        every ``poll_interval`` seconds, and dead-letters a member failure rather than
        stopping the team — so one member's crash never takes the team down. Stop a
        daemon by cancelling the stream (break out of the iteration). Events carry
        ``source = <member name>``.
        """
        mailbox = self._transport_for(self._ctx)
        entry = to or self._entry_name
        if entry not in self._members_by_name:
            raise ValueError(f"Recipient {entry!r} is not a team member.")

        driver: ActorDriver[TeamMessage] = ActorDriver(
            MailboxChannel(mailbox, poll_interval=poll_interval),
            termination="daemon" if daemon else "quiescence",
            max_activations=None if daemon else self._max_hops,
        )
        # Members that raised while handling a message — a bounded run stops on this.
        failed: list[str] = []
        seed = (
            TeamMessage.of_text(sender=USER_SENDER, to=entry, text=str(chat_inputs))
            if chat_inputs is not None
            else None
        )

        yield TeamStartedEvent(source=self._name, data=TeamRunInfo(team=self._name))

        try:
            async for event in self._drive(driver, seed, daemon, failed, run_kwargs):
                yield event
        except asyncio.CancelledError:
            # A daemon stopped by the consumer — honor the cancellation; the
            # consumer is gone, so no closing event is emitted.
            raise
        except Exception:
            logger.exception("AgentTeam %s failed", self._name)
            stop_reason = TeamStopReason.ERROR
        else:
            stop_reason = await self._final_stop_reason(
                failed, driver, mailbox, daemon=daemon
            )

        yield TeamEndedEvent(
            source=self._name,
            data=TeamRunInfo(
                team=self._name,
                activations=driver.activation_count,
                stop_reason=stop_reason,
            ),
        )

    async def _drive(
        self,
        driver: ActorDriver[TeamMessage],
        seed: TeamMessage | None,
        daemon: bool,
        failed: list[str],
        run_kwargs: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        async with driver:
            for member in self._members:
                driver.register_handler(
                    member.name,
                    self._make_handler(member, driver, daemon, failed, run_kwargs),
                )
            if seed is not None:
                await driver.post(seed)
            # Seed posted: an empty bounded run quiesces immediately rather than
            # blocking forever on idle consumers.
            await driver.settle()
            async for event in driver.stream_events():
                yield event

    def _make_handler(
        self,
        member: Processor[Any, Any, CtxT],
        driver: ActorDriver[TeamMessage],
        daemon: bool,
        failed: list[str],
        run_kwargs: dict[str, Any],
    ) -> Callable[[TeamMessage], Awaitable[None]]:
        """
        Build a member's activation: announce the delivery, run the member over the
        one inbound message, bubble its events, and hand any routed output off
        through the transport — the same mailbox path a ``SendMessage`` call takes.
        """

        async def handler(message: TeamMessage) -> None:
            await driver.push_to_stream(
                MessageDeliveredEvent(
                    source=message.sender, destination=member.name, data=message
                )
            )
            try:
                out_packet = await stream_member(
                    member, message, push=driver.push_to_stream, run_kwargs=run_kwargs
                )
                if out_packet is not None and out_packet.routing:
                    await driver.post(TeamMessage.from_packet(out_packet))
            except asyncio.CancelledError:
                raise
            except Exception:
                # The driver acks on return, dead-lettering the message rather than
                # retrying it forever. A bounded run records the failure and stops
                # with MEMBER_ERROR; a daemon logs and keeps serving.
                logger.warning(
                    "Team member %r failed handling a message",
                    member.name,
                    exc_info=True,
                )
                failed.append(member.name)
                if not daemon:
                    await driver.shutdown()

        return handler

    async def _final_stop_reason(
        self,
        failed: list[str],
        driver: ActorDriver[TeamMessage],
        mailbox: MessageTransport,
        *,
        daemon: bool,
    ) -> TeamStopReason:
        if failed:
            logger.warning(
                "AgentTeam %s: members failed: %s", self._name, ", ".join(failed)
            )
            return TeamStopReason.MEMBER_ERROR
        if (
            not daemon
            and driver.activation_count >= self._max_hops
            and await self._any_pending(mailbox)
        ):
            logger.warning(
                "AgentTeam %s reached max_hops=%d with mail still pending",
                self._name,
                self._max_hops,
            )
            return TeamStopReason.HOP_BUDGET_EXHAUSTED
        return TeamStopReason.QUIESCED

    async def _any_pending(self, mailbox: MessageTransport) -> bool:
        for member_name in self._members_by_name:
            if await mailbox.has_mail(member_name):
                return True
        return False
