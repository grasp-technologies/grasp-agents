"""
``AgentTeam`` — a standalone host for a team of peer agents that communicate
asynchronously by sending each other messages.

A sibling to :class:`~grasp_agents.runner.Runner` over the **same** actor runtime
(:class:`~grasp_agents.runtime.ActorDriver` + :class:`~grasp_agents.runtime.
Transport`): where ``Runner`` drives an orchestrated graph to a single result
packet, ``AgentTeam`` lets members each run their own loop and message one another
over a shared mailbox ``Transport`` (in-memory or durable). A *communicator* (an
agent with no static recipients) runs **resident** — one long-lived loop consuming
its inbox between turns; a *transform* (a processor, or an agent with static
recipients) is **triggered** by the shared :class:`ActorDriver`, one activation per
message. Both consume the one transport. The session ends at **quiescence** (no
member running, every inbox empty, no background work) or at the ``max_hops`` budget.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from grasp_agents.inbox import AgentInbox
from grasp_agents.mailbox import CheckpointMailboxTransport, InMemoryMailboxTransport
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
from .sources import WakeupScheduler
from .tools import (
    SCHEDULE_WAKEUP_TOOL_NAME,
    SEND_MESSAGE_TOOL_NAME,
    ScheduleWakeupTool,
    SendMessageTool,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from grasp_agents.agent.llm_agent import LLMAgent
    from grasp_agents.processors.processor import Processor
    from grasp_agents.runtime import Transport
    from grasp_agents.tools.base import BaseTool
    from grasp_agents.types.events import Event

logger = logging.getLogger(__name__)

# A trivial seed satisfying a resident run's input requirement. The resident wait
# fires before any generation, so the agent never reacts to it — it is context, and
# the first real turn is the first inbox message.
_SEED_TEXT = "Team session started."


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
        transport: Transport[TeamMessage] | None = None,
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

        # The one shared mailbox Transport every member views — communicators
        # consume it via their AgentInbox, transforms via the ActorDriver. Explicit,
        # else durable over the session store, else in-memory (single process).
        if transport is not None:
            self._transport: Transport[TeamMessage] = transport
        else:
            store = self._ctx.checkpoint_store
            self._transport = (
                CheckpointMailboxTransport(store, session_key=self._ctx.session_key)
                if store is not None
                else InMemoryMailboxTransport()
            )

        # Partition members by execution mode: a communicator (agent, no static
        # recipients) runs resident; everything else (processors, agents with
        # recipients) is a triggered transform.
        self._communicators: dict[str, LLMAgent[Any, Any, Any]] = {}
        self._transforms: dict[str, Processor[Any, Any, CtxT]] = {}
        for member in self._members:
            if is_llm_agent(member) and not member.recipients:
                self._communicators[member.name] = member
            else:
                self._transforms[member.name] = member

        # The self-wakeup timer source routes through the team (so a wakeup is
        # counted + announced like any delivery); cancelled in aclose.
        self._scheduler = WakeupScheduler(self)

        # Give every communicator the shared messaging + self-wakeup tools, then
        # adopt all members so ctx / path / tracing cascade down (as Runner adopts
        # its procs). SendMessage routes through the team (``self`` is the sink).
        send_tool = cast(
            "BaseTool[BaseModel, Any, CtxT]",
            SendMessageTool(self._cards, transport_resolver=lambda _ctx: self),
        )
        wakeup_tool = cast(
            "BaseTool[BaseModel, Any, CtxT]", ScheduleWakeupTool(self._scheduler)
        )
        for communicator in self._communicators.values():
            communicator.tools[SEND_MESSAGE_TOOL_NAME] = send_tool
            communicator.tools[SCHEDULE_WAKEUP_TOOL_NAME] = wakeup_tool
        for member in self._members:
            member.on_adopted(self)

        # Per-run state (reset in run_stream).
        self._driver: ActorDriver[TeamMessage] | None = None
        self._activations = 0
        self._failed: list[str] = []
        self._hop_exhausted = False
        self._stop_requested = False
        self._daemon = False
        self._poll_interval = 0.05
        self._resident_tasks: list[asyncio.Task[None]] = []

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

    def _resolve_cards(self, cards: Sequence[MemberCard] | None) -> list[MemberCard]:
        if cards is None:
            return [MemberCard(name=n) for n in self._members_by_name]
        unknown = sorted({c.name for c in cards} - set(self._members_by_name))
        if unknown:
            raise ValueError(f"Cards reference non-members {unknown}.")
        provided = {c.name: c for c in cards}
        # Fill any unspecified member with a name-only card so the roster is whole.
        return [provided.get(n, MemberCard(name=n)) for n in self._members_by_name]

    # -- routing (the team is the MessageSink every send goes through) --

    async def post(self, envelope: TeamMessage) -> None:
        """
        Route a message to its recipient(s): announce the delivery, count it against
        the hop budget, then deposit it on the shared transport (where the
        recipient's resident loop or the driver picks it up). This is the single
        interception point — ``SendMessage``, processor hand-off, the entry seed, and
        external sources / wakeups all go through it.

        Outside a live run (no driver) the post is dropped rather than half-applied —
        e.g. a wakeup timer that fires after the run quiesced must not mutate the next
        run's accounting or deposit an orphan message no one consumes.
        """
        if self._driver is None:
            logger.debug(
                "AgentTeam %s: post outside a live run; dropping message from %r",
                self._name,
                envelope.sender,
            )
            return
        for single in envelope.split_by_recipient():
            recipient = single.recipient
            if recipient not in self._members_by_name:
                logger.warning(
                    "AgentTeam %s dropping message to unknown recipient %r",
                    self._name,
                    recipient,
                )
                continue
            if not self._daemon and self._activations >= self._max_hops:
                # Budget spent: refuse further deliveries and ask the run to stop.
                self._hop_exhausted = True
                self._stop_requested = True
                return
            self._activations += 1
            await self._push(
                MessageDeliveredEvent(
                    source=single.sender, destination=recipient, data=single
                )
            )
            await self._transport.post(single)

    async def _push(self, event: Event[Any]) -> None:
        driver = self._driver
        if driver is not None:
            await driver.push_to_stream(event)

    # -- lifecycle (session-scoped; the embedder closes the team) --

    async def aclose(self) -> None:
        await self._scheduler.aclose()
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

        By default the run ends at **quiescence** (no member running, every inbox
        empty, no background work) — or at the ``max_hops`` budget. With
        ``daemon=True`` it never self-terminates: it keeps serving mail (e.g. from an
        external source), and dead-letters a member failure rather than stopping the
        team. Stop a daemon by cancelling the stream. Events carry ``source = <member
        name>``.
        """
        entry = to or self._entry_name
        if entry not in self._members_by_name:
            raise ValueError(f"Recipient {entry!r} is not a team member.")

        self._daemon = daemon
        self._poll_interval = poll_interval
        self._activations = 0
        self._failed = []
        self._hop_exhausted = False
        self._stop_requested = False
        self._resident_tasks = []
        # Cleared until the new driver is wired (a few lines down), so a source /
        # wakeup racing the run start is dropped by ``post`` rather than landing on a
        # stale driver from a prior run.
        self._driver = None

        yield TeamStartedEvent(source=self._name, data=TeamRunInfo(team=self._name))

        stop_reason = TeamStopReason.QUIESCED
        # Daemon mode: the team owns quiescence + the hop budget, so the driver never
        # self-terminates — it just runs the triggered transforms and merges events.
        driver: ActorDriver[TeamMessage] = ActorDriver(
            self._transport, termination="daemon"
        )
        self._driver = driver
        try:
            async for event in self._drive(driver, entry, chat_inputs, run_kwargs):
                yield event
        except asyncio.CancelledError:
            self._driver = None
            raise
        except Exception:
            logger.exception("AgentTeam %s failed", self._name)
            stop_reason = TeamStopReason.ERROR
        else:
            stop_reason = self._final_stop_reason()
        self._driver = None

        yield TeamEndedEvent(
            source=self._name,
            data=TeamRunInfo(
                team=self._name,
                activations=self._activations,
                stop_reason=stop_reason,
            ),
        )

    async def _drive(
        self,
        driver: ActorDriver[TeamMessage],
        entry: str,
        chat_inputs: Any,
        run_kwargs: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        async with driver:
            # Triggered transforms run on the shared driver; resident communicators
            # run their own loop off the same transport.
            for member in self._transforms.values():
                driver.register_handler(
                    member.name, self._make_transform_handler(member, run_kwargs)
                )
            for member_name, communicator in self._communicators.items():
                communicator.inbox = AgentInbox(
                    transport=self._transport, recipient=member_name
                )
                self._resident_tasks.append(
                    asyncio.create_task(
                        self._run_communicator(communicator, run_kwargs),
                        name=f"resident:{member_name}",
                    )
                )
            monitor = asyncio.create_task(
                self._monitor(driver), name=f"{self._name}-monitor"
            )
            try:
                if chat_inputs is not None:
                    await self.post(
                        TeamMessage.of_text(
                            sender=USER_SENDER, to=entry, text=str(chat_inputs)
                        )
                    )
                async for event in driver.stream_events():
                    yield event
            finally:
                monitor.cancel()
                for task in self._resident_tasks:
                    task.cancel()
                await asyncio.gather(
                    monitor, *self._resident_tasks, return_exceptions=True
                )
                # Cancel any wakeup timer still pending so it can't fire into a dead
                # (or a later) run. The scheduler object persists (the tools hold it);
                # only its pending timers are dropped, so a later run schedules anew.
                await self._scheduler.aclose()
                for communicator in self._communicators.values():
                    communicator.inbox = None

    async def _run_communicator(
        self, communicator: LLMAgent[Any, Any, Any], run_kwargs: dict[str, Any]
    ) -> None:
        """
        Run one communicator resident: a single seeded ``run_stream`` whose loop
        consumes its inbox between turns and ends only when this task is cancelled
        (at quiescence / daemon stop). Its events bubble to the team stream.
        """
        try:
            async for event in communicator.run_stream(_SEED_TEXT, **run_kwargs):
                await self._push(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Resident member %r failed", communicator.name, exc_info=True
            )
            self._failed.append(communicator.name)
            if not self._daemon:
                self._stop_requested = True

    def _make_transform_handler(
        self, member: Processor[Any, Any, CtxT], run_kwargs: dict[str, Any]
    ) -> Callable[[TeamMessage], Awaitable[None]]:
        """
        A triggered member's activation: run it over the one inbound message, bubble
        its events, and route any output back through the team.
        """

        async def handler(message: TeamMessage) -> None:
            try:
                out_packet = await stream_member(
                    member, message, push=self._push, run_kwargs=run_kwargs
                )
                if out_packet is not None and out_packet.routing:
                    await self.post(TeamMessage.from_packet(out_packet))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Team member %r failed handling a message",
                    member.name,
                    exc_info=True,
                )
                self._failed.append(member.name)
                if not self._daemon:
                    self._stop_requested = True

        return handler

    async def _monitor(self, driver: ActorDriver[TeamMessage]) -> None:
        """
        Detect quiescence (or a stop request) and tear the run down: cancel the
        resident loops and shut the driver, which ends the event stream.

        Quiescence must hold across two consecutive polls with no new activation in
        between — a cheap guard against the small window between an idle observation
        and a delivery that races it.
        """
        last_idle_activations: int | None = None
        while True:
            await asyncio.sleep(self._poll_interval)
            if self._stop_requested:
                break
            if self._daemon:
                continue
            if not await self._is_quiescent(driver):
                last_idle_activations = None
                continue
            if last_idle_activations == self._activations:
                break
            last_idle_activations = self._activations

        for task in self._resident_tasks:
            task.cancel()
        await driver.shutdown()

    async def _is_quiescent(self, driver: ActorDriver[TeamMessage]) -> bool:
        for communicator in self._communicators.values():
            inbox = communicator.inbox
            if inbox is None or not inbox.is_waiting:
                return False
            if await inbox.has_pending():
                return False
            if self._bg_busy(communicator):
                return False
        return await driver.is_quiescent()

    @staticmethod
    def _bg_busy(communicator: LLMAgent[Any, Any, Any]) -> bool:
        # Mirror the loop's own quiescence notion: answer-blocking work outstanding
        # (``has_pending``) or a completion still waiting to be drained
        # (``has_undelivered_completions``). A non-answer-blocking task that is merely
        # still running (e.g. a backgrounded shell command) must NOT hold the team
        # open — same as it never blocks a lone agent's final answer.
        bg = communicator.background_tasks
        return bg.has_pending or bg.has_undelivered_completions

    def _final_stop_reason(self) -> TeamStopReason:
        if self._failed:
            logger.warning(
                "AgentTeam %s: members failed: %s",
                self._name,
                ", ".join(self._failed),
            )
            return TeamStopReason.MEMBER_ERROR
        if self._hop_exhausted:
            logger.warning(
                "AgentTeam %s reached max_hops=%d with mail still pending",
                self._name,
                self._max_hops,
            )
            return TeamStopReason.HOP_BUDGET_EXHAUSTED
        return TeamStopReason.QUIESCED
