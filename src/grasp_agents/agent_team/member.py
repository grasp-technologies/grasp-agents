"""Drive a single team member off its own mailbox + human input (one per process)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from grasp_agents.runtime import CLOSED, ActorDriver, Closed, Transport

from ._activation import is_llm_agent, stream_member
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

    from grasp_agents.processors.processor import Processor
    from grasp_agents.run_context import RunContext
    from grasp_agents.tools.base import BaseTool
    from grasp_agents.types.events import Event

    from .agent_card import MemberCard

logger = logging.getLogger(__name__)


class _PrioritizedInbox(Transport[TeamMessage]):
    """
    Human input and the member's mailbox as one runtime channel, human first.

    A genuine composite over the mailbox :class:`~grasp_agents.runtime.Transport`
    (not an adapter): human turns are queued in memory and jump ahead of mailbox
    mail; they are not persisted and need no ack (they never entered the mailbox).
    Mailbox mail is consumed and acked by the driver after the turn — at-least-once,
    like any team member.
    """

    def __init__(
        self,
        mailbox: Transport[TeamMessage],
        human_q: asyncio.Queue[str],
        *,
        poll_interval: float,
    ) -> None:
        self._mailbox = mailbox
        self._human_q = human_q
        self._poll_interval = poll_interval
        self._closed = asyncio.Event()
        self._human_ids: set[str] = set()

    def register(self, recipient: str) -> None:
        self._mailbox.register(recipient)

    async def post(self, envelope: TeamMessage) -> None:
        await self._mailbox.post(envelope)

    async def consume(self, recipient: str) -> TeamMessage | Closed:
        while not self._closed.is_set():
            human = self._take_human(recipient)
            if human is not None:
                return human
            if await self._mailbox.has_pending(recipient):
                return await self._mailbox.consume(recipient)
            # Idle: wake on human input, else time out to re-poll the mailbox.
            try:
                text = await asyncio.wait_for(
                    self._human_q.get(), timeout=self._poll_interval
                )
                self._human_q.put_nowait(text)
            except TimeoutError:
                pass
        return CLOSED

    def _take_human(self, recipient: str) -> TeamMessage | None:
        try:
            text = self._human_q.get_nowait()
        except asyncio.QueueEmpty:
            return None
        message = TeamMessage.of_text(sender=USER_SENDER, to=recipient, text=text)
        self._human_ids.add(message.message_id)
        return message

    async def ack(self, recipient: str, envelope: TeamMessage) -> None:
        if envelope.message_id in self._human_ids:
            self._human_ids.discard(envelope.message_id)
        else:
            await self._mailbox.ack(recipient, envelope)

    async def has_pending(self, recipient: str) -> bool:
        return not self._human_q.empty() or await self._mailbox.has_pending(recipient)

    async def shutdown(self) -> None:
        self._closed.set()


class MemberDriver:
    """
    Runs ONE team member as a single serial inbox: human input and incoming mailbox
    messages feed one loop that runs the member one turn at a time (human input takes
    priority), so two turns never interleave into its transcript.

    The reusable core behind a per-process member UI — each member lives in its own
    process, built against a shared (durable) transport, all running on the same
    actor runtime as an in-process :class:`~.agent_team.AgentTeam`. The member may be
    an agent **or** a plain ``Processor``: a communicator agent (no static recipients)
    gets a ``SendMessage`` tool and messages peers; a worker (a processor, or an agent
    with recipients) reacts to inbound packets and hands its output off by name. Both
    reach peers in other processes through the shared mailbox. For a single-process
    team, use :class:`~.agent_team.AgentTeam` instead.
    """

    def __init__(
        self,
        member: Processor[Any, Any, Any],
        *,
        cards: Sequence[MemberCard],
        transport: Transport[TeamMessage],
        poll_interval: float = 0.5,
        run_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._member = member
        self._mailbox = transport
        self._poll_interval = poll_interval
        self._run_kwargs = run_kwargs or {}
        self._human_q: asyncio.Queue[str] = asyncio.Queue()
        self._scheduler = WakeupScheduler(transport)

        # A communicator (an agent with no static recipients) messages peers via
        # SendMessage and schedules its own wakeups via ScheduleWakeup; a worker
        # hands its output off by name and gets no tools.
        if is_llm_agent(member) and not member.recipients:
            send_tool = cast(
                "BaseTool[Any, Any, Any]",
                SendMessageTool(cards, transport_resolver=lambda _ctx: transport),
            )
            wakeup_tool = cast(
                "BaseTool[Any, Any, Any]", ScheduleWakeupTool(self._scheduler)
            )
            for name, tool in (
                (SEND_MESSAGE_TOOL_NAME, send_tool),
                (SCHEDULE_WAKEUP_TOOL_NAME, wakeup_tool),
            ):
                member.tools[name] = tool
                tool.on_adopted(member)

    @property
    def name(self) -> str:
        return self._member.name

    @property
    def ctx(self) -> RunContext[Any]:
        """The member's run context (approval store, skills, etc.)."""
        return self._member.ctx

    def submit_human(self, text: str) -> None:
        """Queue human input as the member's next turn (same event loop)."""
        self._human_q.put_nowait(text)

    async def events(
        self, *, stop_when_idle: bool = False
    ) -> AsyncIterator[Event[Any]]:
        """
        Stream the member's turns. Each activation takes the next human input (if
        any) else the next mailbox message — one group per turn — runs one turn, and
        hands any routed output off through the transport. Runs until cancelled, or
        returns once idle when ``stop_when_idle`` (for batch / tests). A failing turn
        is logged and its message dead-lettered; the loop keeps serving.
        """
        channel = _PrioritizedInbox(
            self._mailbox, self._human_q, poll_interval=self._poll_interval
        )
        driver: ActorDriver[TeamMessage] = ActorDriver(
            channel, termination="quiescence" if stop_when_idle else "daemon"
        )
        try:
            async with driver:
                driver.register_handler(self._member.name, self._make_handler(driver))
                await driver.settle()
                async for event in driver.stream_events():
                    yield event
        finally:
            # Drop any wakeup the member scheduled but that has not yet fired, so a
            # stopped driver leaves no timer firing into a torn-down transport.
            await self._scheduler.aclose()

    def _make_handler(
        self, driver: ActorDriver[TeamMessage]
    ) -> Callable[[TeamMessage], Awaitable[None]]:
        member = self._member

        async def handler(message: TeamMessage) -> None:
            try:
                out_packet = await stream_member(
                    member,
                    message,
                    push=driver.push_to_stream,
                    run_kwargs=self._run_kwargs,
                )
                if out_packet is not None and out_packet.routing:
                    await driver.post(TeamMessage.from_packet(out_packet))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Member %r turn failed", member.name, exc_info=True)

        return handler
