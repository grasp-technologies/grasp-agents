"""Host a single team member off the shared mailbox + human input (one per process)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from grasp_agents.runtime import ActorDriver, Transport

from ._roles import activate_member, is_llm_agent, is_resident, resident_idle
from .message import CONTROL_PRIORITY, USER_SENDER, TeamMessage
from .prompt import (
    make_rewind_notice,
    make_sender_attribution_attachment,
    make_team_section,
)
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
    from grasp_agents.session_context import SessionContext
    from grasp_agents.tools.base import BaseTool
    from grasp_agents.types.content import InputImage
    from grasp_agents.types.events import Event
    from grasp_agents.types.io import LLMPrompt

    from .agent_card import MemberCard

logger = logging.getLogger(__name__)

# Sentinel marking the end of a resident member's event stream.
_DONE = object()


class MemberHost:
    """
    Runs ONE team member off the shared mailbox, one turn at a time so two turns
    never interleave into its transcript.

    The reusable core behind a per-process member UI — each member lives in its own
    process, built against a shared (durable) transport, all running on the same
    actor runtime as an in-process :class:`~.agent_team.AgentTeam`. The member may be
    an agent **or** a plain ``Processor``, and runs on the *same* execution model the
    in-process team uses: a resident (an agent with no static recipients) runs
    **resident** — one long loop consuming its inbox between turns, so peer and human
    messages enter mid-task; a worker (a processor, or an agent with recipients) is
    **triggered** — one activation per inbound message, handing its output off by
    name. Human input is posted to the same mailbox as control-plane mail (it drains
    ahead of peer messages). For a single-process team, use
    :class:`~.agent_team.AgentTeam` instead.
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
        # Peers a rewind notice goes to: everyone except this member and any peer
        # explicitly carded as triggered (activated fresh per message, so it holds
        # no cross-turn view of the filesystem). ``resident=None`` peers are kept —
        # this host cannot run the residency inference on a remote member, and a
        # spurious notice is cheaper than a missed one.
        self._rewind_notice_peers = [
            c.name for c in cards if c.name != member.name and c.resident is not False
        ]

        # This member's card (its per-member team config: accepted input + role).
        card = next((c for c in cards if c.name == member.name), None)

        if card is not None and card.lead:
            # The lead's role — priority mail, rewind right, rewind announcements
            # — presumes a persistent loop; a triggered member is activated fresh
            # per message and cannot hold it.
            if not is_resident(member, card):
                raise ValueError(
                    f"Member {member.name!r} is carded as the lead but runs "
                    "triggered; the lead must run resident (an LLM agent "
                    "consuming its inbox)."
                )
            # The lead holds its session's environment-rewind right (in a
            # shared-environment deployment, every other process declares it via
            # SessionContext(environment_rewinder=...)); when it rewinds, tell
            # the peers over the shared mailbox so they re-verify state instead
            # of panicking over a filesystem that changed under them.
            member.ctx.claim_environment_rewind(member.name)
            member.ctx.add_environment_restored_callback(
                self._notify_environment_rewind
            )

        # A resident runs a persistent loop; a worker is triggered. Hold the narrowed
        # reference so the resident path can use the LLMAgent-only inbox API.
        self._resident: LLMAgent[Any, Any, Any] | None = None

        # A resident messages peers via SendMessage and schedules its own
        # wakeups via ScheduleWakeup; a worker hands its output off by name and gets
        # no tools.
        if is_resident(member, card):
            self._resident = cast("LLMAgent[Any, Any, Any]", member)
            send_tool = SendMessageTool(
                cards, transport_resolver=lambda _ctx: transport
            )
            wakeup_tool = ScheduleWakeupTool()

            for name, tool in (
                (SEND_MESSAGE_TOOL_NAME, send_tool),
                (SCHEDULE_WAKEUP_TOOL_NAME, wakeup_tool),
            ):
                self._resident.tools[name] = cast("BaseTool[Any, Any, Any]", tool)
                tool.on_adopted(member)
            self._resident.add_system_prompt_section(make_team_section(cards))

        # A triggered member renders a peer hand-off through its own input pipeline,
        # which has no sender fence; give every LLM member the attribution attachment
        # so its turns name the teammate they came from (inert for a resident, which
        # gets attribution from the fence on its drained messages).
        if is_llm_agent(member):
            member.add_input_attachment(make_sender_attribution_attachment())

    @property
    def name(self) -> str:
        return self._member.name

    @property
    def ctx(self) -> SessionContext[Any]:
        """The member's run context (approval store, skills, etc.)."""
        return self._member.ctx

    async def submit_message(
        self, chat_inputs: LLMPrompt | Sequence[str | InputImage]
    ) -> None:
        """
        Deliver human input (text, or a mix of text and images) as this member's next
        turn — the per-process counterpart to :meth:`AgentTeam.submit_message`. Posted
        to the shared mailbox as control-plane mail (``CONTROL_PRIORITY``), so it
        drains ahead of queued peer messages and — over a durable transport —
        survives a restart, exactly like a message from any peer.
        """
        await self._mailbox.post(
            TeamMessage.from_input(
                sender=USER_SENDER,
                to=self._member.name,
                chat_inputs=chat_inputs,
                priority=CONTROL_PRIORITY,
            )
        )

    async def _notify_environment_rewind(self, fs_snapshot_ref: str) -> None:
        """
        Announce this member's environment rewind to every peer, control-plane
        (drains ahead of queued mail). Posted straight to the shared mailbox, so
        it reaches peers in other processes; each peer's own host renders it as
        an inbound turn. Only the lead registers this callback — a rewind by
        anyone else happens in that member's process, not here.
        """
        del fs_snapshot_ref
        if not self._rewind_notice_peers:
            return
        await self._mailbox.post(
            TeamMessage.from_text(
                sender=self._member.name,
                to=self._rewind_notice_peers,
                text=make_rewind_notice(self._member.name),
                priority=CONTROL_PRIORITY,
            )
        )

    async def run_stream(
        self, *, stop_when_idle: bool = False
    ) -> AsyncIterator[Event[Any]]:
        """
        Stream the member's turns until cancelled — or until idle when
        ``stop_when_idle`` (for batch / tests). A resident runs a persistent loop
        (consuming its inbox between turns); a worker is triggered once per inbound
        message. A failing turn is logged and its message dead-lettered; the loop
        keeps serving.
        """
        if self._resident is not None:
            stream = self._run_resident(stop_when_idle=stop_when_idle)
        else:
            stream = self._run_triggered(stop_when_idle=stop_when_idle)
        async for event in stream:
            yield event

    # -- resident (persistent loop) --

    async def _run_resident(self, *, stop_when_idle: bool) -> AsyncIterator[Event[Any]]:
        member = self._resident
        assert member is not None
        member.attach_inbox(self._mailbox)
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def run() -> None:
            try:
                async for event in member.run_stream(**self._run_kwargs):
                    queue.put_nowait(event)
            except Exception:
                logger.warning("Resident member %r failed", member.name, exc_info=True)
            finally:
                # Sync put (unbounded queue) so it lands even while being cancelled.
                queue.put_nowait(_DONE)

        run_task = asyncio.create_task(run(), name=f"resident:{member.name}")
        monitor = (
            asyncio.create_task(
                self._monitor_idle(run_task), name=f"{member.name}-monitor"
            )
            if stop_when_idle
            else None
        )
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                yield item
        finally:
            run_task.cancel()
            if monitor is not None:
                monitor.cancel()
            tasks = [run_task] + ([monitor] if monitor is not None else [])
            await asyncio.gather(*tasks, return_exceptions=True)
            member.detach_inbox()

    async def _monitor_idle(self, run_task: asyncio.Task[None]) -> None:
        """
        End the resident once it is idle (parked on an empty inbox, no background
        work), held across two consecutive polls so a delivery racing an idle
        observation isn't dropped.
        """
        member = self._resident
        assert member is not None
        idle_once = False
        while not run_task.done():
            await asyncio.sleep(self._poll_interval)
            if await resident_idle(member):
                if idle_once:
                    run_task.cancel()
                    return
                idle_once = True
            else:
                idle_once = False

    # -- triggered (worker) --

    async def _run_triggered(
        self, *, stop_when_idle: bool
    ) -> AsyncIterator[Event[Any]]:
        driver: ActorDriver[TeamMessage] = ActorDriver(
            self._mailbox, termination="quiescence" if stop_when_idle else "daemon"
        )
        async with driver:
            driver.register_handler(self._member.name, self._make_handler(driver))
            await driver.settle()
            async for event in driver.stream_events():
                yield event

    def _make_handler(
        self, driver: ActorDriver[TeamMessage]
    ) -> Callable[[TeamMessage], Awaitable[None]]:
        member = self._member

        async def handler(message: TeamMessage) -> None:
            try:
                await activate_member(
                    member,
                    message,
                    transport=self._mailbox,
                    run_kwargs=self._run_kwargs,
                    push=driver.push_to_stream,
                    post=driver.post,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Dead-letter the failure; the per-process host keeps serving.
                logger.warning("Member %r turn failed", member.name, exc_info=True)

        return handler
