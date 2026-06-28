"""Drive a single team member off its own mailbox + human input (one per process)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from .message import format_inbound
from .tools import SEND_MESSAGE_TOOL_NAME, SendMessageTool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from grasp_agents.agent.llm_agent import LLMAgent
    from grasp_agents.run_context import RunContext
    from grasp_agents.tools.base import BaseTool
    from grasp_agents.types.events import Event

    from .agent_card import MemberCard
    from .transport import MessageTransport

logger = logging.getLogger(__name__)


class MemberDriver:
    """
    Runs ONE team member as a single serial inbox: human input and incoming
    mailbox messages feed one loop that runs the agent one turn at a time (human
    input takes priority), so two turns never interleave into its transcript.

    The reusable core behind a per-process member UI — each member lives in its
    own process, building its agent against a shared (file/remote) transport.
    For a single-process team, use :class:`~.agent_team.AgentTeam` instead.
    """

    def __init__(
        self,
        agent: LLMAgent[Any, Any, Any],
        *,
        cards: Sequence[MemberCard],
        transport: MessageTransport,
        poll_interval: float = 0.5,
        run_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._agent = agent
        self._transport = transport
        self._poll_interval = poll_interval
        self._run_kwargs = run_kwargs or {}
        self._human_q: asyncio.Queue[str] = asyncio.Queue()

        tool = cast(
            "BaseTool[Any, Any, Any]",
            SendMessageTool(cards, transport_resolver=lambda _ctx: transport),
        )
        agent.tools[SEND_MESSAGE_TOOL_NAME] = tool
        tool.on_adopted(agent)

    @property
    def name(self) -> str:
        return self._agent.name

    @property
    def ctx(self) -> RunContext[Any]:
        """The member agent's run context (approval store, skills, etc.)."""
        return self._agent.ctx

    def submit_human(self, text: str) -> None:
        """Queue human input as the member's next turn (same event loop)."""
        self._human_q.put_nowait(text)

    async def events(
        self, *, stop_when_idle: bool = False
    ) -> AsyncIterator[Event[Any]]:
        """
        Serial inbox loop: take the next human input (if any) else the next
        mailbox message (one group per turn), run one agent turn, and yield its
        events (acking the consumed message). Runs until cancelled, or returns
        once idle when ``stop_when_idle`` (for batch / tests). A failing turn is
        logged and its message dead-lettered; the loop keeps serving.
        """
        name = self._agent.name
        while True:
            text = self._next_human()
            message = None
            if text is not None:
                chat = text
            else:
                message = await self._transport.fetch_next(name)
                chat = format_inbound(message) if message is not None else None

            if chat is None:
                if stop_when_idle:
                    return
                await self._idle_wait()
                continue

            try:
                async for event in self._agent.run_stream(
                    chat_inputs=chat, **self._run_kwargs
                ):
                    yield event
            except Exception:
                logger.warning("Member %r turn failed", name, exc_info=True)
            finally:
                if message is not None:
                    await self._transport.ack(name, [message.message_id])

    def _next_human(self) -> str | None:
        try:
            return self._human_q.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def _idle_wait(self) -> None:
        # Wake on human input, else time out to re-poll the mailbox.
        try:
            text = await asyncio.wait_for(
                self._human_q.get(), timeout=self._poll_interval
            )
        except TimeoutError:
            return
        self._human_q.put_nowait(text)
