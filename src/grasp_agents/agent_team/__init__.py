"""
Asynchronous peer-messaging teams (experimental).

A team of members — :class:`~grasp_agents.agent.LLMAgent`s and/or plain
:class:`~grasp_agents.processors.processor.Processor`s — that communicate by posting
messages to each other over a shared mailbox
:class:`~grasp_agents.runtime.Transport`
(:class:`~grasp_agents.mailbox.InMemoryMailboxTransport` for a single process, or
:class:`~grasp_agents.mailbox.CheckpointMailboxTransport` for durable delivery over
the session checkpoint store). See ``docs/experimental/agent-team``.
"""

from __future__ import annotations

from .agent_card import MemberCard
from .agent_team import AgentTeam, TeamRunResult
from .events import (
    MessageDeliveredEvent,
    TeamEndedEvent,
    TeamRunInfo,
    TeamStartedEvent,
    TeamStopReason,
)
from .member import MemberDriver
from .message import TeamMessage
from .sources import WakeupScheduler, run_interval_source
from .tools import (
    ScheduleWakeupInput,
    ScheduleWakeupTool,
    SendMessageInput,
    SendMessageTool,
)
from .transport import (
    CheckpointMailboxTransport,
    InMemoryMailboxTransport,
    default_transport,
)

__all__ = [
    "AgentTeam",
    "CheckpointMailboxTransport",
    "InMemoryMailboxTransport",
    "MemberCard",
    "MemberDriver",
    "MessageDeliveredEvent",
    "ScheduleWakeupInput",
    "ScheduleWakeupTool",
    "SendMessageInput",
    "SendMessageTool",
    "TeamEndedEvent",
    "TeamMessage",
    "TeamRunInfo",
    "TeamRunResult",
    "TeamStartedEvent",
    "TeamStopReason",
    "WakeupScheduler",
    "default_transport",
    "run_interval_source",
]
