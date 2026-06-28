"""
Asynchronous peer-messaging teams (experimental).

A team of members — :class:`~grasp_agents.agent.LLMAgent`s and/or plain
:class:`~grasp_agents.processors.processor.Processor`s — that communicate by
sending each other messages via a pluggable :class:`MessageTransport` (in-memory
for a single process, or durable over the session checkpoint store). See
``docs/experimental/agent-team``.
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
from .sources import run_interval_source
from .tools import SendMessageInput, SendMessageTool
from .transport import (
    CheckpointMailboxTransport,
    InMemoryMailboxTransport,
    MessageTransport,
    default_transport,
)

__all__ = [
    "AgentTeam",
    "CheckpointMailboxTransport",
    "InMemoryMailboxTransport",
    "MemberCard",
    "MemberDriver",
    "MessageDeliveredEvent",
    "MessageTransport",
    "SendMessageInput",
    "SendMessageTool",
    "TeamEndedEvent",
    "TeamMessage",
    "TeamRunInfo",
    "TeamRunResult",
    "TeamStartedEvent",
    "TeamStopReason",
    "default_transport",
    "run_interval_source",
]
