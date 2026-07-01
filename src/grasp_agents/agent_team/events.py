"""Streamed events for team lifecycle and message delivery."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from grasp_agents.types.events import Event, RoutedEvent

from .message import TeamMessage


class TeamStopReason(StrEnum):
    """Why a team run ended."""

    QUIESCED = "quiesced"  # no member running and no mail left
    HOP_BUDGET_EXHAUSTED = "hop_budget_exhausted"  # max_hops reached, mail pending
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"  # max_tokens reached
    MEMBER_ERROR = "member_error"  # a member's run raised
    CANCELLED = "cancelled"  # the consumer stopped the stream
    ERROR = "error"  # the coordinator itself failed


class TeamRunInfo(BaseModel):
    team: str
    activations: int = 0
    stop_reason: TeamStopReason | None = None


class TeamStartedEvent(Event[TeamRunInfo], frozen=True):
    type: Literal["team.started"] = "team.started"


class TeamEndedEvent(Event[TeamRunInfo], frozen=True):
    type: Literal["team.ended"] = "team.ended"


class MessageDeliveredEvent(RoutedEvent[TeamMessage], frozen=True):
    """A queued message handed to its recipient as the input of a fresh turn."""

    type: Literal["team.message_delivered"] = "team.message_delivered"
