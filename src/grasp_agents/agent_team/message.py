"""
The inter-member message envelope.

The envelope itself now lives in :mod:`grasp_agents.types.message` (a neutral leaf,
so a single agent's durable inbox and a multi-agent host can both carry it); this
module re-exports it for the team API.
"""

from grasp_agents.types.message import (
    USER_SENDER,
    TeamMessage,
    format_inbound,
)

__all__ = ["USER_SENDER", "TeamMessage", "format_inbound"]
