"""
The inter-member message envelope.

The envelope itself now lives in :mod:`grasp_agents.types.message` (a neutral leaf,
so a single agent's durable inbox and a multi-agent host can both carry it); this
module re-exports it for the team API.
"""

from grasp_agents.types.message import (
    CONTROL_PRIORITY,
    USER_SENDER,
    TeamMessage,
)

__all__ = ["CONTROL_PRIORITY", "USER_SENDER", "TeamMessage"]
