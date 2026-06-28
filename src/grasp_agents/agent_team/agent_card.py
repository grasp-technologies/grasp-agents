"""Roster / capability advertisement for a team member."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MemberCard(BaseModel):
    """
    A team member's advertised identity — the roster entry peers see when
    deciding whom to message.

    A trimmed analogue of the Agent2Agent (A2A) protocol's Agent Card: it keeps
    the fields meaningful for in-process discovery (``name`` / ``description`` /
    ``skills``) and omits the transport-specific ones (service URL, security
    schemes, signature) a networked transport would add.
    """

    name: str
    description: str = ""
    skills: list[str] = Field(default_factory=list)
    version: str = "0.1.0"

    def render(self) -> str:
        """A one-line roster entry for the ``SendMessage`` tool description."""
        line = self.name
        if self.description:
            line += f": {self.description}"
        if self.skills:
            line += f" (skills: {', '.join(self.skills)})"
        return line
