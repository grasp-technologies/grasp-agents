from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from pathlib import Path

NAME_REGEX = re.compile(r"^[a-z0-9](?:-?[a-z0-9])*$")
NAME_MAX_LENGTH = 64
DESCRIPTION_MAX_LENGTH = 1024
COMPATIBILITY_MAX_LENGTH = 500


class SkillError(Exception):
    pass


class SkillFormatError(SkillError):
    def __init__(self, path: Path | None, message: str) -> None:
        location = f" [{path}]" if path is not None else ""
        super().__init__(f"{message}{location}")
        self.path = path


class SkillNotFoundError(SkillError):
    pass


class SkillFrontmatter(BaseModel):
    """
    YAML frontmatter for a SKILL.md file.

    Matches the agentskills.io specification: ``name`` and ``description``
    are required; ``license``, ``compatibility``, ``metadata``,
    ``allowed-tools`` are optional. Unknown keys are preserved
    (forward compat with future spec additions).
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str = Field(min_length=1, max_length=NAME_MAX_LENGTH)
    description: str = Field(min_length=1, max_length=DESCRIPTION_MAX_LENGTH)
    license: str | None = None
    compatibility: str | None = Field(default=None, max_length=COMPATIBILITY_MAX_LENGTH)
    metadata: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: str | None = Field(default=None, alias="allowed-tools")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not NAME_REGEX.match(v):
            raise ValueError(
                f"Invalid skill name {v!r}: must contain only lowercase ASCII "
                "letters, digits, and hyphens; cannot start or end with a hyphen "
                "or contain consecutive hyphens"
            )
        return v

    @property
    def grasp_metadata(self) -> dict[str, Any]:
        """Return the ``metadata.grasp`` mapping, or an empty dict."""
        grasp = self.metadata.get("grasp")
        if not isinstance(grasp, dict):
            return {}
        return {str(k): v for k, v in grasp.items()}  # type: ignore[misc]

    @property
    def disable_model_invocation(self) -> bool:
        return bool(self.grasp_metadata.get("disable_model_invocation", False))

    @property
    def inject_body(self) -> bool:
        return bool(self.grasp_metadata.get("inject_body", False))


@dataclass(frozen=True)
class Skill:
    """A loaded skill: parsed frontmatter, markdown body, and source path."""

    frontmatter: SkillFrontmatter
    body: str
    path: Path = field(compare=False)

    @property
    def name(self) -> str:
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def disable_model_invocation(self) -> bool:
        return self.frontmatter.disable_model_invocation

    @property
    def inject_body(self) -> bool:
        return self.frontmatter.inject_body
