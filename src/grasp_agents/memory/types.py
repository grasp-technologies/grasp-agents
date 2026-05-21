from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from pathlib import Path

NAME_REGEX = re.compile(r"^[a-z0-9](?:[-_]?[a-z0-9])*$")
NAME_MAX_LENGTH = 64
DESCRIPTION_MAX_LENGTH = 1024

MemoryType: TypeAlias = Literal["user", "feedback", "project", "reference"]
MEMORY_TYPES: tuple[MemoryType, ...] = get_args(MemoryType)


class MemoryError(Exception):  # noqa: A001
    pass


class MemoryFormatError(MemoryError):
    def __init__(self, path: Path | None, message: str) -> None:
        location = f" [{path}]" if path is not None else ""
        super().__init__(f"{message}{location}")
        self.path = path


class MemoryNotFoundError(MemoryError):
    pass


class MemoryFrontmatter(BaseModel):
    """
    YAML frontmatter for a topic memory file.

    Inspired by Claude Code's memdir schema. ``name`` and ``description`` are
    required; ``type`` is one of ``user|feedback|project|reference`` (closed
    taxonomy with graceful degradation — unknown values are preserved as
    ``raw_type`` but :attr:`memory_type` returns ``None``). Recency comes
    from filesystem ``mtime``, not a frontmatter field.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str = Field(min_length=1, max_length=NAME_MAX_LENGTH)
    description: str = Field(min_length=1, max_length=DESCRIPTION_MAX_LENGTH)
    raw_type: str | None = Field(default=None, alias="type")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not NAME_REGEX.match(v):
            raise ValueError(
                f"Invalid memory name {v!r}: must contain only lowercase ASCII "
                "letters, digits, hyphens, and underscores; cannot start or "
                "end with a separator or contain consecutive separators"
            )
        return v

    @property
    def memory_type(self) -> MemoryType | None:
        """Parsed type or ``None`` if missing/unknown (graceful degradation)."""
        if self.raw_type in MEMORY_TYPES:
            return self.raw_type  # type: ignore[return-value]
        return None


@dataclass(frozen=True)
class MemoryEntry:
    """A loaded topic memory file: frontmatter + body + path + mtime."""

    frontmatter: MemoryFrontmatter
    body: str
    path: Path = field(compare=False)
    mtime_ms: int = field(compare=False, default=0)

    @property
    def name(self) -> str:
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description

    @property
    def memory_type(self) -> MemoryType | None:
        return self.frontmatter.memory_type
