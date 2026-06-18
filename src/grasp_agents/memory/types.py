from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from pathlib import Path

NAME_REGEX = re.compile(r"^[a-z0-9](?:[-_]?[a-z0-9])*$")
NAME_MAX_LENGTH = 64
DESCRIPTION_MAX_LENGTH = 2048

INDEX_FILE_NAME = "MEMORY.md"

MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000
MAX_MEMORY_FILES = 200

DEFAULT_STALE_AFTER = timedelta(days=7)


type MemoryType = Literal["user", "feedback", "project", "reference"]
MEMORY_TYPES: tuple[MemoryType, ...] = get_args(MemoryType.__value__)


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

    ``name`` and ``description`` are required;
    ``type`` is one of ``user|feedback|project|reference`` (closed
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
            return self.raw_type
        return None


@dataclass(frozen=True)
class MemoryEntry:
    """
    A topic memory: frontmatter + (optionally) body, plus a backend handle.

    File-backed entries set ``path`` and populate ``body`` eagerly. Lazy
    backends (e.g. MCP-backed remote stores) set ``uri`` and leave ``body``
    as ``None``; the body is fetched on demand via
    :meth:`MemoryProvider.fetch_body`.
    """

    frontmatter: MemoryFrontmatter
    body: str | None = None
    path: Path | None = field(compare=False, default=None)
    uri: str | None = field(compare=False, default=None)
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
