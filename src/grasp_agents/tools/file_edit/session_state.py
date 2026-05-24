"""
Per-session state for the file-edit tools.

State here is the *minimum needed for correctness*:

``read_file_state``
    Maps a resolved path string to the ``mtime`` the model last observed.
    Path strings come from :meth:`FileBackend.validate_path` so the same
    key shape is used regardless of backend. Drives two invariants:

    - Read-before-write: a ``Write`` to an existing file is refused
      unless a prior ``Read`` is on record in this session.
    - mtime staleness refusal: a ``Write`` to an existing file is
      refused if the file's current ``mtime`` differs from the one
      recorded at the last ``Read``.

``dotfile_overrides``
    Paths the consumer has explicitly whitelisted for this session,
    bypassing the user-dotfile deny list (e.g. the user confirmed
    "yes, the agent may write ``~/.env`` in this session"). Only
    meaningful for local-FS backends.

What is **deliberately absent** — and why:

- **No dedup memo for repeat reads.** A model may legitimately re-read a
  file to bring its content back into the recency window; a
  "file_unchanged" stub would break that pattern.
- **No consecutive-read loop counter.** ``AgentLoop.max_turns`` bounds
  true runaway; per-tool loop detection would interfere with legitimate
  re-reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

# Cap on ``read_file_state`` to bound memory growth on long sessions.
# Eviction policy: oldest-first (dict insertion order in Python 3.7+).
DEFAULT_READ_FILE_STATE_CAP = 1000


@dataclass
class ReadRecord:
    """Snapshot of a file at the time of the last successful ``Read``."""

    mtime: float


@dataclass
class FileEditSessionState:
    """Mutable per-session state shared by the file-edit tools."""

    read_file_state: dict[str, ReadRecord] = field(
        default_factory=dict[str, ReadRecord]
    )

    # Paths the user has explicitly allowed for this session, bypassing
    # the user-dotfile deny list. The system-path baseline is not
    # session-overridable.
    dotfile_overrides: set[str] = field(default_factory=set[str])

    read_file_state_cap: int = DEFAULT_READ_FILE_STATE_CAP

    def record_read(self, resolved_path: PathLike, mtime: float) -> None:
        """Record a successful ``Read`` — updates ``read_file_state``."""
        self.read_file_state[str(resolved_path)] = ReadRecord(mtime=mtime)
        self._cap()

    def get_read_record(self, resolved_path: PathLike) -> ReadRecord | None:
        """Return the most recent ``ReadRecord`` for this path, or ``None``."""
        return self.read_file_state.get(str(resolved_path))

    def record_write(self, resolved_path: PathLike, mtime: float) -> None:
        """
        Refresh ``read_file_state`` after a successful ``Write`` / ``Edit``.

        Without this, consecutive edits of the same file would trip the
        staleness check — the second edit would see ``mtime`` differing
        from the first edit's pre-write record.
        """
        self.read_file_state[str(resolved_path)] = ReadRecord(mtime=mtime)

    def add_dotfile_override(self, resolved_path: PathLike) -> None:
        self.dotfile_overrides.add(str(resolved_path))

    def reset_session(self) -> None:
        """Full reset. Call when starting a new session on a reused toolkit."""
        self.read_file_state.clear()
        self.dotfile_overrides.clear()

    def _cap(self) -> None:
        """Evict oldest entries if ``read_file_state`` exceeds its cap."""
        while len(self.read_file_state) > self.read_file_state_cap:
            try:
                oldest = next(iter(self.read_file_state))
            except StopIteration:  # pragma: no cover — dict was empty mid-evict
                break
            self.read_file_state.pop(oldest, None)
