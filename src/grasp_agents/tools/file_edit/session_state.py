"""
Per-session state for the file-edit tools.

State here is the *minimum needed for correctness*:

``read_file_state``
    Maps a resolved :class:`pathlib.Path` to the ``mtime`` the model last
    observed. Path comes from :meth:`FileBackend.validate_path` so the
    same key shape is used regardless of backend. Drives two invariants:

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

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

    read_file_state: dict[Path, ReadRecord] = field(
        default_factory=dict[Path, ReadRecord]
    )

    # Paths the user has explicitly allowed for this session, bypassing
    # the user-dotfile deny list. The system-path baseline is not
    # session-overridable.
    dotfile_overrides: set[Path] = field(default_factory=set[Path])

    read_file_state_cap: int = DEFAULT_READ_FILE_STATE_CAP

    def record_read(self, resolved_path: Path, mtime: float) -> None:
        """Record a successful ``Read`` — updates ``read_file_state``."""
        self.read_file_state[resolved_path] = ReadRecord(mtime=mtime)
        self._cap()

    def get_read_record(self, resolved_path: Path) -> ReadRecord | None:
        """Return the most recent ``ReadRecord`` for this path, or ``None``."""
        return self.read_file_state.get(resolved_path)

    def record_write(self, resolved_path: Path, mtime: float) -> None:
        """
        Refresh ``read_file_state`` after a successful ``Write`` / ``Edit``.

        Without this, consecutive edits of the same file would trip the
        staleness check — the second edit would see ``mtime`` differing
        from the first edit's pre-write record.
        """
        self.read_file_state[resolved_path] = ReadRecord(mtime=mtime)

    def add_dotfile_override(self, resolved_path: Path) -> None:
        self.dotfile_overrides.add(resolved_path)

    def reset_session(self) -> None:
        """Full reset. Call when starting a new session on a reused toolkit."""
        self.read_file_state.clear()
        self.dotfile_overrides.clear()

    def export_state(self) -> tuple[dict[str, float], list[str]]:
        """
        Dump the ledger as JSON-friendly primitives for checkpointing.

        Returns ``(read_file_state, dotfile_overrides)`` with paths as
        strings; the inverse of :meth:`import_state`.
        """
        return (
            {str(p): rec.mtime for p, rec in self.read_file_state.items()},
            sorted(str(p) for p in self.dotfile_overrides),
        )

    def import_state(
        self,
        read_file_state: Mapping[str, float],
        dotfile_overrides: Iterable[str],
    ) -> None:
        """Replace the ledger from a checkpoint payload (resume path)."""
        self.read_file_state = {
            Path(p): ReadRecord(mtime=mtime) for p, mtime in read_file_state.items()
        }
        self.dotfile_overrides = {Path(p) for p in dotfile_overrides}
        self._cap()

    def _cap(self) -> None:
        """Evict oldest entries if ``read_file_state`` exceeds its cap."""
        while len(self.read_file_state) > self.read_file_state_cap:
            try:
                oldest = next(iter(self.read_file_state))
            except StopIteration:  # pragma: no cover — dict was empty mid-evict
                break
            self.read_file_state.pop(oldest, None)
