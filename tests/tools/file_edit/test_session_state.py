"""
Unit tests for :class:`FileEditSessionState`.

The session state carries only the invariant-related maps now:
``read_file_state`` (for read-before-write + staleness) and
``dotfile_overrides`` (per-session user opt-ins). Dedup and loop
counters were deliberately removed — see ``session_state.py`` docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grasp_agents.tools.file_edit.session_state import (
    FileEditSessionState,
    ReadRecord,
)


@pytest.fixture
def state() -> FileEditSessionState:
    return FileEditSessionState()


def test_record_read_populates_read_file_state(
    state: FileEditSessionState,
) -> None:
    p = Path("/tmp/file_a")
    state.record_read(p, mtime=111.0)
    assert state.read_file_state[p] == ReadRecord(mtime=111.0)


def test_record_read_overwrites_prior_record(
    state: FileEditSessionState,
) -> None:
    """A second Read of the same file updates mtime; no other side effects."""
    p = Path("/tmp/file_a")
    state.record_read(p, 111.0)
    state.record_read(p, 222.0)
    assert state.read_file_state[p].mtime == 222.0


def test_get_read_record_unknown_path_returns_none(
    state: FileEditSessionState,
) -> None:
    assert state.get_read_record(Path("/tmp/nope")) is None


def test_record_write_refreshes_read_state(
    state: FileEditSessionState,
) -> None:
    """
    After Write, readFileState mtime advances so the next Edit doesn't
    see its own prior Write as an external modification.
    """
    p = Path("/tmp/file_a")
    state.record_read(p, 100.0)
    state.record_write(p, mtime=200.0)
    record = state.get_read_record(p)
    assert record is not None
    assert record.mtime == 200.0


def test_reset_session_clears_everything(state: FileEditSessionState) -> None:
    p = Path("/tmp/file_a")
    state.record_read(p, 111.0)
    state.dotfile_overrides.add(Path("/home/u/.env"))

    state.reset_session()

    assert state.read_file_state == {}
    assert state.dotfile_overrides == set()


def test_cap_evicts_oldest_read_file_state() -> None:
    state = FileEditSessionState(read_file_state_cap=3)
    for i in range(5):
        state.record_read(Path(f"/tmp/f{i}"), float(i))

    # Capped at 3 → oldest-first eviction: f0/f1 gone, f2/f3/f4 remain.
    assert len(state.read_file_state) == 3
    assert Path("/tmp/f0") not in state.read_file_state
    assert Path("/tmp/f1") not in state.read_file_state
    assert Path("/tmp/f4") in state.read_file_state
