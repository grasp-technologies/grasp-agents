"""
``expected_mtime`` on ``write_bytes`` / ``delete`` in the local backend.

The in-memory backend's exact behavior is covered in
``test_memory_backend.py``; the E2B and MCP backends' best-effort checks are
in ``test_conditional_writes_e2b.py`` and ``test_conditional_writes_mcp.py``,
which skip without their extras.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from grasp_agents.file_backend import LocalFileBackend, StaleFileError

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


async def test_local_write_with_current_mtime_succeeds(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("v1")
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    _, mtime = await backend.read_text(f)

    await backend.write_bytes(f, b"v2", mode=0o644, expected_mtime=mtime)

    assert f.read_text() == "v2"


async def test_local_write_refuses_a_changed_file(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("v1")
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    _, mtime = await backend.read_text(f)
    os.utime(f, (mtime + 5, mtime + 5))

    with pytest.raises(StaleFileError):
        await backend.write_bytes(f, b"v2", mode=0o644, expected_mtime=mtime)
    assert f.read_text() == "v1"


async def test_local_write_refuses_a_vanished_file(tmp_path: Path) -> None:
    backend = LocalFileBackend(allowed_roots=[tmp_path])

    with pytest.raises(StaleFileError):
        await backend.write_bytes(
            tmp_path / "gone.txt", b"x", mode=0o644, expected_mtime=1.0
        )
    assert not (tmp_path / "gone.txt").exists()


async def test_local_delete_refuses_a_changed_file(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("v1")
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    _, mtime = await backend.read_text(f)
    os.utime(f, (mtime + 5, mtime + 5))

    with pytest.raises(StaleFileError):
        await backend.delete(f, expected_mtime=mtime)
    assert f.exists()

    await backend.delete(f, expected_mtime=mtime + 5)
    assert not f.exists()
