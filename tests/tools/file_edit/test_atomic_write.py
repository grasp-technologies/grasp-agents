"""
Unit tests for the atomic write helper.

Verifies the "no partial / torn files" guarantee: either the full
payload is visible at ``path`` or the original file is unchanged and
no temp artifact is left behind.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path  # noqa: TC003 — used at runtime (Path() calls in tests)
from unittest.mock import patch

import pytest

from grasp_agents.tools.file_edit.atomic_write import (
    atomic_write_bytes,
    atomic_write_text,
)


def test_atomic_write_creates_new_file(tmp_path: Path) -> None:
    target = tmp_path / "new.txt"
    atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("old")
    atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"


def test_atomic_write_refuses_overwrite_when_flag_false(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("old")
    with pytest.raises(FileExistsError):
        atomic_write_bytes(target, b"new", overwrite=False)
    assert target.read_text() == "old"


def test_atomic_write_raises_when_parent_missing(tmp_path: Path) -> None:
    target = tmp_path / "nonexistent_dir" / "f.txt"
    with pytest.raises(FileNotFoundError):
        atomic_write_bytes(target, b"x")


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_atomic_write_applies_mode(tmp_path: Path) -> None:
    target = tmp_path / "secret.txt"
    atomic_write_bytes(target, b"x", mode=0o600)
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_atomic_write_custom_mode(tmp_path: Path) -> None:
    target = tmp_path / "public.txt"
    atomic_write_bytes(target, b"x", mode=0o644)
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o644


def test_atomic_write_cleans_up_tmp_on_write_failure(tmp_path: Path) -> None:
    """
    If the write hits an I/O error after tmpfile creation, the temp
    file is removed — no ``.f.txt.<hash>.tmp`` droppings left behind.
    """
    target = tmp_path / "f.txt"

    with patch("os.replace", side_effect=OSError("simulated replace failure")):
        with pytest.raises(OSError, match="simulated replace failure"):
            atomic_write_bytes(target, b"x")

    # No temp file left.
    stray = [p for p in tmp_path.iterdir() if p.name.startswith(".f.txt.")]
    assert stray == []
    # Target was never created.
    assert not target.exists()


def test_atomic_write_text_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "t.txt"
    atomic_write_text(target, "héllo — world", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "héllo — world"


def test_atomic_write_no_fsync_still_works(tmp_path: Path) -> None:
    target = tmp_path / "nofsync.txt"
    atomic_write_bytes(target, b"x", fsync=False)
    assert target.read_bytes() == b"x"
