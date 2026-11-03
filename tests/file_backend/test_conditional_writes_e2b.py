"""``expected_mtime`` on ``write_bytes`` / ``delete`` in the E2B backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("e2b")

from e2b import NotFoundException

from grasp_agents.file_backend import StaleFileError
from tests.sandbox.test_e2b import _WS, _env, _FakeEntry, _FakeFiles, _FakeSandbox

pytestmark = pytest.mark.asyncio


async def test_e2b_write_refuses_a_changed_file() -> None:
    fake = _FakeSandbox()
    path = f"{_WS}/a.txt"
    fake.files.infos[path] = _FakeEntry(name="a.txt", path=path, mtime_epoch=1500.0)
    backend = _env(fake).file_backend

    with pytest.raises(StaleFileError):
        await backend.write_bytes(Path(path), b"x", mode=0o644, expected_mtime=1000.0)
    assert path not in fake.files.written

    await backend.write_bytes(Path(path), b"x", mode=0o644, expected_mtime=1500.0)
    assert fake.files.written[path] == b"x"


async def test_e2b_delete_refuses_a_changed_file() -> None:
    fake = _FakeSandbox()
    path = f"{_WS}/a.txt"
    fake.files.infos[path] = _FakeEntry(name="a.txt", path=path, mtime_epoch=1500.0)
    backend = _env(fake).file_backend

    with pytest.raises(StaleFileError):
        await backend.delete(Path(path), expected_mtime=1000.0)
    assert fake.files.removed == []


class _StrictFiles(_FakeFiles):
    """Raise E2B's not-found error for unknown paths, like the real API."""

    async def get_info(self, path: str) -> Any:
        if path not in self.infos:
            raise NotFoundException(f"{path} not found")
        return self.infos[path]


async def test_e2b_write_refuses_a_missing_file() -> None:
    fake = _FakeSandbox(files=_StrictFiles())
    backend = _env(fake).file_backend

    with pytest.raises(StaleFileError):
        await backend.write_bytes(
            Path(f"{_WS}/missing.txt"), b"x", mode=0o644, expected_mtime=1000.0
        )
