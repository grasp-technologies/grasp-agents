"""Tests for MemoryProvider and the default backends."""

from __future__ import annotations

import os
import time
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from grasp_agents.memory import (
    DEFAULT_STALE_AFTER,
    FileMemoryProvider,
    InMemoryMemoryProvider,
    MemoryEntry,
    MemoryFrontmatter,
    MemoryProvider,
    MemorySnapshot,
    default_memdir_path,
)
from grasp_agents.memory.provider import GRASP_MEMORY_ENV

if TYPE_CHECKING:
    from pathlib import Path


def _topic_file(path: Path, name: str, body: str = "B") -> Path:
    text = f"---\nname: {name}\ndescription: D\n---\n{body}\n"
    path.write_text(text, encoding="utf-8")
    return path


def _aged_entry(name: str, age_seconds: float, tmp_path: Path) -> MemoryEntry:
    f = tmp_path / f"{name}.md"
    f.write_text("---\nname: " + name + "\ndescription: D\n---\nB\n", encoding="utf-8")
    old = time.time() - age_seconds
    os.utime(f, (old, old))
    fm = MemoryFrontmatter.model_validate({"name": name, "description": "D"})
    return MemoryEntry(
        frontmatter=fm, body="B", path=f, mtime_ms=int(old * 1000)
    )


# ---------- InMemoryMemoryProvider ----------


class TestInMemoryProvider:
    @pytest.mark.anyio
    async def test_default_empty(self) -> None:
        p = InMemoryMemoryProvider()
        snap = await p.load()
        assert snap.is_empty
        assert snap.index is None
        assert snap.entries == ()

    @pytest.mark.anyio
    async def test_with_index(self) -> None:
        p = InMemoryMemoryProvider(index="# hello")
        snap = await p.load()
        assert snap.index == "# hello"

    @pytest.mark.anyio
    async def test_with_entries(self, tmp_path: Path) -> None:
        e = _aged_entry("alpha", age_seconds=10, tmp_path=tmp_path)
        p = InMemoryMemoryProvider(entries=[e])
        snap = await p.load()
        assert len(snap.entries) == 1
        assert snap.get("alpha") is e
        assert snap.get("missing") is None


# ---------- FileMemoryProvider ----------


class TestFileMemoryProvider:
    @pytest.mark.anyio
    async def test_load_from_disk(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("# idx\n", encoding="utf-8")
        _topic_file(tmp_path / "alpha.md", "alpha")
        p = FileMemoryProvider(tmp_path)
        snap = await p.load()
        assert snap.index is not None
        assert "idx" in snap.index
        assert {e.name for e in snap.entries} == {"alpha"}

    @pytest.mark.anyio
    async def test_load_caches(self, tmp_path: Path) -> None:
        _topic_file(tmp_path / "alpha.md", "alpha")
        p = FileMemoryProvider(tmp_path)
        first = await p.load()
        # Mutate disk; without refresh() the cached snapshot wins.
        _topic_file(tmp_path / "beta.md", "beta")
        second = await p.load()
        assert first is second
        assert {e.name for e in second.entries} == {"alpha"}

    @pytest.mark.anyio
    async def test_refresh_picks_up_changes(self, tmp_path: Path) -> None:
        _topic_file(tmp_path / "alpha.md", "alpha")
        p = FileMemoryProvider(tmp_path)
        first = await p.load()
        _topic_file(tmp_path / "beta.md", "beta")
        await p.refresh()
        second = await p.load()
        assert first is not second
        assert {e.name for e in second.entries} == {"alpha", "beta"}

    @pytest.mark.anyio
    async def test_missing_root(self, tmp_path: Path) -> None:
        p = FileMemoryProvider(tmp_path / "missing")
        snap = await p.load()
        assert snap.is_empty

    @pytest.mark.anyio
    async def test_root_property(self, tmp_path: Path) -> None:
        p = FileMemoryProvider(tmp_path)
        assert p.root == tmp_path


# ---------- Freshness ----------


class TestFreshness:
    @pytest.mark.anyio
    async def test_stale_index_gets_warning(self, tmp_path: Path) -> None:
        idx = tmp_path / "MEMORY.md"
        idx.write_text("# idx\n", encoding="utf-8")
        old = time.time() - 30 * 86400  # 30 days
        os.utime(idx, (old, old))
        p = FileMemoryProvider(tmp_path, stale_after=timedelta(days=7))
        snap = await p.load()
        assert snap.index_freshness_warning is not None
        assert "30" in snap.index_freshness_warning
        assert "<system-reminder>" in snap.index_freshness_warning

    @pytest.mark.anyio
    async def test_fresh_index_no_warning(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("# idx\n", encoding="utf-8")
        p = FileMemoryProvider(tmp_path, stale_after=timedelta(days=7))
        snap = await p.load()
        assert snap.index_freshness_warning is None

    @pytest.mark.anyio
    async def test_stale_entries_warned(self, tmp_path: Path) -> None:
        f = _topic_file(tmp_path / "alpha.md", "alpha")
        old = time.time() - 100 * 86400
        os.utime(f, (old, old))
        p = FileMemoryProvider(tmp_path, stale_after=timedelta(days=30))
        snap = await p.load()
        assert "alpha" in snap.entry_freshness_warnings

    @pytest.mark.anyio
    async def test_zero_threshold_disables(self, tmp_path: Path) -> None:
        idx = tmp_path / "MEMORY.md"
        idx.write_text("# idx\n", encoding="utf-8")
        old = time.time() - 30 * 86400
        os.utime(idx, (old, old))
        p = FileMemoryProvider(tmp_path, stale_after=timedelta())
        snap = await p.load()
        assert snap.index_freshness_warning is None
        assert snap.entry_freshness_warnings == {}


# ---------- ABC defaults ----------


class _ROProvider(MemoryProvider):
    async def load(
        self,
        *,
        session_id: str = "",
        ctx: object | None = None,
    ) -> MemorySnapshot:
        del session_id, ctx
        return MemorySnapshot()


class TestProviderDefaults:
    @pytest.mark.anyio
    async def test_refresh_default_noop(self) -> None:
        p = _ROProvider()
        await p.refresh()  # no exception

    def test_root_default_none(self) -> None:
        # Providers that don't expose a memdir return None — callers can
        # check before constructing a file toolkit.
        p = _ROProvider()
        assert p.root is None

    def test_make_file_toolkit_requires_root(self) -> None:
        # No root → can't build a toolkit. Raises with a clear message.
        p = _ROProvider()
        with pytest.raises(ValueError, match="no memdir root"):
            p.make_file_toolkit()

    def test_make_file_toolkit_with_root(self, tmp_path: Path) -> None:
        # FileMemoryProvider exposes its memdir root and can produce a
        # toolkit rooted there.
        p = FileMemoryProvider(tmp_path)
        toolkit = p.make_file_toolkit()
        assert str(tmp_path) in toolkit.allowed_roots


# ---------- default_memdir_path ----------


class TestDefaultMemdirPath:
    def test_default_shape(self, tmp_path: Path) -> None:
        path = default_memdir_path(tmp_path)
        assert path.name == "memory"
        assert path.parent.parent.name == "projects"
        assert ".grasp" in str(path)

    def test_env_override(self, tmp_path: Path) -> None:
        os.environ[GRASP_MEMORY_ENV] = str(tmp_path / "custom")
        try:
            path = default_memdir_path()
            assert path == tmp_path / "custom"
        finally:
            del os.environ[GRASP_MEMORY_ENV]


def test_default_stale_after_seven_days() -> None:
    assert timedelta(days=7) == DEFAULT_STALE_AFTER
