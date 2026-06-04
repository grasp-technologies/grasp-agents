"""Tests for the unified :class:`MemoryProvider` and the InMemory fixture."""

from __future__ import annotations

import os
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.memory import (
    DEFAULT_STALE_AFTER,
    InMemoryMemoryProvider,
    MemoryEntry,
    MemoryFrontmatter,
    MemoryProvider,
    default_memdir_path,
)
from grasp_agents.memory.default_path import GRASP_MEMORY_ENV
from grasp_agents.run_context import RunContext
from grasp_agents.tools.file_edit.local_backend import LocalFileBackend

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
    return MemoryEntry(frontmatter=fm, body="B", path=f, mtime_ms=int(old * 1000))


def _make_ctx(memdir: Path) -> RunContext[Any]:
    """Build a ctx wired to LocalFileBackend + MemoryProvider over memdir."""
    backend = LocalFileBackend(allowed_roots=[memdir])
    return RunContext[Any](file_backend=backend, memory=MemoryProvider(memdir))


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


# ---------- MemoryProvider over LocalFileBackend ----------


class TestMemoryProviderLocal:
    @pytest.mark.anyio
    async def test_load_from_disk(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("# idx\n", encoding="utf-8")
        _topic_file(tmp_path / "alpha.md", "alpha")
        ctx = _make_ctx(tmp_path)
        assert ctx.memory is not None
        snap = await ctx.memory.load()
        assert snap.index is not None
        assert "idx" in snap.index
        assert {e.name for e in snap.entries} == {"alpha"}

    @pytest.mark.anyio
    async def test_load_caches(self, tmp_path: Path) -> None:
        _topic_file(tmp_path / "alpha.md", "alpha")
        ctx = _make_ctx(tmp_path)
        assert ctx.memory is not None
        first = await ctx.memory.load()
        # Mutate disk; without refresh() the cached snapshot wins.
        _topic_file(tmp_path / "beta.md", "beta")
        second = await ctx.memory.load()
        assert first is second
        assert {e.name for e in second.entries} == {"alpha"}

    @pytest.mark.anyio
    async def test_refresh_picks_up_changes(self, tmp_path: Path) -> None:
        _topic_file(tmp_path / "alpha.md", "alpha")
        ctx = _make_ctx(tmp_path)
        assert ctx.memory is not None
        first = await ctx.memory.load()
        _topic_file(tmp_path / "beta.md", "beta")
        await ctx.memory.refresh()
        second = await ctx.memory.load()
        assert first is not second
        assert {e.name for e in second.entries} == {"alpha", "beta"}

    @pytest.mark.anyio
    async def test_missing_root(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        # Use the parent dir for allowed_roots so validation accepts the
        # nested-missing path; with auto-create off the provider returns an
        # empty snapshot rather than bootstrapping an index.
        backend = LocalFileBackend(allowed_roots=[tmp_path])
        ctx = RunContext[Any](
            file_backend=backend,
            memory=MemoryProvider(missing, auto_create_index=False),
        )
        assert ctx.memory is not None
        snap = await ctx.memory.load()
        assert snap.is_empty

    @pytest.mark.anyio
    async def test_auto_creates_index(self, tmp_path: Path) -> None:
        """Default auto-create bootstraps MEMORY.md (and the dir) when absent."""
        memdir = tmp_path / "mem"
        backend = LocalFileBackend(allowed_roots=[tmp_path])
        ctx = RunContext[Any](file_backend=backend, memory=MemoryProvider(memdir))
        assert ctx.memory is not None
        snap = await ctx.memory.load()
        assert (memdir / "MEMORY.md").is_file()
        assert snap.index is not None
        assert not snap.is_empty

    @pytest.mark.anyio
    async def test_auto_create_disabled_leaves_index_absent(
        self, tmp_path: Path
    ) -> None:
        memdir = tmp_path / "mem2"
        memdir.mkdir()
        backend = LocalFileBackend(allowed_roots=[tmp_path])
        ctx = RunContext[Any](
            file_backend=backend,
            memory=MemoryProvider(memdir, auto_create_index=False),
        )
        assert ctx.memory is not None
        snap = await ctx.memory.load()
        assert not (memdir / "MEMORY.md").exists()
        assert snap.index is None

    @pytest.mark.anyio
    async def test_root_property(self, tmp_path: Path) -> None:
        p = MemoryProvider(tmp_path)
        assert p.root == tmp_path

    @pytest.mark.anyio
    async def test_load_requires_file_backend(self, tmp_path: Path) -> None:
        # Validator catches missing backend at RunContext construction,
        # but call ``load`` directly with a hand-rolled namespace to
        # exercise the runtime guard too.
        p = MemoryProvider(tmp_path)
        with pytest.raises(ValueError, match="file_backend"):
            await p.load()


class TestMemdirAdmission:
    """The RunContext validator admits the memdir into the backend roots."""

    def test_memdir_outside_allowed_roots_is_auto_added(
        self, tmp_path: Path
    ) -> None:
        # Backend rooted at a sibling dir that does NOT contain the memdir.
        work = tmp_path / "work"
        work.mkdir()
        memdir = tmp_path / "mem"
        backend = LocalFileBackend(allowed_roots=[work])
        # No raise — the memdir is admitted into allowed_roots instead.
        ctx = RunContext[Any](
            file_backend=backend,
            memory=MemoryProvider(memdir, auto_create_index=False),
        )
        assert ctx.file_backend is not None
        assert any(
            memdir == r or r in memdir.parents
            for r in ctx.file_backend.allowed_roots
        )

    def test_admission_is_idempotent_when_already_covered(
        self, tmp_path: Path
    ) -> None:
        backend = LocalFileBackend(allowed_roots=[tmp_path])
        before = len(backend.allowed_roots)
        RunContext[Any](
            file_backend=backend,
            memory=MemoryProvider(tmp_path / "mem", auto_create_index=False),
        )
        # memdir already nested under tmp_path → no new root appended.
        assert len(backend.allowed_roots) == before


# ---------- Freshness ----------


class TestFreshness:
    @pytest.mark.anyio
    async def test_stale_index_gets_warning(self, tmp_path: Path) -> None:
        idx = tmp_path / "MEMORY.md"
        idx.write_text("# idx\n", encoding="utf-8")
        old = time.time() - 30 * 86400  # 30 days
        os.utime(idx, (old, old))
        backend = LocalFileBackend(allowed_roots=[tmp_path])
        ctx = RunContext[Any](
            file_backend=backend,
            memory=MemoryProvider(tmp_path, stale_after=timedelta(days=7)),
        )
        assert ctx.memory is not None
        snap = await ctx.memory.load()
        assert snap.index_freshness_warning is not None
        assert "30" in snap.index_freshness_warning
        assert "<system-reminder>" in snap.index_freshness_warning

    @pytest.mark.anyio
    async def test_fresh_index_no_warning(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("# idx\n", encoding="utf-8")
        backend = LocalFileBackend(allowed_roots=[tmp_path])
        ctx = RunContext[Any](
            file_backend=backend,
            memory=MemoryProvider(tmp_path, stale_after=timedelta(days=7)),
        )
        assert ctx.memory is not None
        snap = await ctx.memory.load()
        assert snap.index_freshness_warning is None

    @pytest.mark.anyio
    async def test_stale_entries_warned(self, tmp_path: Path) -> None:
        f = _topic_file(tmp_path / "alpha.md", "alpha")
        old = time.time() - 100 * 86400
        os.utime(f, (old, old))
        backend = LocalFileBackend(allowed_roots=[tmp_path])
        ctx = RunContext[Any](
            file_backend=backend,
            memory=MemoryProvider(tmp_path, stale_after=timedelta(days=30)),
        )
        assert ctx.memory is not None
        snap = await ctx.memory.load()
        assert "alpha" in snap.entry_freshness_warnings

    @pytest.mark.anyio
    async def test_zero_threshold_disables(self, tmp_path: Path) -> None:
        idx = tmp_path / "MEMORY.md"
        idx.write_text("# idx\n", encoding="utf-8")
        old = time.time() - 30 * 86400
        os.utime(idx, (old, old))
        backend = LocalFileBackend(allowed_roots=[tmp_path])
        ctx = RunContext[Any](
            file_backend=backend,
            memory=MemoryProvider(tmp_path, stale_after=timedelta()),
        )
        assert ctx.memory is not None
        snap = await ctx.memory.load()
        assert snap.index_freshness_warning is None
        assert snap.entry_freshness_warnings == {}


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
