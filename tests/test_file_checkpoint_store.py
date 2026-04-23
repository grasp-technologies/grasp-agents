"""Unit tests for :class:`FileCheckpointStore`."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from grasp_agents.durability import CheckpointStore, FileCheckpointStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


async def test_save_load_round_trip(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    await store.save("agent/session-1", b'{"hello":"world"}')
    got = await store.load("agent/session-1")
    assert got == b'{"hello":"world"}'


async def test_conforms_to_checkpoint_store_protocol(tmp_path: Path) -> None:
    """Drop-in substitutability for any :class:`CheckpointStore` caller."""
    store: CheckpointStore = FileCheckpointStore(tmp_path)
    await store.save("agent/s", b"v")
    assert await store.load("agent/s") == b"v"
    assert await store.list_keys("agent/") == ["agent/s"]
    await store.delete("agent/s")
    assert await store.load("agent/s") is None


async def test_load_missing_returns_none(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    assert await store.load("agent/nope") is None


async def test_overwrite_replaces_prior(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    await store.save("agent/s", b"v1")
    await store.save("agent/s", b"v2")
    assert await store.load("agent/s") == b"v2"


async def test_delete_removes(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    await store.save("agent/s", b"v")
    await store.delete("agent/s")
    assert await store.load("agent/s") is None


async def test_delete_missing_is_noop(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    await store.delete("agent/s")  # no file — must not raise


async def test_nested_keys(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    await store.save("workflow/sess/sub/leaf", b"x")
    assert await store.load("workflow/sess/sub/leaf") == b"x"
    assert (tmp_path / "workflow" / "sess" / "sub" / "leaf.json").exists()


async def test_root_created_lazily(tmp_path: Path) -> None:
    target = tmp_path / "store-root"
    assert not target.exists()
    store = FileCheckpointStore(target)
    await store.save("agent/s", b"v")
    assert target.is_dir()


# ---------------------------------------------------------------------------
# list_keys
# ---------------------------------------------------------------------------


async def test_list_keys_prefix_match(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    await store.save("agent/a", b"1")
    await store.save("agent/b", b"2")
    await store.save("workflow/w", b"3")

    agent_keys = sorted(await store.list_keys("agent/"))
    assert agent_keys == ["agent/a", "agent/b"]

    all_keys = sorted(await store.list_keys(""))
    assert all_keys == ["agent/a", "agent/b", "workflow/w"]


async def test_list_keys_nested(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    await store.save("task/session-1/t1", b"1")
    await store.save("task/session-1/t2", b"2")
    await store.save("task/session-2/t3", b"3")

    keys = sorted(await store.list_keys("task/session-1/"))
    assert keys == ["task/session-1/t1", "task/session-1/t2"]


async def test_list_keys_empty_root(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path / "fresh")
    assert await store.list_keys("") == []


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


async def test_stale_tmp_does_not_affect_load(tmp_path: Path) -> None:
    """
    A lingering ``.tmp`` from a prior crashed write must not be picked up
    by ``load()`` — it only reads the final path.
    """
    store = FileCheckpointStore(tmp_path)
    await store.save("agent/s", b"good")

    # Simulate a crashed writer leaving behind a partial tmp file in the
    # target directory.
    stale = tmp_path / "agent" / "s.json.crash.tmp"
    stale.write_bytes(b"partial-garbage")

    assert await store.load("agent/s") == b"good"
    # Stale file still exists — no auto-sweep is promised by this layer.
    assert stale.exists()


async def test_no_tmp_files_left_after_success(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    await store.save("agent/s", b"v")
    # Glob the key's directory for any leftover .tmp files.
    leftovers = list((tmp_path / "agent").glob("*.tmp"))
    assert leftovers == []


async def test_concurrent_same_key_serializes(tmp_path: Path) -> None:
    """
    Running many concurrent saves on the same key must leave one of the
    payloads fully intact — no interleaved / torn bytes.
    """
    store = FileCheckpointStore(tmp_path)
    payloads = [f"payload-{i}".encode().ljust(1024, b".") for i in range(20)]

    await asyncio.gather(*(store.save("agent/s", p) for p in payloads))
    final = await store.load("agent/s")

    assert final in payloads


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "/agent/s",
        "agent/../escape",
        "agent/./here",
        "agent//double",
        "agent/with\x00null",
        "agent\\windows",
    ],
)
async def test_invalid_keys_rejected(tmp_path: Path, bad_key: str) -> None:
    store = FileCheckpointStore(tmp_path)
    with pytest.raises(ValueError):  # noqa: PT011
        await store.save(bad_key, b"x")
    with pytest.raises(ValueError):  # noqa: PT011
        await store.load(bad_key)


async def test_symlink_escape_is_caught(tmp_path: Path) -> None:
    """
    A symlink planted inside the store that points outside the root must
    not let a write escape.
    """
    outside = tmp_path.parent / "outside-store"
    outside.mkdir(exist_ok=True)
    try:
        root = tmp_path / "store"
        root.mkdir()
        (root / "agent").symlink_to(outside, target_is_directory=True)

        store = FileCheckpointStore(root)
        with pytest.raises(ValueError, match="outside store root"):
            await store.save("agent/s", b"x")
    finally:
        for p in outside.glob("*"):
            p.unlink()
        outside.rmdir()
