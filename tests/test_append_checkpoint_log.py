"""
Append-only message-log persistence (schema v5).

Covers the store primitive (``append_messages`` / ``read_messages`` /
``rewrite_messages``) across all three backends — including a minimal custom
store exercising the whole-blob defaults — plus the head/log split, the commit
watermark, torn-tail tolerance, in-place file appends, and agent resume parity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from grasp_agents.durability import (
    AgentCheckpoint,
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
)
from grasp_agents.types.items import InputItem, InputMessageItem

from .test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    _make_agent,
    _text_response,
    load_agent_checkpoint,
    save_agent_checkpoint,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msgs(*texts: str) -> list[InputItem]:
    return [InputMessageItem.from_text(t, role="user") for t in texts]


def _texts(messages: Sequence[InputItem]) -> list[str]:
    return [m.text for m in messages if isinstance(m, InputMessageItem)]


@pytest.fixture(params=["memory", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> CheckpointStore:
    kind: str = request.param
    if kind == "memory":
        return InMemoryCheckpointStore()
    return FileCheckpointStore(tmp_path)


# ---------------------------------------------------------------------------
# Store primitive — both backends
# ---------------------------------------------------------------------------


KEY = "s1/agent/test_agent"


async def test_read_empty_log_is_empty(store: CheckpointStore) -> None:
    assert await store.read_messages(KEY) == []


async def test_append_then_read_roundtrip(store: CheckpointStore) -> None:
    await store.append_messages(KEY, _msgs("a", "b"))
    await store.append_messages(KEY, _msgs("c"))
    assert _texts(await store.read_messages(KEY)) == ["a", "b", "c"]


async def test_append_empty_is_noop(store: CheckpointStore) -> None:
    await store.append_messages(KEY, _msgs("a"))
    await store.append_messages(KEY, [])
    assert _texts(await store.read_messages(KEY)) == ["a"]


async def test_rewrite_replaces_log(store: CheckpointStore) -> None:
    await store.append_messages(KEY, _msgs("a", "b", "c"))
    await store.rewrite_messages(KEY, _msgs("x"))
    assert _texts(await store.read_messages(KEY)) == ["x"]


async def test_rewrite_empty_clears_log(store: CheckpointStore) -> None:
    await store.append_messages(KEY, _msgs("a"))
    await store.rewrite_messages(KEY, [])
    assert await store.read_messages(KEY) == []


async def test_log_is_isolated_per_key(store: CheckpointStore) -> None:
    await store.append_messages("s1/agent/a", _msgs("a"))
    await store.append_messages("s1/agent/b", _msgs("b1", "b2"))
    assert _texts(await store.read_messages("s1/agent/a")) == ["a"]
    assert _texts(await store.read_messages("s1/agent/b")) == ["b1", "b2"]


# ---------------------------------------------------------------------------
# Head/log split + watermark (all backends)
# ---------------------------------------------------------------------------


async def test_head_blob_excludes_messages(store: CheckpointStore) -> None:
    """The overwrite head must not embed the transcript (that is the log)."""
    cp = AgentCheckpoint(
        session_key="s1", processor_name="test_agent", messages=_msgs("x", "y")
    )
    await save_agent_checkpoint(store, KEY, cp)

    head_blob = await store.load(KEY)
    assert head_blob is not None
    head_only = AgentCheckpoint.model_validate_json(head_blob)
    assert head_only.messages == []  # transcript lives in the log, not the head
    assert head_only.message_count == 2  # …but the watermark is recorded


async def test_save_load_agent_checkpoint_roundtrip(store: CheckpointStore) -> None:
    cp = AgentCheckpoint(
        session_key="s1",
        processor_name="test_agent",
        messages=_msgs("hello", "there"),
        turn=3,
    )
    await save_agent_checkpoint(store, KEY, cp)

    loaded = await load_agent_checkpoint(store, KEY)
    assert loaded is not None
    assert _texts(loaded.messages) == ["hello", "there"]
    assert loaded.message_count == 2
    assert loaded.turn == 3


async def test_load_missing_checkpoint_is_none(store: CheckpointStore) -> None:
    assert await load_agent_checkpoint(store, "nope/agent/x") is None


async def test_watermark_drops_uncommitted_tail(store: CheckpointStore) -> None:
    """
    A crash between log-append and head-save leaves records past the head's
    watermark; load must ignore them.
    """
    cp = AgentCheckpoint(
        session_key="s1", processor_name="test_agent", messages=_msgs("a", "b")
    )
    await save_agent_checkpoint(store, KEY, cp)  # head.message_count == 2

    # Simulate the uncommitted tail: extra records the head doesn't count.
    await store.append_messages(KEY, _msgs("uncommitted"))
    assert len(await store.read_messages(KEY)) == 3  # physically present

    loaded = await load_agent_checkpoint(store, KEY)
    assert loaded is not None
    assert _texts(loaded.messages) == ["a", "b"]  # watermark wins


# ---------------------------------------------------------------------------
# File backend specifics
# ---------------------------------------------------------------------------


async def test_file_tolerates_torn_tail(tmp_path: Path) -> None:
    """A half-written final record (no newline) is discarded, not fatal."""
    store = FileCheckpointStore(tmp_path)
    await store.append_messages("log-test", _msgs("a", "b"))

    log_path = tmp_path / "log-test.jsonl"
    with log_path.open("ab") as f:
        f.write(b'{"type":"message","role":"user","content_part')  # torn line

    assert _texts(await store.read_messages("log-test")) == ["a", "b"]


async def test_file_appends_in_place(tmp_path: Path) -> None:
    """Appends extend the file rather than rewriting it (the whole point)."""
    store = FileCheckpointStore(tmp_path)
    await store.append_messages("log-test", _msgs("first"))
    after_first = (tmp_path / "log-test.jsonl").read_bytes()

    await store.append_messages("log-test", _msgs("second"))
    after_second = (tmp_path / "log-test.jsonl").read_bytes()

    # The original bytes are untouched; only new bytes were appended.
    assert after_second.startswith(after_first)
    assert len(after_second) > len(after_first)


async def test_file_log_not_listed_as_checkpoint_key(tmp_path: Path) -> None:
    """``list_keys`` globs ``*.json`` — the ``.jsonl`` log must not appear."""
    store = FileCheckpointStore(tmp_path)
    cp = AgentCheckpoint(
        session_key="s1", processor_name="test_agent", messages=_msgs("a")
    )
    await save_agent_checkpoint(store, KEY, cp)

    assert await store.list_keys("s1/") == [KEY]


async def test_file_delete_removes_log(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    cp = AgentCheckpoint(
        session_key="s1", processor_name="test_agent", messages=_msgs("a", "b")
    )
    await save_agent_checkpoint(store, KEY, cp)
    assert (tmp_path / "s1" / "agent" / "test_agent.jsonl").exists()

    await store.delete(KEY)
    assert not (tmp_path / "s1" / "agent" / "test_agent.jsonl").exists()
    assert await store.read_messages(KEY) == []


async def test_memory_delete_removes_log() -> None:
    store = InMemoryCheckpointStore()
    await store.append_messages(KEY, _msgs("a"))
    await store.delete(KEY)
    assert await store.read_messages(KEY) == []


# ---------------------------------------------------------------------------
# Agent end-to-end: incremental append + resume parity
# ---------------------------------------------------------------------------


async def test_agent_appends_incrementally_and_resumes(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)

    agent1, _ = _make_agent(
        [_text_response("one")],
        session_key="sess",
        store=store,  # type: ignore[arg-type]
    )
    await agent1.run("first")
    after_turn1 = (tmp_path / "sess" / "agent" / "test_agent.jsonl").read_bytes()
    log1 = await store.read_messages("sess/agent/test_agent")

    agent2, _ = _make_agent(
        [_text_response("two")],
        session_key="sess",
        store=store,  # type: ignore[arg-type]
    )
    await agent2.run("second")
    after_turn2 = (tmp_path / "sess" / "agent" / "test_agent.jsonl").read_bytes()
    log2 = await store.read_messages("sess/agent/test_agent")

    # The second run appended onto the first run's records (no full rewrite).
    assert after_turn2.startswith(after_turn1)
    assert len(log2) > len(log1)

    # A fresh agent on the same session reconstructs the full transcript.
    agent3, _ = _make_agent(
        [_text_response("three")],
        session_key="sess",
        store=store,  # type: ignore[arg-type]
    )
    await agent3.load_checkpoint()
    assert agent3.transcript.messages == log2


async def test_reset_transcript_rewrites_log(tmp_path: Path) -> None:
    """reset_transcript_on_run must replace the log, not append onto it."""
    store = FileCheckpointStore(tmp_path)
    agent1, _ = _make_agent(
        [_text_response("one")],
        session_key="sess",
        store=store,  # type: ignore[arg-type]
    )
    await agent1.run("first message")
    first_count = len(await store.read_messages("sess/agent/test_agent"))
    assert first_count > 0

    agent2, _ = _make_agent(
        [_text_response("two")],
        session_key="sess",
        store=store,  # type: ignore[arg-type]
        reset_transcript_on_run=True,
    )
    await agent2.run("second message")

    # The rewrite went to a fresh generation — read via the head, like resume.
    head = await load_agent_checkpoint(store, "sess/agent/test_agent")
    assert head is not None
    user_texts = [
        m.text
        for m in head.messages
        if isinstance(m, InputMessageItem) and m.role == "user"
    ]
    assert user_texts == ["second message"]  # the first run's records are gone
    # The superseded generation-0 file is removed.
    assert await store.read_messages("sess/agent/test_agent") == []
