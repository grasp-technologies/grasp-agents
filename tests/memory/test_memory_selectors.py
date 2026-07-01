"""Tests for the reference LLM-based relevance selector."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.memory import (
    InMemoryMemoryProvider,
    MemoryEntry,
    MemoryFrontmatter,
    extract_latest_user_text,
    format_manifest,
    make_llm_relevance_selector,
)
from grasp_agents.types.content import InputImage
from grasp_agents.types.items import InputMessageItem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grasp_agents.types.items import InputItem


# ---- Test fixtures -----------------------------------------------------------


def _entry(name: str, *, mtime_ms: int = 0, type_: str | None = None) -> MemoryEntry:
    return MemoryEntry(
        frontmatter=MemoryFrontmatter(
            name=name,
            description=f"description for {name}",
            type=type_,  # type: ignore[arg-type]
        ),
        body=f"body for {name}",
        mtime_ms=mtime_ms,
    )


class _FakeLLM:
    """Stub LLM that returns a canned JSON output."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict[str, Any]] = []

    async def generate_response(
        self,
        *,
        input: Any,
        output_schema: Any = None,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {
                "input": list(input),
                "output_schema": output_schema,
                "max_output_tokens": max_output_tokens,
                **kwargs,
            }
        )

        class _Resp:
            output_text = self._response_text

        return _Resp()


# ---- Helper-function tests ---------------------------------------------------


class TestFormatManifest:
    def test_includes_type_filename_ts_description(self) -> None:
        ts_ms = int(datetime(2025, 1, 15, 12, 0, tzinfo=UTC).timestamp() * 1000)
        out = format_manifest([_entry("alpha", mtime_ms=ts_ms, type_="user")])
        assert "[user]" in out
        assert "alpha.md" in out
        assert "2025-01-15T12:00:00Z" in out
        assert "description for alpha" in out

    def test_missing_type_uses_placeholder(self) -> None:
        out = format_manifest([_entry("beta")])
        assert "[?]" in out
        assert "beta.md" in out

    def test_empty_entries(self) -> None:
        assert format_manifest([]) == ""


class TestExtractLatestUserText:
    def test_returns_last_user_message(self) -> None:
        msgs: Sequence[InputItem] = [
            InputMessageItem.from_text("hi", role="user"),
            InputMessageItem.from_text("system note", role="developer"),
            InputMessageItem.from_text("the real query", role="user"),
        ]
        assert extract_latest_user_text(msgs) == "the real query"

    def test_returns_empty_for_no_user(self) -> None:
        msgs: Sequence[InputItem] = [
            InputMessageItem.from_text("system note", role="system"),
        ]
        assert extract_latest_user_text(msgs) == ""

    def test_handles_none(self) -> None:
        assert extract_latest_user_text(None) == ""


# ---- Selector behavior tests -------------------------------------------------


@pytest.mark.asyncio
async def test_selector_picks_listed_entries() -> None:
    llm = _FakeLLM('{"selected_memories": ["alpha.md", "beta.md"]}')
    selector = make_llm_relevance_selector(llm, max_select=5)  # type: ignore[arg-type]

    entries = (_entry("alpha"), _entry("beta"), _entry("gamma"))
    messages: Sequence[InputItem] = [
        InputMessageItem.from_text("what about alpha and beta?", role="user")
    ]
    picked = await selector(entries=entries, messages=messages)

    names = [e.name for e in picked]
    assert names == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_selector_caps_at_max_select() -> None:
    llm = _FakeLLM(
        '{"selected_memories": ["a.md", "b.md", "c.md", "d.md", "e.md", "f.md"]}'
    )
    selector = make_llm_relevance_selector(llm, max_select=3)  # type: ignore[arg-type]

    entries = tuple(_entry(c) for c in "abcdef")
    messages: Sequence[InputItem] = [
        InputMessageItem.from_text("all of them please", role="user")
    ]
    picked = await selector(entries=entries, messages=messages)

    assert len(picked) == 3
    assert [e.name for e in picked] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_selector_empty_query_returns_empty() -> None:
    llm = _FakeLLM('{"selected_memories": ["a.md"]}')
    selector = make_llm_relevance_selector(llm)  # type: ignore[arg-type]

    picked = await selector(entries=(_entry("a"),), messages=[])
    assert picked == ()
    # LLM should NOT have been called — selector short-circuits when
    # there's no query to anchor selection.
    assert llm.calls == []


@pytest.mark.asyncio
async def test_selector_empty_entries_returns_empty() -> None:
    llm = _FakeLLM('{"selected_memories": []}')
    selector = make_llm_relevance_selector(llm)  # type: ignore[arg-type]

    picked = await selector(
        entries=(),
        messages=[InputMessageItem.from_text("hi", role="user")],
    )
    assert picked == ()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_selector_skips_unknown_names() -> None:
    llm = _FakeLLM('{"selected_memories": ["alpha.md", "nope.md"]}')
    selector = make_llm_relevance_selector(llm)  # type: ignore[arg-type]

    picked = await selector(
        entries=(_entry("alpha"),),
        messages=[InputMessageItem.from_text("a?", role="user")],
    )
    assert [e.name for e in picked] == ["alpha"]


@pytest.mark.asyncio
async def test_selector_failure_returns_empty() -> None:
    class _BrokenLLM:
        async def generate_response(self, **_: Any) -> Any:
            raise RuntimeError("boom")

    selector = make_llm_relevance_selector(_BrokenLLM())  # type: ignore[arg-type]
    picked = await selector(
        entries=(_entry("alpha"),),
        messages=[InputMessageItem.from_text("a?", role="user")],
    )
    # Selection is non-essential — failure falls back to "surface nothing"
    # rather than crashing the parent turn.
    assert picked == ()


@pytest.mark.asyncio
async def test_selector_input_shape() -> None:
    """The selector sends only the system prompt + a single user message."""
    llm = _FakeLLM('{"selected_memories": []}')
    selector = make_llm_relevance_selector(llm, max_tokens=128)  # type: ignore[arg-type]

    await selector(
        entries=(_entry("alpha"),),
        messages=[
            InputMessageItem.from_text("old conversation turn", role="user"),
            InputMessageItem.from_text("intermediate", role="developer"),
            InputMessageItem.from_text("the actual query", role="user"),
        ],
    )

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["max_output_tokens"] == 128
    inputs = call["input"]
    assert len(inputs) == 2
    assert inputs[0].role == "system"
    assert inputs[1].role == "user"
    # Only the LATEST user message should be in the call body —
    # conversation history is NOT replayed to the selector.
    assert "the actual query" in inputs[1].text
    assert "old conversation turn" not in inputs[1].text
    assert "alpha.md" in inputs[1].text


@pytest.mark.asyncio
async def test_selector_parses_fenced_json() -> None:
    """Fenced / non-bare JSON from the model is still parsed (shared validator)."""
    llm = _FakeLLM('```json\n{"selected_memories": ["alpha.md"]}\n```')
    selector = make_llm_relevance_selector(llm)  # type: ignore[arg-type]
    picked = await selector(
        entries=(_entry("alpha"),),
        messages=[InputMessageItem.from_text("a?", role="user")],
    )
    assert [e.name for e in picked] == ["alpha"]


@pytest.mark.asyncio
async def test_selector_unparseable_output_returns_empty() -> None:
    """Unparseable model output falls back to surfacing nothing, not a crash."""
    llm = _FakeLLM("not json at all")
    selector = make_llm_relevance_selector(llm)  # type: ignore[arg-type]
    picked = await selector(
        entries=(_entry("alpha"),),
        messages=[InputMessageItem.from_text("a?", role="user")],
    )
    assert picked == ()


@pytest.mark.asyncio
async def test_selector_forwards_image_only_query() -> None:
    """An image-only turn anchors selection: the image is sent, not dropped."""
    llm = _FakeLLM('{"selected_memories": ["alpha.md"]}')
    selector = make_llm_relevance_selector(llm)  # type: ignore[arg-type]
    image = InputImage(image_url="https://example.com/chart.png")
    picked = await selector(
        entries=(_entry("alpha"),),
        messages=[InputMessageItem(content_parts=[image], role="user")],
    )
    assert [e.name for e in picked] == ["alpha"]
    assert len(llm.calls) == 1
    user_msg = llm.calls[0]["input"][1]
    assert len(user_msg.images) == 1  # the image reached the selector LLM
    assert "alpha.md" in user_msg.text  # manifest still present alongside it


# ---- select_relevant seen-path filter ----------------------------------------


def _path_entry(name: str, path: Any) -> MemoryEntry:
    return MemoryEntry(
        frontmatter=MemoryFrontmatter(name=name, description=f"desc {name}"),
        body=f"body {name}",
        path=path,
    )


@pytest.mark.asyncio
async def test_select_relevant_filters_seen_paths(tmp_path: Any) -> None:
    """Entries already read this session are dropped before selection."""
    a = _path_entry("a", tmp_path / "a.md")
    b = _path_entry("b", tmp_path / "b.md")
    lazy = _path_entry("c", None)  # remote/lazy entry: no path to dedup on
    provider = InMemoryMemoryProvider(entries=[a, b, lazy])
    provider.set_selector(
        lambda *, entries, **_: entries  # type: ignore[arg-type,return-value]
    )
    snapshot = await provider.load()

    picked = await provider.select_relevant(snapshot, seen_paths={tmp_path / "a.md"})

    # "a" is filtered (seen); "b" survives; the path-less entry is never dropped.
    assert {e.name for e in picked} == {"b", "c"}


@pytest.mark.asyncio
async def test_select_relevant_no_seen_paths_keeps_all() -> None:
    """Empty/None seen-set is a no-op — the filter never over-suppresses."""
    a = _path_entry("a", None)
    b = _path_entry("b", None)
    provider = InMemoryMemoryProvider(entries=[a, b])
    provider.set_selector(
        lambda *, entries, **_: entries  # type: ignore[arg-type,return-value]
    )
    snapshot = await provider.load()

    assert len(await provider.select_relevant(snapshot)) == 2
    assert len(await provider.select_relevant(snapshot, seen_paths=set())) == 2


# ---- Integration with renderer ------------------------------------------------


@pytest.mark.asyncio
async def test_per_entry_staleness_warning_surfaced() -> None:
    """The relevance renderer must surface entry_freshness_warnings."""
    from grasp_agents.memory.injection import (
        _compute_relevant_memories,
    )

    # Build a provider with one stale entry (mtime far in the past).
    stale_ms = int(
        (datetime.now(UTC) - timedelta(days=30)).timestamp() * 1000
    )
    entry = _entry("stale_topic", mtime_ms=stale_ms, type_="reference")
    provider = InMemoryMemoryProvider(entries=[entry])
    # Set an identity selector so every entry is surfaced.
    provider.set_selector(
        lambda *, entries, **_: entries  # type: ignore[arg-type,return-value]
    )

    class _Ctx:
        memory = provider

    rendered = await _compute_relevant_memories(
        input_message=InputMessageItem.from_text("q", role="user"),  # type: ignore[arg-type]
        ctx=_Ctx(),  # type: ignore[arg-type]
        exec_id=None,
        messages=[InputMessageItem.from_text("q", role="user")],
    )

    assert rendered is not None
    assert "days old" in rendered  # freshness warning text
    # Warning lands above the body, below the heading.
    body_idx = rendered.find("body for stale_topic")
    warn_idx = rendered.find("days old")
    head_idx = rendered.find("stale_topic")
    assert head_idx < warn_idx < body_idx
