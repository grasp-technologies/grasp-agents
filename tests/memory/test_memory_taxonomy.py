"""A host-defined memory taxonomy and instructions template."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.file_backend import InMemoryFileBackend
from grasp_agents.memory import (
    MEMORY_TYPES,
    InMemoryMemoryProvider,
    MemoryEntry,
    MemoryFrontmatter,
    MemoryProvider,
    format_manifest,
    make_memory_section,
    render_memory_instructions,
)
from grasp_agents.memory.injection import (
    _format_entry_heading,  # pyright: ignore[reportPrivateUsage]
)
from grasp_agents.session_context import SessionContext

pytestmark = pytest.mark.asyncio

MEMDIR = Path("/learner/memory")
LEARNER_TYPES = ("learner", "goal", "preference", "struggle", "feedback")


async def _section_text(provider: MemoryProvider) -> str:
    backend = InMemoryFileBackend(
        [MEMDIR], files={str(MEMDIR / "MEMORY.md"): "# Memory\n"}
    )
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, memory=provider)
    result = make_memory_section().compute(ctx=ctx, exec_id=None, agent_ctx=None)
    if inspect.isawaitable(result):
        result = await result
    assert isinstance(result, str)
    return result


async def test_default_taxonomy_is_unchanged() -> None:
    provider = MemoryProvider(root=MEMDIR)

    text = await _section_text(provider)

    assert provider.memory_types == MEMORY_TYPES
    assert "one of: user, feedback, project, reference" in text


async def test_custom_types_reach_the_instructions() -> None:
    provider = MemoryProvider(root=MEMDIR, memory_types=LEARNER_TYPES)

    text = await _section_text(provider)

    assert "one of: learner, goal, preference, struggle, feedback" in text


async def test_custom_template_replaces_the_instructions() -> None:
    template = (
        "# Learner memory at{memdir}\n"
        "Types: {memory_types}. Index: {index_path} ({index_file}), "
        "{max_lines} lines / {max_bytes} bytes.{selector_instructions}"
    )
    provider = MemoryProvider(
        root=MEMDIR, memory_types=LEARNER_TYPES, instructions_template=template
    )

    text = await _section_text(provider)

    assert text.startswith(
        "# Learner memory at `/learner/memory`\nTypes: learner, goal"
    )
    assert "Index: /learner/memory/MEMORY.md (MEMORY.md)" in text
    assert "## Types of memory" not in text


async def test_render_memory_instructions_accepts_types_and_template() -> None:
    text = render_memory_instructions(
        memory_types=("a", "b"), template="types={memory_types}{selector_instructions}"
    )

    assert text == "types=a, b"


def _entry(type_value: str | None) -> MemoryEntry:
    return MemoryEntry(
        frontmatter=MemoryFrontmatter.model_validate(
            {"name": "topic", "description": "about the topic", "type": type_value}
        ),
        body="body",
        mtime_ms=0,
    )


async def test_headings_and_manifest_show_custom_types() -> None:
    goal = _entry("goal")
    untyped = _entry(None)

    assert _format_entry_heading(goal).startswith("### [goal] topic")
    assert _format_entry_heading(untyped).startswith("### topic")
    assert format_manifest([goal, untyped]).splitlines() == [
        "- [goal] topic.md: about the topic",
        "- [?] topic.md: about the topic",
    ]


async def test_in_memory_provider_keeps_the_default_taxonomy() -> None:
    assert InMemoryMemoryProvider().memory_types == MEMORY_TYPES
