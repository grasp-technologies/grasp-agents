"""
Reference relevance selector for cross-session memory.

Issues a small system prompt plus a single user message containing the
latest query and a manifest of available memories (filename + type +
timestamp + description, no bodies), with a tight JSON-schema-bounded
output. The model picks up to :data:`DEFAULT_MAX_SELECT` filenames; the
selector filters the snapshot down to those entries.

Cost discipline (per call, at Sonnet rates):

* Input: ~300-800 tokens (system + manifest of ~200-cap entries).
* Output: capped at 256 tokens via :data:`DEFAULT_MAX_TOKENS`.
* No conversation history is sent — only the latest user query and
  the metadata-only manifest. Bodies are never serialized into the
  selector call.

Approximate cost: ~$0.005 per turn at Sonnet 4.5 list prices.

Usage::

    from grasp_agents import LLMAgent
    from grasp_agents.memory import (
        MemoryProvider,
        make_llm_relevance_selector,
    )
    from grasp_agents.file_backend import LocalFileBackend

    selector_llm = OpenAILLM(model="gpt-4o-mini")
    memory = MemoryProvider(root="/path/to/memdir")
    memory.set_selector(make_llm_relevance_selector(selector_llm))

    backend = LocalFileBackend(allowed_roots=[memory.root])
    agent = LLMAgent(...)
    ctx = SessionContext(state=..., file_backend=backend, memory=memory)
    await agent.run(..., ctx=ctx)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from grasp_agents.types.content import InputPart, InputText
from grasp_agents.types.errors import JSONSchemaValidationError
from grasp_agents.types.items import InputMessageItem
from grasp_agents.utils.validation import validate_obj_from_json_or_py_string

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from grasp_agents.llm.llm import LLM
    from grasp_agents.session_context import SessionContext
    from grasp_agents.types.items import InputItem

    from .provider import MemorySelector
    from .types import MemoryEntry


logger = logging.getLogger(__name__)


DEFAULT_MAX_SELECT = 5
DEFAULT_MAX_TOKENS = 256


SELECT_MEMORIES_SYSTEM_PROMPT = """\
You are selecting memories that will be useful for an agent's current turn.

You will receive:
1. The user's most recent query (it may include attached images or files).
2. A list of available memory files — each one tagged with its type
   ([user], [feedback], [project], [reference]), filename, last-updated
   timestamp, and a short description.

Return the filenames of the memories that will *clearly* be useful for
answering or acting on the query. Be conservative — only include
memories you are confident will help, based on their description and
type. Up to {max_select} selections; fewer is better when the query is
narrow. If nothing is clearly relevant, return an empty list.
"""


class _MemorySelectionResult(BaseModel):
    """JSON-schema-bounded output: at most ``max_select`` filenames."""

    selected_memories: list[str] = Field(default_factory=list[str])


def format_manifest(entries: Iterable[MemoryEntry]) -> str:
    """
    Render a manifest line per entry — type, filename, timestamp, description.
    """
    lines: list[str] = []
    for e in entries:
        type_tag = f"[{e.memory_type}]" if e.memory_type else "[?]"
        ts = _format_mtime(e.mtime_ms)
        ts_part = f" ({ts})" if ts else ""
        lines.append(f"- {type_tag} {e.name}.md{ts_part}: {e.description}")

    return "\n".join(lines)


def extract_latest_user_message(
    messages: Sequence[InputItem] | None,
) -> InputMessageItem | None:
    """Return the most recent ``user``-role message item, or ``None``."""
    if not messages:
        return None
    for item in reversed(list(messages)):
        if isinstance(item, InputMessageItem) and item.role == "user":
            return item
    return None


def extract_latest_user_text(messages: Sequence[InputItem] | None) -> str:
    """Return the text of the most recent ``user``-role message, or empty."""
    if not messages:
        return ""
    for item in reversed(list(messages)):
        if not isinstance(item, InputMessageItem):
            continue
        if item.role != "user":
            continue
        text = item.text.strip()
        if text:
            return text
    return ""


def make_llm_relevance_selector(
    llm: LLM,
    *,
    max_select: int = DEFAULT_MAX_SELECT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str = SELECT_MEMORIES_SYSTEM_PROMPT,
) -> MemorySelector:
    """
    Build a relevance selector that asks ``llm`` which memories to surface.

    Per call: small system prompt + a single user message with the latest
    query and a metadata-only manifest, capped at ~256 output tokens via
    JSON schema. No conversation history is sent — only the most recent user
    message.

    Args:
        llm: The selector LLM. A small/cheap model is appropriate
            (e.g. Sonnet, Haiku, ``gpt-4o-mini``). Independent of the
            agent's main LLM.
        max_select: Hard cap on the number of memories the selector
            may return. Default 5.
        max_tokens: Output-token budget for the selector call. Default
            256 — enough for ~5 filenames with formatting overhead.
        system_prompt: Override the built-in selector instructions. The
            default reads ``{max_select}`` as a format variable.

    Returns:
        A :data:`MemorySelector` ready to pass to
        :meth:`MemoryProvider.set_selector`.

    """
    rendered_system = system_prompt.format(max_select=max_select)

    async def select(
        *,
        entries: Sequence[MemoryEntry],
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        messages: Sequence[InputItem] | None = None,
    ) -> tuple[MemoryEntry, ...]:
        del ctx, exec_id

        if not entries:
            return ()

        latest = extract_latest_user_message(messages)
        if latest is None:
            return ()
        query_text = latest.text.strip()
        # Forward the query's images/files to the selector too, so an
        # image-only turn still anchors selection instead of being dropped.
        attachments: list[InputPart] = [*latest.images, *latest.files]
        if not query_text and not attachments:
            # Nothing to anchor selection on — surfacing nothing is safer than
            # dumping every body into the context window.
            return ()

        manifest = format_manifest(entries)
        lead = f"Query: {query_text}" if query_text else "Query (see attached content):"
        query_parts: list[InputPart] = [
            InputText(text=lead),
            *attachments,
            InputText(text=f"\n\nAvailable memories:\n{manifest}"),
        ]

        try:
            response = await llm.generate_response(
                input=[
                    InputMessageItem.from_text(rendered_system, role="system"),
                    InputMessageItem(content_parts=query_parts, role="user"),
                ],
                output_schema=_MemorySelectionResult,
                max_output_tokens=max_tokens,
            )
        except Exception as exc:
            # Selection is non-essential; if the call fails, fall back to
            # "surface nothing" rather than crashing the parent turn.
            logger.warning("relevance selector LLM call failed: %s", exc)
            return ()

        try:
            names = validate_obj_from_json_or_py_string(
                response.output_text, schema=_MemorySelectionResult
            ).selected_memories
        except JSONSchemaValidationError:
            logger.debug(
                "relevance selector: could not parse output: %r",
                response.output_text[:200],
            )
            names = []
        by_name = {e.name: e for e in entries}
        picked: list[MemoryEntry] = []
        for raw in names[:max_select]:
            base = raw.removesuffix(".md").strip()
            entry = by_name.get(base)
            if entry is not None and entry not in picked:
                picked.append(entry)

        return tuple(picked)

    return select


# ---- internals --------------------------------------------------------------


def _format_mtime(mtime_ms: int) -> str:
    if mtime_ms <= 0:
        return ""
    return (
        datetime.fromtimestamp(mtime_ms / 1000, tz=UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
