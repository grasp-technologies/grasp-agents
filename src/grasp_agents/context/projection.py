"""
View-time mechanisms the loop applies when projecting the model-facing view.

The transcript is the immutable append-only log; the per-turn view is derived
from it without mutating it, so step rollback stays valid across compaction. Two
built-ins live here:

* :func:`apply_folds` replaces summarized spans (persisted :class:`FoldSpec`s)
  with their pinned user messages and summary message, and
* :func:`repair_tool_call_pairing` fixes a ``tool_call`` / ``tool_result`` pair a
  projection may have orphaned — the view-time companion to
  :meth:`LLMAgentTranscript.validate_tool_call_pairing`, which validates the log.
"""

import logging
import re
from collections.abc import Sequence

from grasp_agents.types.content import InputText
from grasp_agents.types.folds import FoldSpec
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
)

from .system_reminder import is_system_reminder

logger = logging.getLogger(__name__)

# Stub result for a call whose own result a projection dropped. Phrased as data,
# not an error, so the model treats the turn as resolved rather than retrying.
_ELIDED_RESULT = "[Tool result omitted from this view by context management.]"


def repair_tool_call_pairing(messages: Sequence[InputItem]) -> list[InputItem]:
    """
    Return a copy of ``messages`` in which every tool call is resolved.

    Two repairs, mirroring the invariant
    :meth:`LLMAgentTranscript.validate_tool_call_pairing` enforces on the log:

    * **drop** a ``FunctionToolOutputItem`` with no matching open call (a
      free-floating or duplicate result the projection left behind), and
    * **synthesize** a stub ``FunctionToolOutputItem`` for a call whose result
      the projection dropped, inserted before the next input message (or at the
      end) so the call is resolved immediately.

    Tool-pairing only: same-turn reasoning / output items are not adjusted
    (reasoning preservation is a separate, provider-side concern). A projection
    that cuts on turn boundaries leaves no orphans, so this is a no-op there.
    """
    repaired: list[InputItem] = []
    open_calls: list[str] = []

    def _resolve_open() -> None:
        for call_id in open_calls:
            repaired.append(
                FunctionToolOutputItem.from_tool_result(
                    call_id=call_id, output=_ELIDED_RESULT
                )
            )
        open_calls.clear()

    for item in messages:
        if isinstance(item, FunctionToolCallItem):
            open_calls.append(item.call_id)
            repaired.append(item)
        elif isinstance(item, FunctionToolOutputItem):
            # Pair it with an open call; otherwise it is free-floating
            # (no call in this view) or a duplicate — drop it.
            if item.call_id in open_calls:
                open_calls.remove(item.call_id)
                repaired.append(item)
        elif isinstance(item, InputMessageItem):
            # A tool call must be resolved before any input message.
            _resolve_open()
            repaired.append(item)
        else:
            repaired.append(item)

    _resolve_open()  # dangling calls at the tail
    return repaired


SUMMARY_TAG = "conversation_summary"

# Framework tags a summary must not be able to open or close: it is model-written
# text that may echo untrusted content.
_FRAMEWORK_TAG_RE = re.compile(
    r"<\s*/?\s*(?:conversation_summary|compaction_request|untrusted_content"
    r"|system-reminder)\b[^>]*>",
    re.IGNORECASE,
)


def neutralize_framework_tags(text: str) -> str:
    """Escape the angle brackets of any framework tag embedded in ``text``."""
    return _FRAMEWORK_TAG_RE.sub(
        lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), text
    )


def fold_summary_text(summary: str) -> str:
    """The wrapped text a fold's summary is injected as — what the agent sees."""
    return f"<{SUMMARY_TAG}>\n{neutralize_framework_tags(summary)}\n</{SUMMARY_TAG}>"


def is_fold_summary(item: InputItem) -> bool:
    """True if ``item`` is a summary message :func:`apply_folds` injected."""
    return isinstance(item, InputMessageItem) and item.text.lstrip().startswith(
        f"<{SUMMARY_TAG}>"
    )


def user_authored(item: InputItem) -> InputMessageItem | None:
    """
    ``item`` as the user wrote it — a user message without its
    ``<system-reminder>`` parts (input attachments) — or ``None`` when it is
    not a user message or nothing but framework notes is left.
    """
    if not isinstance(item, InputMessageItem) or item.role != "user":
        return None
    parts = [
        p
        for p in item.content
        if not (isinstance(p, InputText) and is_system_reminder(p.text))
    ]
    if not any(not isinstance(p, InputText) or p.text.strip() for p in parts):
        return None
    if len(parts) == len(item.content):
        return item
    return item.model_copy(update={"content": parts})


def active_folds(folds: Sequence[FoldSpec], message_count: int) -> list[FoldSpec]:
    """
    The folds the view applies, in log order: the newest fold within the log,
    then, before its start, the newest fold before that, and so on. A newer fold
    that covers an older one supersedes it, so after a rollback the newest
    surviving fold applies on its own.
    """
    active: list[FoldSpec] = []
    floor = message_count
    for fold in reversed(folds):
        if fold.end <= floor:
            active.append(fold)
            floor = fold.start
    active.reverse()
    return active


def apply_folds(
    messages: Sequence[InputItem], folds: Sequence[FoldSpec]
) -> list[InputItem]:
    """
    Replace each active fold's span of ``messages`` (see :func:`active_folds`)
    with its pinned user messages, then its summary message.

    A fold reaching past the end of ``messages`` is skipped with a warning: the
    producer keeps folds in range and rollback drops folds past a truncation,
    so it signals a diverged fold list. With no folds the messages pass through
    unchanged.
    """
    if not folds:
        return list(messages)
    for fold in folds:
        if fold.end > len(messages):
            logger.warning(
                "Skipping out-of-bounds fold [%d, %d): log has %d messages",
                fold.start,
                fold.end,
                len(messages),
            )
    result: list[InputItem] = []
    cursor = 0
    for fold in active_folds(folds, len(messages)):
        result.extend(messages[cursor : fold.start])
        for i in fold.pinned:
            pinned = user_authored(messages[i])
            if pinned is not None:
                result.append(pinned)
        result.append(
            InputMessageItem.from_text(fold_summary_text(fold.summary), role="user")
        )
        cursor = fold.end
    result.extend(messages[cursor:])
    return result
