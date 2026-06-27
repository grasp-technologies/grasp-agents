"""
View-time mechanisms the loop applies when projecting the model-facing view.

The transcript is the immutable append-only log; the per-turn view is derived
from it without mutating it, so step rollback stays valid across compaction. Two
built-ins live here:

* :func:`apply_folds` replaces summarized spans (persisted :class:`FoldSpec`s)
  with their summary message, and
* :func:`repair_tool_call_pairing` fixes a ``tool_call`` / ``tool_result`` pair a
  projection may have orphaned — the view-time companion to
  :meth:`LLMAgentTranscript.validate_tool_call_pairing`, which validates the log.
"""

import logging
from collections.abc import Sequence

from grasp_agents.types.folds import FoldSpec
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
)

from .system_reminder import wrap_in_system_reminder

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


SUMMARY_SUBJECT = "your previous conversation, summarized to compact the input context"


def fold_summary_text(summary: str) -> str:
    """The wrapped text a fold's summary is injected as — what the agent sees."""
    return wrap_in_system_reminder(summary, subject=SUMMARY_SUBJECT)


def _fold_summary_message(summary: str) -> InputMessageItem:
    return InputMessageItem.from_text(fold_summary_text(summary), role="user")


def apply_folds(
    messages: Sequence[InputItem], folds: Sequence[FoldSpec]
) -> list[InputItem]:
    """
    Replace each folded span of ``messages`` with its summary message.

    ``folds`` index the log; each ``[start, end)`` span becomes a single summary
    message. Folds are applied in order; one that is out of bounds or overlaps an
    earlier fold is skipped (defensive — the producer keeps them disjoint and
    in range, and rollback drops folds past a truncation). With no folds the
    messages pass through unchanged.
    """
    if not folds:
        return list(messages)
    result: list[InputItem] = []
    cursor = 0
    for fold in sorted(folds, key=lambda f: f.start):
        # A skip should not happen in normal operation (the compactor keeps folds
        # disjoint and in range, and rollback drops invalidated ones), so it
        # signals a diverged fold list — surface it rather than silently dropping.
        if fold.end > len(messages):
            logger.warning(
                "Skipping out-of-bounds fold [%d, %d): log has %d messages",
                fold.start,
                fold.end,
                len(messages),
            )
            continue
        if fold.start < cursor:
            logger.warning(
                "Skipping overlapping fold [%d, %d): overlaps a fold ending at %d",
                fold.start,
                fold.end,
                cursor,
            )
            continue
        result.extend(messages[cursor : fold.start])
        result.append(_fold_summary_message(fold.summary))
        cursor = fold.end
    result.extend(messages[cursor:])
    return result
