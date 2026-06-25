"""
View-time provider-invariant repair for the model-facing transcript view.

The transcript is the immutable append-only log; a :class:`ViewProjector`
derives the per-turn view the LLM sees (pruning, collapsing tool outputs,
summaries) without mutating the log, so step rollback stays valid across
compaction. A projection can
drop or reorder messages and orphan a ``tool_call`` / ``tool_result`` pair,
which strict providers reject. This module repairs the projected view right
before the provider call — the view-time companion to
:meth:`LLMAgentTranscript.validate_tool_call_pairing`, which validates the log.
"""

from collections.abc import Sequence

from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
)

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
