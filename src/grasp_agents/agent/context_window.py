"""
The agent's context-window manager: derives the model-facing view from the
immutable transcript log, sizes it against the budget, and drives compaction.

The agent loop and ``LLMAgent`` delegate every context-window concern here —
view projection, token accounting, summary folds — and decide only *when* to
project or compact. This keeps the loop focused on the generate → tools cycle and
``LLMAgent`` on orchestration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from grasp_agents.context.compaction import (
    MAX_COMPACTION_FAILURES,
    Budgeted,
    Compactor,
    ContextBudget,
)
from grasp_agents.context.projection import apply_folds, repair_tool_call_pairing
from grasp_agents.llm.model_info import get_model_capabilities
from grasp_agents.llm.token_counting import estimate_input_tokens
from grasp_agents.types.items import (
    FunctionToolOutputItem,
    InputMessageItem,
)

from .llm_agent_transcript import LLMAgentTranscript

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grasp_agents.hooks import CompactionSummarizer, ViewProjector
    from grasp_agents.llm.model_info import ModelCapabilities
    from grasp_agents.types.folds import FoldSpec
    from grasp_agents.types.items import InputItem
    from grasp_agents.types.response import ResponseUsage

logger = logging.getLogger(__name__)


class ContextWindowManager:
    """
    Owns the transcript log and the model-facing view derived from it.

    The log is created here and exposed as a read-only view
    (:attr:`transcript`), re-exposed by ``AgentContext`` to tools; every write
    — appends included — is a method of this class, so the view state derived
    from the log is repaired alongside. Composes the view (ephemeral
    ``initial_context`` header + summary folds + the view-projector pipeline over
    the log), tracks its input-token cost against the budget, and records
    compaction folds under pressure. Projections and folds shape only the view,
    never the log, so step rollback and resume keep the full history. Held by
    :class:`AgentContext`; configured by ``LLMAgent`` hooks.
    """

    def __init__(
        self,
        *,
        model_name: str,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        # The log. Mutated in place only — ``transcript`` is a read-only view
        # sharing this list, so it is never rebound.
        self._messages: list[InputItem] = []
        self._transcript = LLMAgentTranscript(self._messages)

        # Budget sizing. ``capabilities`` should be the LLM's own (a composed LLM
        # reports its conservative merge — a fallback cascade's smallest member
        # window); omitted, they are looked up by name.
        self._model_name = model_name
        self._capabilities = capabilities

        # Ephemeral system-prompt header (+ leading messages), rebuilt per step by
        # ``LLMAgent`` and prepended to the view; never written to the log.
        self.initial_context: list[InputItem] = []

        # Stacked projector pipeline, the compactor and an optional summarizer
        # that writes its summaries (set via LLMAgent).
        self.view_projectors: list[ViewProjector] = []
        self.compactor: Compactor | None = None
        self.compaction_summarizer: CompactionSummarizer[Any] | None = None

        # Summarized log spans applied to the view; persisted, survive resume,
        # dropped past a rollback rewind.
        self.folds: list[FoldSpec] = []

        # The latest response's reported usage, for the request that carried the
        # first ``_usage_basis`` log messages: its input_tokens anchor the budget,
        # and newer messages are estimated on top. Cleared when a fold or a cut
        # changes the view it measured. ``_pending_view_count`` is the basis of
        # the view being built, promoted when its response reports usage.
        self.last_usage: ResponseUsage | None = None
        self._usage_basis = 0
        self._pending_view_count = 0

        # Failed compactions in a row, and the log index the last failed one
        # would have folded up to (it is not retried at the same point).
        self._compaction_failures = 0
        self.failed_fold_end: int | None = None

        # Model-derived budget, built lazily and shared into registered strategies
        # that don't bring their own (so the agent never has to construct one).
        self._budget: ContextBudget | None = None

    @property
    def transcript(self) -> LLMAgentTranscript:
        """Read-only view of the log (also ``agent.transcript``)."""
        return self._transcript

    def add_messages(self, messages: Sequence[InputItem]) -> int:
        """
        Append ``messages`` to the log; returns its new length (the 1-based
        position of the last message appended). Appends need no view repair:
        folds and the usage anchor index a prefix the append leaves intact.
        """
        self._messages.extend(list(messages))  # ``messages`` may be the log itself
        return len(self._messages)

    # --- registration (called by LLMAgent's hooks) ---

    def default_budget(self) -> ContextBudget:
        """The agent's model-derived budget, cached."""
        if self._budget is None:
            caps = self._capabilities
            if caps is None:
                caps = get_model_capabilities(self._model_name)
            self._budget = ContextBudget.from_capabilities(self._model_name, caps)
        return self._budget

    def add_view_projector(self, projector: ViewProjector) -> None:
        # Supply the model-derived budget to a budget-gated projector that didn't
        # bring its own (a proactive one carries the slot too but ignores it).
        if isinstance(projector, Budgeted) and projector.budget is None:
            projector.budget = self.default_budget()
        self.view_projectors.append(projector)

    def set_compactor(self, compactor: Compactor) -> None:
        if compactor.budget is None:
            compactor.budget = self.default_budget()
        self.compactor = compactor

    # --- view derivation ---

    async def project_view(self, *, exec_id: str) -> Sequence[InputItem]:
        """
        Build the transient model-facing view for this turn.

        The ephemeral ``initial_context`` header is prepended to the projected
        conversation log. The transcript is the immutable append-only log;
        registered view projectors derive what the LLM sees (pruning, collapsing
        tool outputs, summaries) from the conversation without mutating the log,
        so step rollback stays valid across compaction. They run as a pipeline in
        registration order; with none registered the conversation passes through
        unchanged. A projection can orphan a ``tool_call`` / ``tool_result`` pair,
        so the final view is repaired once before the call. The view is never
        written back to the transcript.
        """
        self._pending_view_count = len(self._messages)
        if not self.folds and not self.view_projectors:
            # No compaction: the log is pairing-valid and the header adds no tool
            # calls, so the prepended view needs no repair.
            return [*self.initial_context, *self._messages]
        body = await self.build_view_body(exec_id=exec_id)
        return repair_tool_call_pairing([*self.initial_context, *body])

    def folded_messages(self) -> list[InputItem]:
        """
        The log with folds applied: the conversation as the model sees it,
        before projectors.
        """
        return apply_folds(self._messages, self.folds)

    async def build_view_body(
        self, *, exec_id: str, end: int | None = None
    ) -> Sequence[InputItem]:
        """
        The conversation body of the view — folds then the projector pipeline,
        without the header or pairing repair — over the log, or over its first
        ``end`` messages. Folds (summarized spans) apply first; projectors then
        trim what remains, each given the same ``input_tokens``: the size before
        any projector ran. Both shape the view, never the log.
        """
        body: Sequence[InputItem] = (
            self._messages if end is None else self._messages[:end]
        )
        if self.folds:
            body = apply_folds(body, self.folds)
        if self.view_projectors:
            input_tokens = self.effective_input_tokens()
            for project in self.view_projectors:
                body = await project(
                    list(body), exec_id=exec_id, input_tokens=input_tokens
                )
        return body

    # --- token accounting ---

    def effective_input_tokens(self) -> int:
        """
        The view's token cost for the compaction budget.

        Anchored on the latest response's reported ``input_tokens`` plus an
        estimate of the messages appended since (the turn's new input). With no
        anchor — before any reported usage, or after a fold or rollback —
        estimates the whole current (folded) view. Estimates are
        :func:`~grasp_agents.llm.token_counting.estimate_input_tokens`.
        """
        usage = self.last_usage
        if usage is not None and usage.input_tokens > 0:
            delta = self._messages[self._usage_basis :]
            return usage.input_tokens + (estimate_input_tokens(delta) if delta else 0)
        return estimate_input_tokens([*self.initial_context, *self.folded_messages()])

    async def projected_input_tokens(self, *, exec_id: str) -> int:
        """
        Token cost of the view that would actually be sent — folds and
        projectors applied. With a valid anchor it is the anchored estimate (the
        provider already counted the projected view it was sent); otherwise the
        projected view is estimated.
        """
        anchored = self.last_usage is not None and self.last_usage.input_tokens > 0
        if anchored or not self.view_projectors:
            return self.effective_input_tokens()
        body = await self.build_view_body(exec_id=exec_id)
        return estimate_input_tokens([*self.initial_context, *body])

    def note_response_usage(self, usage: ResponseUsage) -> None:
        """Record a response's reported usage; it anchors the budget."""
        self.last_usage = usage
        self._usage_basis = self._pending_view_count

    # --- transcript surgery ---
    # Cutting or replacing the log repairs everything keyed to it — summary
    # folds and the usage anchor — in the same step.

    def truncate_transcript(self, message_count: int) -> None:
        """
        Cut the transcript to its first ``message_count`` messages (rollback
        rewind, interrupt settling). Folds whose span extends past the cut are
        dropped (those messages are gone; the view falls back to the
        originals), and so is the usage anchor when the cut removes the
        response it came from. A count past the end keeps everything.
        """
        if message_count < 0:
            raise ValueError(f"message_count must be >= 0, got {message_count}")
        del self._messages[message_count:]
        self.folds = [f for f in self.folds if f.end <= message_count]
        if self._usage_basis >= message_count:
            # The response that reported it is gone.
            self.last_usage = None
        if self.failed_fold_end is not None and self.failed_fold_end > message_count:
            self.failed_fold_end = None
        self._pending_view_count = min(self._pending_view_count, message_count)

    def replace_transcript(
        self,
        messages: Sequence[InputItem],
        *,
        folds: Sequence[FoldSpec] = (),
        usage: ResponseUsage | None = None,
    ) -> None:
        """
        Replace the transcript wholesale (resume from a checkpoint) together
        with its folds — lossy summaries are carried so resume doesn't
        re-summarize — and the latest response's ``usage``, which anchors the
        budget on the request that preceded that response in the log. A fold
        reaching past the new log is dropped: once the log grew past it, it
        would summarize messages it never saw.
        """
        self._messages[:] = list(messages)
        kept = [f for f in folds if f.end <= len(self._messages)]
        if len(kept) < len(folds):
            logger.warning(
                "dropped %d summary fold(s) reaching past the %d-message log",
                len(folds) - len(kept),
                len(self._messages),
            )
        self.folds = kept
        basis = _latest_response_start(self._messages) if usage is not None else None
        self.last_usage = usage if basis is not None else None
        self._usage_basis = basis or 0
        self._pending_view_count = 0
        self._compaction_failures = 0
        self.failed_fold_end = None

    def clear_transcript(self) -> None:
        """Start a fresh conversation: empty transcript, no folds, anchor reset."""
        self.replace_transcript([])

    # --- compaction ---

    @property
    def compaction_failures(self) -> int:
        """Failed compactions in a row since the last fold."""
        return self._compaction_failures

    def record_fold(self, fold: FoldSpec) -> None:
        """Record a compaction's fold; the view now summarizes its span."""
        self.folds.append(fold)
        self._compaction_failures = 0
        self.failed_fold_end = None
        # The fold shrank the view, so the latest usage no longer measures it.
        self.last_usage = None

    def note_compaction_failure(self, fold_end: int) -> None:
        """Count a compaction that recorded no fold ending at ``fold_end``."""
        self._compaction_failures += 1
        self.failed_fold_end = fold_end
        logger.warning(
            "context compaction failed (%d in a row)%s",
            self._compaction_failures,
            "; turn-boundary compaction stops"
            if self._compaction_failures >= MAX_COMPACTION_FAILURES
            else "",
        )

    @property
    def context_window(self) -> int | None:
        """The input-token window compaction targets — from a registered budget."""
        for strategy in (self.compactor, *self.view_projectors):
            budget = getattr(strategy, "budget", None)
            window = getattr(budget, "max_input_tokens", None)
            if window is not None:
                return window
        return None


def _latest_response_start(messages: Sequence[InputItem]) -> int | None:
    """
    Log index where the latest model response begins: the start of the last
    run of response items (everything but input messages and tool results).
    The request that produced it carried exactly the messages before it.
    """
    end = len(messages)
    while end > 0 and isinstance(
        messages[end - 1], (InputMessageItem, FunctionToolOutputItem)
    ):
        end -= 1
    if end == 0:
        return None
    start = end
    while start > 0 and not isinstance(
        messages[start - 1], (InputMessageItem, FunctionToolOutputItem)
    ):
        start -= 1
    return start
