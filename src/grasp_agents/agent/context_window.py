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
from typing import TYPE_CHECKING

from grasp_agents.context.compaction import Budgeted, ContextBudget, count_turns
from grasp_agents.context.projection import apply_folds, repair_tool_call_pairing
from grasp_agents.llm.model_info import get_model_capabilities
from grasp_agents.llm.token_counting import count_input_tokens
from grasp_agents.types.events import CompactionEvent, CompactionInfo

from .llm_agent_transcript import LLMAgentTranscript

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grasp_agents.hooks import Compactor, ViewProjector
    from grasp_agents.llm.model_info import ModelCapabilities
    from grasp_agents.types.folds import FoldSpec
    from grasp_agents.types.items import InputItem

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
        source: str,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        # The log. Mutated in place only — ``transcript`` is a read-only view
        # sharing this list, so it is never rebound.
        self._messages: list[InputItem] = []
        self._transcript = LLMAgentTranscript(self._messages)

        # Tokenizer selection + budget sizing. ``capabilities`` should be the
        # LLM's own (a composed LLM reports its conservative merge — a fallback
        # cascade's smallest member window); omitted, they are looked up by name.
        self._model_name = model_name
        self._capabilities = capabilities

        self._source = source  # agent name, stamped on emitted compaction events

        # Ephemeral system-prompt header (+ leading messages), rebuilt per step by
        # ``LLMAgent`` and prepended to the view; never written to the log.
        self.initial_context: list[InputItem] = []

        # Stacked projector pipeline + single compactor (set via LLMAgent hooks).
        self.view_projectors: list[ViewProjector] = []
        self.compactor: Compactor | None = None

        # Summarized log spans applied to the view; persisted, survive resume,
        # dropped past a rollback rewind.
        self.folds: list[FoldSpec] = []

        # Budget anchor: the last response's exact reported input_tokens over the
        # first ``_last_counted_len`` log messages; the turn's new messages are
        # counted on top via litellm. ``_pending_view_count`` is the basis of the
        # view being built, promoted when its response reports usage.
        self._last_input_tokens = 0
        self._last_counted_len = 0
        self._pending_view_count = 0

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
        folds and the token anchor index a prefix the append leaves intact.
        """
        self._messages.extend(messages)
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
        if isinstance(compactor, Budgeted) and compactor.budget is None:
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
        body = await self._build_view_body(exec_id=exec_id)
        return repair_tool_call_pairing([*self.initial_context, *body])

    async def _build_view_body(self, *, exec_id: str) -> Sequence[InputItem]:
        """
        The conversation body of the view — folds then the projector pipeline,
        without the header or pairing repair. Folds (summarized spans) apply
        first; projectors then trim what remains. Both shape the view, never the
        log.
        """
        body: Sequence[InputItem] = self._messages
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

        Anchored on the last response's exact reported ``input_tokens`` plus a
        single litellm count of the messages appended since (the turn's new
        input). Before any reported usage — or after a fold/rollback drops the
        anchor — counts the whole current (folded) view via litellm, which itself
        falls back to a chars-per-token estimate when no tokenizer is available.
        """
        model = self._model_name
        messages = self._messages
        if self._anchor_valid():
            delta = messages[self._last_counted_len :]
            if not delta:
                return self._last_input_tokens
            return self._last_input_tokens + count_input_tokens(model, delta)
        view = [*self.initial_context, *apply_folds(messages, self.folds)]
        return count_input_tokens(model, view)

    def _anchor_valid(self) -> bool:
        # The anchor is valid only while it still indexes within the live log
        # (the log is append + suffix-truncate only). A rollback / resume that
        # cut below it falls through to a full recount rather than mis-slicing.
        return self._last_input_tokens > 0 and self._last_counted_len <= len(
            self._messages
        )

    async def projected_input_tokens(self, *, exec_id: str) -> int:
        """
        Token cost of the view that would actually be sent — folds AND projectors
        applied — so the compaction gate measures the cheap projector tier (tool-
        output collapse) before the expensive compactor (summary fold) fires;
        folding only escalates when projection alone can't get under budget.

        Uses the fast anchor count when valid (the provider already counted the
        projected view it was sent); only the projector-blind recount fallback
        re-measures the projected view here.
        """
        if self._anchor_valid() or not self.view_projectors:
            return self.effective_input_tokens()
        body = await self._build_view_body(exec_id=exec_id)
        return count_input_tokens(self._model_name, [*self.initial_context, *body])

    def note_response_usage(self, input_tokens: int) -> None:
        """Promote the budget anchor from a response's reported usage."""
        self._last_input_tokens = input_tokens
        self._last_counted_len = self._pending_view_count

    def reset_anchor(self) -> None:
        """
        Drop the budget anchor so the next count is recomputed over the current
        view — used after a rollback rewinds the transcript, on resume, and on a
        fresh transcript.
        """
        self._last_input_tokens = 0
        self._last_counted_len = 0

    # --- transcript surgery ---
    # Cutting or replacing the log repairs everything keyed to it — summary
    # folds and the token anchor — in the same step.

    def truncate_transcript(self, message_count: int) -> None:
        """
        Cut the transcript to its first ``message_count`` messages (rollback
        rewind, interrupt settling). Folds whose span extends past the cut are
        dropped (those messages are gone; the view falls back to the
        originals); the token anchor survives unless the cut falls below its
        counted prefix.
        """
        del self._messages[max(message_count, 0) :]
        self.folds = [f for f in self.folds if f.end <= message_count]
        if self._last_counted_len > message_count:
            self.reset_anchor()
        self._pending_view_count = min(self._pending_view_count, message_count)

    def replace_transcript(
        self, messages: Sequence[InputItem], *, folds: Sequence[FoldSpec] = ()
    ) -> None:
        """
        Replace the transcript wholesale (resume from a checkpoint) together
        with its folds — lossy summaries are carried so resume doesn't
        re-summarize — and recount the budget anchor.
        """
        self._messages[:] = list(messages)
        self.folds = list(folds)
        self.reset_anchor()
        self._pending_view_count = 0

    def clear_transcript(self) -> None:
        """Start a fresh conversation: empty transcript, no folds, anchor reset."""
        self.replace_transcript([])

    # --- compaction ---

    async def maybe_compact(
        self, *, exec_id: str, force: bool = False
    ) -> FoldSpec | None:
        """
        Run the compactor at the turn boundary; record a fold if it returns one.

        Gates on the *projected* view size (collapse already applied), so a
        summary fold fires only when the cheap projector tier can't free enough.
        The compactor is a no-op while the view is under budget; ``force`` (the
        context-window-error recovery path) folds regardless. Returns the fold it
        recorded, or ``None``.
        """
        if self.compactor is None:
            return None
        fold = await self.compactor(
            self._messages,
            input_tokens=await self.projected_input_tokens(exec_id=exec_id),
            folds=self.folds,
            exec_id=exec_id,
            force=force,
        )
        if fold is None:
            return None
        self.folds.append(fold)
        # The fold shrank the view; drop the stale anchor so the next budget count
        # is recomputed over the folded view.
        self._last_input_tokens = 0
        logger.debug(
            "recorded fold [%d, %d) (folds=%d)", fold.start, fold.end, len(self.folds)
        )
        return fold

    def compaction_event(self, fold: FoldSpec, *, exec_id: str) -> CompactionEvent:
        """The event announcing a recorded fold (the reduced view size + summary)."""
        messages = self._messages
        budget = getattr(self.compactor, "budget", None)
        return CompactionEvent(
            source=self._source,
            exec_id=exec_id,
            data=CompactionInfo(
                folded_turns=count_turns(messages[fold.start : fold.end]),
                preserved_turns=count_turns(messages[fold.end :]),
                context_tokens=self.effective_input_tokens(),
                context_window=getattr(budget, "max_input_tokens", None),
                summary=fold.summary,
            ),
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
