"""
Interactive compaction + rollback demo for the TUI.

A single research assistant with one verbose tool (``recall_notes``) and an
*artificially small* context window, so the running token meter (above the input
box) climbs fast and context-window compaction visibly kicks in within a couple
of turns: older turns are folded into a summary, shown as a ``⊙ context
compacted`` panel (with the summary text), after which the meter drops.

It also wires rollback: type ``/rollback`` (or press ``ctrl+r``) to pick an
earlier message and rewind to it — that message and everything after it are
discarded, and the on-screen transcript truncates to match.

Try, for example (the meter climbs each turn; after the 2nd-3rd question the
``⊙ context compacted`` panel appears and the meter drops)::

    Tell me about the history of the internet.
    Now summarize the key protocols.
    What about email specifically?
    /rollback                             # rewind to an earlier question

Run (needs ``OPENAI_API_KEY`` in ``.env``)::

    python -m grasp_agents.examples.tui.compaction_copilot

Requires the ``tui`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from grasp_agents import LLMAgent, SessionContext
from grasp_agents.context import ContextBudget, LLMSummarizer, SummarizingCompactor
from grasp_agents.durability import FileCheckpointStore
from grasp_agents.llm_providers.openai_responses import (
    OpenAIResponsesLLM,
    OpenAIResponsesLLMSettings,
)
from grasp_agents.tools.function_tool import function_tool

DEFAULT_MODEL = "gpt-5.4-nano"

# An artificially small window so summarization fires within a couple of turns.
# The meter is shown against ARTIFICIAL_WINDOW; compaction triggers at the soft
# limit (window - buffer), folding older turns into a summary while keeping the
# last KEEP_RECENT_TURNS turns verbatim.
ARTIFICIAL_WINDOW = 6000
BUFFER_TOKENS = 500
KEEP_RECENT_TURNS = 2

# Persist the session so you can watch the transcript evolve on disk: the
# append-only message log + the head (with its summary folds) are written here.
# Delete this directory to start a fresh session (a re-run otherwise resumes it).
_CHECKPOINTS = Path(".grasp") / "compaction_copilot"

_SYS = """\
You are a knowledgeable researcher. For every question, FIRST call \
`recall_notes` with the topic to pull reference material, then answer in 2-3 \
sentences citing what you found. Always use the tool before answering."""

_LOREM = (
    "The subject has a long, interleaved history shaped by many contributing "
    "figures, competing standards, and turning points that repeatedly reframed "
    "how practitioners weighed the core tradeoffs against real-world constraints. "
)


@function_tool
def recall_notes(topic: str) -> str:
    """Return detailed reference notes on a topic (deliberately verbose)."""
    body = _LOREM * 4
    return "\n\n".join(f"Note {i} — {topic}: {body}" for i in range(1, 13))


def build_copilot(
    *, model: str = DEFAULT_MODEL
) -> tuple[LLMAgent[str, str, None], SessionContext[None]]:
    """Build the assistant with a tiny compaction budget and its context."""
    llm = OpenAIResponsesLLM(
        model_name=model,
        llm_settings=cast(
            "OpenAIResponsesLLMSettings",
            {"reasoning": {"effort": "low", "summary": "detailed"}},
        ),
    )
    ctx = SessionContext[None](
        state=None,
        checkpoint_store=FileCheckpointStore(_CHECKPOINTS),
        session_key="compaction-demo",
    )
    agent = LLMAgent[str, str, None](
        name="researcher",
        ctx=ctx,
        llm=llm,
        tools=[recall_notes],
        sys_prompt=_SYS,
        max_turns=12,
        stream_llm=True,
    )
    # Normally compaction is one line — ``agent.add_compaction()`` — with the
    # budget auto-derived from the model (``ContextBudget.for_model``). This demo
    # *overrides* it with a tiny artificial window so summarization fires within a
    # couple of turns, and registers the summarizing compactor directly (rather
    # than the full Compaction bundle, which would also collapse tool outputs
    # first) so the lossy summary is the visible effect — a ⊙ context-compacted
    # panel. The summarizer's own input cap is still inferred from its model.
    budget = ContextBudget(
        model=model, max_input_tokens=ARTIFICIAL_WINDOW, buffer_tokens=BUFFER_TOKENS
    )
    agent.add_compactor(
        SummarizingCompactor(
            summarizer=LLMSummarizer(llm),
            budget=budget,
            keep_recent_turns=KEEP_RECENT_TURNS,
        )
    )
    return agent, ctx


def main() -> None:
    from grasp_agents.ui import run_tui_interactive  # noqa: PLC0415

    agent, _ = build_copilot()
    # Passing the agent auto-wires stepped delivery (so /rollback can rewind), the
    # token meter, and the meter window — inferred from the agent's compaction
    # budget (the artificial window set in build_copilot).
    run_tui_interactive(agent)


if __name__ == "__main__":
    main()
