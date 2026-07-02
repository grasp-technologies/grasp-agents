"""
Interactive approval copilot — a tool-approval-dialog demo for the TUI.

An "ops" assistant over a tiny in-memory records database. Two of its tools are
destructive and gated behind human approval: ``delete_record`` and
``update_record`` can't run until you say so. Safe reads (``list_records``) run
without prompting.

When the agent calls a gated tool, a dialog pops up — **once**, **session**
(that tool then stops prompting), **always** (persisted to disk, so it survives
restarts), or **deny** (the model is told it was denied and continues). ``esc``
on the dialog denies; ``esc`` with no dialog open interrupts the whole run.

"Always" decisions are written to ``.grasp/approval_copilot.json`` — allow a
tool permanently, restart the demo, and it won't prompt for that tool again.

Try, for example::

    Delete record 3, then archive record 1.

Run (needs ``OPENAI_API_KEY`` in ``.env``)::

    python -m grasp_agents.examples.tui.approval_copilot

Requires the ``tui`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from grasp_agents import LLMAgent, SessionContext
from grasp_agents.agent.approval_store import build_store_approval
from grasp_agents.llm_providers.openai_responses import (
    OpenAIResponsesLLM,
    OpenAIResponsesLLMSettings,
)
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.ui import TuiApprovalStore

DEFAULT_MODEL = "gpt-5.4-nano"

# "Allow always" decisions are written here, so they survive restarts: re-run
# the demo and a tool you allowed permanently is no longer prompted for.
APPROVALS_PATH = Path(".grasp") / "approval_copilot.json"

# A tiny in-memory "production database" the assistant operates on — the demo is
# entirely self-contained (no real I/O), so the approval flow is the star.
_RECORDS: dict[int, dict[str, str]] = {
    1: {"name": "alice", "status": "active", "created": "2019"},
    2: {"name": "bob", "status": "active", "created": "2021"},
    3: {"name": "carol", "status": "archived", "created": "2018"},
}

# Destructive tools whose calls go through the approval dialog (reads do not).
GATED_TOOLS = {"delete_record", "update_record"}

_SYS = """\
You are an operations assistant managing a small records database. Use the tools \
to read and modify records. Do not ask the user for any approvals explicitly, they \
are handled at system level."""


@function_tool
def list_records() -> str:
    """List every record id with its fields. Safe — runs without approval."""
    return "\n".join(f"{rid}: {fields}" for rid, fields in sorted(_RECORDS.items()))


@function_tool
def delete_record(record_id: int) -> str:
    """Permanently delete the record with this id. Requires approval."""
    if record_id not in _RECORDS:
        return f"No record {record_id}."
    del _RECORDS[record_id]
    return f"Deleted record {record_id}."


@function_tool
def update_record(record_id: int, field: str, value: str) -> str:
    """Set one field on a record (e.g. status=archived). Requires approval."""
    if record_id not in _RECORDS:
        return f"No record {record_id}."
    _RECORDS[record_id][field] = value
    return f"Set {field}={value!r} on record {record_id}."


def build_copilot(
    *, model: str = DEFAULT_MODEL, persist_path: Path | None = APPROVALS_PATH
) -> tuple[LLMAgent[str, str, None], SessionContext[None]]:
    """Build the ops assistant (gated tools) and its approval-enabled context."""
    llm = OpenAIResponsesLLM(
        model_name=model,
        llm_settings=cast(
            "OpenAIResponsesLLMSettings",
            {"reasoning": {"effort": "medium", "summary": "auto"}},
        ),
    )
    # The TUI drains this store's pending queue and pops an approval dialog per
    # gated call; session_key scopes "allow for session" decisions, and
    # persist_path makes "allow always" decisions survive restarts.
    store = TuiApprovalStore(persist_path=persist_path)
    ctx = SessionContext[None](state=None, approval_store=store, session_key="ops-demo")
    agent = LLMAgent[str, str, None](
        name="ops_assistant",
        ctx=ctx,
        llm=llm,
        tools=[list_records, delete_record, update_record],
        sys_prompt=_SYS,
        stream_llm=True,
    )
    # The gate is a before-tool hook: it consults ctx.approval_store for the
    # named tools and parks each call until the dialog resolves it.
    agent.add_before_tool_hook(build_store_approval(tool_names=GATED_TOOLS))
    return agent, ctx


def main() -> None:
    from grasp_agents.ui import run_tui_interactive  # noqa: PLC0415

    agent, _ = build_copilot()
    run_tui_interactive(agent)


if __name__ == "__main__":
    main()
