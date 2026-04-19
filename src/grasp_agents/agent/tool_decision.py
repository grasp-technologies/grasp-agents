"""
Per-call decisions a :class:`BeforeToolHook` can return to control
tool execution in a batch: run normally, reject with a synthetic
output, or abort the whole batch by raising.

The hook return type is ``Mapping[str, ToolCallDecision] | None`` keyed
by ``FunctionToolCallItem.call_id``. Missing keys default to
:class:`AllowTool`; a ``None`` return is equivalent to allowing every
call.

Usage::

    @agent.add_before_tool_hook
    async def guard(*, tool_calls, ctx, exec_id):
        decisions: dict[str, ToolCallDecision] = {}
        for call in tool_calls:
            if call.name == "delete_all" and not ctx.state.admin:
                decisions[call.call_id] = RejectToolContent(
                    content="Permission denied: admin-only tool."
                )
        return decisions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class AllowTool:
    """Allow the tool to run normally (the default)."""


@dataclass(frozen=True, slots=True)
class RejectToolContent:
    """
    Skip tool execution; surface ``content`` to the LLM as the tool's
    output for that ``call_id``.

    The loop fabricates a :class:`FunctionToolOutputItem`, appends it
    to memory, and emits a :class:`ToolResultEvent` so downstream
    consumers (printer, telemetry) see a normal tool round.
    """

    content: str


@dataclass(frozen=True, slots=True)
class RaiseToolException:
    """
    Abort the entire tool batch by raising ``exception``.

    The exception propagates out of the agentic loop; no tools in this
    batch execute, including ones marked :class:`AllowTool`. Intended
    for hard stops (policy violation, kill-switch) rather than per-tool
    rejection — use :class:`RejectToolContent` for the latter.
    """

    exception: Exception


ToolCallDecision: TypeAlias = AllowTool | RejectToolContent | RaiseToolException
