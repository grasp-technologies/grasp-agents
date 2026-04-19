"""
Blocking tool-call approval helper.

Wraps a user-supplied approver callback into a :class:`BeforeToolHook`,
gating each matched tool call on a yes/no answer. The approver is
awaited inline — suited for CLI prompts, blocking HTTP calls to a
policy service, or any same-task check.

For approvals that cross asynchronous boundaries (UI click minutes
later, persistent allow-always across runs) use
:mod:`grasp_agents.agent.approval_store` instead.

Usage::

    async def approve(call, *, ctx, exec_id) -> bool:
        ans = input(f"Run {call.name}({call.arguments})? [y/N] ").strip()
        return ans.lower() == "y"

    agent.add_before_tool_hook(
        build_callback_approval(approve, tool_names={"delete_file"})
    )

Denied calls surface to the LLM as :class:`RejectToolContent` so the
model sees a rejection message rather than silently losing the call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..run_context import CtxT
from .tool_decision import RejectToolContent, ToolCallDecision

if TYPE_CHECKING:
    from collections.abc import Container, Mapping, Sequence

    from ..run_context import RunContext
    from ..types.hooks import BeforeToolHook
    from ..types.items import FunctionToolCallItem


DEFAULT_DENY_MESSAGE = "User denied tool call '{name}'."


class ApprovalCallback(Protocol[CtxT]):
    """
    User-supplied approver: returns ``True`` to let the call run,
    ``False`` to deny and surface :class:`RejectToolContent` to the LLM.

    The approver is awaited sequentially, once per gated call in a
    batch, with the call, run context, and exec id — enough to render
    a prompt, consult ``ctx.state``, or log a decision.
    """

    async def __call__(
        self,
        call: FunctionToolCallItem,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> bool: ...


def build_callback_approval(
    approve: ApprovalCallback[CtxT],
    *,
    tool_names: Container[str] | None = None,
    deny_message: str = DEFAULT_DENY_MESSAGE,
) -> BeforeToolHook[CtxT]:
    """
    Wrap ``approve`` into a :class:`BeforeToolHook`.

    Every call in a batch whose ``name`` is in ``tool_names`` is passed
    to ``approve``; unmatched calls default to
    :class:`~grasp_agents.agent.tool_decision.AllowTool`. When
    ``tool_names`` is ``None`` every call is gated.

    ``deny_message`` is formatted via ``str.format(name=..., arguments=...)``
    with the call's ``name`` and ``arguments``, so the template can
    reference either field.

    The returned hook returns ``None`` when no denials occurred so the
    agent loop takes the fast "allow all" path without iterating an
    empty decision map.
    """

    async def hook(
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> Mapping[str, ToolCallDecision] | None:
        decisions: dict[str, ToolCallDecision] = {}
        for call in tool_calls:
            if tool_names is not None and call.name not in tool_names:
                continue
            if await approve(call, ctx=ctx, exec_id=exec_id):
                continue
            decisions[call.call_id] = RejectToolContent(
                content=deny_message.format(
                    name=call.name, arguments=call.arguments
                )
            )
        return decisions or None

    return hook
