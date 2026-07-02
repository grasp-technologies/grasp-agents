"""
SessionContext.record_response bounding (prevents unbounded growth of
``responses`` on a long-lived shared ctx) + the deprecated pre-rename aliases.
"""

from __future__ import annotations

from grasp_agents.session_context import SessionContext
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import OutputMessageItem
from grasp_agents.types.response import Response


def _response(text: str = "x") -> Response:
    return Response(
        model="test-model",
        output_items=[
            OutputMessageItem(
                content_parts=[OutputMessageText(text=text)],
                status="completed",
            )
        ],
    )


def test_responses_unbounded_by_default() -> None:
    ctx: SessionContext[None] = SessionContext(state=None)
    for _ in range(5):
        ctx.record_response("a", _response())
    assert len(ctx.responses["a"]) == 5


def test_responses_trimmed_to_cap_keeping_most_recent() -> None:
    ctx: SessionContext[None] = SessionContext(state=None, max_responses_per_agent=2)
    responses = [_response(str(i)) for i in range(5)]
    for r in responses:
        ctx.record_response("a", r)

    bucket = ctx.responses["a"]
    assert len(bucket) == 2
    assert bucket[0] is responses[3]
    assert bucket[1] is responses[4]


def test_cap_is_per_agent() -> None:
    ctx: SessionContext[None] = SessionContext(state=None, max_responses_per_agent=1)
    ctx.record_response("a", _response())
    ctx.record_response("b", _response())
    ctx.record_response("a", _response())
    assert len(ctx.responses["a"]) == 1
    assert len(ctx.responses["b"]) == 1


def test_deprecated_run_context_aliases() -> None:
    # Pre-rename names must keep working (root export and old module path)
    # until the deprecation window closes.
    import grasp_agents
    import grasp_agents.run_context as old_module
    from grasp_agents.session_context import (
        current_session_context,
        reset_default_session_context,
    )

    assert grasp_agents.RunContext is SessionContext
    assert old_module.RunContext is SessionContext
    assert old_module.current_run_context is current_session_context
    assert old_module.reset_default_run_context is reset_default_session_context
