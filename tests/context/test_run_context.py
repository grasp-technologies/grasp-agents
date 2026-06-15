"""
RunContext.record_response bounding (prevents unbounded growth of
``responses`` on a long-lived shared ctx).
"""

from __future__ import annotations

from grasp_agents.run_context import RunContext
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
    ctx: RunContext[None] = RunContext(state=None)
    for _ in range(5):
        ctx.record_response("a", _response())
    assert len(ctx.responses["a"]) == 5


def test_responses_trimmed_to_cap_keeping_most_recent() -> None:
    ctx: RunContext[None] = RunContext(state=None, max_responses_per_agent=2)
    responses = [_response(str(i)) for i in range(5)]
    for r in responses:
        ctx.record_response("a", r)

    bucket = ctx.responses["a"]
    assert len(bucket) == 2
    assert bucket[0] is responses[3]
    assert bucket[1] is responses[4]


def test_cap_is_per_agent() -> None:
    ctx: RunContext[None] = RunContext(state=None, max_responses_per_agent=1)
    ctx.record_response("a", _response())
    ctx.record_response("b", _response())
    ctx.record_response("a", _response())
    assert len(ctx.responses["a"]) == 1
    assert len(ctx.responses["b"]) == 1
