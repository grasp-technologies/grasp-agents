"""
Tools must be stateless and safe to share across agents/sessions.

Regression guard for the ``ProcessorTool`` cross-session footgun: a single
tool instance attached to several hosts (manager/worker graphs, LRU-cached
agents) used to bind its wrapped template to whichever host adopted last, so a
dispatch could resolve to a stale session. Binding now happens per call, from
the call-time ``ctx`` — the clone that actually runs always wins.
"""

from __future__ import annotations

from pydantic import BaseModel

from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.tools.processor_tool import ProcessorTool


class _Msg(BaseModel):
    text: str = ""


class _Pass(Processor[_Msg, _Msg, None]):
    """Minimal passthrough processor used as a template."""


class _Host:
    """Duck-typed BaseTool-parent exposing only tracing settings."""

    tracing_enabled = True
    tracing_exclude_input_fields = None


def _tool() -> ProcessorTool[_Msg, _Msg, None]:
    return ProcessorTool(
        processor=_Pass(name="worker"),
        name="worker_tool",
        description="wraps a worker",
    )


def test_resolve_binds_each_call_to_its_own_ctx() -> None:
    tool = _tool()
    ctx_a: RunContext[None] = RunContext(state=None)
    ctx_b: RunContext[None] = RunContext(state=None)

    # The same tool instance is attached to two different hosts.
    tool.on_adopted(_Host())
    tool.on_adopted(_Host())

    # Each dispatch clones the template and binds the clone to that call's ctx.
    proc_a = tool._resolve_processor(ctx=ctx_a)
    proc_b = tool._resolve_processor(ctx=ctx_b)

    assert proc_a.ctx is ctx_a
    assert proc_b.ctx is ctx_b
    # Distinct clones, never the shared template.
    assert proc_a is not proc_b
    assert proc_a is not tool.processor
    assert proc_b is not tool.processor


def test_call_ctx_overrides_adoption_ctx() -> None:
    """
    The cross-session guard: even when the tool was adopted under one host's
    ctx, a dispatch with a different ctx resolves to the *call* ctx.
    """
    tool = _tool()
    adopt_ctx: RunContext[None] = RunContext(state=None)
    call_ctx: RunContext[None] = RunContext(state=None)

    class _HostWithCtx:
        tracing_enabled = True
        tracing_exclude_input_fields = None
        ctx = adopt_ctx
        path = ["host"]  # noqa: RUF012

    tool.on_adopted(_HostWithCtx())  # template adopted under adopt_ctx

    proc = tool._resolve_processor(ctx=call_ctx)
    assert proc.ctx is call_ctx
    assert proc.ctx is not adopt_ctx
