"""
Tests for tracing-setting propagation to children (PR #13 + #15).

A field a parent masks from traces must stay masked in every descendant —
direct tools, sub-agents wrapped as tools (:class:`ProcessorTool`),
dynamically-spawned subagents (:class:`AgentTool`), and container subprocs.
Inheritance is a union, so a child's own excludes are never shadowed.

``tracing_enabled`` follows the same downward-restriction rule: a parent that
disables tracing forces it off in every descendant, but a parent that leaves
it on never re-enables a child that opted out.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.agent_tool import AgentTool, AgentToolInput
from grasp_agents.agent.function_tool import function_tool
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.processor_tool import ProcessorTool
from grasp_agents.llm.llm import LLM
from grasp_agents.run_context import RunContext
from grasp_agents.types.llm_events import LlmEvent
from grasp_agents.types.response import Response
from grasp_agents.types.tool import BaseTool
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow

# ruff: noqa: ARG002, SLF001


@dataclass(frozen=True)
class _StubLLM(LLM):
    """Construction-only stub — the generate methods are never called."""

    model_name: str = "mock"

    async def _generate_response_once(
        self,
        input: Sequence[Any],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        raise NotImplementedError

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        raise NotImplementedError
        yield  # pragma: no cover  (marks this an async generator)


def _llm() -> _StubLLM:
    return _StubLLM()


def test_direct_tool_inherits_on_construction() -> None:
    @function_tool
    async def search(query: str) -> str:
        return query

    agent = LLMAgent[str, str, None](
        name="agent",
        llm=_llm(),
        tools=[search],
        tracing_exclude_input_fields={"secret"},
    )
    (tool,) = agent._loop.tools.values()
    assert tool.tracing_exclude_input_fields == {"secret"}


def test_processor_tool_wrapped_proc_inherits() -> None:
    child = LLMAgent[str, str, None](name="child", llm=_llm())
    ptool = ProcessorTool[Any, Any, None](
        processor=child, name="child_tool", description="wrapped"
    )
    LLMAgent[str, str, None](
        name="parent",
        llm=_llm(),
        tools=[ptool],
        tracing_exclude_input_fields={"secret"},
    )
    # The tool's own span and the wrapped processor both inherit.
    assert ptool.tracing_exclude_input_fields == {"secret"}
    assert ptool.processor.tracing_exclude_input_fields == {"secret"}


@pytest.mark.anyio
async def test_agent_tool_inherits_and_threads_to_spawned_child() -> None:
    sub = AgentTool[None](name="sub", description="subagent", llm=_llm())
    LLMAgent[str, str, None](
        name="parent",
        llm=_llm(),
        tools=[sub],
        tracing_exclude_input_fields={"secret"},
    )
    assert sub.tracing_exclude_input_fields == {"secret"}

    # The fresh child agent spawned per invocation inherits it too.
    child, _ = await sub._prepare_child(
        AgentToolInput(prompt="hi"), ctx=RunContext[None](), exec_id="e", path=None
    )
    assert child.tracing_exclude_input_fields == {"secret"}


def test_workflow_subprocs_inherit() -> None:
    a = LLMAgent[str, str, None](name="a", llm=_llm())
    b = LLMAgent[str, str, None](name="b", llm=_llm())
    SequentialWorkflow[str, str, None](
        subprocs=[a, b],
        name="wf",
        tracing_exclude_input_fields={"secret"},
    )
    assert a.tracing_exclude_input_fields == {"secret"}
    assert b.tracing_exclude_input_fields == {"secret"}


def test_union_does_not_clobber_child_excludes() -> None:
    child = LLMAgent[str, str, None](
        name="child", llm=_llm(), tracing_exclude_input_fields={"child_only"}
    )
    ptool = ProcessorTool[Any, Any, None](
        processor=child, name="child_tool", description="wrapped"
    )
    LLMAgent[str, str, None](
        name="parent",
        llm=_llm(),
        tools=[ptool],
        tracing_exclude_input_fields={"parent_only"},
    )
    # Both the parent's and the child's own excludes survive.
    assert ptool.processor.tracing_exclude_input_fields == {"child_only", "parent_only"}


def test_inherit_is_noop_for_attrless_or_none_parent() -> None:
    @function_tool
    async def search(query: str) -> str:
        return query

    search.tracing_exclude_input_fields = {"own"}

    # Parent without the attribute at all → unchanged.
    search.on_adopted(object())
    assert search.tracing_exclude_input_fields == {"own"}

    # Parent with the attribute set to None → unchanged.
    class _Parent:
        tracing_exclude_input_fields = None

    search.on_adopted(_Parent())
    assert search.tracing_exclude_input_fields == {"own"}


# --- tracing_enabled (PR #15) ---


def test_disabled_parent_forces_subprocs_off() -> None:
    a = LLMAgent[str, str, None](name="a", llm=_llm())
    b = LLMAgent[str, str, None](name="b", llm=_llm())
    SequentialWorkflow[str, str, None](
        subprocs=[a, b], name="wf", tracing_enabled=False
    )
    assert a.tracing_enabled is False
    assert b.tracing_enabled is False


def test_enabled_parent_does_not_reenable_optedout_child() -> None:
    a = LLMAgent[str, str, None](name="a", llm=_llm(), tracing_enabled=False)
    b = LLMAgent[str, str, None](name="b", llm=_llm())
    # Workflow leaves tracing on (default) -> a's own opt-out must survive.
    SequentialWorkflow[str, str, None](subprocs=[a, b], name="wf")
    assert a.tracing_enabled is False
    assert b.tracing_enabled is True


def test_enabled_is_noop_for_parent_without_the_flag() -> None:
    agent = LLMAgent[str, str, None](name="a", llm=_llm())

    # A Runner-like parent has ctx + path but no ``tracing_enabled`` attribute,
    # so it imposes no tracing restriction.
    class _RunnerLike:
        ctx = RunContext[None]()
        path = ["runner"]

    agent.on_adopted(_RunnerLike())
    assert agent.tracing_enabled is True


@pytest.mark.anyio
async def test_agent_tool_threads_disabled_tracing_to_spawned_child() -> None:
    sub = AgentTool[None](name="sub", description="subagent", llm=_llm())
    LLMAgent[str, str, None](
        name="parent",
        llm=_llm(),
        tools=[sub],
        tracing_enabled=False,
    )
    # The tool inherits the host's disabled tracing...
    assert sub.tracing_enabled is False

    # ...and threads it (via on_adopted) onto the freshly-spawned child.
    child, _ = await sub._prepare_child(
        AgentToolInput(prompt="hi"), ctx=RunContext[None](), exec_id="e", path=None
    )
    assert child.tracing_enabled is False
