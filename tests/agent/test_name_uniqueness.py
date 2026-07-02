"""
Construction-time name-uniqueness guards.

Tool dispatch and event routing/capture are all keyed by ``source == name``, so
collisions are rejected at construction (fail-fast) rather than silently
aliasing one entity's events to another's:

- an agent's tool names must be unique and must not equal the agent's own name
  (a sub-agent is itself a tool here, so this covers sub-agents too);
- a runner's / workflow's processor names must be unique and must not equal the
  container's own name;
- a node's tool name must not equal another processor's name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.llm.llm import LLM
from grasp_agents.processors.processor import Processor
from grasp_agents.runner.runner import END_PROC_NAME, Runner
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.errors import RunnerError, WorkflowConstructionError
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from grasp_agents.types.io import ProcName
    from grasp_agents.types.response import Response


# --- minimal stubs (no real LLM / network) ----------------------------------


class _Pass(Processor[str, str, None]):
    """Minimal str->str passthrough processor."""


def _proc(name: str, recipients: list[ProcName] | None = None) -> _Pass:
    return _Pass(name=name, recipients=recipients)


@dataclass(frozen=True)
class _StubLLM(LLM):
    """A never-called LLM, just to construct an LLMAgent."""

    model_name: str = "stub"

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> Response:
        raise NotImplementedError

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[Any]:
        raise NotImplementedError
        yield


class _ToolInput(BaseModel):
    pass


class _Tool(BaseTool[_ToolInput, str, Any]):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, description="x")

    async def _run(
        self,
        inp: _ToolInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
        path: Any = None,
        agent_ctx: Any = None,
    ) -> str:
        return "ok"


def _agent(name: str, tool_names: list[str]) -> LLMAgent[str, str, None]:
    return LLMAgent[str, str, None](
        name=name,
        ctx=SessionContext[None](state=None),
        llm=_StubLLM(),
        tools=[_Tool(t) for t in tool_names],
    )


# --- agent: tool names -------------------------------------------------------


def test_agent_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="duplicate tool names"):
        _agent("a", ["t", "t"])


def test_agent_rejects_tool_named_like_the_agent() -> None:
    with pytest.raises(ValueError, match="own name"):
        _agent("a", ["a"])


def test_agent_accepts_unique_tool_names() -> None:
    _agent("a", ["t1", "t2"])  # no raise


# --- runner: processor names -------------------------------------------------


def test_runner_rejects_duplicate_processor_names() -> None:
    a = _proc("dup", recipients=[END_PROC_NAME])
    b = _proc("dup", recipients=["dup"])
    with pytest.raises(RunnerError, match="Duplicate processor names"):
        Runner(entry_proc=a, procs=[a, b])


def test_runner_rejects_node_named_like_the_runner() -> None:
    a = _proc("node", recipients=[END_PROC_NAME])
    with pytest.raises(RunnerError, match="runner's own name"):
        Runner(entry_proc=a, procs=[a], name="node")


def test_runner_rejects_tool_name_colliding_with_a_node_name() -> None:
    # node "A" has a tool named "B", and "B" is also a node → collision
    a = LLMAgent[str, str, None](
        name="A",
        ctx=SessionContext[None](state=None),
        llm=_StubLLM(),
        tools=[_Tool("B")],
        recipients=[END_PROC_NAME],
    )
    b = _proc("B", recipients=["A"])
    with pytest.raises(RunnerError, match="colliding with processor names"):
        Runner(entry_proc=a, procs=[a, b])


# --- workflow: subprocessor names --------------------------------------------


def test_workflow_rejects_duplicate_subproc_names() -> None:
    with pytest.raises(WorkflowConstructionError, match="Duplicate subprocessor"):
        SequentialWorkflow(name="w", subprocs=[_proc("x"), _proc("x")])


def test_workflow_rejects_subproc_named_like_the_workflow() -> None:
    with pytest.raises(WorkflowConstructionError, match="own name"):
        SequentialWorkflow(name="w", subprocs=[_proc("a"), _proc("w")])
