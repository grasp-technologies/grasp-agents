"""
Non-background ``AgentTool`` / ``ProcessorTool`` session-key isolation.

Previously, the child agent inside a non-bg resumable tool ran under the
parent's session key with no distinguishing ``path`` — which meant
the child's checkpoint key collided with the parent's, clobbering it.

The fix is in ``agent_loop.py``'s tool-invocation site: resumable tools
receive ``path = make_tool_call_path(parent_path, call.call_id)``,
routed to the child via :meth:`Processor.set_path`. The child
inherits the parent's ``session_key`` (approval / file-edit / usage
scopes stay shared) but lives at a nested checkpoint path
``"<session_key>/agent/<parent_path>/tc_<call_id>"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.agent_tool import AgentTool
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import (
    AgentCheckpoint,
    InMemoryCheckpointStore,
)
from grasp_agents.durability.checkpoints import ParallelCheckpoint
from grasp_agents.packet import Packet
from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.run_context import RunContext
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import ProcPacketOutEvent
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
)
from grasp_agents.types.response import Response
from grasp_agents.types.tool import BaseTool

if TYPE_CHECKING:
    from grasp_agents.llm.llm import LLM

from .test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
    _text_response,
    _tool_call_response,
)

pytestmark = pytest.mark.anyio


def _make_child_tool(responses: list[Any]) -> AgentTool[None]:
    return AgentTool[None](
        name="child_agent",
        description="Delegates to a sub-agent",
        llm=MockLLM(responses_queue=responses),
        sys_prompt="You are a helper.",
        max_turns=2,
    )


def _make_parent_agent(
    parent_llm: LLM,
    child_tool: AgentTool[None],
) -> LLMAgent[str, str, None]:
    return LLMAgent[str, str, None](
        name="parent",
        llm=parent_llm,
        tools=[child_tool],
        stream_llm=True,
    )


async def test_parent_and_child_keys_dont_collide() -> None:
    """
    After a non-bg AgentTool call, both parent and child checkpoints
    are intact at distinct keys.
    """
    store = InMemoryCheckpointStore()

    parent_llm = MockLLM(
        responses_queue=[
            _tool_call_response("child_agent", '{"prompt":"do work"}', "call_abc"),
            _text_response("parent final"),
        ]
    )
    child_tool = _make_child_tool([_text_response("child final")])
    parent = _make_parent_agent(parent_llm, child_tool)

    ctx: RunContext[None] = RunContext(checkpoint_store=store, session_key="s1")
    await parent.run("start", ctx=ctx)

    # Parent checkpoint at "<session>/agent/parent" (root processor uses
    # its own name as the first path segment).
    parent_data = await store.load("s1/agent/parent")
    assert parent_data is not None
    parent_snap = AgentCheckpoint.model_validate_json(parent_data)
    assert parent_snap.processor_name == "parent"

    # Child checkpoint at "<session>/agent/<parent_path>/tc_<call_id>".
    child_data = await store.load("s1/agent/parent/tc_call_abc")
    assert child_data is not None
    child_snap = AgentCheckpoint.model_validate_json(child_data)
    # Child's processor_name matches the AgentTool's name (deterministic
    # across resumes — no exec_id suffix).
    assert child_snap.processor_name == "child_agent"


async def test_sibling_child_calls_dont_collide() -> None:
    """
    Two AgentTool calls in the same parent turn get distinct call_ids
    → distinct checkpoint keys → no collision between siblings.
    """
    store = InMemoryCheckpointStore()

    parent_llm = MockLLM(
        responses_queue=[
            # Single turn with two tool calls — the framework emits them
            # as separate responses; serialize them to make assertion easy.
            _tool_call_response("child_agent", '{"prompt":"a"}', "call_one"),
            _tool_call_response("child_agent", '{"prompt":"b"}', "call_two"),
            _text_response("parent done"),
        ]
    )
    child_tool = _make_child_tool(
        [_text_response("child one"), _text_response("child two")]
    )
    parent = _make_parent_agent(parent_llm, child_tool)

    ctx: RunContext[None] = RunContext(checkpoint_store=store, session_key="s2")
    await parent.run("start", ctx=ctx)

    assert await store.load("s2/agent/parent/tc_call_one") is not None
    assert await store.load("s2/agent/parent/tc_call_two") is not None
    # Parent's own checkpoint is still intact.
    assert await store.load("s2/agent/parent") is not None


async def test_parallel_processor_replicas_dont_collide() -> None:
    """
    Each ParallelProcessor replica gets its own path (suffix
    = replica index) so their checkpoint keys are disjoint. Without the
    fix, all N copies would overwrite each other at the same key.
    """
    store = InMemoryCheckpointStore()
    subproc = LLMAgent[str, str, None](
        name="worker",
        llm=MockLLM(
            responses_queue=[
                _text_response("out-0"),
                _text_response("out-1"),
                _text_response("out-2"),
            ]
        ),
        stream_llm=True,
    )
    par = ParallelProcessor[str, str, None](subproc=subproc)
    ctx: RunContext[None] = RunContext(
        checkpoint_store=store, session_key="par-s", state=None
    )

    await par.run(in_args=["a", "b", "c"], ctx=ctx)

    # ParallelProcessor's own checkpoint lives at
    # ``"<session>/parallel/<parallel_name>"``; each replica's LLMAgent
    # checkpoint at ``"<session>/agent/<parallel_name>/<subproc>_<idx>"``
    # (combined segment encodes subproc name + replica index).
    assert await store.load("par-s/parallel/worker_par") is not None
    assert await store.load("par-s/agent/worker_par/worker_0") is not None
    assert await store.load("par-s/agent/worker_par/worker_1") is not None
    assert await store.load("par-s/agent/worker_par/worker_2") is not None


# ---------------------------------------------------------------------------
# Multi-step subproc: each replica resumes from its OWN checkpoint
# ---------------------------------------------------------------------------


class _EchoInput(BaseModel):
    text: str


class _EchoTool(BaseTool[_EchoInput, str, Any]):
    """Tool that echoes back its input — drives a tool-call turn."""

    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes text back.")

    async def _run(
        self,
        inp: _EchoInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
    ) -> str:
        del ctx, exec_id, progress_callback
        return f"echo: {inp.text}"


async def test_parallel_replicas_resume_multistep_from_own_checkpoints() -> None:
    """
    Multi-step subproc, partial completion: indices 0 and 2 already
    finished; index 1 stopped mid-run at AFTER_TOOL_RESULT. On resume,
    index 1 loads *its own* per-replica checkpoint (turn 1 memory with
    tool_call + tool_result already in place) and issues only the final
    LLM call. Indices 0 and 2 are redelivered from the ParallelProcessor
    checkpoint without being re-run.

    Uses pre-planted state rather than an actual crashed first-phase
    run, because ``MockLLM`` shares its response queue across replicas
    (``LLM.__deepcopy__`` returns self — an intentional sharing choice
    on the real LLM classes).
    """
    store = InMemoryCheckpointStore()

    # Plant the ParallelProcessor checkpoint: indices 0 and 2 done.
    input_packet: Packet[str] = Packet(sender="worker_par", payloads=["a", "b", "c"])
    par_cp = ParallelCheckpoint(
        session_key="par-ms",
        processor_name="worker_par",
        input_packet=input_packet,
        completed={
            0: Packet[str](sender="worker", payloads=["phase1-final-0"]),
            2: Packet[str](sender="worker", payloads=["phase1-final-2"]),
        },
    )
    await store.save("par-ms/parallel/worker_par", par_cp.model_dump_json().encode())

    # Plant replica 1's mid-run checkpoint at the new combined-segment
    # path "<session>/agent/<parallel_name>/<subproc>_1": tool has fired,
    # waiting for the final answer.
    call_id = "call_mid"
    r1_cp = AgentCheckpoint(
        session_key="par-ms",
        processor_name="worker",
        messages=[
            InputMessageItem.from_text("system", role="system"),
            InputMessageItem.from_text("b", role="user"),
            FunctionToolCallItem(
                call_id=call_id, name="echo", arguments='{"text":"b"}'
            ),
            FunctionToolOutputItem.from_tool_result(call_id=call_id, output="echo: b"),
        ],
        turn=1,
    )
    await store.save(
        "par-ms/agent/worker_par/worker_1", r1_cp.model_dump_json().encode()
    )

    # Resume: fresh subproc; its LLM only needs the final answer for
    # replica 1 (indices 0 and 2 are redelivered from the parallel cp).
    subproc = LLMAgent[str, str, None](
        name="worker",
        llm=MockLLM(
            responses_queue=[
                Response(
                    model="mock",
                    output_items=[
                        OutputMessageItem(
                            content_parts=[OutputMessageText(text="phase2-final")],
                            status="completed",
                        )
                    ],
                )
            ]
        ),
        tools=[_EchoTool()],
        stream_llm=True,
    )
    par = ParallelProcessor[str, str, None](subproc=subproc)
    ctx: RunContext[None] = RunContext(
        checkpoint_store=store, session_key="par-ms", state=None
    )

    payloads: list[str] = []
    async for event in par.run_stream(ctx=ctx, exec_id="resume", step=0):
        if isinstance(event, ProcPacketOutEvent) and event.source == par.name:
            payloads = list(event.data.payloads)

    assert len(payloads) == 3
    # Indices 0 and 2 redelivered from the planted parallel checkpoint;
    # index 1 resumed and completed with phase2-final.
    assert payloads.count("phase1-final-0") == 1
    assert payloads.count("phase1-final-2") == 1
    assert payloads.count("phase2-final") == 1
