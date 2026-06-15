"""
Regression tests for the P2 orchestration / composition fixes
(consolidated audit 2026-06-11, §3 items 25-26).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.runner.event_bus import MAX_QUEUE_SIZE, EventBus
from grasp_agents.runner.runner import END_PROC_NAME, Runner
from grasp_agents.types.errors import ProcInputValidationError, RunnerError
from grasp_agents.types.events import (
    Event,
    ProcPacketOutEvent,
    ProcPayloadOutEvent,
    RoutedEvent,
)
from grasp_agents.types.packet import Packet
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow

from .test_runner import (  # type: ignore[attr-defined]
    AppendProcessor,
    FailOnCallProcessor,
    run_runner,
)
from .test_sessions import (  # type: ignore[attr-defined]
    MockLLM,
    _text_response,
)

# ---------- Item 26: parameterized-generic InT ----------


class ListProcessor(Processor[list[int], list[int], None]):
    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[list[int]] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for arg in in_args or []:
            yield ProcPayloadOutEvent(data=arg, source=self.name, exec_id=exec_id)


class TestParameterizedGenericInType:
    def test_single_list_arg_no_typeerror(self) -> None:
        proc = ListProcessor(name="p")
        out = proc.validate_inputs(exec_id="e", in_args=[1, 2, 3])
        # One argument of the declared list type — not three int args.
        assert out == [[1, 2, 3]]

    def test_packet_payloads_coerced(self) -> None:
        proc = ListProcessor(name="p")
        packet = Packet[Any](sender="x", payloads=[[1, 2], [3]])
        out = proc.validate_inputs(exec_id="e", in_packet=packet)
        assert out == [[1, 2], [3]]


# ---------- Item 25: multi-payload fan-in to an LLMAgent ----------


class TestLLMAgentFanIn:
    def test_multi_payload_packet_aggregates_into_list_input(self) -> None:
        agent = LLMAgent[list[str], str, None](
            name="t",
            llm=MockLLM(responses_queue=[_text_response("ok")]),
            env_info=False,
        )
        packet = Packet[Any](sender="par", payloads=["a", "b", "c"])
        result = agent.validate_inputs(exec_id="e", in_packet=packet)
        assert result == [["a", "b", "c"]]

    def test_non_list_agent_raises_with_hint(self) -> None:
        agent = LLMAgent[str, str, None](
            name="t",
            llm=MockLLM(responses_queue=[]),
            env_info=False,
        )
        packet = Packet[Any](sender="par", payloads=["a", "b"])
        with pytest.raises(
            ProcInputValidationError, match=r"list\[\.\.\.\] to fan in"
        ):
            agent.validate_inputs(exec_id="e", in_packet=packet)


# ---------- Item 26: bounded event-bus queues ----------


class TestEventBusBounds:
    def test_queues_are_bounded(self) -> None:
        bus = EventBus()
        assert bus._streamed_event_queue.maxsize == MAX_QUEUE_SIZE

        async def handler(event: Event[Any], **kwargs: Any) -> None:
            del event, kwargs

        bus.register_event_handler("x", handler)
        assert bus._routed_event_queues["x"].maxsize == MAX_QUEUE_SIZE

    @pytest.mark.asyncio
    async def test_shutdown_with_full_queue_does_not_hang(self) -> None:
        bus = EventBus()

        async def handler(event: Event[Any], **kwargs: Any) -> None:
            del event, kwargs

        bus.register_event_handler("x", handler)
        queue = bus._routed_event_queues["x"]
        dummy = ProcPayloadOutEvent(data="d", source="x", exec_id="e")
        while not queue.full():
            queue.put_nowait(dummy)

        # The consumer is gone (bus never entered) — shutdown must still
        # complete by making room for the sentinel.
        await asyncio.wait_for(bus.shutdown(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_crashing_handler_retrieves_future_exception(self) -> None:
        # A crashing handler delivers its error via the TaskGroup; the
        # final-result future's identical exception must be retrieved on exit
        # so a GC'd future can't trip asyncio's "Future exception was never
        # retrieved" warning in an unrelated later task.
        bus = EventBus()

        async def boom(event: Event[Any], **kwargs: Any) -> None:
            del event, kwargs
            raise RuntimeError("handler boom")

        async def drive() -> None:
            async with bus:
                bus.register_event_handler("p", boom)
                await bus.post(RoutedEvent(type="t", data="x", destination="p"))
                async for _ in bus.stream_events():
                    pass

        with pytest.raises(BaseExceptionGroup):
            await drive()

        fut = bus._final_result_fut
        assert fut is not None
        assert fut.done()
        assert fut._log_traceback is False  # retrieved during __aexit__
        assert isinstance(fut.exception(), RuntimeError)


# ---------- Item 26: Runner[OutT] validates final payloads ----------


class IntProducer(Processor[str, int, None]):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, recipients=[END_PROC_NAME])

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        yield ProcPayloadOutEvent(data=42, source=self.name, exec_id=exec_id)


class TestRunnerOutTypeValidation:
    @pytest.mark.asyncio
    async def test_mismatched_final_payload_raises(self) -> None:
        p = IntProducer("P")
        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](entry_proc=p, procs=[p], ctx=ctx, name="r")

        with pytest.raises(Exception) as excinfo:
            await run_runner(runner, chat_inputs="s")
        assert excinfo.group_contains(
            RunnerError, match="does not match the runner's output type"
        )

    @pytest.mark.asyncio
    async def test_matching_final_payload_passes(self) -> None:
        a = AppendProcessor("A", recipients=[END_PROC_NAME])
        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner[str, None](entry_proc=a, procs=[a], ctx=ctx, name="r")

        result = await run_runner(runner, chat_inputs="s")
        assert result == ["s->A"]

    @pytest.mark.asyncio
    async def test_unsubscripted_runner_skips_validation(self) -> None:
        p = IntProducer("P")
        ctx: RunContext[None] = RunContext(state=None)
        runner = Runner(entry_proc=p, procs=[p], ctx=ctx, name="r")

        payloads: list[Any] = []
        async for event in runner.run_stream(chat_inputs="s"):
            if type(event).__name__ == "RunPacketOutEvent":
                payloads = list(event.data.payloads)  # type: ignore[attr-defined]
        assert payloads == [42]


# ---------- Item 26: sequential resume with re-delivered chat_inputs ----------


class TestSequentialResumeChatInputs:
    @pytest.mark.asyncio
    async def test_resume_with_chat_inputs_uses_checkpoint_packet(self) -> None:
        store = InMemoryCheckpointStore()

        async def drain(wf: SequentialWorkflow[str, str, None]) -> list[str]:
            payloads: list[str] = []
            async for event in wf.run_stream(chat_inputs="start", exec_id="t"):
                if (
                    isinstance(event, ProcPacketOutEvent)
                    and event.source == wf.name
                ):
                    payloads = list(event.data.payloads)
            return payloads

        # First run: A succeeds, B crashes — checkpoint has completed_step=0.
        a1 = AppendProcessor("A")
        b1 = FailOnCallProcessor("B", fail_on_call=1)
        c1 = AppendProcessor("C")
        wf1 = SequentialWorkflow[str, str, None](name="wf", subprocs=[a1, b1, c1])
        wf1.on_adopted(
            ctx=RunContext[None](
                state=None, checkpoint_store=store, session_key="seq-ci"
            )
        )
        with pytest.raises(Exception, match="Processor run failed"):
            await drain(wf1)

        # Resume re-delivering the same chat_inputs: the checkpointed packet
        # supersedes them — previously a dual-input validation error.
        a2 = AppendProcessor("A")
        b2 = FailOnCallProcessor("B", fail_on_call=99)
        c2 = AppendProcessor("C")
        wf2 = SequentialWorkflow[str, str, None](name="wf", subprocs=[a2, b2, c2])
        wf2.on_adopted(
            ctx=RunContext[None](
                state=None, checkpoint_store=store, session_key="seq-ci"
            )
        )
        result = await drain(wf2)
        assert result == ["start->A->B->C"]
