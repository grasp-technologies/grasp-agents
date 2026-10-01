from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.evals import (
    Dataset,
    EvalContext,
    Example,
    ProcessorTask,
    evaluate,
    evaluator,
)
from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.processors.processor import Processor
from grasp_agents.session_context import SessionContext
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import Event, LLMStreamEvent, ProcPayloadOutEvent
from grasp_agents.types.items import InputMessageItem, OutputMessageItem
from grasp_agents.types.packet import Packet
from grasp_agents.types.response import Response
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow
from tests._helpers import MockLLM, _make_usage


@dataclass(frozen=True)
class EchoLLM(MockLLM):
    """Answers with the upper-cased last user message, whatever the call order."""

    async def _generate_response_once(self, input: Sequence[Any], **_: Any) -> Response:
        text = ""
        for item in reversed(list(input)):
            if isinstance(item, InputMessageItem) and item.role == "user":
                text = " ".join(
                    getattr(part, "text", "") for part in item.content
                ).strip()
                break
        return Response(
            model="echo-1",
            output=[
                OutputMessageItem(
                    content=[OutputMessageText(text=text.upper())], status="completed"
                )
            ],
            usage=_make_usage(),
        )


class Counter(BaseModel):
    seen: list[int] = []


class _AddOne(Processor[int, int, Any]):
    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[int] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for value in in_args or []:
            state = self.ctx.state
            if isinstance(state, Counter):
                state.seen.append(value)
            yield ProcPayloadOutEvent(data=value + 1, source=self.name, exec_id=exec_id)


def _ints(n: int = 3) -> Dataset[int, int]:
    return Dataset([Example(id=f"i{i}", input=i, reference=i + 2) for i in range(n)])


@evaluator
def matches(ctx: EvalContext[Any, Any, Any]) -> bool:
    return ctx.output == ctx.reference


class TestAnyProcessor:
    @pytest.mark.asyncio
    async def test_workflow_template_is_copied_per_trial(self) -> None:
        workflow = SequentialWorkflow[int, int, Any](
            name="plus_two", subprocs=[_AddOne(name="a"), _AddOne(name="b")]
        )
        run = await evaluate(ProcessorTask(workflow), _ints(), [matches], persist=False)
        assert run.metric("pass_rate(matches)").value == pytest.approx(1.0)  # type: ignore[union-attr]
        assert run.task.name == "plus_two"
        assert run.task.kind.endswith("SequentialWorkflow")

    @pytest.mark.asyncio
    async def test_each_trial_gets_its_own_session(self) -> None:
        sessions: list[SessionContext[Any]] = []

        def ctx_factory(example: Example[Any, Any]) -> SessionContext[Any]:
            ctx = SessionContext[Counter](state=Counter())
            sessions.append(ctx)
            return ctx

        @evaluator
        def outcome(ctx: EvalContext[int, int, int]) -> bool:
            assert ctx.session is not None
            return ctx.session.state.seen == [ctx.input]

        task = ProcessorTask(_AddOne(name="add"), ctx_factory=ctx_factory)
        run = await evaluate(task, _ints(), [outcome], concurrency=3, persist=False)
        assert len(sessions) == 3
        assert len({id(s) for s in sessions}) == 3
        assert all(t.score("outcome").value is True for t in run.trials)  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_output_fn_reads_state(self) -> None:
        def final_state(packet: Packet[Any], ctx: SessionContext[Any]) -> list[int]:
            return list(ctx.state.seen)

        task = ProcessorTask(
            _AddOne(name="add"),
            ctx_factory=lambda _: SessionContext[Counter](state=Counter()),
            output_fn=final_state,
        )
        run = await evaluate(task, _ints(2), persist=False)
        assert [t.output for t in run.trials] == [[0], [1]]

    @pytest.mark.asyncio
    async def test_factory_builds_inside_trial_session(self) -> None:
        built: list[Processor[int, int, Any]] = []

        def factory() -> Processor[int, int, Any]:
            proc = _AddOne(name="fresh")
            built.append(proc)
            return proc

        run = await evaluate(ProcessorTask(factory), _ints(), [matches], persist=False)
        assert len(built) == 3
        assert len({id(p.ctx) for p in built}) == 3
        assert run.counts.task_errors == 0

    @pytest.mark.asyncio
    async def test_parallel_fan_out_returns_list(self) -> None:
        parallel = ParallelProcessor[int, int, Any](_AddOne(name="add"))
        dataset = Dataset(
            [
                Example[list[int], list[int]](
                    id="p", input=[1, 2, 3], reference=[2, 3, 4]
                )
            ]
        )
        task = ProcessorTask(parallel, input_fn=list)
        run = await evaluate(task, dataset, [matches], persist=False)
        assert run.trials[0].output == [2, 3, 4]
        assert run.trials[0].score("matches").value is True  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_processor_failure_is_a_task_error(self) -> None:
        class Boom(_AddOne):
            async def _process_stream(
                self, *args: Any, **kwargs: Any
            ) -> AsyncIterator[Event[Any]]:
                raise RuntimeError("pipeline broke")
                yield  # pragma: no cover

        run = await evaluate(
            ProcessorTask(Boom(name="boom")), _ints(1), [matches], persist=False
        )
        error = run.trials[0].error
        assert error is not None
        assert "pipeline broke" in error.message


class TestLLMAgent:
    @pytest.mark.asyncio
    async def test_agent_trials_are_isolated_and_observed(self) -> None:
        agent = LLMAgent[str, str, None](name="shouter", llm=EchoLLM())
        dataset = Dataset(
            [
                Example[str, str](id=w, input=w, reference=w.upper())
                for w in ["alpha", "beta", "gamma"]
            ]
        )

        @evaluator
        def transcript_has_no_deltas(ctx: EvalContext[str, str, str]) -> bool:
            return not any(isinstance(e, LLMStreamEvent) for e in ctx.events) and bool(
                ctx.events
            )

        run = await evaluate(
            ProcessorTask(agent, input_mode="chat"),
            dataset,
            [matches, transcript_has_no_deltas],
            concurrency=3,
            persist=False,
        )
        assert run.counts.task_errors == 0, [t.error for t in run.trials]
        assert all(t.score("matches").value is True for t in run.trials)  # type: ignore[union-attr]
        assert all(
            t.score("transcript_has_no_deltas").value is True for t in run.trials
        )  # type: ignore[union-attr]
        trial = run.trials[0]
        assert trial.usage.input_tokens == 10
        assert trial.usage.output_tokens == 5
        assert trial.usage_by_agent.keys() == {"shouter"}
        assert trial.models == {"shouter": ["echo-1"]}
        assert run.provenance.observed_models == {"shouter": ["echo-1"]}
        # The template agent itself never ran.
        assert len(agent.transcript.messages) == 0
