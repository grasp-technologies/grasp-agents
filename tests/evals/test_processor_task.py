from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.evals import (
    Dataset,
    Example,
    ProcessorTask,
    ScoreContext,
    evaluate,
    scorer,
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


@scorer
def matches(ctx: ScoreContext[Any, Any, Any]) -> bool:
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

        @scorer
        def outcome(ctx: ScoreContext[int, int, int]) -> bool:
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

        @scorer
        def transcript_has_no_deltas(ctx: ScoreContext[str, str, str]) -> bool:
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


@dataclass(frozen=True)
class CountingLLM(MockLLM):
    """Answers with how many user messages the conversation holds."""

    async def _generate_response_once(self, input: Sequence[Any], **_: Any) -> Response:
        users = sum(
            1
            for item in input
            if isinstance(item, InputMessageItem) and item.role == "user"
        )
        return Response(
            model="count-1",
            output=[
                OutputMessageItem(
                    content=[OutputMessageText(text=str(users))], status="completed"
                )
            ],
            usage=_make_usage(),
        )


class TestTrialIsolation:
    @pytest.mark.asyncio
    async def test_processors_built_during_a_trial_join_its_session(self) -> None:
        bindings: list[tuple[SessionContext[Any], SessionContext[Any]]] = []

        class Builder(_AddOne):
            async def _process_stream(
                self,
                chat_inputs: Any | None = None,
                *,
                in_args: list[int] | None = None,
                exec_id: str,
                step: int | None = None,
            ) -> AsyncIterator[Event[Any]]:
                helper = _AddOne(name="helper")  # built while the trial runs
                bindings.append((helper.ctx, self.ctx))
                async for event in super()._process_stream(
                    chat_inputs, in_args=in_args, exec_id=exec_id, step=step
                ):
                    yield event

        run = await evaluate(
            ProcessorTask(Builder(name="builder")),
            _ints(2),
            [matches],
            concurrency=2,
            persist=False,
        )
        assert run.counts.task_errors == 0
        assert all(helper is trial for helper, trial in bindings)
        assert len({id(trial) for _, trial in bindings}) == 2

    @pytest.mark.asyncio
    async def test_a_used_template_does_not_leak_its_conversation(self) -> None:
        agent = LLMAgent[str, str, None](name="counter", llm=CountingLLM())
        await agent.run(chat_inputs="warm up")
        dataset = Dataset([Example[str, str](id=w, input=w) for w in ["a", "b"]])
        run = await evaluate(
            ProcessorTask(agent, input_mode="chat"), dataset, persist=False
        )
        assert [t.output for t in run.trials] == ["1", "1"]

    @pytest.mark.asyncio
    async def test_input_fn_failures_still_close_the_processor(self) -> None:
        closed: list[str] = []

        class Tracked(_AddOne):
            async def aclose(self) -> None:
                closed.append(self.name)
                await super().aclose()

        def explode(value: int) -> int:
            raise ValueError("bad input")

        task = ProcessorTask(lambda: Tracked(name="t"), input_fn=explode)
        run = await evaluate(task, _ints(2), persist=False)
        assert run.counts.task_errors == 2
        assert closed == ["t", "t"]


class TestTaskIdentity:
    def test_fingerprint_follows_models_and_settings(self) -> None:
        def task(model: str) -> ProcessorTask[str, str]:
            return ProcessorTask(
                LLMAgent[str, str, None](name="a", llm=EchoLLM(model_name=model))
            )

        assert task("m1").describe().fingerprint == task("m1").describe().fingerprint
        assert task("m1").describe().fingerprint != task("m2").describe().fingerprint

    def test_fingerprint_follows_prompts_and_turn_limits(self) -> None:
        def task(**options: Any) -> str | None:
            agent = LLMAgent[str, str, None](
                name="a", llm=EchoLLM(model_name="m"), **options
            )
            return ProcessorTask(agent).describe().fingerprint

        base = task(in_prompt="Grade: {answer}")
        assert base == task(in_prompt="Grade: {answer}")
        assert base != task(in_prompt="Score: {answer}")
        assert base != task(in_prompt="Grade: {answer}", max_turns=3)

    def test_factories_are_named_and_typed_by_their_declaration(self) -> None:
        def build_adder() -> Processor[int, int, Any]:
            return _AddOne(name="adder")

        task = ProcessorTask(build_adder)
        assert task.describe().kind.endswith("build_adder")
        assert task.output_type is int
        assert ProcessorTask(build_adder, output_type=str).output_type is str
