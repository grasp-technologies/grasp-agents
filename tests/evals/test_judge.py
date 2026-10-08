import functools
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.evals import (
    Dataset,
    Example,
    JudgedOutput,
    JudgedPair,
    LocalRunStore,
    PairwiseVerdict,
    ProcessorPairwiseJudge,
    ProcessorScorer,
    evaluate,
    pairwise,
    rescore,
)
from grasp_agents.processors.processor import Processor
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import Event, ProcPayloadOutEvent
from grasp_agents.types.items import InputMessageItem, OutputMessageItem
from grasp_agents.types.response import Response
from tests._helpers import MockLLM, _make_usage


class Verdict(BaseModel):
    passed: bool
    explanation: str


@dataclass(frozen=True)
class VerdictLLM(MockLLM):
    """Passes outputs whose rendered input mentions "good"."""

    async def _generate_response_once(self, input: Sequence[Any], **_: Any) -> Response:
        object.__setattr__(self, "_call_count", self.call_count + 1)
        text = ""
        for item in reversed(list(input)):
            if isinstance(item, InputMessageItem) and item.role == "user":
                text = " ".join(getattr(part, "text", "") for part in item.content)
                break
        verdict = {"passed": "good" in text, "explanation": "looked for 'good'"}
        return Response(
            model=self.model_name,
            output=[
                OutputMessageItem(
                    content=[OutputMessageText(text=json.dumps(verdict))],
                    status="completed",
                )
            ],
            usage=_make_usage(),
        )


def _passed(verdict: Verdict) -> bool:
    return verdict.passed


def _judge_agent(
    llm: VerdictLLM | None = None, sys_prompt: str = "Is the answer good?"
) -> LLMAgent[JudgedOutput[str, str, Any], Verdict, None]:
    return LLMAgent[JudgedOutput[str, str, Any], Verdict, None](
        name="judge", llm=llm or VerdictLLM(model_name="judge-1"), sys_prompt=sys_prompt
    )


def _letters() -> Dataset[str, Any]:
    return Dataset([Example(id=c, input=c) for c in "abc"], name="letters")


async def _answer(x: str) -> str:
    return "bad" if x == "b" else f"{x} is good"


@pytest.fixture
def store(tmp_path: Path) -> LocalRunStore:
    return LocalRunStore(tmp_path / "evals")


class TestProcessorScorer:
    @pytest.mark.asyncio
    async def test_runs_an_isolated_judge_per_trial_and_records_its_usage(
        self,
    ) -> None:
        llm = VerdictLLM(model_name="judge-1")
        agent = _judge_agent(llm)
        judge = ProcessorScorer(agent, name="quality", to_scores=_passed)
        run = await evaluate(_answer, _letters(), [judge], persist=False)
        values = {t.example_id: t.score("quality") for t in run.trials}
        assert {k: v.value if v else None for k, v in values.items()} == {
            "a": True,
            "b": False,
            "c": True,
        }
        assert llm.call_count == 3
        # The template never runs: every call is a fresh copy.
        assert agent.transcript.messages == []
        for trial in run.trials:
            usage = trial.scorer_usage["quality"]
            assert (usage.input_tokens, usage.output_tokens) == (10, 5)
            assert trial.usage.is_empty
        assert run.scorers[0].annotator == "LLM"

    @pytest.mark.asyncio
    async def test_default_input_is_a_typed_judged_output(self) -> None:
        seen: list[JudgedOutput[str, str, str]] = []

        class Recorder(Processor[JudgedOutput[str, str, str], bool, Any]):
            async def _process_stream(
                self,
                chat_inputs: Any | None = None,
                *,
                in_args: list[JudgedOutput[str, str, str]] | None = None,
                exec_id: str,
                step: int | None = None,
            ) -> AsyncIterator[Event[Any]]:
                for item in in_args or []:
                    seen.append(item)
                    yield ProcPayloadOutEvent(
                        data=item.output == item.reference,
                        source=self.name,
                        exec_id=exec_id,
                    )

        dataset = Dataset(
            [Example(id="x", input="q", reference="Q", metadata={"k": 1})]
        )

        async def upper(x: str) -> str:
            return x.upper()

        judge = ProcessorScorer(Recorder(name="recorder"), name="matches")
        run = await evaluate(upper, dataset, [judge], persist=False)
        assert run.trials[0].score("matches") is not None
        assert run.trials[0].score("matches").value is True  # type: ignore[union-attr]
        assert seen[0].input == "q"
        assert seen[0].output == "Q"
        assert seen[0].metadata == {"k": 1}

    def test_identity_follows_prompt_model_and_code(self) -> None:
        def build(**kwargs: Any) -> ProcessorScorer[Any, Any, Any, Any, Any]:
            return ProcessorScorer(_judge_agent(**kwargs), to_scores=_passed)

        base = build().describe()
        assert base == build().describe()
        assert base.fingerprint is not None
        assert build(sys_prompt="Be strict.").describe().fingerprint != base.fingerprint
        other_model = build(llm=VerdictLLM(model_name="judge-2")).describe()
        assert other_model.fingerprint != base.fingerprint
        relabeled = ProcessorScorer(
            _judge_agent(), to_scores=lambda v: v.passed
        ).describe()
        assert relabeled.fingerprint == base.fingerprint
        assert relabeled.source != base.source

    @pytest.mark.asyncio
    async def test_rescoring_reruns_a_judge_whose_prompt_changed(
        self, store: LocalRunStore
    ) -> None:
        first = VerdictLLM(model_name="judge-1")
        judge = ProcessorScorer(_judge_agent(first), name="q", to_scores=_passed)
        run = await evaluate(_answer, _letters(), [judge], store=store)
        assert first.call_count == 3

        same = VerdictLLM(model_name="judge-1")
        unchanged = ProcessorScorer(_judge_agent(same), name="q", to_scores=_passed)
        await rescore(run, [unchanged], store=store)
        assert same.call_count == 0

        edited = VerdictLLM(model_name="judge-1")
        reworded = ProcessorScorer(
            _judge_agent(edited, sys_prompt="Is the answer good? Be strict."),
            name="q",
            to_scores=_passed,
        )
        await rescore(run, [reworded], store=store)
        assert edited.call_count == 3

    @pytest.mark.asyncio
    async def test_a_failing_judge_is_an_scorer_failure(self) -> None:
        class Broken(Processor[Any, bool, Any]):
            async def _process_stream(
                self,
                chat_inputs: Any | None = None,
                *,
                in_args: list[Any] | None = None,
                exec_id: str,
                step: int | None = None,
            ) -> AsyncIterator[Event[Any]]:
                raise RuntimeError("judge exploded")
                yield  # pragma: no cover

        judge = ProcessorScorer(Broken(name="broken"))
        run = await evaluate(_answer, _letters(), [judge], persist=False)
        assert run.counts.scorer_failures == 3
        assert all(t.ok for t in run.trials)
        assert "judge exploded" in run.trials[0].scorer_failures[0].error.message


class _Longer(Processor[JudgedPair[str, str, Any], PairwiseVerdict, Any]):
    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[JudgedPair[str, str, Any]] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for pair in in_args or []:
            first, second = len(pair.first), len(pair.second)
            winner = (
                "tie" if first == second else "first" if first > second else "second"
            )
            yield ProcPayloadOutEvent(
                data=PairwiseVerdict(winner=winner), source=self.name, exec_id=exec_id
            )


class TestProcessorPairwiseJudge:
    @pytest.mark.asyncio
    async def test_judges_pairs_in_both_orders(self, store: LocalRunStore) -> None:
        async def short(x: str) -> str:
            return x

        async def long(x: str) -> str:
            return x * 3

        base = await evaluate(short, _letters(), store=store)
        candidate = await evaluate(long, _letters(), store=store)
        judge = ProcessorPairwiseJudge(_Longer(name="longer"), annotator="CODE")
        run = await pairwise(base, candidate, judge, store=store)
        win_rate = run.metric("win_rate(longer)")
        assert win_rate is not None
        assert win_rate.value == 1.0
        consistent = run.metric("pass_rate(longer.position_consistent)")
        assert consistent is not None
        assert consistent.value == 1.0
        assert run.scorers[0].config["judge"]["fingerprint"] is not None

    @pytest.mark.asyncio
    async def test_pairwise_judge_usage_reaches_the_trial(
        self, store: LocalRunStore
    ) -> None:
        @dataclass(frozen=True)
        class FirstLLM(MockLLM):
            async def _generate_response_once(
                self, input: Sequence[Any], **_: Any
            ) -> Response:
                return Response(
                    model="pair-1",
                    output=[
                        OutputMessageItem(
                            content=[OutputMessageText(text='{"winner": "first"}')],
                            status="completed",
                        )
                    ],
                    usage=_make_usage(),
                )

        agent = LLMAgent[JudgedPair[str, str, Any], PairwiseVerdict, None](
            name="pair_judge", llm=FirstLLM(), sys_prompt="Which is better?"
        )
        base = await evaluate(_answer, _letters(), store=store)
        candidate = await evaluate(_answer, _letters(), store=store)
        run = await pairwise(
            base, candidate, ProcessorPairwiseJudge(agent), store=store
        )
        usage = run.trials[0].scorer_usage["pair_judge"]
        # One call per order.
        assert usage.input_tokens == 20
        # "first" in both orders: position-inconsistent, a tie.
        assert run.trials[0].score("pair_judge.winner").value == "inconsistent"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_outputs_other_than_verdicts_need_to_verdict(
        self, store: LocalRunStore
    ) -> None:
        class Says(Processor[Any, str, Any]):
            async def _process_stream(
                self,
                chat_inputs: Any | None = None,
                *,
                in_args: list[Any] | None = None,
                exec_id: str,
                step: int | None = None,
            ) -> AsyncIterator[Event[Any]]:
                yield ProcPayloadOutEvent(
                    data="first", source=self.name, exec_id=exec_id
                )

        base = await evaluate(_answer, _letters(), store=store)
        candidate = await evaluate(_answer, _letters(), store=store)
        bare = await pairwise(
            base, candidate, ProcessorPairwiseJudge(Says(name="s")), store=store
        )
        assert bare.counts.scorer_failures == 3
        assert "to_verdict" in bare.trials[0].scorer_failures[0].error.message
        mapped = ProcessorPairwiseJudge(
            Says(name="s"), to_verdict=lambda w: PairwiseVerdict(winner=w)
        )
        run = await pairwise(base, candidate, mapped, store=store)
        assert run.counts.scorer_failures == 0


def _judge_with(
    *, tools: list[Any] | None = None, **kwargs: Any
) -> LLMAgent[JudgedOutput[str, str, Any], Verdict, None]:
    return LLMAgent[JudgedOutput[str, str, Any], Verdict, None](
        name="judge",
        llm=VerdictLLM(model_name="judge-1"),
        sys_prompt="Is the answer good?",
        tools=tools,
        **kwargs,
    )


def _render(*_: Any, **__: Any) -> str:
    return "rendered"


def _render_differently(*_: Any, **__: Any) -> str:
    return "rendered differently"


class TestJudgeIdentity:
    def _source(self, agent: Any, to_scores: Any = _passed) -> str | None:
        return ProcessorScorer(agent, to_scores=to_scores).describe().source

    def test_hooks_are_part_of_the_identity(self) -> None:
        plain = _judge_with()
        built = _judge_with()
        built.add_input_content_builder(_render)
        rebuilt = _judge_with()
        rebuilt.add_input_content_builder(_render_differently)
        sources = {self._source(a) for a in (plain, built, rebuilt)}
        assert len(sources) == 3
        same = _judge_with()
        same.add_input_content_builder(_render)
        assert self._source(same) == self._source(built)

    def test_tools_are_part_of_the_identity(self) -> None:
        from grasp_agents.tools.function_tool import function_tool  # noqa: PLC0415

        def lookup(query: str) -> str:
            """Look a term up."""
            return query

        def lookup_v2(query: str) -> str:
            """Look a term up."""
            return query.upper()

        first = _judge_with(tools=[function_tool(lookup, name="lookup")])
        described = _judge_with(
            tools=[function_tool(lookup, name="lookup", description="Find a term")]
        )
        recoded = _judge_with(tools=[function_tool(lookup_v2, name="lookup")])
        fingerprints = {
            ProcessorScorer(a).describe().fingerprint for a in (first, described)
        }
        assert len(fingerprints) == 2
        assert self._source(first) != self._source(recoded)

    def test_partials_and_closure_constants_count_mutable_state_does_not(
        self,
    ) -> None:
        def at_least(verdict: Verdict, *, bar: float) -> bool:
            return verdict.passed and bar > 0

        loose = self._source(_judge_with(), functools.partial(at_least, bar=0.5))
        strict = self._source(_judge_with(), functools.partial(at_least, bar=0.9))
        assert loose != strict

        def make(bar: float) -> Any:
            seen: list[Verdict] = []

            def scores(verdict: Verdict) -> bool:
                seen.append(verdict)
                return verdict.passed and bar > 0

            return scores

        scorer = make(0.5)
        before = self._source(_judge_with(), scorer)
        scorer(Verdict(passed=True, explanation=""))  # mutates the closed-over list
        assert self._source(_judge_with(), scorer) == before
        assert self._source(_judge_with(), make(0.9)) != before

    def test_operational_flags_and_late_changes(self) -> None:
        agent = _judge_with()
        judge = ProcessorScorer(agent, to_scores=_passed)
        before = judge.describe()
        agent.tracing_enabled = False
        assert judge.describe() == before
        # Changes after the judge is built count too.
        agent.add_before_llm_hook(_render)  # type: ignore[arg-type]
        assert judge.describe().source != before.source


class TestJudgeSpend:
    @pytest.mark.asyncio
    async def test_judge_models_are_recorded(self, store: LocalRunStore) -> None:
        judge = ProcessorScorer(_judge_agent(), name="q", to_scores=_passed)
        run = await evaluate(_answer, _letters(), [judge], store=store)
        assert run.trials[0].models == {"judge": ["judge-1"]}
        assert run.provenance.observed_models == {"judge": ["judge-1"]}

    @pytest.mark.asyncio
    async def test_unpriced_judges_are_reported(self, store: LocalRunStore) -> None:
        judge = ProcessorScorer(_judge_agent(), name="q", to_scores=_passed)
        run = await evaluate(
            _answer, _letters(), [judge], store=store, max_cost_usd=1.0
        )
        assert run.metadata["unpriced_agents"] == ["scorer q"]

    @pytest.mark.asyncio
    async def test_a_failing_judgment_stops_the_others(self) -> None:
        import asyncio  # noqa: PLC0415

        from grasp_agents.evals import (  # noqa: PLC0415
            EvalContext,
            Perturbation,
            scorer,
        )
        from grasp_agents.evals.validation import ProbeTask  # noqa: PLC0415

        finished: list[str] = []

        @scorer(name="slow", annotator="LLM")
        async def slow(ctx: EvalContext[str, str, Any]) -> bool:
            if ctx.output.endswith("!"):
                raise RuntimeError("judge failed")
            await asyncio.sleep(0.2)
            finished.append(ctx.output)
            return True

        task = ProbeTask(
            slow,
            [
                Perturbation("loud", lambda i: i.output + "!", "same"),
                Perturbation("quiet", lambda i: i.output + ".", "same"),
            ],
        )
        dataset = Dataset.from_records(
            [{"id": "x", "input": {"input": "q", "output": "fine"}}],
            input_type=task.input_type,
        )
        run = await evaluate(task, dataset, persist=False)
        assert run.counts.task_errors == 1
        assert "judge failed" in run.trials[0].error.message  # type: ignore[union-attr]
        await asyncio.sleep(0.3)
        assert finished == []
