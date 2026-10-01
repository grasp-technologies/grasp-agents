import textwrap
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.evals import (
    Dataset,
    DatasetError,
    EvalContext,
    Evaluation,
    Example,
    FunctionTask,
    LocalRunStore,
    evaluator,
    list_evaluations,
    load_evaluation,
)


class Problem(BaseModel):
    a: int
    b: int


async def add(problem: Problem) -> int:
    return problem.a + problem.b


@evaluator
def exact(ctx: EvalContext[Problem, int, int]) -> bool:
    return ctx.output == ctx.reference


def _write_dataset(path: Path) -> Path:
    Dataset(
        [
            Example(
                id=f"p{i}",
                input=Problem(a=i, b=i),
                reference=2 * i,
                splits=["dev" if i < 3 else "test"],
            )
            for i in range(5)
        ],
        name="sums",
    ).save(path)
    return path


class TestEvaluation:
    @pytest.mark.asyncio
    async def test_run_from_file_with_selection(self, tmp_path: Path) -> None:
        evaluation = Evaluation(
            name="adder",
            description="Does the adder add?",
            task=FunctionTask(add),
            dataset=_write_dataset(tmp_path / "sums.jsonl"),
            evaluators=[exact],
            reference_type=int,
            sealed_splits=["test"],
        )
        assert evaluation.resolved_input_type is Problem
        run = await evaluation.run(split="dev", store=LocalRunStore(tmp_path / "evals"))
        assert run.name == "adder"
        assert run.description == "Does the adder add?"
        assert run.dataset.selection == ["split=dev"]
        assert run.dataset.selected_size == 3
        assert run.metric("pass_rate(exact)").value == pytest.approx(1.0)  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_dataset_checks_fail_fast(self, tmp_path: Path) -> None:
        def has_reference(example: Example[Any, Any]) -> str | None:
            return None if example.reference is not None else "missing reference"

        dataset = Dataset([Example(id="x", input=Problem(a=1, b=1))])
        evaluation = Evaluation(
            name="checked",
            task=FunctionTask(add),
            dataset=dataset,
            dataset_checks=[has_reference],
        )
        with pytest.raises(DatasetError, match="missing reference"):
            await evaluation.run(persist=False)

    @pytest.mark.asyncio
    async def test_lazy_task_factory(self, tmp_path: Path) -> None:
        built = 0

        def make_task() -> FunctionTask[Problem, int]:
            nonlocal built
            built += 1
            return FunctionTask(add)

        evaluation = Evaluation(
            name="lazy",
            task=make_task,
            dataset=_write_dataset(tmp_path / "d.jsonl"),
            evaluators=[exact],
        )
        assert built == 0
        await evaluation.run(limit=1, persist=False)
        await evaluation.run(limit=1, persist=False)
        assert built == 1

    def test_load_from_file_and_module(self, tmp_path: Path) -> None:
        module = tmp_path / "my_evals.py"
        module.write_text(
            textwrap.dedent(
                """
                from grasp_agents.evals import (
                    Dataset, Evaluation, Example, FunctionTask,
                )

                async def echo(x: str) -> str:
                    return x

                first = Evaluation(
                    name="first",
                    task=FunctionTask(echo),
                    dataset=Dataset([Example(input="a")]),
                )

                def second():
                    return Evaluation(
                        name="second",
                        task=FunctionTask(echo),
                        dataset=Dataset([Example(input="b")]),
                    )
                """
            )
        )
        loaded = load_evaluation(f"{module}:first")
        assert loaded.name == "first"
        assert loaded.spec == f"{module}:first"
        assert load_evaluation(f"{module}:second").name == "second"
        assert load_evaluation(str(module)).name == "first"
        assert set(list_evaluations(str(module))) == {"first"}
        with pytest.raises(LookupError):
            load_evaluation(f"{module}:missing")


class Verdict(BaseModel):
    value: int


@evaluator(name="typed", version="2")
def typed_exact(ctx: EvalContext[Problem, int, Verdict]) -> bool:
    # Attribute access fails unless stored examples are re-validated.
    return ctx.output == ctx.input.a + ctx.input.b == ctx.reference.value  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_rescore_from_disk_sees_typed_examples(tmp_path: Path) -> None:
    @evaluator(name="typed", version="1")
    def typed_v1(ctx: EvalContext[Problem, int, Verdict]) -> bool:
        return ctx.reference is not None and ctx.output == ctx.reference.value

    dataset = Dataset(
        [
            Example(id=f"p{i}", input=Problem(a=i, b=1), reference=Verdict(value=i + 1))
            for i in range(3)
        ]
    )
    store = LocalRunStore(tmp_path)
    first = Evaluation(
        name="typed",
        task=FunctionTask(add),
        dataset=dataset,
        evaluators=[typed_v1],
        reference_type=Verdict,
    )
    run = await first.run(store=store)
    second = Evaluation(
        name="typed",
        task=FunctionTask(add),
        dataset=dataset,
        evaluators=[typed_exact],
        reference_type=Verdict,
    )
    child = await second.rescore(run.id, store=store)
    assert child.counts.evaluator_failures == 0
    assert child.metric("pass_rate(typed)").value == pytest.approx(1.0)  # type: ignore[union-attr]
