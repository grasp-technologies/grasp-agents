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
    SpecError,
    evaluator,
    list_evaluations,
    load_evaluation,
    load_object,
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
        with pytest.raises(SpecError, match="no attribute 'missing'"):
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


class TestSpecFiles:
    def test_files_inside_packages_load_as_their_module(self) -> None:
        import sys

        import grasp_agents.examples.evals.grader_evals as module_form

        spec = "src/grasp_agents/examples/evals/grader_evals.py:grader_v1"
        by_path = load_evaluation(spec)
        assert by_path is module_form.grader_v1
        assert by_path.spec is not None
        assert Path(by_path.spec.rpartition(":")[0]).is_absolute()
        assert not any(
            name.startswith("_grasp_evals_grader_evals") for name in sys.modules
        )

    def test_standalone_specs_import_their_siblings_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "specs"
        folder.mkdir()
        (folder / "helpers_for_spec.py").write_text("VALUE = 7\n")
        (folder / "spec_module.py").write_text(
            textwrap.dedent(
                """
                from grasp_agents.evals import Dataset, Evaluation, Example
                import helpers_for_spec

                LOADS = []
                LOADS.append(1)

                async def task(x: int) -> int:
                    return x + helpers_for_spec.VALUE

                spec = Evaluation(
                    name="sibling",
                    task=task,
                    dataset=Dataset([Example(id="a", input=1)]),
                )
                """
            )
        )
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        first = load_object(f"{folder / 'spec_module.py'}:LOADS")
        again = load_object(f"{folder / 'spec_module.py'}:LOADS")
        assert first is again
        assert first == [1]
        evaluation = load_evaluation(f"{folder / 'spec_module.py'}:spec")
        assert evaluation.name == "sibling"


_SPEC = """
from grasp_agents.evals import Dataset, Evaluation, Example

{imports}

async def task(x: int) -> int:
    return x

spec = Evaluation(name="spec", task=task, dataset=Dataset([Example(id="a", input=1)]))
"""


class TestSpecNames:
    def test_package_specs_keep_their_package_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import uuid

        package = f"pkg_{uuid.uuid4().hex[:8]}"
        root = tmp_path / package
        (root / "specs").mkdir(parents=True)
        (root / "__init__.py").write_text("")
        (root / "specs" / "__init__.py").write_text("")
        (root / "models.py").write_text("VALUE = 7\n")
        spec_file = root / "specs" / "spec.py"
        spec_file.write_text(_SPEC.format(imports="from ..models import VALUE"))
        monkeypatch.setattr(sys, "path", list(sys.path))
        # Run from inside the package: its directory is on sys.path too.
        sys.path[:0] = [str(root / "specs"), str(tmp_path)]
        evaluation = load_evaluation(f"{spec_file}:spec")
        assert evaluation is sys.modules[f"{package}.specs.spec"].spec
        assert "spec" not in sys.modules or sys.modules["spec"].__file__ != str(
            spec_file
        )

    def test_loose_specs_are_imported_under_their_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib
        import sys
        import uuid

        stem = f"loose_spec_{uuid.uuid4().hex[:8]}"
        (tmp_path / f"{stem}.py").write_text(
            _SPEC.format(imports="class Grade:\n    pass")
        )
        (tmp_path / f"judges_{stem}.py").write_text(f"from {stem} import Grade\n")
        monkeypatch.setattr(sys, "path", list(sys.path))
        evaluation = load_evaluation(f"{tmp_path / f'{stem}.py'}:spec")
        spec_module = sys.modules[stem]
        assert evaluation is spec_module.spec
        judges = importlib.import_module(f"judges_{stem}")
        assert judges.Grade is spec_module.Grade
