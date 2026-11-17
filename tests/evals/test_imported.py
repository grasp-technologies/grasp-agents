import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from grasp_agents.evals import (
    Dataset,
    DatasetError,
    Evaluation,
    Example,
    FunctionTask,
    LocalRunStore,
    RunStatus,
    ScoreContext,
    compare,
    evaluate,
    import_outputs,
    read_outputs,
    scorer,
)


class Answer(BaseModel):
    value: int


def _numbers(n: int = 4) -> Dataset[int, int]:
    return Dataset(
        [Example(id=f"n{i}", input=i, reference=i * 2) for i in range(n)],
        name="numbers",
    )


@scorer
def correct(ctx: ScoreContext[int, Answer, int]) -> bool:
    return ctx.output.value == ctx.reference


async def double(x: int) -> Answer:
    return Answer(value=x * 2)


@pytest.fixture
def store(tmp_path: Path) -> LocalRunStore:
    return LocalRunStore(tmp_path / "evals")


def _write(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.mark.asyncio
async def test_an_imported_run_compares_with_a_native_one(
    tmp_path: Path, store: LocalRunStore
) -> None:
    native = await evaluate(double, _numbers(), [correct], store=store)
    path = _write(
        tmp_path / "opus.jsonl",
        [
            {
                "example_id": f"n{i}",
                "output": {"value": -1 if i == 3 else i * 2},
                "duration_s": 2.0,
                "cost_usd": 0.01,
                "measurements": {"tool_calls": i},
                "models": {"claude": ["opus"]},
            }
            for i in range(4)
        ],
    )
    imported = await import_outputs(
        read_outputs(path),
        _numbers(),
        [correct],
        task="opus",
        task_version="cli-1",
        output_type=Answer,
        store=store,
    )
    assert imported.kind == "import"
    assert imported.status == RunStatus.COMPLETED
    assert imported.task.kind == "imported"
    assert imported.usage.cost_usd == pytest.approx(0.04)
    assert imported.provenance.observed_models == {"claude": ["opus"]}
    reloaded = store.load(imported.id)
    assert reloaded.trial("n1") is not None
    result = compare(native, reloaded)
    assert result.n_paired_examples == 4
    target = result.target("correct")
    assert target is not None
    assert (target.improved, target.regressed) == (0, 1)


@pytest.mark.asyncio
async def test_imported_outputs_must_match_the_dataset(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="lacks"):
        await import_outputs(
            [{"example_id": "zz", "output": {"value": 0}}],
            _numbers(),
            task="t",
            persist=False,
        )
    with pytest.raises(DatasetError, match="Two outputs"):
        await import_outputs(
            [{"example_id": "n0", "output": {}}, {"example_id": "n0", "output": {}}],
            _numbers(),
            task="t",
            persist=False,
        )
    partial = await import_outputs(
        [
            {"example_id": "n0", "output": {"value": 0}},
            {"example_id": "n1", "error": "timed out"},
        ],
        _numbers(),
        [correct],
        task="t",
        output_type=Answer,
        persist=False,
    )
    assert partial.status == RunStatus.PARTIAL
    assert partial.counts.task_errors == 1
    with pytest.raises(DatasetError, match="example_id"):
        read_outputs(_write(tmp_path / "bad.jsonl", [{"output": 1}]))


@pytest.mark.asyncio
async def test_an_evaluation_imports_on_its_own_selection(
    tmp_path: Path, store: LocalRunStore
) -> None:
    data = Dataset(
        [
            Example(id=f"n{i}", input=i, reference=i * 2, splits=[split])
            for i, split in enumerate(["dev", "dev", "test", "test"])
        ],
        name="numbers",
    )
    evaluation = Evaluation(
        name="numbers",
        task=FunctionTask(double),
        dataset=data,
        scorers=[correct],
        sealed_splits=["test"],
    )
    path = _write(
        tmp_path / "vendor.jsonl",
        [{"example_id": f"n{i}", "output": {"value": i * 2}} for i in range(4)],
    )
    run = await evaluation.import_outputs(path, store=store)
    # Sealed splits are left out as for a run; their outputs are ignored.
    assert sorted(t.example_id for t in run.trials) == ["n0", "n1"]
    assert run.task.name == "vendor"
    assert run.metadata["imported_from"] == str(path)
    assert run.metric("pass_rate(correct)") is not None


@pytest.mark.asyncio
async def test_an_included_sealed_split_needs_every_output() -> None:
    data = Dataset(
        [
            Example(id=f"n{i}", input=i, reference=i * 2, splits=[split])
            for i, split in enumerate(["dev", "dev", "test", "test"])
        ],
        name="numbers",
    )
    outputs = [{"example_id": f"n{i}", "output": {"value": i * 2}} for i in range(3)]
    with pytest.raises(DatasetError, match="1 of its 2"):
        await import_outputs(
            outputs,
            data,
            task="t",
            sealed_splits=["test"],
            include_sealed=True,
            persist=False,
        )
    failed = [*outputs, {"example_id": "n3", "error": "timed out"}]
    run = await import_outputs(
        failed,
        data,
        task="t",
        sealed_splits=["test"],
        include_sealed=True,
        persist=False,
    )
    assert len(run.trials) == 4
