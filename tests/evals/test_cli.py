import json
import textwrap
from pathlib import Path

import pytest

from grasp_agents.evals.cli import main

_MODULE = """
from grasp_agents.evals import (
    Dataset, EvalContext, Evaluation, Example, FunctionTask, evaluator,
)

async def solve(x: int) -> int:
    return x * 2 if x != 3 else -1

async def solve_fixed(x: int) -> int:
    return x * 2

@evaluator
def correct(ctx: EvalContext[int, int, int]) -> bool:
    return ctx.output == ctx.reference

@evaluator(name="correct", version="2")
def correct_v2(ctx: EvalContext[int, int, int]) -> float:
    return 1.0 if ctx.output == ctx.reference else 0.0

DATA = Dataset(
    [
        Example(id=f"x{i}", input=i, reference=i * 2, splits=["test"] if i >= 4 else [])
        for i in range(6)
    ],
    name="doubling",
)

def doubling(solver, evaluators, sealed=("test",)):
    return Evaluation(
        name="doubling",
        task=FunctionTask(solver, name="solver"),
        dataset=DATA,
        evaluators=evaluators,
        sealed_splits=sealed,
    )

buggy = doubling(solve, [correct])
fixed = doubling(solve_fixed, [correct])
rejudged = doubling(solve, [correct_v2], sealed=())
"""


@pytest.fixture
def module(tmp_path: Path) -> Path:
    path = tmp_path / "cli_evals.py"
    path.write_text(textwrap.dedent(_MODULE))
    return path


def _run_json(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, dict]:  # type: ignore[type-arg]
    code = main(list(args))
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_run_show_compare_rescore(
    tmp_path: Path, module: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = str(tmp_path / "evals")
    code, base = _run_json(
        capsys, "--root", root, "run", f"{module}:buggy", "--json", "-q"
    )
    assert code == 0
    assert base["metrics"]["pass_rate(correct)"]["value"] == pytest.approx(5 / 6)
    assert Path(base["path"], "trials.jsonl").exists()

    code, gated = _run_json(
        capsys,
        "--root",
        root,
        "run",
        f"{module}:buggy",
        "--json",
        "-q",
        "--fail-under",
        "pass_rate(correct)=0.9",
    )
    assert code == 1
    assert gated["gate_failures"] == ["pass_rate(correct) = 0.8333333333333334 < 0.9"]

    code, candidate = _run_json(
        capsys,
        "--root",
        root,
        "run",
        f"{module}:fixed",
        "--json",
        "-q",
        "--baseline",
        base["id"],
        "--fail-on-regression",
    )
    assert code == 0
    target = next(
        t for t in candidate["comparison"]["targets"] if t["target"] == "correct"
    )
    assert target["improved"] == 1
    assert target["regressed"] == 0

    code, compared = _run_json(
        capsys, "--root", root, "compare", base["id"], "latest", "--json"
    )
    assert compared["candidate_run"] == candidate["id"]

    code, shown = _run_json(
        capsys, "--root", root, "show", base["id"], "--failures", "--json"
    )
    assert code == 0
    assert shown["trials"] == []  # wrong answers are scores, not failures

    code, detail = _run_json(
        capsys, "--root", root, "show", base["id"], "--example", "x3"
    )
    assert detail["trials"][0]["scores"]["correct"]["value"] is False
    assert detail["example"]["reference"] == 6

    assert main(["--root", root, "show", base["id"], "--example", "x5"]) == 2
    assert "sealed" in capsys.readouterr().err

    code, child = _run_json(
        capsys,
        "--root",
        root,
        "rescore",
        base["id"],
        "--spec",
        f"{module}:rejudged",
        "--json",
        "-q",
    )
    assert child["kind"] == "rescore"
    assert child["parent_run_id"] == base["id"]
    assert child["evaluators"] == {"correct": "2"}

    code, runs = _run_json(capsys, "--root", root, "runs", "--json")
    assert len(runs) == 4


def test_datasets_commands(
    tmp_path: Path, module: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, schema = _run_json(capsys, "datasets", "schema", f"{module}:buggy")
    assert code == 0
    assert schema["required"] == ["input"]

    code, report = _run_json(capsys, "datasets", "validate", f"{module}:buggy")
    assert code == 0
    assert report["valid"]
    assert report["examples"] == 6

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "z", "input": "not an int"}) + "\n")
    code, report = _run_json(
        capsys, "datasets", "validate", f"{module}:buggy", "--dataset", str(bad)
    )
    assert code == 1
    assert not report["valid"]


def test_usage_errors_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--root", str(tmp_path), "show", "nope"]) == 2
    assert "No run matching" in capsys.readouterr().err
