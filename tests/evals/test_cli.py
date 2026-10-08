import json
import textwrap
from pathlib import Path

import pytest

from grasp_agents.evals.cli import main

_MODULE = """
from grasp_agents.evals import (
    Dataset, EvalContext, Evaluation, Example, FunctionTask, scorer,
)

async def solve(x: int) -> int:
    return x * 2 if x != 3 else -1

async def solve_fixed(x: int) -> int:
    return x * 2

@scorer
def correct(ctx: EvalContext[int, int, int]) -> bool:
    return ctx.output == ctx.reference

@scorer(name="correct", version="2")
def correct_v2(ctx: EvalContext[int, int, int]) -> float:
    return 1.0 if ctx.output == ctx.reference else 0.0

DATA = Dataset(
    [
        Example(id=f"x{i}", input=i, reference=i * 2, splits=["test"] if i >= 4 else [])
        for i in range(6)
    ],
    name="doubling",
)

def doubling(solver, scorers, sealed=("test",)):
    return Evaluation(
        name="doubling",
        task=FunctionTask(solver, name="solver"),
        dataset=DATA,
        scorers=scorers,
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
    # Failures are errors and failed scores: x3 is answered wrongly.
    assert [t["example_id"] for t in shown["trials"]] == ["x3"]

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
    assert child["scorers"] == {"correct": "2"}

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


def test_baseline_latest_is_the_previous_run(
    tmp_path: Path, module: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = str(tmp_path / "evals")
    code, base = _run_json(
        capsys, "--root", root, "run", f"{module}:fixed", "--json", "-q"
    )
    assert code == 0
    code, candidate = _run_json(
        capsys,
        "--root",
        root,
        "run",
        f"{module}:buggy",
        "--json",
        "-q",
        "--baseline",
        "latest",
        "--fail-on-regression",
    )
    assert candidate["comparison"]["base_run"] == base["id"]
    assert candidate["comparison"]["candidate_run"] == candidate["id"]


def test_gate_flags_need_a_baseline(
    tmp_path: Path, module: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _run_json(
        capsys,
        "--root",
        str(tmp_path),
        "run",
        f"{module}:buggy",
        "--fail-on-regression",
        "--json",
    )
    assert code == 2
    assert "--baseline" in payload["error"]["message"]


def test_errors_are_json_documents_with_usage_codes(
    tmp_path: Path, module: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = str(tmp_path / "evals")
    for args, expected in [
        (["run", str(tmp_path / "missing.py:spec")], "No spec file"),
        (["run", f"{module}:nope"], "no attribute"),
        (["run", f"{module}:buggy", "--split", "tset"], "No split 'tset'"),
        (["run", f"{module}:buggy", "--ids", "x4"], "sealed split"),
    ]:
        code, payload = _run_json(capsys, "--root", root, *args, "--json", "-q")
        assert code == 2, args
        assert expected in payload["error"]["message"], args


def test_a_failed_push_still_reports_the_run(
    tmp_path: Path,
    module: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOENIX_BASE_URL", raising=False)
    code, payload = _run_json(
        capsys,
        "--root",
        str(tmp_path),
        "run",
        f"{module}:buggy",
        "--push",
        "--json",
        "-q",
    )
    assert code == 2  # a configuration problem, not a Phoenix failure
    assert payload["id"]
    assert "PHOENIX_BASE_URL" in payload["push_error"]


def test_listing_and_dataset_commands_take_json(
    tmp_path: Path, module: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = str(tmp_path / "evals")
    _run_json(capsys, "--root", root, "run", f"{module}:buggy", "--json", "-q")
    code, runs = _run_json(capsys, "--root", root, "runs", "--json")
    assert code == 0
    assert runs[0]["kind"] == "evaluation"
    assert runs[0]["evaluation"].endswith(":buggy")
    code, latest = _run_json(capsys, "--root", root, "show", "latest:buggy", "--json")
    assert latest["id"] == runs[0]["id"]
    code, schema = _run_json(capsys, "datasets", "schema", f"{module}:buggy", "--json")
    assert code == 0
    assert schema["additionalProperties"] is False


def test_counts_must_be_positive(
    tmp_path: Path, module: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--root", str(tmp_path), "run", f"{module}:buggy", "-r", "0"]) == 2
    assert "must be >= 1" in capsys.readouterr().err


def test_argument_errors_are_json_documents_under_json(
    tmp_path: Path, module: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _run_json(
        capsys, "--root", str(tmp_path), "run", f"{module}:buggy", "-r", "0", "--json"
    )
    assert code == 2
    assert payload["error"]["exit"] == 2
    assert "repetitions" in payload["error"]["message"]


def test_progress_can_be_json_lines(
    tmp_path: Path, module: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["--root", str(tmp_path), "run", f"{module}:buggy", "--progress", "json"]
    )
    assert code == 0
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()[:6]]
    assert [e["done"] for e in events] == [1, 2, 3, 4, 5, 6]
    assert all(e["event"] == "trial" and e["total"] == 6 for e in events)
    shown = {e["trial"].get("example_id") for e in events}
    assert shown == {"x0", "x1", "x2", "x3", None}  # sealed trials are redacted
    assert events[-1]["task_errors"] == 0


_JUDGED = """
import json
from pathlib import Path

from grasp_agents.evals import (
    Dataset, EvalContext, Evaluation, Example, FunctionTask, PassRate,
    ValidationGate, scorer, judge_validation,
)

LABELS = Path(__file__).with_name("labels.jsonl")
LABELS.write_text("".join(
    json.dumps({"id": f"o{i}", "input": {"input": i, "output": i * 2},
                "reference": i % 2 == 0, "splits": ["test" if i > 3 else "dev"]}) + "\\n"
    for i in range(8)
))

@scorer(name="even", annotator="LLM")
def even(ctx: EvalContext[int, int, bool]) -> bool:
    return ctx.output % 4 == 0

async def double(x: int) -> int:
    return x * 2

judged = Evaluation(
    name="judged",
    task=FunctionTask(double),
    dataset=Dataset([Example(id=f"x{i}", input=i) for i in range(4)]),
    scorers=[even],
    validation_gates={"even": ValidationGate(min_accuracy=0.0)},
)
even_validation = judge_validation(even, LABELS, input_type=int, output_type=int)
listed = Evaluation(
    name="listed",
    task=FunctionTask(double),
    dataset=Dataset([Example(id=f"x{i}", input=i) for i in range(4)]),
    scorers=[even],
    metrics=[PassRate("even")],
    validation_gates={"even": ValidationGate(min_accuracy=0.0)},
)
"""


def test_unvalidated_judges_fail_the_run_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = tmp_path / "judged_evals.py"
    module.write_text(textwrap.dedent(_JUDGED))
    root = str(tmp_path / "evals")
    code, payload = _run_json(
        capsys, "--root", root, "run", f"{module}:judged", "--json", "-q"
    )
    assert code == 1
    assert payload["error"]["type"] == "UnvalidatedJudgeError"
    code, allowed = _run_json(
        capsys,
        "--root",
        root,
        "run",
        f"{module}:judged",
        "--allow-unvalidated",
        "--json",
        "-q",
    )
    assert code == 0
    assert allowed["status"] == "completed"
    code, _ = _run_json(
        capsys,
        "--root",
        root,
        "run",
        f"{module}:even_validation",
        "--split",
        "test",
        "--json",
        "-q",
    )
    assert code == 0
    code, validated = _run_json(
        capsys, "--root", root, "run", f"{module}:judged", "--json", "-q"
    )
    assert code == 0
    assert "corrected_pass_rate(even)" in validated["metrics"]
    code, listed = _run_json(
        capsys,
        "--root",
        root,
        "run",
        f"{module}:listed",
        "--fail-under",
        "corrected_pass_rate(even)=0.0",
        "--json",
        "-q",
    )
    assert code == 0
    assert listed["gate_failures"] == []
