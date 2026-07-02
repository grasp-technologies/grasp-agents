"""
Tests for the auto-research demo harness (``examples/tui/autoresearch.py``):
the stratified split, the hidden-holdout scoring protocol, and the
experiment log carried in session state.
"""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.examples.tui.autoresearch import (
    ExperimentRecord,
    ResearchState,
    accuracy,
    balanced_accuracy,
    make_research_tools,
    read_csv_rows,
    stratified_split,
)
from grasp_agents.session_context import SessionContext
from grasp_agents.types.events import ToolErrorInfo

if TYPE_CHECKING:
    from grasp_agents.tools.function_tool import FunctionTool

# ---------- Fixtures ----------


def _make_rows(n: int = 40, positive_fraction: float = 0.4) -> list[dict[str, str]]:
    rng = random.Random(7)  # noqa: S311 - deterministic fixture data
    rows: list[dict[str, str]] = []
    n_pos = round(n * positive_fraction)
    for i in range(1, n + 1):
        rows.append(
            {
                "PassengerId": str(i),
                "Survived": "1" if i <= n_pos else "0",
                "Pclass": str(rng.choice([1, 2, 3])),
                "Sex": rng.choice(["male", "female"]),
                "Age": str(rng.randint(1, 80)),
            }
        )
    rng.shuffle(rows)
    return rows


def _rows_to_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "titanic.csv").write_text(_rows_to_csv(_make_rows()))
    return tmp_path


@pytest.fixture
def tools(workdir: Path) -> dict[str, FunctionTool]:
    return {t.name: t for t in make_research_tools(workdir)}


@pytest.fixture
def ctx() -> SessionContext[ResearchState]:
    return SessionContext[ResearchState](state=ResearchState())


# ---------- Pure helpers ----------


class TestStratifiedSplit:
    def test_deterministic_and_partitioning(self) -> None:
        rows = _make_rows(50)
        train1, holdout1 = stratified_split(
            rows, label_col="Survived", holdout_fraction=0.2, seed=42
        )
        train2, holdout2 = stratified_split(
            rows, label_col="Survived", holdout_fraction=0.2, seed=42
        )
        assert train1 == train2
        assert holdout1 == holdout2

        ids = {r["PassengerId"] for r in rows}
        train_ids = {r["PassengerId"] for r in train1}
        holdout_ids = {r["PassengerId"] for r in holdout1}
        assert train_ids | holdout_ids == ids
        assert not train_ids & holdout_ids
        assert len(holdout1) == pytest.approx(len(rows) * 0.2, abs=1)

    def test_stratification_preserves_class_balance(self) -> None:
        rows = _make_rows(100, positive_fraction=0.3)
        _, holdout = stratified_split(
            rows, label_col="Survived", holdout_fraction=0.2, seed=1
        )
        positives = sum(1 for r in holdout if r["Survived"] == "1")
        assert positives == pytest.approx(len(holdout) * 0.3, abs=1)

    def test_different_seeds_differ(self) -> None:
        rows = _make_rows(50)
        _, h1 = stratified_split(
            rows, label_col="Survived", holdout_fraction=0.2, seed=1
        )
        _, h2 = stratified_split(
            rows, label_col="Survived", holdout_fraction=0.2, seed=2
        )
        assert {r["PassengerId"] for r in h1} != {r["PassengerId"] for r in h2}


class TestAccuracy:
    def test_perfect_and_partial(self) -> None:
        labels = {"1": 1, "2": 0, "3": 1, "4": 0}
        assert accuracy(dict(labels), labels) == 1.0
        flipped = dict(labels) | {"1": 0}
        assert accuracy(flipped, labels) == 0.75

    def test_missing_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing 1 holdout ids"):
            accuracy({"1": 1}, {"1": 1, "2": 0})
        with pytest.raises(ValueError, match="missing 1 holdout ids"):
            balanced_accuracy({"1": 1}, {"1": 1, "2": 0})

    def test_extra_ids_ignored(self) -> None:
        labels = {"1": 1}
        assert accuracy({"1": 1, "99": 0}, labels) == 1.0

    def test_balanced_punishes_majority_predictor(self) -> None:
        # 3:1 imbalance; predicting the majority class everywhere looks fine
        # on accuracy but scores 0.5 balanced.
        labels = {"1": 0, "2": 0, "3": 0, "4": 1}
        all_majority = dict.fromkeys(labels, 0)
        assert accuracy(all_majority, labels) == 0.75
        assert balanced_accuracy(all_majority, labels) == 0.5
        assert balanced_accuracy(dict(labels), labels) == 1.0


# ---------- split_dataset tool ----------


@pytest.mark.asyncio
class TestSplitDataset:
    async def test_split_writes_files_and_hides_labels(
        self,
        workdir: Path,
        tools: dict[str, FunctionTool],
        ctx: SessionContext[ResearchState],
    ) -> None:
        out = await tools["split_dataset"](ctx=ctx)
        assert "data/train.csv" in str(out)

        train_fields, train_rows = read_csv_rows(
            (workdir / "data" / "train.csv").read_text()
        )
        holdout_fields, holdout_rows = read_csv_rows(
            (workdir / "data" / "holdout.csv").read_text()
        )
        assert "Survived" in train_fields
        assert "Survived" not in holdout_fields
        assert len(train_rows) + len(holdout_rows) == 40

        # Labels live only in session state, keyed by PassengerId.
        assert set(ctx.state.holdout_labels) == {r["PassengerId"] for r in holdout_rows}
        assert set(ctx.state.holdout_labels.values()) <= {0, 1}

    async def test_resplit_blocked_after_submission(
        self,
        tools: dict[str, FunctionTool],
        ctx: SessionContext[ResearchState],
    ) -> None:
        await tools["split_dataset"](ctx=ctx)
        ctx.state.submissions_used = 1
        result = await tools["split_dataset"](ctx=ctx)
        assert isinstance(result, ToolErrorInfo)
        assert "cannot re-split" in result.error

    async def test_path_escape_rejected(
        self,
        tools: dict[str, FunctionTool],
        ctx: SessionContext[ResearchState],
    ) -> None:
        result = await tools["split_dataset"](csv_path="../outside.csv", ctx=ctx)
        assert isinstance(result, ToolErrorInfo)
        assert "escapes the workspace" in result.error

    async def test_missing_dataset_reported(
        self,
        tools: dict[str, FunctionTool],
        ctx: SessionContext[ResearchState],
    ) -> None:
        result = await tools["split_dataset"](csv_path="data/nope.csv", ctx=ctx)
        assert isinstance(result, ToolErrorInfo)
        assert "download it first" in result.error


# ---------- submit_predictions tool ----------


def _write_predictions(workdir: Path, labels: dict[str, int], *, flip: int = 0) -> str:
    rows = [
        {"PassengerId": pid, "Survived": str(label)} for pid, label in labels.items()
    ]
    for row in rows[:flip]:
        row["Survived"] = str(1 - int(row["Survived"]))
    path = workdir / "submissions" / "preds.csv"
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["PassengerId", "Survived"])
        writer.writeheader()
        writer.writerows(rows)
    return "submissions/preds.csv"


@pytest.mark.asyncio
class TestSubmitPredictions:
    async def test_scores_against_hidden_labels(
        self,
        workdir: Path,
        tools: dict[str, FunctionTool],
        ctx: SessionContext[ResearchState],
    ) -> None:
        await tools["split_dataset"](ctx=ctx)
        n = len(ctx.state.holdout_labels)
        rel = _write_predictions(workdir, ctx.state.holdout_labels, flip=2)

        out = await tools["submit_predictions"](
            csv_path=rel, experiment_name="exp1", ctx=ctx
        )
        expected = (n - 2) / n
        assert f"accuracy {expected:.4f}" in str(out)
        assert "balanced accuracy" in str(out)
        assert ctx.state.submissions_used == 1
        assert ctx.state.experiments[0].name == "exp1"
        assert ctx.state.experiments[0].holdout_accuracy == pytest.approx(expected)
        assert ctx.state.experiments[0].holdout_balanced_accuracy is not None
        assert ctx.state.best_holdout == pytest.approx(expected)

    async def test_requires_split_first(
        self,
        workdir: Path,
        tools: dict[str, FunctionTool],
        ctx: SessionContext[ResearchState],
    ) -> None:
        rel = _write_predictions(workdir, {"1": 1})
        result = await tools["submit_predictions"](
            csv_path=rel, experiment_name="x", ctx=ctx
        )
        assert isinstance(result, ToolErrorInfo)
        assert "split_dataset first" in result.error

    async def test_budget_enforced(
        self,
        workdir: Path,
        tools: dict[str, FunctionTool],
        ctx: SessionContext[ResearchState],
    ) -> None:
        await tools["split_dataset"](ctx=ctx)
        rel = _write_predictions(workdir, ctx.state.holdout_labels)
        ctx.state.submissions_used = ctx.state.submissions_cap
        result = await tools["submit_predictions"](
            csv_path=rel, experiment_name="x", ctx=ctx
        )
        assert isinstance(result, ToolErrorInfo)
        assert "budget exhausted" in result.error

    async def test_incomplete_predictions_rejected(
        self,
        workdir: Path,
        tools: dict[str, FunctionTool],
        ctx: SessionContext[ResearchState],
    ) -> None:
        await tools["split_dataset"](ctx=ctx)
        partial = dict(list(ctx.state.holdout_labels.items())[:-1])
        rel = _write_predictions(workdir, partial)
        result = await tools["submit_predictions"](
            csv_path=rel, experiment_name="x", ctx=ctx
        )
        assert isinstance(result, ToolErrorInfo)
        assert "missing" in result.error
        # A rejected submission must not burn budget.
        assert ctx.state.submissions_used == 0


# ---------- record_experiment / research_status ----------


@pytest.mark.asyncio
class TestExperimentLog:
    async def test_record_and_update(
        self,
        tools: dict[str, FunctionTool],
        ctx: SessionContext[ResearchState],
    ) -> None:
        out = await tools["record_experiment"](
            name="baseline", cv_accuracy=0.78, notes="logreg", ctx=ctx
        )
        assert "baseline=0.7800" in str(out)

        await tools["record_experiment"](name="baseline", cv_accuracy=0.81, ctx=ctx)
        assert len(ctx.state.experiments) == 1
        assert ctx.state.experiments[0].cv_accuracy == pytest.approx(0.81)
        assert ctx.state.experiments[0].notes == "logreg"

    async def test_status_reports_log_and_budget(
        self,
        tools: dict[str, FunctionTool],
        ctx: SessionContext[ResearchState],
    ) -> None:
        ctx.state.experiments.append(
            ExperimentRecord(name="rf", cv_accuracy=0.83, holdout_accuracy=0.81)
        )
        ctx.state.submissions_used = 2
        out = str(await tools["research_status"](ctx=ctx))
        assert "Holdout split ready: no" in out
        assert "Submissions: 2/8 (best holdout: 0.8100)" in out
        assert "rf | 0.8300 | 0.8100 | - | -" in out


# ---------- State durability ----------


class TestStateRoundTrip:
    def test_state_survives_json_round_trip(self) -> None:
        state = ResearchState(
            holdout_labels={"5": 1, "9": 0},
            experiments=[ExperimentRecord(name="a", cv_accuracy=0.8)],
            submissions_used=3,
        )
        restored = ResearchState.model_validate_json(state.model_dump_json())
        assert restored == state
        assert restored.holdout_labels == {"5": 1, "9": 0}


# ---------- Tool surface ----------


def test_tool_names_and_schemas(tools: dict[str, Any]) -> None:
    assert set(tools) == {
        "split_dataset",
        "record_experiment",
        "submit_predictions",
        "research_status",
    }
    # ctx is injected by the loop, never exposed to the model.
    for tool in tools.values():
        assert "ctx" not in tool.in_type.model_fields
