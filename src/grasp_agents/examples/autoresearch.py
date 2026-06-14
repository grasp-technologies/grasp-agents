"""
Auto-research demo — an agent that trains and iteratively improves a classic
ML model inside a real Jupyter notebook.

A single ``researcher`` agent works like a careful scientist in a sandboxed
workspace:

- gets a **private venv** preloaded with numpy / pandas / scikit-learn /
  seaborn — the notebook kernel runs on it; extra packages it installs itself
  via ``Bash`` (network egress is allowlisted to the dataset mirror + PyPI);
- downloads the Titanic dataset and calls ``split_dataset``, which hides the
  holdout labels in (checkpointed) session state;
- experiments **in ``research.ipynb`` only** via ``NotebookRead`` /
  ``NotebookEdit`` / ``RunCell`` — kernel state persists across cells; Bash and
  file tools are reserved for downloads and file manipulation;
- tracks progress with ``record_experiment`` and a **limited** budget of
  ``submit_predictions`` calls scored against the hidden holdout;
- checkpoints to ``.grasp/checkpoints`` — re-running the script resumes the
  session (transcript, experiment log, and label vault included).

Run interactively in the TUI (needs the ``tui`` + ``notebook`` extras, the
``srt`` CLI, and ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` in ``.env``)::

    python -m grasp_agents.examples.autoresearch

or headless::

    python -m grasp_agents.examples.autoresearch --headless
    python -m grasp_agents.examples.autoresearch --headless --resume
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import random
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from grasp_agents import LLMAgent, ProcPacketOutEvent, RunContext, function_tool
from grasp_agents.durability import FileCheckpointStore
from grasp_agents.sandbox import NetworkPolicy, local_environment
from grasp_agents.tools.bash import bash_tools
from grasp_agents.tools.file_toolkit import FileToolkit
from grasp_agents.tools.notebook_exec import RunCell

DEFAULT_MODEL = "gpt-5.5"

TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
SANDBOX_ALLOWED_DOMAINS = ("raw.githubusercontent.com",)

# The agent venv's package requirements, declared on the sandbox env. With
# provision=True the env creates the venv and installs these; on later runs it
# just verifies them. ipykernel is required for the notebook/RunPython kernel.
VENV_PACKAGES = ("ipykernel", "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn")

DEFAULT_GOAL = (
    "Build the most accurate Titanic survival classifier you can within your "
    f"turn budget. The dataset CSV is at {TITANIC_URL}."
)


# ---------------------------------------------------------------------------
# Session state — the experiment log and the hidden holdout labels
# ---------------------------------------------------------------------------


class ExperimentRecord(BaseModel):
    name: str
    cv_accuracy: float | None = None
    holdout_accuracy: float | None = None
    holdout_balanced_accuracy: float | None = None
    notes: str = ""


class ResearchState(BaseModel):
    """Checkpointed with the session — survives crashes and restarts."""

    holdout_labels: dict[str, int] = Field(default_factory=dict)
    experiments: list[ExperimentRecord] = Field(default_factory=list)
    submissions_used: int = 0
    submissions_cap: int = 8

    @property
    def best_holdout(self) -> float | None:
        scores = [
            e.holdout_accuracy
            for e in self.experiments
            if e.holdout_accuracy is not None
        ]
        return max(scores) if scores else None


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without an agent)
# ---------------------------------------------------------------------------


def read_csv_rows(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader.fieldnames or []), list(reader)


def stratified_split(
    rows: list[dict[str, str]],
    *,
    label_col: str,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split rows into (train, holdout), stratified by ``label_col``."""
    rng = random.Random(seed)  # noqa: S311 - reproducible split, not crypto
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row[label_col], []).append(row)
    train: list[dict[str, str]] = []
    holdout: list[dict[str, str]] = []
    for group in by_label.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_holdout = round(len(shuffled) * holdout_fraction)
        holdout.extend(shuffled[:n_holdout])
        train.extend(shuffled[n_holdout:])
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


def _aligned(
    predictions: dict[str, int], labels: dict[str, int]
) -> tuple[list[int], list[int]]:
    if not labels:
        raise ValueError("no holdout labels to score against")
    missing = sorted(set(labels) - set(predictions))
    if missing:
        raise ValueError(
            f"predictions are missing {len(missing)} holdout ids "
            f"(e.g. {missing[:5]}); predict every row of holdout.csv"
        )
    ids = list(labels)
    return [labels[i] for i in ids], [predictions[i] for i in ids]


def accuracy(predictions: dict[str, int], labels: dict[str, int]) -> float:
    y_true, y_pred = _aligned(predictions, labels)
    hits = sum(t == p for t, p in zip(y_true, y_pred, strict=True))
    return hits / len(y_true)


def balanced_accuracy(predictions: dict[str, int], labels: dict[str, int]) -> float:
    """Mean per-class recall — honest under the dataset's class imbalance."""
    y_true, y_pred = _aligned(predictions, labels)
    recalls: list[float] = []
    for cls in set(y_true):
        total = sum(t == cls for t in y_true)
        hit = sum(t == cls and p == cls for t, p in zip(y_true, y_pred, strict=True))
        recalls.append(hit / total)
    return sum(recalls) / len(recalls)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)


def _format_experiments(state: ResearchState) -> str:
    if not state.experiments:
        return "No experiments recorded yet."

    def fmt(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "-"

    lines = ["name | cv_accuracy | holdout_accuracy | holdout_balanced | notes"]
    lines.extend(
        f"{e.name} | {fmt(e.cv_accuracy)} | {fmt(e.holdout_accuracy)} | "
        f"{fmt(e.holdout_balanced_accuracy)} | {e.notes or '-'}"
        for e in state.experiments
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Research tools — the harness the agent works against
# ---------------------------------------------------------------------------


def make_research_tools(workdir: Path) -> list[Any]:
    """Build the dataset-split / scoring / experiment-tracking tools."""
    workdir = workdir.resolve()

    def _resolve(rel: str) -> Path:
        path = Path(rel)
        resolved = (path if path.is_absolute() else workdir / path).resolve()
        if not resolved.is_relative_to(workdir):
            raise ValueError(f"path escapes the workspace: {rel}")
        return resolved

    @function_tool
    def split_dataset(
        csv_path: str = "data/titanic.csv",
        holdout_fraction: float = 0.2,
        seed: int = 42,
        *,
        ctx: RunContext[ResearchState],
    ) -> str:
        """
        Split the downloaded dataset into a labeled ``data/train.csv`` and an
        UNLABELED ``data/holdout.csv`` (stratified by Survived). The holdout
        labels are stored out of your reach and used only by
        ``submit_predictions``. Call this once, right after downloading.
        """
        state = ctx.state
        if state.submissions_used > 0:
            raise ValueError(
                "cannot re-split after submissions were scored — the holdout "
                "is fixed for this session"
            )
        source = _resolve(csv_path)
        if not source.is_file():
            raise ValueError(f"dataset not found at {csv_path}; download it first")
        fields, rows = read_csv_rows(source.read_text())
        for required in ("PassengerId", "Survived"):
            if required not in fields:
                raise ValueError(f"expected a '{required}' column, got: {fields}")
        if not 0.05 <= holdout_fraction <= 0.5:
            raise ValueError("holdout_fraction must be in [0.05, 0.5]")

        train, holdout = stratified_split(
            rows, label_col="Survived", holdout_fraction=holdout_fraction, seed=seed
        )
        holdout_fields = [f for f in fields if f != "Survived"]
        _write_csv(workdir / "data" / "train.csv", fields, train)
        _write_csv(workdir / "data" / "holdout.csv", holdout_fields, holdout)
        state.holdout_labels = {
            row["PassengerId"]: int(row["Survived"]) for row in holdout
        }
        return (
            f"Split {len(rows)} rows -> data/train.csv ({len(train)} labeled rows) "
            f"+ data/holdout.csv ({len(holdout)} rows, Survived column removed). "
            f"Train on train.csv only; predict holdout.csv and score via "
            f"submit_predictions."
        )

    @function_tool
    def record_experiment(
        name: str,
        cv_accuracy: float,
        notes: str = "",
        *,
        ctx: RunContext[ResearchState],
    ) -> str:
        """
        Log an experiment and its cross-validation accuracy on train.csv.
        Call this after EVERY evaluated idea so progress survives restarts.
        """
        state = ctx.state
        for existing in state.experiments:
            if existing.name == name:
                existing.cv_accuracy = cv_accuracy
                if notes:
                    existing.notes = notes
                break
        else:
            state.experiments.append(
                ExperimentRecord(name=name, cv_accuracy=cv_accuracy, notes=notes)
            )
        ranked = sorted(
            (e for e in state.experiments if e.cv_accuracy is not None),
            key=lambda e: e.cv_accuracy or 0.0,
            reverse=True,
        )
        top = ", ".join(f"{e.name}={e.cv_accuracy:.4f}" for e in ranked[:5])
        return f"Recorded '{name}' (cv={cv_accuracy:.4f}). Leaderboard: {top}"

    @function_tool
    def submit_predictions(
        csv_path: str,
        experiment_name: str,
        *,
        ctx: RunContext[ResearchState],
    ) -> str:
        """
        Score a predictions CSV (columns: PassengerId, Survived) against the
        hidden holdout labels: accuracy (the primary metric) plus balanced
        accuracy. Submissions are LIMITED — submit only when cross-validation
        suggests a real improvement.
        """
        state = ctx.state
        if not state.holdout_labels:
            raise ValueError("no holdout exists yet — call split_dataset first")
        if state.submissions_used >= state.submissions_cap:
            raise ValueError(
                f"submission budget exhausted "
                f"({state.submissions_used}/{state.submissions_cap}); "
                "finish with your best CV-validated model"
            )
        source = _resolve(csv_path)
        if not source.is_file():
            raise ValueError(f"predictions file not found: {csv_path}")
        fields, rows = read_csv_rows(source.read_text())
        for required in ("PassengerId", "Survived"):
            if required not in fields:
                raise ValueError(f"expected a '{required}' column, got: {fields}")
        try:
            predictions = {
                row["PassengerId"]: int(float(row["Survived"])) for row in rows
            }
        except ValueError as exc:
            raise ValueError(f"Survived must be 0/1 integers: {exc}") from exc

        score = accuracy(predictions, state.holdout_labels)
        balanced = balanced_accuracy(predictions, state.holdout_labels)
        state.submissions_used += 1
        for existing in state.experiments:
            if existing.name == experiment_name:
                existing.holdout_accuracy = score
                existing.holdout_balanced_accuracy = balanced
                break
        else:
            state.experiments.append(
                ExperimentRecord(
                    name=experiment_name,
                    holdout_accuracy=score,
                    holdout_balanced_accuracy=balanced,
                )
            )
        best = max(score, state.best_holdout or 0.0)
        left = state.submissions_cap - state.submissions_used
        return (
            f"Holdout scores for '{experiment_name}': accuracy {score:.4f}, "
            f"balanced accuracy {balanced:.4f} "
            f"(best accuracy so far: {best:.4f}; {left} submissions left)"
        )

    @function_tool
    def research_status(*, ctx: RunContext[ResearchState]) -> str:
        """
        See the current session's state: the experiment log and the submission budget.
        """
        state = ctx.state
        best = state.best_holdout
        return (
            f"Holdout split ready: {'yes' if state.holdout_labels else 'no'}"
            f"\nSubmissions: {state.submissions_used}/{state.submissions_cap}"
            f" (best holdout: {f'{best:.4f}' if best is not None else 'n/a'})"
            "\n\nExperiments:\n" + _format_experiments(state)
        )

    return [split_dataset, record_experiment, submit_predictions, research_status]


# ---------------------------------------------------------------------------
# Workspace + agent assembly
# ---------------------------------------------------------------------------

_EMPTY_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "intro",
            "metadata": {},
            "source": (
                "# Research notebook\n\nLab notebook for the auto-research "
                "session. One section per experiment."
            ),
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def prepare_workspace(workdir: Path) -> None:
    """
    Lay out the workspace: the data/submissions dirs + an empty
    ``research.ipynb``. Idempotent. The agent's venv is created and populated
    by the sandbox env (``provision=True`` in :func:`build_researcher`).
    """
    workdir = workdir.resolve()
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    (workdir / "submissions").mkdir(parents=True, exist_ok=True)

    notebook = workdir / "research.ipynb"
    if not notebook.is_file():
        notebook.write_text(json.dumps(_EMPTY_NOTEBOOK, indent=1))


_RESEARCHER_SYS = """\
You are an autonomous ML research agent working in a sandboxed workspace
(your cwd). The rules of the game are strict:

ENVIRONMENT
- research.ipynb is your ONLY computation environment: all Python — EDA,
  features, training, evaluation, predictions — runs in its cells via
  NotebookRead / NotebookEdit / RunCell. Kernel state persists across calls;
  displayed plots come back to you (`%matplotlib inline` once first). Keep the
  notebook re-runnable top to bottom; fix failing cells in place.
- The kernel runs on .venv — numpy, pandas, scikit-learn, seaborn and
  matplotlib are preinstalled.
- Bash is ONLY for downloads (`curl -fsSL` — quiet, fails on error) and file
  manipulation — never for running Python.
- Use only the available Python packages; installing extras is forbidden.

EXPERIMENT PROTOCOL
- Call split_dataset once after downloading the data: it writes labeled
  data/train.csv + unlabeled data/holdout.csv and hides the holdout labels
  from you. Never attempt to reconstruct them; validate on train.csv only.
- The metric is balanced accuracy.
- Iterate using k-fold cross-validation on train.csv only to avoid overfitting.
- Log EVERY evaluated idea with record_experiment (metrics + a note).
- submit_predictions scores a submissions/<name>.csv (PassengerId,Survived
 for every holdout row) against the hidden labels. The budget is strictly limited: 
    submit only on real CV improvements.
- When you are done (while turn budget remains), present you results as follows:
    * Write a concise research report as a Markdown cell in research.ipynb.
    * Make a nice seaborn plot that shows the evolution of your experiments', 
indicating the frontier of the best CV and holdout scores as a "ladder" plot. 
"""

def build_researcher(
    workdir: Path,
    *,
    model: str = DEFAULT_MODEL,
    confinement: Literal["none", "seatbelt", "srt"] = "srt",
    max_turns: int = 100,
    session_key: str = "autores",
) -> tuple[LLMAgent[str, str, ResearchState], RunContext[ResearchState]]:
    """Build the researcher agent bound to a durable, sandboxed session."""
    workdir = workdir.resolve()
    prepare_workspace(workdir)

    # provision=True creates <workdir>/.venv (the first allowed_root) and
    # installs VENV_PACKAGES into it; the notebook/RunPython kernel runs on it.
    env = local_environment(
        allowed_roots=[workdir],
        confinement=confinement,
        network=NetworkPolicy.ALLOWLIST,
        allowed_domains=SANDBOX_ALLOWED_DOMAINS,
        packages=VENV_PACKAGES,
        provision=True,
        env={"MPLCONFIGDIR": str(workdir / ".mpl")},
    )
    ctx = RunContext[ResearchState](
        state=ResearchState(),
        serialize_state=True,
        checkpoint_store=FileCheckpointStore(workdir / ".grasp" / "checkpoints"),
        session_key=session_key,
        environment=env,
    )
    toolkit = FileToolkit(include_notebook=True)
    researcher = LLMAgent[str, str, ResearchState](
        name="researcher",
        ctx=ctx,
        llm=_make_llm(model),
        tools=[
            *bash_tools(auto_background_at=90.0),
            RunCell(default_timeout=240.0),
            *toolkit.tools(),
            *make_research_tools(workdir),
        ],
        sys_prompt=_RESEARCHER_SYS,
        max_turns=max_turns,
        stream_llm=True,
    )
    return researcher, ctx


def _make_llm(model: str) -> Any:
    if model.startswith("claude"):
        from grasp_agents.llm_providers.anthropic.anthropic_llm import (  # noqa: PLC0415
            AnthropicLLM,
            AnthropicLLMSettings,
        )

        return AnthropicLLM(
            model_name=model,
            llm_settings=AnthropicLLMSettings(max_tokens=16_000),
        )
    from grasp_agents.llm_providers.openai_responses import (  # noqa: PLC0415
        OpenAIResponsesLLM,
        OpenAIResponsesLLMSettings,
    )

    return OpenAIResponsesLLM(
        model_name=model,
        llm_settings=OpenAIResponsesLLMSettings(
            reasoning={"effort": "medium", "summary": "detailed"}
        ),
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def run_headless(
    researcher: LLMAgent[str, str, ResearchState],
    *,
    goal: str | None,
) -> str | None:
    """Run one research session in the console; ``goal=None`` resumes."""
    from grasp_agents import stream_events  # noqa: PLC0415

    final: str | None = None
    async with researcher:
        async for event in stream_events(
            researcher.run_stream(goal), show_thinking=True
        ):
            if isinstance(event, ProcPacketOutEvent):
                final = str(event.data.payloads[0])
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default="./autoresearch_workdir")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument(
        "--confinement", default="srt", choices=["none", "seatbelt", "srt", "auto"]
    )
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument(
        "--headless", action="store_true", help="run in the console instead of the TUI"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="headless: continue the checkpointed session instead of a new goal",
    )
    args = parser.parse_args()

    load_dotenv()
    researcher, _ctx = build_researcher(
        Path(args.workdir),
        model=args.model,
        confinement=args.confinement,
        max_turns=args.max_turns,
    )

    if args.headless:
        final = asyncio.run(
            run_headless(researcher, goal=None if args.resume else args.goal)
        )
        print(f"\n=== final report ===\n{final}")
        return

    from grasp_agents.ui import run_tui_interactive  # noqa: PLC0415

    run_tui_interactive(researcher.run_stream, main_agent=researcher.name)


if __name__ == "__main__":
    main()
