"""
Evaluations for grasp-agents processors.

Build a :class:`Dataset` of :class:`Example` s, run any processor (or async
function) over it with :func:`evaluate`, score each :class:`Trial` with
:class:`Scorer` s, aggregate with :class:`Metric` s, and compare runs with
:func:`compare`. Every run is an append-only, on-disk :class:`EvaluationRun`
whose results never change once completed; retrying, rescoring (:func:`rescore`)
and pairwise judging (:func:`pairwise`) create new runs from stored outputs.
Package an evaluation as an :class:`Evaluation` to run it from the
``grasp-evals`` CLI.
"""

from ._execution import TrialProgress
from .compare import Comparison, ExampleDelta, TargetComparison, compare
from .dataset import (
    Dataset,
    DatasetCheck,
    DatasetError,
    DatasetProblem,
    example_json_schema,
)
from .evaluation import (
    Evaluation,
    SpecError,
    list_evaluations,
    load_evaluation,
    load_object,
)
from .metrics import (
    Distribution,
    ErrorRate,
    Mean,
    Measure,
    Metric,
    PassAtK,
    PassHatK,
    PassRate,
    Percentile,
    Proportion,
    Total,
    compute_metrics,
    default_metrics,
)
from .pairwise import (
    FunctionPairwiseJudge,
    OrderSwapped,
    PairwiseContext,
    PairwiseJudge,
    PairwiseVerdict,
    WinRate,
    pairwise,
)
from .report import render_comparison_markdown, render_run_markdown, run_summary
from .runner import ResumeError, SealedSelectionError, evaluate, rescore
from .scorer import (
    FunctionScorer,
    ScoreContext,
    Scorer,
    ScorerOutput,
    scorer,
)
from .store import LocalRunStore, RunNotFoundError, RunStore
from .task import FunctionTask, ProcessorTask, Task, TrialContext
from .types import (
    ComponentInfo,
    DatasetRef,
    ErrorInfo,
    EvaluationRun,
    Example,
    MetricResult,
    Provenance,
    RunStatus,
    Score,
    ScoreReason,
    ScorerFailure,
    Trial,
    Usage,
)

__all__ = [
    "Comparison",
    "ComponentInfo",
    "Dataset",
    "DatasetCheck",
    "DatasetError",
    "DatasetProblem",
    "DatasetRef",
    "Distribution",
    "ErrorInfo",
    "ErrorRate",
    "Evaluation",
    "EvaluationRun",
    "Example",
    "ExampleDelta",
    "FunctionPairwiseJudge",
    "FunctionScorer",
    "FunctionTask",
    "LocalRunStore",
    "Mean",
    "Measure",
    "Metric",
    "MetricResult",
    "OrderSwapped",
    "PairwiseContext",
    "PairwiseJudge",
    "PairwiseVerdict",
    "PassAtK",
    "PassHatK",
    "PassRate",
    "Percentile",
    "ProcessorTask",
    "Proportion",
    "Provenance",
    "ResumeError",
    "RunNotFoundError",
    "RunStatus",
    "RunStore",
    "Score",
    "ScoreContext",
    "ScoreReason",
    "Scorer",
    "ScorerFailure",
    "ScorerOutput",
    "SealedSelectionError",
    "SpecError",
    "TargetComparison",
    "Task",
    "Total",
    "Trial",
    "TrialContext",
    "TrialProgress",
    "Usage",
    "WinRate",
    "compare",
    "compute_metrics",
    "default_metrics",
    "evaluate",
    "example_json_schema",
    "list_evaluations",
    "load_evaluation",
    "load_object",
    "pairwise",
    "render_comparison_markdown",
    "render_run_markdown",
    "rescore",
    "run_summary",
    "scorer",
]
