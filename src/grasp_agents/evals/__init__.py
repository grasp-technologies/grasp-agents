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

Judges are scorers too: :class:`ProcessorScorer` turns any processor
(an ``LLMAgent`` with a structured verdict) into one, and
:func:`judge_validation` / :func:`judge_probes` measure a judge with the same
machinery — agreement with labels collected by :func:`sample_for_labeling` and
:func:`import_labels`, and sensitivity to :class:`Perturbation` s. A
:class:`ValidationGate` keeps an evaluation from using a judge that has not
passed validation.
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
    TraceExport,
    list_evaluations,
    load_evaluation,
    load_object,
)
from .judge import (
    JudgedPair,
    ProcessorPairwiseJudge,
    ProcessorScorer,
    judge_probes,
    judge_validation,
)
from .labeling import import_labels, judged_outputs, sample_for_labeling
from .metrics import (
    ClassRecall,
    CohenKappa,
    ConfusionMatrix,
    Consistency,
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
from .online import (
    AnnotationError,
    AnnotationsRejectedError,
    Extracted,
    Extractor,
    SpanRecord,
    TraceAnnotation,
    TraceItem,
    TraceQuery,
    TraceSource,
    annotate_run,
    default_extractor,
    evaluate_traces,
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
from .runner import (
    ResumeError,
    SealedSelectionError,
    evaluate,
    evaluate_trials,
    rescore,
)
from .scorer import (
    EvalContext,
    FunctionScorer,
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
    JudgedOutput,
    MetricResult,
    Provenance,
    RunStatus,
    Score,
    ScoreReason,
    ScorerFailure,
    TraceWindow,
    Trial,
    Usage,
)
from .validation import (
    CorrectedPassRate,
    JudgeErrorRates,
    JudgeValidation,
    Perturbation,
    UnvalidatedJudgeError,
    ValidationGate,
)

__all__ = [
    "AnnotationError",
    "AnnotationsRejectedError",
    "ClassRecall",
    "CohenKappa",
    "Comparison",
    "ComponentInfo",
    "ConfusionMatrix",
    "Consistency",
    "CorrectedPassRate",
    "Dataset",
    "DatasetCheck",
    "DatasetError",
    "DatasetProblem",
    "DatasetRef",
    "Distribution",
    "ErrorInfo",
    "ErrorRate",
    "EvalContext",
    "Evaluation",
    "EvaluationRun",
    "Example",
    "ExampleDelta",
    "Extracted",
    "Extractor",
    "FunctionPairwiseJudge",
    "FunctionScorer",
    "FunctionTask",
    "JudgeErrorRates",
    "JudgeValidation",
    "JudgedOutput",
    "JudgedPair",
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
    "Perturbation",
    "ProcessorPairwiseJudge",
    "ProcessorScorer",
    "ProcessorTask",
    "Proportion",
    "Provenance",
    "ResumeError",
    "RunNotFoundError",
    "RunStatus",
    "RunStore",
    "Score",
    "ScoreReason",
    "Scorer",
    "ScorerFailure",
    "ScorerOutput",
    "SealedSelectionError",
    "SpanRecord",
    "SpecError",
    "TargetComparison",
    "Task",
    "Total",
    "TraceAnnotation",
    "TraceExport",
    "TraceItem",
    "TraceQuery",
    "TraceSource",
    "TraceWindow",
    "Trial",
    "TrialContext",
    "TrialProgress",
    "UnvalidatedJudgeError",
    "Usage",
    "ValidationGate",
    "WinRate",
    "annotate_run",
    "compare",
    "compute_metrics",
    "default_extractor",
    "default_metrics",
    "evaluate",
    "evaluate_traces",
    "evaluate_trials",
    "example_json_schema",
    "import_labels",
    "judge_probes",
    "judge_validation",
    "judged_outputs",
    "list_evaluations",
    "load_evaluation",
    "load_object",
    "pairwise",
    "render_comparison_markdown",
    "render_run_markdown",
    "rescore",
    "run_summary",
    "sample_for_labeling",
    "scorer",
]
