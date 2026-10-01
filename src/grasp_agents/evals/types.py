import math
import traceback
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grasp_agents.utils.errors import format_error_chain, root_cause

from ._util import canonical_json, short_hash

type ScoreValue = bool | float | str


class ScoreReason:
    """
    Conventional ``Score.reason`` values. The first group blames the system
    under test (a real failure); the second blames the instrument (the score
    is missing, not low).
    """

    INVALID_RESPONSE_FORMAT = "invalid_response_format"
    REFUSAL = "refusal"
    NO_RESPONSE = "no_response"
    GRADER_FAILED = "grader_failed"
    SCORING_FAILED = "scoring_failed"


class Example[InT, RefT](BaseModel):
    """
    One evaluation case: an input for the task, an optional reference (ground
    truth, expected output, rubric target) and free-form metadata.

    ``id`` must stay stable across dataset versions — it is how runs are
    paired for comparison and resumed. When omitted it defaults to a hash of
    the input, so editing the input of such an example makes it a new example;
    give curated examples explicit ids.
    """

    id: str = ""
    input: InT
    reference: RefT | None = None
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])
    # Split membership (e.g. "dev", "test"), versioned with the content.
    splits: list[str] = Field(default_factory=list[str])

    @model_validator(mode="after")
    def _default_id(self) -> Self:
        if not self.id:
            self.id = short_hash(canonical_json(self.input))
        return self

    def content_hash(self) -> str:
        """Hash of everything an evaluation can depend on (not split membership)."""
        return short_hash(
            canonical_json(
                {
                    "input": self.input,
                    "reference": self.reference,
                    "metadata": self.metadata,
                }
            )
        )


class Score(BaseModel):
    """
    One per-example judgment produced by an evaluator.

    ``value`` is a pass/fail (``bool``), a number, or a categorical label
    (``str``). ``None`` means *unscored*: the evaluator ran but could not
    produce a judgment (e.g. an unparseable judge verdict). Unscored values are
    excluded from metrics and counted separately — never turned into a number.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    value: ScoreValue | None
    explanation: str | None = None
    # Machine-readable cause for an unscored value or a degenerate output,
    # e.g. "invalid_response_format", "refusal", "no_response".
    reason: str | None = None
    # Name of the evaluator that produced this score (stamped by the runner).
    evaluator: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])

    @classmethod
    def unscored(
        cls, name: str, reason: str, explanation: str | None = None
    ) -> "Score":
        return cls(name=name, value=None, reason=reason, explanation=explanation)

    @property
    def scored(self) -> bool:
        return self.value is not None

    def as_float(self) -> float | None:
        """Numeric view: ``True``/``False`` → 1.0/0.0; labels have none."""
        value = self.value
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, float | int):
            return float(value) if math.isfinite(value) else None
        return None


class ErrorInfo(BaseModel):
    # Type of the root cause (framework wrappers such as ``ProcRunError`` are
    # skipped), so errors group by what actually went wrong.
    type: str
    # Complete description, type included ("ValueError: ..."); for chained
    # exceptions the whole cause chain, outermost first.
    message: str
    traceback: str | None = None

    @classmethod
    def from_exception(cls, exc: BaseException, *, max_frames: int = 8) -> Self:
        frames = traceback.format_exception(exc, limit=max_frames)
        return cls(
            type=type(root_cause(exc)).__name__,
            message=format_error_chain(exc),
            traceback="".join(frames)[-8000:],
        )


class EvaluatorFailure(BaseModel):
    evaluator: str
    error: ErrorInfo


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: "Usage") -> "Usage":
        cost = (
            None
            if self.cost_usd is None and other.cost_usd is None
            else (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        )
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cost_usd=cost,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def is_empty(self) -> bool:
        return self.total_tokens == 0 and self.cost_usd is None


class Trial(BaseModel):
    """
    One execution of the task on one example (one repetition), and its scores.

    ``output`` is stored in JSON form; ``error`` is set instead when the task
    raised or timed out (evaluators are then skipped).
    """

    example_id: str
    repetition: int = 0
    example_hash: str
    output: Any = None
    error: ErrorInfo | None = None
    started_at: datetime
    duration_s: float
    usage: Usage = Field(default_factory=Usage)
    usage_by_agent: dict[str, Usage] = Field(default_factory=dict[str, Usage])
    models: dict[str, list[str]] = Field(default_factory=dict[str, list[str]])
    trace_id: str | None = None
    measurements: dict[str, float] = Field(default_factory=dict[str, float])
    scores: list[Score] = Field(default_factory=list[Score])
    evaluator_failures: list[EvaluatorFailure] = Field(
        default_factory=list[EvaluatorFailure]
    )
    # Evaluators that completed on this trial (including "not applicable").
    evaluated: list[str] = Field(default_factory=list[str])
    evaluator_usage: Usage = Field(default_factory=Usage)
    # The example belongs to a sealed (held-out) split: reports show this
    # trial only in aggregate.
    sealed: bool = False

    @property
    def key(self) -> tuple[str, int]:
        return (self.example_id, self.repetition)

    @property
    def ok(self) -> bool:
        return self.error is None

    def score(self, name: str) -> Score | None:
        for score in self.scores:
            if score.name == name:
                return score
        return None

    @property
    def total_usage(self) -> Usage:
        return self.usage + self.evaluator_usage


class MetricResult(BaseModel):
    name: str
    value: float | None
    # Units the value aggregates (examples, or trials for trial-level metrics).
    n: int
    # Units with no usable value: the task failed, the score is unscored, or
    # its evaluator failed. Never silently counted as zero.
    n_missing: int = 0
    # Units the score does not apply to (the evaluator returned nothing).
    n_na: int = 0
    stderr: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    details: dict[str, Any] = Field(default_factory=dict[str, Any])
    groups: dict[str, "MetricResult"] = Field(default_factory=dict[str, "MetricResult"])


class DatasetRef(BaseModel):
    """What was evaluated: the source dataset and the subset selected from it."""

    name: str
    source: str | None = None
    # Store-assigned version (e.g. a Phoenix dataset version id); ``None`` for
    # files, whose content fingerprint identifies them.
    version: str | None = None
    fingerprint: str
    size: int
    # Derivation steps applied to the source, e.g. ["split=dev", "sample=20 seed=0"].
    selection: list[str] = Field(default_factory=list[str])
    selected_fingerprint: str
    selected_size: int


class ComponentInfo(BaseModel):
    """Declared identity of a task or evaluator, for provenance and diffs."""

    name: str
    # Import path of the implementing class or function.
    kind: str
    version: str | None = None
    config: dict[str, Any] = Field(default_factory=dict[str, Any])
    # Evaluators only: who produced the judgments.
    annotator: Literal["CODE", "LLM", "HUMAN"] | None = None


class Provenance(BaseModel):
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    # Hash of the uncommitted diff of tracked files: two runs with the same
    # commit and diff hash ran the same code.
    git_diff_hash: str | None = None
    python: str
    grasp_agents: str | None = None
    # Models seen in task responses, per agent name, across all trials.
    observed_models: dict[str, list[str]] = Field(default_factory=dict[str, list[str]])


class RunConfig(BaseModel):
    repetitions: int = 1
    concurrency: int = 4
    timeout_s: float | None = None
    max_cost_usd: float | None = None
    max_error_rate: float | None = None
    score: bool = True
    group_by: list[str] = Field(default_factory=list[str])
    cluster_by: str | None = None
    sealed_splits: list[str] = Field(default_factory=list[str])


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    # Stopped before every trial ran (cost budget exhausted).
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunCounts(BaseModel):
    examples: int = 0
    trials_expected: int = 0
    trials_done: int = 0
    task_errors: int = 0
    evaluator_failures: int = 0
    unscored: int = 0


class PhoenixLink(BaseModel):
    base_url: str
    dataset_id: str
    dataset_version_id: str | None = None
    experiment_id: str | None = None
    # Trial keys ("<example_id>#<repetition>") already logged as experiment runs.
    logged_trials: list[str] = Field(default_factory=list[str])


type RunKind = Literal["evaluation", "rescore", "pairwise"]


class EvaluationRun(BaseModel):
    """
    The record of one execution of an evaluation.

    Immutable once finished: rescoring produces a child run (``parent_run_id``)
    instead of editing this one. ``trials`` and ``examples`` are stored next to
    the header (``trials.jsonl`` / ``examples.jsonl``), not inside ``run.json``.
    """

    schema_version: int = 1
    id: str
    name: str
    kind: RunKind = "evaluation"
    status: RunStatus = RunStatus.RUNNING
    created_at: datetime
    finished_at: datetime | None = None
    description: str | None = None
    # Import spec of the ``Evaluation`` this run came from, if any.
    evaluation: str | None = None
    parent_run_id: str | None = None
    dataset: DatasetRef
    task: ComponentInfo
    evaluators: list[ComponentInfo] = Field(default_factory=list[ComponentInfo])
    config: RunConfig = Field(default_factory=RunConfig)
    provenance: Provenance
    config_hash: str
    counts: RunCounts = Field(default_factory=RunCounts)
    usage: Usage = Field(default_factory=Usage)
    metrics: list[MetricResult] = Field(default_factory=list[MetricResult])
    # Set when the run finished but its numbers should not be trusted (e.g.
    # the task error rate exceeded ``config.max_error_rate``).
    invalid_reason: str | None = None
    tags: list[str] = Field(default_factory=list[str])
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])
    phoenix: PhoenixLink | None = None

    trials: list[Trial] = Field(default_factory=list[Trial], exclude=True)
    examples: list[Example[Any, Any]] = Field(
        default_factory=list[Example[Any, Any]], exclude=True
    )

    @property
    def finished(self) -> bool:
        return self.status != RunStatus.RUNNING

    def example(self, example_id: str) -> Example[Any, Any] | None:
        for example in self.examples:
            if example.id == example_id:
                return example
        return None

    def trial(self, example_id: str, repetition: int = 0) -> Trial | None:
        for trial in self.trials:
            if trial.example_id == example_id and trial.repetition == repetition:
                return trial
        return None

    def metric(self, name: str) -> MetricResult | None:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def score_names(self) -> list[str]:
        names: dict[str, None] = {}
        for trial in self.trials:
            for score in trial.scores:
                names.setdefault(score.name, None)
        return list(names)
