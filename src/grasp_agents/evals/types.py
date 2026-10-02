import dataclasses
import math
import traceback
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property
from typing import Any, Literal, NamedTuple, Self, cast, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticSerializationError, to_jsonable_python

from grasp_agents.utils.errors import format_error_chain, root_cause

from ._util import canonical_json, short_hash, to_jsonable

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
    NON_FINITE_VALUE = "non_finite_value"


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        # Defaulted fields are left out, so adding a field with a default to
        # an input model keeps existing ids and hashes.
        return _plain(value.model_dump(mode="python", exclude_defaults=True))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in cast("Mapping[Any, Any]", value).items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in cast("Iterable[Any]", value)]
    if isinstance(value, set | frozenset):
        members = (_plain(v) for v in cast("Iterable[Any]", value))
        return sorted(members, key=canonical_json)
    return value


def record_form(value: Any) -> Any:
    """
    ``value`` as stored in dataset files: plain JSON with model fields left at
    their defaults omitted, set members sorted and bytes base64-encoded.
    """
    try:
        return to_jsonable_python(
            _plain(value), inf_nan_mode="constants", bytes_mode="base64"
        )
    except PydanticSerializationError as exc:
        raise TypeError(
            f"Cannot store a {type(value).__name__} value: it is not JSON-serializable"
        ) from exc


def input_digest(raw_input: Any) -> str:
    """Default example id: a hash of the input as written in the dataset."""
    return short_hash(canonical_json(raw_input))


def content_digest(raw_input: Any, raw_reference: Any, metadata: Any) -> str:
    """Hash of everything an evaluation can depend on (not split membership)."""
    return short_hash(
        canonical_json(
            {"input": raw_input, "reference": raw_reference, "metadata": metadata}
        )
    )


_CONTENT_FIELDS = frozenset({"input", "reference", "metadata"})


class ExampleRecord(NamedTuple):
    """An example's content as stored in dataset files and Phoenix."""

    input: Any
    reference: Any
    metadata: dict[str, Any]


class Example[InT, RefT](BaseModel):
    """
    One evaluation case: an input for the task, an optional reference (ground
    truth, expected output, rubric target) and free-form metadata.

    ``id`` must stay stable across dataset versions — it is how runs are
    paired for comparison and resumed. When omitted it defaults to a hash of
    the input, so editing the input of such an example makes it a new example;
    give curated examples explicit ids.

    ``content_hash`` identifies the example's content (input, reference and
    metadata, not splits); runs pair examples only when it is unchanged. It
    hashes :attr:`record`, the content as stored: loaders keep each record
    exactly as read, so changing the task's types does not re-identify
    examples, and saving or pushing a dataset keeps its hashes; examples built
    in code are stored as plain JSON with defaulted model fields left out.
    Examples are immutable: derive changed ones with
    ``model_copy(update=...)``, which rehashes them.
    """

    model_config = ConfigDict(frozen=True)

    id: str = ""
    input: InT
    reference: RefT | None = None
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])
    # Split membership (e.g. "dev", "test"), versioned with the content.
    splits: list[str] = Field(default_factory=list[str])
    content_hash: str = ""

    @model_validator(mode="wrap")
    @classmethod
    def _identity(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        if not isinstance(data, Mapping) or "input" not in data:
            return handler(data)
        fields = dict(cast("Mapping[str, Any]", data))
        record: ExampleRecord | None = None
        if not fields.get("id") or not fields.get("content_hash"):
            record = ExampleRecord(
                record_form(fields["input"]),
                record_form(fields.get("reference")),
                record_form(fields.get("metadata") or {}),
            )
            if not fields.get("id"):
                fields["id"] = input_digest(record.input)
        if record is not None and not fields.get("content_hash"):
            fields["content_hash"] = content_digest(*record)
            return with_record(handler(fields), record)
        return handler(fields)

    @cached_property
    def record(self) -> ExampleRecord:
        """The content as stored in dataset files and Phoenix."""
        return ExampleRecord(
            record_form(self.input),
            record_form(self.reference),
            record_form(self.metadata),
        )

    @override
    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        changes = dict(update or {})
        changed = changes.keys() & _CONTENT_FIELDS
        record: ExampleRecord | None = None
        if changed and "content_hash" not in changes:
            record = self.record._replace(
                **{name: record_form(changes[name]) for name in changed}
            )
            changes["content_hash"] = content_digest(*record)
        copied = super().model_copy(update=changes, deep=deep)
        if changed:
            vars(copied).pop("record", None)
        return copied if record is None else with_record(copied, record)


def with_record[E: Example[Any, Any]](example: E, record: ExampleRecord) -> E:
    """``example`` with ``record`` as its stored content (for loaders)."""
    vars(example)["record"] = record
    return example


class JudgedOutput[InT, OutT, RefT](BaseModel):
    """
    An output together with the example it answers: what a judge reads, and
    the input of judge validation, probes and labeling.
    """

    input: InT
    output: OutT
    reference: RefT | None = None
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])


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
    # Model usage reported by each evaluator (see ``EvalContext.record_usage``).
    evaluator_usage: dict[str, Usage] = Field(default_factory=dict[str, Usage])
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
        return sum(self.evaluator_usage.values(), self.usage)


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
    confidence: float = 0.95
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
    # Tasks and processor judges: hash of what the processor is made of
    # (models, settings, prompts, tools, structure) when it can be read before
    # running.
    fingerprint: str | None = None
    # Evaluators only: hash of the evaluator's own code (its function or
    # class). Outside the config hash; rescoring re-runs an evaluator whose
    # code changed, and comparisons warn about it.
    source: str | None = None

    @field_validator("config")
    @classmethod
    def _stored_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        # Kept as stored on disk, so a reloaded run's components compare equal.
        return cast("dict[str, Any]", to_jsonable(_plain(config)))


class Provenance(BaseModel):
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    # Hash of the uncommitted diff of tracked files: two runs with the same
    # commit and diff hash ran the same code.
    git_diff_hash: str | None = None
    # Hash of the source files defining the task and evaluators, tracked by
    # git or not.
    source_hash: str | None = None
    python: str
    grasp_agents: str | None = None
    # Models seen in task responses, per agent name, across all trials.
    observed_models: dict[str, list[str]] = Field(default_factory=dict[str, list[str]])


class RunConfig(BaseModel):
    repetitions: int = 1
    concurrency: int = 4
    timeout_s: float | None = None
    evaluator_timeout_s: float | None = None
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
    # Trial key ("<example_id>#<repetition>") → digest of what was logged for
    # it, so trials that changed since (resumed, re-scored) are logged again.
    logged_trials: dict[str, str] = Field(default_factory=dict[str, str])


type RunKind = Literal["evaluation", "rescore", "pairwise", "retry", "online"]


class TraceWindow(BaseModel):
    """
    The ``[start, end)`` span start times an online run read from a trace
    store (UTC when given without a timezone).
    """

    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def _aware(cls, moment: datetime) -> datetime:
        return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


class EvaluationRun(BaseModel):
    """
    The record of one execution of an evaluation.

    A completed run's results are never modified (pushing it to Phoenix only
    records the link): resuming it creates a ``retry`` child and rescoring a
    ``rescore`` child (``parent_run_id``). Runs that did not complete
    (running, cancelled, partial, failed) are continued in place. An
    ``online`` run scores outputs read from production traces in ``window``
    instead of running the task.
    ``trials`` and ``examples`` are stored next to the header
    (``trials.jsonl`` / ``examples.jsonl``), not inside ``run.json``.
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
    window: TraceWindow | None = None
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

    @property
    def completed(self) -> bool:
        return self.status == RunStatus.COMPLETED

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
