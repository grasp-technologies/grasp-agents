"""
Online evaluation: score what the system did in production.

Spans are read from a trace store for a time window, grouped into items (a
span, a trace or a session), sampled deterministically, turned into trials by
an extractor, scored by reference-free evaluators like any run, and the scores
written back to the store as annotations.
"""

import asyncio
import hashlib
import inspect
import json
import logging
import math
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from operator import itemgetter
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from grasp_agents.telemetry import attributes as span_attrs

from ._execution import ProgressCallback, resolve_store
from ._util import (
    canonical_json,
    code_hash,
    qualified_name,
    short_hash,
    to_jsonable,
    utc_now,
)
from .evaluator import Evaluator
from .metrics import MetricsSpec
from .report import render_run_markdown
from .runner import evaluate_trials
from .store import RunStore
from .types import (
    ComponentInfo,
    DatasetRef,
    ErrorInfo,
    EvaluationRun,
    Example,
    RunStatus,
    Score,
    TraceWindow,
    Trial,
)

logger = logging.getLogger(__name__)

type TraceScope = Literal["span", "trace", "session"]
type AnnotatorKind = Literal["LLM", "CODE", "HUMAN"]

# Spans per batch when fetching by trace id.
TRACE_BATCH = 100
# Sessions read at a time.
SESSION_CONCURRENCY = 8


class SpanRecord(BaseModel):
    """One span as read from a trace store; ``attributes`` use dotted keys."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    span_id: str
    parent_id: str | None = None
    name: str
    start_time: datetime
    end_time: datetime | None = None
    # "OK", "ERROR" or "UNSET".
    status: str = "UNSET"
    status_message: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict[str, Any])

    @property
    def session_id(self) -> str | None:
        value = self.attributes.get(span_attrs.ATTR_SESSION_ID)
        return None if value is None else str(value)

    @property
    def errored(self) -> bool:
        return self.status == "ERROR"

    @property
    def input(self) -> Any:
        """The recorded ``input.value`` (parsed when JSON), or ``None``."""
        return _payload(
            self.attributes,
            span_attrs.ATTR_INPUT_VALUE,
            span_attrs.ATTR_INPUT_MIME_TYPE,
        )

    @property
    def output(self) -> Any:
        """The recorded ``output.value`` (parsed when JSON), or ``None``."""
        return _payload(
            self.attributes,
            span_attrs.ATTR_OUTPUT_VALUE,
            span_attrs.ATTR_OUTPUT_MIME_TYPE,
        )

    @property
    def truncated(self) -> bool:
        """Whether a recorded payload was shortened to fit the attribute limit."""
        return bool(
            self.attributes.get(span_attrs.ATTR_INPUT_TRUNCATED)
            or self.attributes.get(span_attrs.ATTR_OUTPUT_TRUNCATED)
        )

    @property
    def has_payloads(self) -> bool:
        """Whether an input or output was recorded (content tracing was on)."""
        return (
            span_attrs.ATTR_INPUT_VALUE in self.attributes
            or span_attrs.ATTR_OUTPUT_VALUE in self.attributes
        )

    @property
    def cancelled(self) -> bool:
        return bool(self.attributes.get(span_attrs.ATTR_CANCELLED))

    @property
    def from_evaluation(self) -> bool:
        """Recorded by an evaluation (a trial or its scoring), not production."""
        return span_attrs.ATTR_EVAL_RUN_ID in self.attributes


def _payload(attributes: Mapping[str, Any], key: str, mime_key: str) -> Any:
    value = attributes.get(key)
    if not isinstance(value, str) or attributes.get(mime_key) != (
        span_attrs.JSON_MIME_TYPE
    ):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def flatten_attributes(
    attributes: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """Dotted keys for attributes a store returns nested (``{"grasp": {...}}``)."""
    flat: dict[str, Any] = {}
    for key, value in attributes.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(
                flatten_attributes(cast("Mapping[str, Any]", value), f"{name}.")
            )
        else:
            flat[name] = value
    return flat


class TraceItem(BaseModel):
    """
    What one online trial scores: a span, or the selected spans of one trace
    or one session, ordered by start time.
    """

    model_config = ConfigDict(frozen=True)

    scope: TraceScope
    # The span id, trace id or session id.
    id: str
    spans: tuple[SpanRecord, ...]

    @property
    def first(self) -> SpanRecord:
        return self.spans[0]

    @property
    def trace_id(self) -> str:
        return self.first.trace_id

    @property
    def session_id(self) -> str | None:
        return self.id if self.scope == "session" else self.first.session_id

    @property
    def started_at(self) -> datetime:
        return self.first.start_time

    @property
    def ended_at(self) -> datetime:
        ends = [s.end_time or s.start_time for s in self.spans]
        return max(ends)

    @property
    def sampling_key(self) -> str:
        # The spans of one trace (or session) are sampled together, so every
        # evaluation sees the same traces.
        return self.id if self.scope == "session" else self.trace_id


class Extracted(BaseModel):
    """What an extractor reads from a trace item."""

    input: Any = None
    output: Any = None
    # Added to the provenance an online example carries (``scope``,
    # ``trace_id``, ``span_id``, ``session_id``, ``started_at``,
    # ``processor``, ``version``, ``model``, ``truncated``), which it cannot
    # replace.
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])
    # The system failed on this item: the trial is a task error.
    error: ErrorInfo | None = None


type Extractor = Callable[[TraceItem], Extracted | Awaitable[Extracted | None] | None]
"""Reads an item's input and output; ``None`` skips the item."""


def _span_error(span: SpanRecord) -> ErrorInfo | None:
    if span.errored:
        return ErrorInfo(
            type=str(span.attributes.get(span_attrs.ATTR_ERROR_TYPE) or "Error"),
            message=span.status_message or "the span failed",
        )
    if span.cancelled:
        return ErrorInfo(type="CancelledError", message="the run was cancelled")
    return None


def default_extractor(item: TraceItem) -> Extracted | None:
    """
    The recorded payloads: a span's (the first selected span's, for a trace)
    ``input.value`` / ``output.value``, where a failed or cancelled span is a
    task error; for a session, the lists of its spans' inputs and outputs.
    Items without recorded payloads (content tracing off) are skipped.
    """
    if not any(s.has_payloads for s in item.spans):
        return None
    if item.scope == "session":
        return Extracted(
            input=[s.input for s in item.spans],
            output=[s.output for s in item.spans],
        )
    span = item.first
    error = _span_error(span)
    if error is not None:
        return Extracted(input=span.input, error=error)
    return Extracted(input=span.input, output=span.output)


@dataclass(frozen=True)
class TraceQuery:
    """
    Which production spans an online evaluation reads and how they become
    trials.

    Spans of ``project`` whose start falls in the window are selected by
    processor name (``grasp.processor.name``), attribute equality and status;
    spans recorded by evaluations are left out. ``scope`` groups them: one
    trial per span, per trace, or per session (``session.id``) — a session is
    scored once, in the window in which it has been idle for
    ``session_idle_s``. A trace belongs to the window its first selected span
    started in, so ``"trace"`` suits a trace per request: the runs of a
    session share one trace when ``SessionContext.session_trace_grouping`` is
    on (the default) — score those per span or per session.

    ``sample_rate`` keeps a deterministic share of traces (sessions), and
    ``max_items`` caps the count — whole traces, spread evenly over the
    values of the ``strata`` span attributes (``"status"``: ok or error). A
    window may end no later than ``completion_buffer_s`` ago: spans are
    exported when they end, so the buffer must exceed the longest run plus
    the export delay, or such runs are missed.

    ``extractor`` maps an item to its input and output; a custom one can
    fetch full artifacts from the application's database by the ids in the
    span attributes, and is where data that must not be stored is removed.
    """

    project: str
    processor: str | None = None
    attributes: Mapping[str, str | int | float | bool] = field(
        default_factory=dict[str, str | int | float | bool]
    )
    status: Literal["ok", "error"] | None = None
    scope: TraceScope = "span"
    extractor: Extractor = default_extractor
    sample_rate: float = 1.0
    max_items: int | None = None
    strata: Sequence[str] = ()
    completion_buffer_s: float = 3600.0
    session_idle_s: float = 1800.0

    def __post_init__(self) -> None:
        if not 0.0 < self.sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1], got {self.sample_rate}")
        if self.max_items is not None and self.max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {self.max_items}")
        if self.completion_buffer_s < 0 or self.session_idle_s < 0:
            raise ValueError("completion_buffer_s and session_idle_s must be >= 0")

    @property
    def span_attributes(self) -> dict[str, str | int | float | bool]:
        """The attribute equalities a store filters spans by."""
        attributes = dict(self.attributes)
        if self.processor is not None:
            attributes[span_attrs.ATTR_PROCESSOR_NAME] = self.processor
        return attributes

    def selection(self) -> dict[str, Any]:
        """What is read: two queries with the same selection continue each other."""
        selection: dict[str, Any] = {
            "project": self.project,
            "attributes": dict(sorted(self.span_attributes.items())),
            "status": self.status,
            "scope": self.scope,
        }
        if self.scope == "session":
            selection["session_idle_s"] = self.session_idle_s
        return selection

    def describe(self) -> ComponentInfo:
        return ComponentInfo(
            name=self.processor or self.project,
            kind=qualified_name(type(self)),
            config={
                **self.selection(),
                "extractor": qualified_name(self.extractor),
                "sample_rate": self.sample_rate,
                "max_items": self.max_items,
                "strata": list(self.strata),
                "session_idle_s": self.session_idle_s,
            },
            fingerprint=short_hash(canonical_json(self.selection())),
            source=code_hash(self.extractor),
        )


class TraceAnnotation(BaseModel):
    """A score to write back onto the span, trace or session it judged."""

    scope: TraceScope
    target_id: str
    name: str
    annotator: AnnotatorKind = "CODE"
    label: str | None = None
    score: float | None = None
    explanation: str | None = None
    # Annotations with the same name, target and identifier replace each
    # other, so writing a run's scores again changes nothing.
    identifier: str
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])


class TraceSource(Protocol):
    """A trace store online evaluations read from and annotate."""

    @property
    def location(self) -> str:
        """Where the traces are, e.g. ``phoenix:<server>/<project>``."""
        ...

    async def spans(
        self,
        query: TraceQuery,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        trace_ids: Sequence[str] | None = None,
    ) -> list[SpanRecord]:
        """
        Spans matching ``query``'s filters (its attributes, processor and
        status), started in ``[start, end)``, of ``trace_ids`` when given.
        """
        ...

    async def annotate(
        self, project: str, annotations: Sequence[TraceAnnotation]
    ) -> int:
        """
        Write ``annotations`` and return how many were stored — those whose
        target no longer exists are left out. Raises
        :class:`AnnotationsRejectedError` when the store refuses them (a
        retry cannot help), and other errors when it could not be reached.
        """
        ...


class AnnotationsRejectedError(RuntimeError):
    """The trace store refused annotations: writing them again cannot help."""


class AnnotationError(RuntimeError):
    """A scored online run whose annotations could not be written."""

    def __init__(self, run: EvaluationRun, cause: BaseException) -> None:
        super().__init__(
            f"Run {run.id} was scored, but writing its annotations failed: "
            f"{type(cause).__name__}: {cause}"
        )
        self.run = run


# --- Selection ---


def _production(spans: Iterable[SpanRecord]) -> list[SpanRecord]:
    return [s for s in spans if not s.from_evaluation]


def _by_start(spans: Iterable[SpanRecord]) -> tuple[SpanRecord, ...]:
    return tuple(sorted(spans, key=lambda s: (s.start_time, s.span_id)))


def _in(moment: datetime, start: datetime, end: datetime) -> bool:
    return start <= moment < end


async def _spans_of_traces(
    source: TraceSource, query: TraceQuery, trace_ids: Sequence[str]
) -> list[SpanRecord]:
    found: list[SpanRecord] = []
    for offset in range(0, len(trace_ids), TRACE_BATCH):
        batch = trace_ids[offset : offset + TRACE_BATCH]
        found.extend(await source.spans(query, trace_ids=batch))
    return _production(found)


async def collect_items(
    source: TraceSource, query: TraceQuery, window: TraceWindow
) -> list[TraceItem]:
    """The items ``query`` selects in ``window``, by start time."""
    start, end = window.start, window.end
    if query.scope == "span":
        spans = _production(await source.spans(query, start=start, end=end))
        items = [TraceItem(scope="span", id=s.span_id, spans=(s,)) for s in spans]
    elif query.scope == "trace":
        seeds = _production(await source.spans(query, start=start, end=end))
        trace_ids = sorted({s.trace_id for s in seeds})
        grouped: dict[str, list[SpanRecord]] = defaultdict(list)
        for span in await _spans_of_traces(source, query, trace_ids):
            grouped[span.trace_id].append(span)
        items = [
            TraceItem(scope="trace", id=trace_id, spans=_by_start(spans))
            for trace_id, spans in grouped.items()
        ]
        # A trace belongs to the window its first selected span started in.
        items = [i for i in items if _in(i.started_at, start, end)]
    else:
        items = await _session_items(source, query, window)
    return sorted(items, key=lambda i: (i.started_at, i.id))


async def _session_items(
    source: TraceSource, query: TraceQuery, window: TraceWindow
) -> list[TraceItem]:
    idle = timedelta(seconds=query.session_idle_s)
    seeds = _production(
        await source.spans(query, start=window.start - idle, end=window.end - idle)
    )
    semaphore = asyncio.Semaphore(SESSION_CONCURRENCY)

    async def session(session_id: str) -> TraceItem | None:
        async with semaphore:
            of_session = replace(
                query,
                attributes={**query.attributes, span_attrs.ATTR_SESSION_ID: session_id},
            )
            spans = _production(await source.spans(of_session))
        if not spans:
            return None
        item = TraceItem(scope="session", id=session_id, spans=_by_start(spans))
        # Scored once: in the window in which its last selected span has been
        # idle for ``session_idle_s``.
        last = item.spans[-1].start_time
        return item if _in(last + idle, window.start, window.end) else None

    ids = sorted({s.session_id for s in seeds if s.session_id})
    found = await asyncio.gather(*(session(i) for i in ids))
    return [item for item in found if item is not None]


def _unit_hash(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / 16**16


def _stratum(item: TraceItem, strata: Sequence[str]) -> tuple[Any, ...]:
    span = item.first
    return tuple(
        ("error" if span.errored else "ok")
        if key == "status"
        else span.attributes.get(key)
        for key in strata
    )


def sample_items(
    items: Sequence[TraceItem],
    *,
    rate: float = 1.0,
    max_items: int | None = None,
    strata: Sequence[str] = (),
) -> list[TraceItem]:
    """
    A deterministic sample: an item is kept when the hash of its trace (or
    session) id falls below ``rate``, so every evaluation sampling at the same
    rate sees the same traces. ``max_items`` keeps whole traces with the
    smallest hashes, taken in turn from each stratum, up to that many items.
    """
    groups: dict[str, list[TraceItem]] = {}
    for item in items:
        groups.setdefault(item.sampling_key, []).append(item)
    ranked = sorted(
        (
            (_unit_hash(key), key, group)
            for key, group in groups.items()
            if _unit_hash(key) < rate
        ),
        key=itemgetter(0, 1),
    )
    chosen = [group for _, _, group in ranked]
    if max_items is not None and sum(len(g) for g in chosen) > max_items:
        queues: dict[tuple[Any, ...], list[list[TraceItem]]] = {}
        for group in chosen:
            queues.setdefault(_stratum(group[0], strata), []).append(group)
        picked: list[list[TraceItem]] = []
        count = 0
        depth = 0
        while any(depth < len(q) for q in queues.values()):
            for queue in queues.values():
                if depth < len(queue) and count + len(queue[depth]) <= max_items:
                    picked.append(queue[depth])
                    count += len(queue[depth])
            depth += 1
        chosen = picked
    return sorted(
        (item for group in chosen for item in group),
        key=lambda i: (i.started_at, i.id),
    )


# --- Trials ---


def provenance(item: TraceItem) -> dict[str, Any]:
    """Where an online example came from, as flat metadata keys."""
    span = item.first
    metadata: dict[str, Any] = {
        "scope": item.scope,
        "trace_id": item.trace_id,
        "started_at": item.started_at.isoformat(),
    }
    if item.scope == "span":
        metadata["span_id"] = span.span_id
    if item.session_id is not None:
        metadata["session_id"] = item.session_id
    for key, attribute in (
        ("processor", span_attrs.ATTR_PROCESSOR_NAME),
        ("version", span_attrs.ATTR_PROCESSOR_VERSION),
        ("model", span_attrs.ATTR_AGENT_MODEL),
    ):
        value = span.attributes.get(attribute)
        if value is not None:
            metadata[key] = value
    if any(s.truncated for s in item.spans):
        metadata["truncated"] = True
    return metadata


async def _extract(extractor: Extractor, item: TraceItem) -> Extracted | None:
    extracted: Any = extractor(item)
    if inspect.isawaitable(extracted):
        extracted = await extracted
    if extracted is not None and not isinstance(extracted, Extracted):
        raise TypeError(
            f"An extractor returns Extracted or None, not {type(extracted).__name__}"
        )
    return extracted


class Extraction(BaseModel):
    """The examples and trials read from a window's items."""

    examples: list[Example[Any, Any]] = Field(default_factory=list[Example[Any, Any]])
    trials: list[Trial] = Field(default_factory=list[Trial])
    # Items the extractor returned nothing for.
    skipped: list[str] = Field(default_factory=list[str])
    # Item id → why extracting it failed (the items are left out).
    failures: dict[str, str] = Field(default_factory=dict[str, str])


async def extract_trials(
    items: Sequence[TraceItem], extractor: Extractor, *, concurrency: int = 8
) -> Extraction:
    """Run ``extractor`` over ``items`` (``concurrency`` at a time)."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: TraceItem) -> Extracted | Exception | None:
        async with semaphore:
            try:
                return await _extract(extractor, item)
            except Exception as exc:
                return exc

    results = await asyncio.gather(*(one(item) for item in items))
    extraction = Extraction()
    for item, result in zip(items, results, strict=True):
        if isinstance(result, Exception):
            extraction.failures[item.id] = f"{type(result).__name__}: {result}"
            continue
        if result is None:
            extraction.skipped.append(item.id)
            continue
        example = Example[Any, Any](
            id=item.id,
            input=result.input,
            metadata={**result.metadata, **provenance(item)},
        )
        model = example.metadata.get("model")
        processor = example.metadata.get("processor")
        extraction.examples.append(example)
        extraction.trials.append(
            Trial(
                example_id=example.id,
                example_hash=example.content_hash,
                output=None if result.error is not None else to_jsonable(result.output),
                error=result.error,
                started_at=item.started_at,
                duration_s=max(0.0, (item.ended_at - item.started_at).total_seconds()),
                models={str(processor): [str(model)]} if processor and model else {},
                trace_id=item.trace_id,
            )
        )
    if extraction.failures:
        first = next(iter(extraction.failures.items()))
        logger.warning(
            "Extracting %d of %d items failed (%s: %s)",
            len(extraction.failures),
            len(items),
            *first,
        )
    return extraction


def dataset_ref(
    source: TraceSource,
    query: TraceQuery,
    window: TraceWindow,
    items: Sequence[TraceItem],
    examples: Sequence[Example[Any, Any]],
) -> DatasetRef:
    """What an online run read: the window's items and the sampled examples."""
    selection = [
        f"window={window.start.isoformat()}/{window.end.isoformat()}",
        f"scope={query.scope}",
        *(f"{k}={v}" for k, v in sorted(query.span_attributes.items())),
    ]
    if query.status is not None:
        selection.append(f"status={query.status}")
    if query.sample_rate < 1.0:
        selection.append(f"sample_rate={query.sample_rate:g}")
    if query.max_items is not None:
        selection.append(f"max_items={query.max_items}")
    return DatasetRef(
        name=f"traces:{query.project}",
        source=source.location,
        fingerprint=short_hash(*sorted(i.id for i in items)),
        size=len(items),
        selection=selection,
        selected_fingerprint=short_hash(
            *sorted(f"{e.id}:{e.content_hash}" for e in examples)
        ),
        selected_size=len(examples),
    )


# --- Annotations ---


def annotation_identifier(evaluator: str, version: str | None) -> str:
    """The identifier of an evaluator version's annotations."""
    return f"grasp-evals:{evaluator}@{version or '1'}"


def score_result(score: Score) -> tuple[str | None, float | None] | None:
    """``(label, score)`` of a scored value; ``None`` when unscored."""
    value = score.value
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, bool):
        return ("pass" if value else "fail"), (1.0 if value else 0.0)
    if isinstance(value, str):
        return value, None
    return None, float(value)


def run_annotations(run: EvaluationRun) -> list[TraceAnnotation]:
    """The scores of a run of traces as annotations of what they judged."""
    scope = run.task.config.get("scope")
    if run.window is None or scope not in {"span", "trace", "session"}:
        return []
    evaluators = {e.name: e for e in run.evaluators}
    annotations: list[TraceAnnotation] = []
    for trial in run.trials:
        for score in trial.scores:
            result = score_result(score)
            if result is None:
                continue
            evaluator = score.evaluator or score.name
            info = evaluators.get(evaluator)
            version = info.version if info is not None else None
            metadata: dict[str, Any] = {
                "run_id": run.id,
                "evaluation": run.name,
                "evaluator": evaluator,
                "evaluator_version": version,
            }
            if score.reason:
                metadata["reason"] = score.reason
            annotations.append(
                TraceAnnotation(
                    scope=cast("TraceScope", scope),
                    target_id=trial.example_id,
                    name=score.name,
                    annotator=(info.annotator if info is not None else None) or "CODE",
                    label=result[0],
                    score=result[1],
                    explanation=score.explanation,
                    identifier=annotation_identifier(evaluator, version),
                    metadata=metadata,
                )
            )
    return annotations


def _record_annotations(
    run: EvaluationRun, store: RunStore | None, record: dict[str, Any]
) -> None:
    run.metadata = {**run.metadata, "annotations": record}
    if store is not None:
        store.save(run)


async def annotate_run(
    source: TraceSource, run: EvaluationRun, *, store: RunStore | None = None
) -> int:
    """
    Write the scores of a run of traces (online, or a rescore of one) onto
    what they judged — again harmlessly: annotations replace their earlier
    copies. Records the outcome in ``run.metadata["annotations"]`` (saved to
    ``store``) and returns the number written. A failure is recorded —
    ``pending`` when a retry may succeed, ``rejected`` when the store refused
    them — and raised.
    """
    if run.window is None:
        raise ValueError(f"Run {run.id} did not score traces: nothing to annotate")
    project = str(run.task.config.get("project") or "")
    annotations = run_annotations(run)
    try:
        written = await source.annotate(project, annotations) if annotations else 0
    except Exception as exc:
        outcome = "rejected" if isinstance(exc, AnnotationsRejectedError) else "pending"
        _record_annotations(
            run,
            store,
            {
                outcome: len(annotations),
                "location": source.location,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    record: dict[str, Any] = {"written": written, "location": source.location}
    if written < len(annotations):
        # Their spans, traces or sessions no longer exist.
        record["missing_targets"] = len(annotations) - written
    _record_annotations(run, store, record)
    return written


def annotations_pending(run: EvaluationRun) -> bool:
    """Whether writing the run's annotations failed and may succeed later."""
    record = run.metadata.get("annotations")
    return isinstance(record, Mapping) and "pending" in record


def annotations_location(run: EvaluationRun) -> str | None:
    record = run.metadata.get("annotations")
    if not isinstance(record, Mapping):
        return None
    location = cast("Mapping[str, Any]", record).get("location")
    return None if location is None else str(location)


def _judge_extraction(
    run: EvaluationRun, extraction: Extraction, extracted_items: int
) -> None:
    # Items that could not be read are not failures of the system: they
    # make the run invalid, and when nothing could be read (an outage of
    # what the extractor reads from) the run failed, so the window is read
    # again next time.
    failed = len(extraction.failures)
    if not failed:
        return
    reason = (
        f"extracting {failed} of {extracted_items} items failed "
        f"({next(iter(extraction.failures.values()))})"
    )
    if failed == extracted_items:
        run.status = RunStatus.FAILED
        run.invalid_reason = reason
        return
    limit = run.config.max_error_rate or 0.0
    if failed / extracted_items > limit and run.invalid_reason is None:
        run.invalid_reason = reason


async def evaluate_traces(
    source: TraceSource,
    query: TraceQuery,
    evaluators: Sequence[Evaluator[Any, Any, Any]],
    metrics: MetricsSpec = None,
    *,
    window: TraceWindow,
    name: str,
    output_type: Any = Any,
    description: str | None = None,
    concurrency: int = 4,
    evaluator_timeout_s: float | None = None,
    max_cost_usd: float | None = None,
    max_error_rate: float | None = None,
    group_by: Sequence[str] = (),
    cluster_by: str | None = None,
    annotate: bool = True,
    store: RunStore | None = None,
    persist: bool = True,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    evaluation: str | None = None,
    progress: ProgressCallback | None = None,
) -> EvaluationRun:
    """
    Score the items ``query`` selects in ``window`` as an ``online`` run and,
    with ``annotate``, write the scores back onto the traces. A window with
    nothing in it still makes an (empty) run, so the next window starts where
    this one ended. Items the extractor fails on make the run invalid (more
    than ``max_error_rate`` of them, default any), all of them failed — the
    window is read again by the next run. Raises :class:`AnnotationError`,
    holding the scored run, when the annotations could not be written.
    """
    items = await collect_items(source, query, window)
    sampled = sample_items(
        items, rate=query.sample_rate, max_items=query.max_items, strata=query.strata
    )
    extraction = await extract_trials(sampled, query.extractor)
    run_store = resolve_store(store, persist)
    run = await evaluate_trials(
        extraction.examples,
        extraction.trials,
        evaluators,
        metrics,
        name=name,
        task=query.describe(),
        dataset=dataset_ref(source, query, window, items, extraction.examples),
        window=window,
        output_type=output_type,
        description=description,
        concurrency=concurrency,
        evaluator_timeout_s=evaluator_timeout_s,
        max_cost_usd=max_cost_usd,
        max_error_rate=max_error_rate,
        group_by=group_by,
        cluster_by=cluster_by,
        store=run_store,
        persist=persist,
        tags=tags,
        metadata={
            **(metadata or {}),
            "traces": {
                "location": source.location,
                "items": len(items),
                "sampled": len(sampled),
                "skipped": extraction.skipped,
                "extraction_failures": extraction.failures,
            },
        },
        evaluation=evaluation,
        progress=progress,
    )
    if query.status == "error" and run.invalid_reason == "every trial failed":
        # Failures are what this selection reads.
        run.invalid_reason = None
    _judge_extraction(run, extraction, len(sampled))
    if run_store is not None:
        run_store.save(run)
        run_store.write_report(run.id, render_run_markdown(run))
    if annotate and run.status != RunStatus.FAILED:
        try:
            await annotate_run(source, run, store=run_store)
        except Exception as exc:
            raise AnnotationError(run, exc) from exc
    return run


# --- Windows ---

# Runs that read their whole window: a budget-capped (partial) run scored what
# it could afford, and the rest of its window is not read again.
_COVERING = frozenset({RunStatus.COMPLETED, RunStatus.PARTIAL})


def last_window_end(
    runs: Iterable[EvaluationRun], name: str, query: TraceQuery
) -> datetime | None:
    """
    Where the next window of ``name`` starts: the end of the latest window an
    online run of the same selection covered (completed, or stopped by its
    budget).
    """
    fingerprint = query.describe().fingerprint
    ends = [
        run.window.end
        for run in runs
        if run.kind == "online"
        and run.name == name
        and run.status in _COVERING
        and run.window is not None
        and run.task.fingerprint == fingerprint
    ]
    return max(ends, default=None)


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


def resolve_window(
    query: TraceQuery,
    *,
    name: str,
    runs: Iterable[EvaluationRun] = (),
    start: datetime | None = None,
    end: datetime | None = None,
    now: datetime | None = None,
) -> TraceWindow | None:
    """
    The window to evaluate: from ``start`` (default: where the latest online
    run of ``name`` that covered its window ended) to ``end`` (default: the
    completion buffer before ``now``). ``None`` when it is empty. Times
    without a timezone are UTC.
    """
    latest = _utc(now or utc_now()) - timedelta(seconds=query.completion_buffer_s)
    end = _utc(end) if end is not None else latest
    if end > latest:
        raise ValueError(
            f"The window must end by {latest.isoformat()}: spans that started "
            f"later may still be running or exporting (completion buffer "
            f"{query.completion_buffer_s:g}s)"
        )
    if start is None:
        start = last_window_end(runs, name, query)
        if start is None:
            raise ValueError(
                f"No earlier online run of {name!r} to continue from: give "
                "the window start"
            )
    start = _utc(start)
    if start >= end:
        return None
    return TraceWindow(start=start, end=end)
