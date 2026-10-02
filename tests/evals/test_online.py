import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.evals import (
    Dataset,
    EvalContext,
    Evaluation,
    Example,
    Extracted,
    LocalRunStore,
    PassRate,
    RunStatus,
    SpanRecord,
    TraceAnnotation,
    TraceItem,
    TraceQuery,
    TraceWindow,
    Usage,
    annotate_run,
    default_extractor,
    evaluate_traces,
    evaluate_trials,
    evaluator,
    rescore,
)
from grasp_agents.evals.online import (
    AnnotationError,
    AnnotationsRejectedError,
    annotation_identifier,
    annotations_pending,
    collect_items,
    dataset_ref,
    extract_trials,
    flatten_attributes,
    last_window_end,
    resolve_window,
    run_annotations,
    sample_items,
)
from grasp_agents.telemetry import attributes as span_attrs

T0 = datetime(2026, 9, 1, tzinfo=UTC)


def _span(
    span_id: str,
    *,
    trace_id: str = "t1",
    minute: float = 0.0,
    processor: str = "writer",
    input: Any = "question",  # noqa: A002
    output: Any = "answer",
    session: str | None = None,
    status: str = "UNSET",
    error_type: str | None = None,
    eval_run: str | None = None,
    **attributes: Any,
) -> SpanRecord:
    attrs: dict[str, Any] = {
        span_attrs.ATTR_SPAN_KIND: "agent",
        span_attrs.ATTR_PROCESSOR_NAME: processor,
        span_attrs.ATTR_AGENT_MODEL: "gpt-x",
        span_attrs.ATTR_PROCESSOR_VERSION: "v3",
        span_attrs.ATTR_INPUT_VALUE: json.dumps(input),
        span_attrs.ATTR_INPUT_MIME_TYPE: span_attrs.JSON_MIME_TYPE,
        **attributes,
    }
    if output is not None:
        attrs[span_attrs.ATTR_OUTPUT_VALUE] = json.dumps(output)
        attrs[span_attrs.ATTR_OUTPUT_MIME_TYPE] = span_attrs.JSON_MIME_TYPE
    if session is not None:
        attrs[span_attrs.ATTR_SESSION_ID] = session
    if error_type is not None:
        attrs[span_attrs.ATTR_ERROR_TYPE] = error_type
    if eval_run is not None:
        attrs[span_attrs.ATTR_EVAL_RUN_ID] = eval_run
    start = T0 + timedelta(minutes=minute)
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        name=processor,
        start_time=start,
        end_time=start + timedelta(seconds=30),
        status=status,
        status_message="ValueError: empty chunk" if status == "ERROR" else None,
        attributes=attrs,
    )


class MemorySource:
    location = "memory:test"

    def __init__(self, spans: Sequence[SpanRecord]) -> None:
        self.stored = list(spans)
        self.annotations: list[TraceAnnotation] = []
        self.fail_annotations = False
        self.reject_annotations = False

    async def spans(
        self,
        query: TraceQuery,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        trace_ids: Sequence[str] | None = None,
    ) -> list[SpanRecord]:
        found: list[SpanRecord] = []
        for span in self.stored:
            if start is not None and span.start_time < start:
                continue
            if end is not None and span.start_time >= end:
                continue
            if trace_ids is not None and span.trace_id not in trace_ids:
                continue
            if any(
                span.attributes.get(k) != v for k, v in query.span_attributes.items()
            ):
                continue
            if query.status == "error" and not span.errored:
                continue
            if query.status == "ok" and span.errored:
                continue
            found.append(span)
        return found

    async def annotate(
        self, project: str, annotations: Sequence[TraceAnnotation]
    ) -> int:
        if self.reject_annotations:
            raise AnnotationsRejectedError("name not allowed")
        if self.fail_annotations:
            raise RuntimeError("store unavailable")
        self.annotations.extend(annotations)
        return len(annotations)


def _window(start: float = 0.0, end: float = 60.0) -> TraceWindow:
    return TraceWindow(
        start=T0 + timedelta(minutes=start), end=T0 + timedelta(minutes=end)
    )


@evaluator
def answered(ctx: EvalContext[Any, Any, Any]) -> bool:
    return ctx.output == "answer"


@evaluator(name="tone")
def tone(ctx: EvalContext[Any, Any, Any]) -> str | None:
    return None if ctx.output == "skip" else "polite"


class TestSpanRecords:
    def test_payloads_parse_as_json_and_flags_read(self) -> None:
        span = _span(
            "s1",
            input={"q": 1},
            output=[1, 2],
            **{span_attrs.ATTR_OUTPUT_TRUNCATED: True},
        )
        assert span.input == {"q": 1}
        assert span.output == [1, 2]
        assert span.truncated
        assert not span.from_evaluation
        assert _span("s2", eval_run="r1").from_evaluation

    def test_payload_without_json_mime_type_stays_text(self) -> None:
        span = SpanRecord(
            trace_id="t",
            span_id="s",
            name="x",
            start_time=T0,
            attributes={span_attrs.ATTR_INPUT_VALUE: '{"q": 1}'},
        )
        assert span.input == '{"q": 1}'
        assert span.output is None

    def test_nested_attributes_flatten_to_dotted_keys(self) -> None:
        nested = {"grasp": {"processor": {"name": "writer"}}, "session.id": "s1"}
        assert flatten_attributes(nested) == {
            "grasp.processor.name": "writer",
            "session.id": "s1",
        }


class TestCollect:
    @pytest.mark.asyncio
    async def test_span_scope_reads_the_window_and_skips_evaluation_spans(
        self,
    ) -> None:
        source = MemorySource(
            [
                _span("early", minute=-1),
                _span("a", minute=1),
                _span("b", trace_id="t2", minute=2),
                _span("eval", trace_id="t3", minute=3, eval_run="r1"),
                _span("other", trace_id="t4", minute=4, processor="reviewer"),
                _span("late", trace_id="t5", minute=60),
            ]
        )
        query = TraceQuery(project="p", processor="writer")
        items = await collect_items(source, query, _window())
        assert [i.id for i in items] == ["a", "b"]
        assert all(i.scope == "span" for i in items)

    @pytest.mark.asyncio
    async def test_trace_scope_groups_by_trace_where_it_started(self) -> None:
        source = MemorySource(
            [
                _span("a1", trace_id="ta", minute=-5),
                _span("a2", trace_id="ta", minute=5),
                _span("b1", trace_id="tb", minute=10),
                _span("b2", trace_id="tb", minute=70),
            ]
        )
        query = TraceQuery(project="p", processor="writer", scope="trace")
        items = await collect_items(source, query, _window())
        # ``ta`` started in the previous window; ``tb`` here, with its later span.
        assert [(i.id, [s.span_id for s in i.spans]) for i in items] == [
            ("tb", ["b1", "b2"])
        ]

    @pytest.mark.asyncio
    async def test_a_session_is_scored_in_the_window_it_went_idle(self) -> None:
        source = MemorySource(
            [
                # Idle 30 min after minute 10: due in [0, 60).
                _span("q1", trace_id="t1", minute=0, session="done"),
                _span("q2", trace_id="t2", minute=10, session="done"),
                # Last turn at 50: idle at 80, so the next window scores it.
                _span("r1", trace_id="t3", minute=20, session="active"),
                _span("r2", trace_id="t4", minute=50, session="active"),
                # Went idle before this window: already scored.
                _span("s1", trace_id="t5", minute=-40, session="old"),
                _span("lone", trace_id="t6", minute=5),
            ]
        )
        query = TraceQuery(
            project="p", processor="writer", scope="session", session_idle_s=1800
        )
        items = await collect_items(source, query, _window(0, 60))
        assert [(i.id, len(i.spans)) for i in items] == [("done", 2)]
        later = await collect_items(source, query, _window(60, 120))
        assert [i.id for i in later] == ["active"]


class TestSampling:
    def _items(self, n: int) -> list[TraceItem]:
        return [
            TraceItem(
                scope="span",
                id=f"s{i}",
                spans=(_span(f"s{i}", trace_id=f"t{i}", minute=i % 50),),
            )
            for i in range(n)
        ]

    def test_hash_sampling_is_deterministic_and_nested(self) -> None:
        items = self._items(400)
        half = {i.id for i in sample_items(items, rate=0.5)}
        assert half == {i.id for i in sample_items(items, rate=0.5)}
        assert 150 < len(half) < 250
        tenth = {i.id for i in sample_items(items, rate=0.1)}
        assert tenth < half

    def test_spans_of_one_trace_are_sampled_together(self) -> None:
        items = [
            TraceItem(
                scope="span", id=f"s{i}", spans=(_span(f"s{i}", trace_id="same"),)
            )
            for i in range(20)
        ]
        kept = sample_items(items, rate=0.5)
        assert len(kept) in {0, 20}

    def test_max_items_spreads_over_strata(self) -> None:
        items = [
            TraceItem(
                scope="span",
                id=f"s{i}",
                spans=(
                    _span(
                        f"s{i}",
                        trace_id=f"t{i}",
                        status="ERROR" if i < 5 else "UNSET",
                    ),
                ),
            )
            for i in range(100)
        ]
        picked = sample_items(items, max_items=10, strata=["status"])
        assert sum(1 for i in picked if i.first.errored) == 5
        assert len(picked) == 10


class TestExtraction:
    def test_default_extractor_reads_payloads_and_failures(self) -> None:
        ok = TraceItem(scope="span", id="s", spans=(_span("s"),))
        assert default_extractor(ok) == Extracted(input="question", output="answer")
        failed = TraceItem(
            scope="span",
            id="f",
            spans=(_span("f", status="ERROR", output=None, error_type="ValueError"),),
        )
        extracted = default_extractor(failed)
        assert extracted.input == "question"
        assert extracted.error is not None
        assert extracted.error.type == "ValueError"
        assert extracted.error.message == "ValueError: empty chunk"
        session = TraceItem(
            scope="session",
            id="c1",
            spans=(_span("a", output="hi"), _span("b", input="more", output="bye")),
        )
        assert default_extractor(session) == Extracted(
            input=["question", "more"], output=["hi", "bye"]
        )

    @pytest.mark.asyncio
    async def test_trials_carry_provenance_and_extraction_problems(self) -> None:
        items = [
            TraceItem(scope="span", id=f"s{i}", spans=(_span(f"s{i}", session="c"),))
            for i in range(3)
        ]

        async def extractor(item: TraceItem) -> Extracted | None:
            if item.id == "s1":
                return None
            if item.id == "s2":
                raise LookupError("no such lesson")
            return Extracted(input="q", output="a", metadata={"lesson": 7})

        extraction = await extract_trials(items, extractor)
        assert extraction.skipped == ["s1"]
        assert extraction.failures == {"s2": "LookupError: no such lesson"}
        (example,) = extraction.examples
        assert example.id == "s0"
        assert example.metadata == {
            "scope": "span",
            "trace_id": "t1",
            "span_id": "s0",
            "session_id": "c",
            "started_at": T0.isoformat(),
            "processor": "writer",
            "version": "v3",
            "model": "gpt-x",
            "lesson": 7,
        }
        (trial,) = extraction.trials
        assert trial.output == "a"
        assert trial.trace_id == "t1"
        assert trial.models == {"writer": ["gpt-x"]}
        assert trial.duration_s == 30.0


class TestEvaluateTraces:
    @pytest.mark.asyncio
    async def test_scores_annotate_what_they_judged(self, tmp_path: Path) -> None:
        source = MemorySource(
            [
                _span("a", trace_id="ta", minute=1),
                _span("b", trace_id="tb", minute=2, output="wrong"),
                _span("c", trace_id="tc", minute=3, output="skip"),
                _span("f", trace_id="tf", minute=4, status="ERROR", output=None),
            ]
        )
        store = LocalRunStore(tmp_path)
        run = await evaluate_traces(
            source,
            TraceQuery(project="p", processor="writer"),
            [answered, tone],
            [PassRate("answered")],
            window=_window(),
            name="writer-online",
            store=store,
        )
        assert run.kind == "online"
        assert run.completed
        assert run.window == _window()
        assert run.dataset.name == "traces:p"
        assert run.dataset.source == "memory:test"
        assert run.counts.task_errors == 1
        # The production failure counts as a failure.
        assert run.metric("pass_rate(answered)").value == pytest.approx(1 / 4)  # type: ignore[union-attr]
        by_target = {(a.target_id, a.name): a for a in source.annotations}
        assert by_target["a", "answered"].label == "pass"
        assert by_target["a", "answered"].score == 1.0
        assert by_target["b", "answered"].label == "fail"
        assert by_target["a", "tone"].label == "polite"
        assert ("c", "tone") not in by_target
        assert ("f", "answered") not in by_target
        assert by_target["a", "answered"].identifier == "grasp-evals:answered@1"
        assert by_target["a", "answered"].metadata["run_id"] == run.id
        assert {a.scope for a in source.annotations} == {"span"}
        assert run.metadata["annotations"] == {"written": 5, "location": "memory:test"}
        stored = store.load(run.id)
        assert stored.window == run.window
        assert stored.metadata["annotations"]["written"] == 5

    @pytest.mark.asyncio
    async def test_a_failed_annotation_write_is_recorded_as_pending(
        self, tmp_path: Path
    ) -> None:
        source = MemorySource([_span("a")])
        source.fail_annotations = True
        store = LocalRunStore(tmp_path)
        with pytest.raises(RuntimeError, match="store unavailable"):
            await evaluate_traces(
                source,
                TraceQuery(project="p"),
                [answered],
                window=_window(),
                name="w",
                store=store,
            )
        (header,) = store.list_runs()
        assert header.completed
        assert header.metadata["annotations"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_an_empty_window_still_makes_a_run(self) -> None:
        run = await evaluate_traces(
            MemorySource([]),
            TraceQuery(project="p"),
            [answered],
            window=_window(),
            name="w",
            persist=False,
        )
        assert run.completed
        assert not run.trials
        assert run.invalid_reason is None

    def test_annotations_follow_the_item_scope(self) -> None:
        assert annotation_identifier("judge", "2") == "grasp-evals:judge@2"
        assert annotation_identifier("judge", None) == "grasp-evals:judge@1"


class TestWindows:
    def test_the_next_window_starts_where_the_last_completed_run_ended(
        self,
    ) -> None:
        query = TraceQuery(project="p", processor="writer", completion_buffer_s=600)
        now = T0 + timedelta(hours=3)
        with pytest.raises(ValueError, match="give the window start"):
            resolve_window(query, name="w", now=now)
        first = resolve_window(query, name="w", start=T0, now=now)
        assert first == TraceWindow(start=T0, end=now - timedelta(minutes=10))
        with pytest.raises(ValueError, match="must end by"):
            resolve_window(query, name="w", start=T0, end=now, now=now)

    @pytest.mark.asyncio
    async def test_runs_of_another_selection_do_not_continue_each_other(
        self,
    ) -> None:
        query = TraceQuery(project="p", processor="writer")
        run = await evaluate_traces(
            MemorySource([]),
            query,
            [answered],
            window=_window(0, 60),
            name="w",
            persist=False,
            annotate=False,
        )
        assert last_window_end([run], "w", query) == _window(0, 60).end
        # Sampling is not the selection: a sampled query continues the window.
        sampled = TraceQuery(project="p", processor="writer", sample_rate=0.5)
        assert last_window_end([run], "w", sampled) == _window(0, 60).end
        other = TraceQuery(project="p", processor="reviewer")
        assert last_window_end([run], "w", other) is None
        assert last_window_end([run], "another", query) is None


class TestEvaluationOnline:
    def _evaluation(self, **query: Any) -> Evaluation:
        return Evaluation(
            name="writer-online",
            evaluators=[answered],
            traces=TraceQuery(
                project="p", processor="writer", completion_buffer_s=0, **query
            ),
        )

    @pytest.mark.asyncio
    async def test_consecutive_runs_cover_consecutive_windows(
        self, tmp_path: Path
    ) -> None:
        source = MemorySource(
            [_span("a", trace_id="ta", minute=10), _span("b", trace_id="tb", minute=70)]
        )
        store = LocalRunStore(tmp_path)
        evaluation = self._evaluation()
        first = await evaluation.run_online(
            source=source, start=T0, now=T0 + timedelta(hours=1), store=store
        )
        assert first is not None
        assert [t.example_id for t in first.trials] == ["a"]
        nothing = await evaluation.run_online(
            source=source, now=T0 + timedelta(hours=1), store=store
        )
        assert nothing is None
        second = await evaluation.run_online(
            source=source, now=T0 + timedelta(hours=2), store=store
        )
        assert second is not None
        assert second.window == _window(60, 120)
        assert [t.example_id for t in second.trials] == ["b"]

    @pytest.mark.asyncio
    async def test_pending_annotations_are_written_by_the_next_run(
        self, tmp_path: Path
    ) -> None:
        source = MemorySource(
            [_span("a", trace_id="ta", minute=10), _span("b", trace_id="tb", minute=70)]
        )
        store = LocalRunStore(tmp_path)
        evaluation = self._evaluation()
        source.fail_annotations = True
        with pytest.raises(RuntimeError):
            await evaluation.run_online(
                source=source, start=T0, now=T0 + timedelta(hours=1), store=store
            )
        source.fail_annotations = False
        await evaluation.run_online(
            source=source, now=T0 + timedelta(hours=2), store=store
        )
        assert sorted(a.target_id for a in source.annotations) == ["a", "b"]
        assert not any(
            "pending" in r.metadata.get("annotations", {}) for r in store.list_runs()
        )

    @pytest.mark.asyncio
    async def test_an_offline_only_evaluation_cannot_run_online(self) -> None:
        evaluation = Evaluation(name="x", evaluators=[answered])
        with pytest.raises(LookupError, match="defines no traces"):
            await evaluation.run_online(source=MemorySource([]), persist=False)

    @pytest.mark.asyncio
    async def test_traces_become_deduplicated_dataset_examples(self) -> None:
        source = MemorySource(
            [
                _span("a", trace_id="ta", minute=1, input="q1"),
                _span("b", trace_id="tb", minute=2, input="q1"),
                _span("c", trace_id="tc", minute=3, input="q2"),
                _span(
                    "f",
                    trace_id="tf",
                    minute=4,
                    input="q3",
                    status="ERROR",
                    output=None,
                ),
            ]
        )
        evaluation = self._evaluation()
        export = await evaluation.dataset_from_traces(
            source=source, start=T0, end=T0 + timedelta(hours=1)
        )
        assert [e.input for e in export.dataset] == ["q1", "q2", "q3"]
        assert export.duplicates == 1
        assert export.dataset[2].metadata["error"] == "ValueError: empty chunk"
        assert export.dataset[0].metadata["trace_id"] == "ta"
        failed = await evaluation.dataset_from_traces(
            source=source,
            start=T0,
            end=T0 + timedelta(hours=1),
            status="error",
            exclude=[export.dataset[0]],
        )
        assert [e.input for e in failed.dataset] == ["q3"]
        reloaded = Dataset.from_records(export.dataset.to_records())
        assert reloaded.ids == export.dataset.ids


def test_run_annotations_skip_trials_that_are_not_trace_items() -> None:
    from grasp_agents.evals import EvaluationRun

    run = EvaluationRun.model_validate(
        {
            "id": "r",
            "name": "n",
            "created_at": T0,
            "dataset": {
                "name": "d",
                "fingerprint": "f",
                "size": 0,
                "selected_fingerprint": "f",
                "selected_size": 0,
            },
            "task": {"name": "t", "kind": "k"},
            "provenance": {"python": "3.12"},
            "config_hash": "h",
        }
    )
    assert run_annotations(run) == []


# --- Review regressions ---


@evaluator(name="costly")
def costly(ctx: EvalContext[Any, Any, Any]) -> bool:
    ctx.record_usage(Usage(input_tokens=10, cost_usd=1.0))
    return True


def _spans(n: int, *, minute: float = 10.0) -> list[SpanRecord]:
    return [_span(f"s{i}", trace_id=f"t{i}", minute=minute + i * 0.1) for i in range(n)]


class TestWindowsKeepMoving:
    @pytest.mark.asyncio
    async def test_a_budget_capped_run_still_covers_its_window(
        self, tmp_path: Path
    ) -> None:
        store = LocalRunStore(tmp_path)
        evaluation = Evaluation(
            name="w",
            evaluators=[costly],
            max_cost_usd=1.5,
            traces=TraceQuery(project="p", completion_buffer_s=0),
        )
        first = await evaluation.run_online(
            source=MemorySource(_spans(10)),
            start=T0,
            now=T0 + timedelta(hours=1),
            store=store,
            concurrency=1,
        )
        assert first is not None
        assert first.status == RunStatus.PARTIAL
        later = await evaluation.run_online(
            source=MemorySource(_spans(10)), now=T0 + timedelta(hours=2), store=store
        )
        assert later is not None
        assert later.window == _window(60, 120)

    @pytest.mark.asyncio
    async def test_an_extraction_outage_fails_the_run_and_keeps_the_window(
        self, tmp_path: Path
    ) -> None:
        def outage(item: TraceItem) -> Extracted:
            raise ConnectionError("database unavailable")

        store = LocalRunStore(tmp_path)
        query = TraceQuery(project="p", extractor=outage, completion_buffer_s=0)
        run = await evaluate_traces(
            MemorySource(_spans(3)),
            query,
            [answered],
            window=_window(),
            name="w",
            store=store,
        )
        assert run.status == RunStatus.FAILED
        assert "database unavailable" in (run.invalid_reason or "")
        assert last_window_end(store.list_runs(), "w", query) is None

    @pytest.mark.asyncio
    async def test_some_extraction_failures_make_the_run_invalid(self) -> None:
        def flaky(item: TraceItem) -> Extracted:
            if item.id == "s1":
                raise LookupError("lesson deleted")
            return Extracted(input="q", output="answer")

        run = await evaluate_traces(
            MemorySource(_spans(3)),
            TraceQuery(project="p", extractor=flaky),
            [answered],
            window=_window(),
            name="w",
            persist=False,
        )
        assert run.completed
        assert run.invalid_reason is not None
        assert run.invalid_reason.startswith("extracting 1 of 3 items failed")
        tolerant = await evaluate_traces(
            MemorySource(_spans(3)),
            TraceQuery(project="p", extractor=flaky),
            [answered],
            window=_window(),
            name="w",
            persist=False,
            max_error_rate=0.5,
        )
        assert tolerant.invalid_reason is None

    @pytest.mark.asyncio
    async def test_an_all_error_selection_is_not_invalid(self) -> None:
        source = MemorySource([_span("f", status="ERROR", output=None)])
        run = await evaluate_traces(
            source,
            TraceQuery(project="p", status="error"),
            [answered],
            window=_window(),
            name="w",
            persist=False,
        )
        assert run.counts.task_errors == 1
        assert run.invalid_reason is None

    def test_naive_times_are_utc(self) -> None:
        naive = datetime(2026, 9, 1, 1, 0)
        window = resolve_window(
            TraceQuery(project="p", completion_buffer_s=0),
            name="w",
            start=naive - timedelta(hours=1),
            end=naive,
            now=naive,
        )
        assert window == TraceWindow(start=T0, end=T0 + timedelta(hours=1))
        assert TraceWindow(start=naive, end=naive).start.tzinfo is UTC

    def test_the_session_idle_time_is_part_of_the_selection(self) -> None:
        short = TraceQuery(project="p", scope="session", session_idle_s=60)
        long = TraceQuery(project="p", scope="session", session_idle_s=600)
        assert short.describe().fingerprint != long.describe().fingerprint
        spans = TraceQuery(project="p", session_idle_s=60)
        assert (
            spans.describe().fingerprint
            == TraceQuery(project="p").describe().fingerprint
        )


class TestAnnotationFailures:
    @pytest.mark.asyncio
    async def test_a_failing_retry_does_not_block_the_next_window(
        self, tmp_path: Path
    ) -> None:
        source = MemorySource(
            [_span("a", trace_id="ta", minute=10), _span("b", trace_id="tb", minute=70)]
        )
        store = LocalRunStore(tmp_path)
        evaluation = TestEvaluationOnline()._evaluation()
        source.fail_annotations = True
        with pytest.raises(AnnotationError) as raised:
            await evaluation.run_online(
                source=source, start=T0, now=T0 + timedelta(hours=1), store=store
            )
        assert raised.value.run.completed
        # Still failing: the retry is logged, the new window is scored.
        with pytest.raises(AnnotationError):
            await evaluation.run_online(
                source=source, now=T0 + timedelta(hours=2), store=store
            )
        runs = store.list_runs()
        assert [r.window.end for r in runs if r.window] == [
            _window(60, 120).end,
            _window(0, 60).end,
        ]

    @pytest.mark.asyncio
    async def test_rejected_annotations_are_not_retried(self, tmp_path: Path) -> None:
        source = MemorySource([_span("a", minute=10)])
        source.reject_annotations = True
        store = LocalRunStore(tmp_path)
        with pytest.raises(AnnotationError):
            await evaluate_traces(
                source,
                TraceQuery(project="p"),
                [answered],
                window=_window(),
                name="w",
                store=store,
            )
        (run,) = store.list_runs()
        assert run.metadata["annotations"]["rejected"] == 1
        assert not annotations_pending(run)

    @pytest.mark.asyncio
    async def test_pending_runs_of_another_store_location_are_left_alone(
        self, tmp_path: Path
    ) -> None:
        store = LocalRunStore(tmp_path)
        evaluation = TestEvaluationOnline()._evaluation()
        elsewhere = MemorySource([_span("a", trace_id="ta", minute=10)])
        elsewhere.location = "memory:elsewhere"
        elsewhere.fail_annotations = True
        with pytest.raises(AnnotationError):
            await evaluation.run_online(
                source=elsewhere, start=T0, now=T0 + timedelta(hours=1), store=store
            )
        here = MemorySource([_span("b", trace_id="tb", minute=70)])
        await evaluation.run_online(
            source=here, now=T0 + timedelta(hours=2), store=store
        )
        assert [a.target_id for a in here.annotations] == ["b"]


class TestProvenanceAndRouting:
    @pytest.mark.asyncio
    async def test_extractor_metadata_cannot_replace_provenance(self) -> None:
        def business(item: TraceItem) -> Extracted:
            return Extracted(
                input="q",
                output="answer",
                metadata={"scope": "lesson", "trace_id": "x", "lesson": 7},
            )

        source = MemorySource([_span("a")])
        run = await evaluate_traces(
            source,
            TraceQuery(project="p", extractor=business),
            [answered],
            window=_window(),
            name="w",
            persist=False,
        )
        (example,) = run.examples
        assert example.metadata["scope"] == "span"
        assert example.metadata["trace_id"] == "t1"
        assert example.metadata["lesson"] == 7
        assert [(a.scope, a.target_id) for a in source.annotations] == [("span", "a")]

    @pytest.mark.asyncio
    async def test_a_rescored_online_run_can_be_annotated(self, tmp_path: Path) -> None:
        source = MemorySource([_span("a")])
        store = LocalRunStore(tmp_path)
        run = await evaluate_traces(
            source,
            TraceQuery(project="p"),
            [answered],
            window=_window(),
            name="w",
            store=store,
        )
        child = await rescore(run, [tone], store=store)
        assert child.window == run.window
        assert await annotate_run(source, child, store=store) == 2

    @pytest.mark.asyncio
    async def test_a_run_of_traces_is_never_pushed_as_an_experiment(self) -> None:
        from grasp_agents.evals.phoenix import push_run  # noqa: PLC0415

        run = await evaluate_traces(
            MemorySource([_span("a")]),
            TraceQuery(project="p"),
            [answered],
            window=_window(),
            name="w",
            persist=False,
            annotate=False,
        )
        with pytest.raises(ValueError, match="annotations"):
            await push_run(object(), run)  # type: ignore[arg-type]


class TestExtractionEdges:
    def test_spans_without_payloads_are_skipped(self) -> None:
        bare = _span("a", output=None)
        dark = bare.model_copy(
            update={
                "attributes": {
                    k: v
                    for k, v in bare.attributes.items()
                    if not k.startswith("input.")
                }
            }
        )
        assert default_extractor(TraceItem(scope="span", id="a", spans=(dark,))) is None

    def test_a_cancelled_run_is_a_task_error(self) -> None:
        span = _span("a", output=None, **{span_attrs.ATTR_CANCELLED: True})
        extracted = default_extractor(TraceItem(scope="span", id="a", spans=(span,)))
        assert extracted is not None
        assert extracted.error is not None
        assert extracted.error.type == "CancelledError"

    @pytest.mark.asyncio
    async def test_an_extractor_returning_the_wrong_type_is_a_failure(self) -> None:
        extraction = await extract_trials(
            [TraceItem(scope="span", id="a", spans=(_span("a"),))],
            lambda item: {"input": "q"},  # type: ignore[arg-type,return-value]
        )
        assert extraction.failures["a"].startswith("TypeError")

    def test_max_items_keeps_whole_traces(self) -> None:
        items = [
            TraceItem(
                scope="span",
                id=f"s{i}-{j}",
                spans=(_span(f"s{i}-{j}", trace_id=f"t{i}"),),
            )
            for i in range(10)
            for j in range(3)
        ]
        picked = sample_items(items, max_items=7)
        per_trace: dict[str, int] = {}
        for item in picked:
            per_trace[item.trace_id] = per_trace.get(item.trace_id, 0) + 1
        assert set(per_trace.values()) == {3}
        assert len(picked) == 6

    def test_status_strata_are_ok_and_error(self) -> None:
        items = [
            TraceItem(
                scope="span",
                id=f"s{i}",
                spans=(
                    _span(
                        f"s{i}",
                        trace_id=f"t{i}",
                        status=("ERROR", "OK", "UNSET")[i % 3],
                    ),
                ),
            )
            for i in range(60)
        ]
        picked = sample_items(items, max_items=10, strata=["status"])
        assert sum(1 for i in picked if i.first.errored) == 5

    @pytest.mark.asyncio
    async def test_duplicate_trials_are_refused(self) -> None:
        extraction = await extract_trials(
            [TraceItem(scope="span", id="a", spans=(_span("a"),))], default_extractor
        )
        with pytest.raises(ValueError, match="same example id"):
            await evaluate_trials(
                extraction.examples,
                [*extraction.trials, *extraction.trials],
                [answered],
                name="w",
                task=TraceQuery(project="p").describe(),
                dataset=dataset_ref(
                    MemorySource([]),
                    TraceQuery(project="p"),
                    _window(),
                    [],
                    extraction.examples,
                ),
                persist=False,
            )


@pytest.mark.asyncio
async def test_curated_examples_exclude_their_inputs_from_trace_exports() -> None:
    curated = Dataset([Example(id="bio-1", input="q1")])
    export = (
        await TestEvaluationOnline()
        ._evaluation()
        .dataset_from_traces(
            source=MemorySource(
                [_span("a", input="q1", minute=1), _span("b", input="q2", minute=2)]
            ),
            start=T0,
            end=T0 + timedelta(hours=1),
            exclude=curated,
        )
    )
    assert [e.input for e in export.dataset] == ["q2"]
    assert export.duplicates == 1
