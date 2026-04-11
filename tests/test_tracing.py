"""Tests for the tracing decorator system and telemetry setup."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ReadableSpan,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from grasp_agents.telemetry import SpanKind, traced
from grasp_agents.telemetry.decorators import (
    ATTR_ENTITY_INPUT,
    ATTR_ENTITY_NAME,
    ATTR_ENTITY_OUTPUT,
    ATTR_SPAN_KIND,
    ATTR_WORKFLOW_NAME,
    _resolve_span_kind,
    _should_send_prompts,
    _to_plain,
    _truncate_if_needed,
)

# ---------------------------------------------------------------------------
# Minimal in-memory span exporter (MemoryExporter removed in OTel 1.39)
# ---------------------------------------------------------------------------


class MemoryExporter(SpanExporter):
    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()

    def shutdown(self) -> None:
        self.clear()


# ---------------------------------------------------------------------------
# Module-level OTel setup (one provider for all tests, exporter cleared per test)
# ---------------------------------------------------------------------------

_exporter = MemoryExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))

# Reset the set-once guard so we can install our test provider.
# This is the same approach OTel's own test suite uses.
import opentelemetry.trace as _trace_mod  # noqa: E402

_trace_mod._TRACER_PROVIDER_SET_ONCE = _trace_mod.Once()  # type: ignore[attr-defined]
_trace_mod._TRACER_PROVIDER = None  # type: ignore[attr-defined]
trace.set_tracer_provider(_provider)


@pytest.fixture(autouse=True)
def _clear_exporter() -> None:
    _exporter.clear()


# ========================================================================= #
#  Decorator correctness — sync function (the bug fix)                       #
# ========================================================================= #


class TestSyncFunction:
    def test_returns_value_not_generator(self) -> None:
        @traced(name="compute")
        def compute(x: int) -> int:
            return x + 1

        result = compute(5)
        assert result == 6
        assert not hasattr(result, "__next__"), "Must return value, not generator"

    def test_span_created(self) -> None:
        @traced(name="add_one")
        def add_one(x: int) -> int:
            return x + 1

        add_one(5)

        spans = _exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "add_one.task"
        assert spans[0].attributes[ATTR_SPAN_KIND] == "task"
        assert spans[0].attributes[ATTR_ENTITY_NAME] == "add_one"

    def test_exception_propagates_and_sets_error_status(self) -> None:
        @traced(name="failing")
        def failing() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing()

        spans = _exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == trace.StatusCode.ERROR


# ========================================================================= #
#  Decorator correctness — sync generator                                    #
# ========================================================================= #


class TestSyncGenerator:
    def test_yields_correctly(self) -> None:
        @traced(name="gen")
        def gen_items() -> Any:
            yield 1
            yield 2
            yield 3

        assert list(gen_items()) == [1, 2, 3]

    def test_span_captures_last_item(self) -> None:
        @traced(name="gen")
        def gen_items() -> Any:
            yield "a"
            yield "b"

        list(gen_items())

        spans = _exporter.get_finished_spans()
        assert len(spans) == 1
        # Output should be the last yielded item
        assert spans[0].attributes is not None
        output = spans[0].attributes.get(ATTR_ENTITY_OUTPUT)
        assert output is not None
        assert "b" in str(output)


# ========================================================================= #
#  Decorator correctness — async function                                    #
# ========================================================================= #


class TestAsyncFunction:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_returns_value(self) -> None:
        @traced(name="async_add")
        async def add(x: int) -> int:
            return x + 1

        assert await add(5) == 6

    @pytest.mark.asyncio(loop_scope="function")
    async def test_span_created(self) -> None:
        @traced(name="async_add")
        async def add(x: int) -> int:
            return x + 1

        await add(5)

        spans = _exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "async_add.task"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_exception_propagates(self) -> None:
        @traced(name="async_fail")
        async def fail() -> None:
            raise RuntimeError("async boom")

        with pytest.raises(RuntimeError, match="async boom"):
            await fail()

        spans = _exporter.get_finished_spans()
        assert spans[0].status.status_code == trace.StatusCode.ERROR


# ========================================================================= #
#  Decorator correctness — async generator                                   #
# ========================================================================= #


class TestAsyncGenerator:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_yields_correctly(self) -> None:
        @traced(name="async_gen")
        async def gen() -> Any:
            yield 1
            yield 2
            yield 3

        assert [item async for item in gen()] == [1, 2, 3]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_span_created(self) -> None:
        @traced(name="async_gen")
        async def gen() -> Any:
            yield 1

        [item async for item in gen()]

        spans = _exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "async_gen.task"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_exception_propagates(self) -> None:
        @traced(name="async_gen_fail")
        async def gen() -> Any:
            yield 1
            raise ValueError("gen boom")

        with pytest.raises(ValueError, match="gen boom"):
            async for _ in gen():
                pass

        spans = _exporter.get_finished_spans()
        assert spans[0].status.status_code == trace.StatusCode.ERROR


# ========================================================================= #
#  Dynamic span kind resolution via _span_kind                               #
# ========================================================================= #


class TestSpanKindResolution:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_instance_span_kind_overrides_decorator(self) -> None:
        """A class with _span_kind = AGENT overrides @task's default TASK."""

        class MyAgent:
            _span_kind = SpanKind.AGENT
            name = "planner"
            tracing_enabled = True

            @traced(name="run")
            async def run(self) -> str:
                return "done"

        await MyAgent().run()

        spans = _exporter.get_finished_spans()
        assert spans[0].attributes[ATTR_SPAN_KIND] == "agent"
        assert spans[0].attributes[ATTR_WORKFLOW_NAME] == "run"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_falls_back_to_decorator_default(self) -> None:
        """Without _span_kind, the decorator's span_kind is used."""

        class Plain:
            name = "proc"
            tracing_enabled = True

            @traced(name="do_work", span_kind=SpanKind.WORKFLOW)
            async def do_work(self) -> str:
                return "ok"

        await Plain().do_work()

        spans = _exporter.get_finished_spans()
        assert spans[0].attributes[ATTR_SPAN_KIND] == "workflow"

    def test_resolve_span_kind_helper(self) -> None:
        class WithKind:
            _span_kind = SpanKind.AGENT

        assert _resolve_span_kind(WithKind(), SpanKind.TASK) == SpanKind.AGENT
        assert _resolve_span_kind(None, SpanKind.TASK) == SpanKind.TASK
        assert _resolve_span_kind(object(), SpanKind.WORKFLOW) == SpanKind.WORKFLOW


# ========================================================================= #
#  tracing_enabled opt-out                                                   #
# ========================================================================= #


class TestTracingOptOut:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_disabled_skips_span(self) -> None:
        class Disabled:
            tracing_enabled = False

            @traced(name="noop")
            async def run(self) -> str:
                return "done"

        result = await Disabled().run()
        assert result == "done"
        assert len(_exporter.get_finished_spans()) == 0

    def test_disabled_sync(self) -> None:
        class Disabled:
            tracing_enabled = False

            @traced(name="noop")
            def run(self) -> str:
                return "done"

        assert Disabled().run() == "done"
        assert len(_exporter.get_finished_spans()) == 0


# ========================================================================= #
#  Span attributes                                                           #
# ========================================================================= #


class TestSpanAttributes:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_workflow_sets_workflow_name(self) -> None:
        @traced(name="my_wf", span_kind=SpanKind.WORKFLOW)
        async def wf() -> str:
            return "ok"

        await wf()

        spans = _exporter.get_finished_spans()
        assert spans[0].attributes[ATTR_WORKFLOW_NAME] == "my_wf"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_task_does_not_set_workflow_name(self) -> None:
        @traced(name="my_task")
        async def t() -> str:
            return "ok"

        await t()

        spans = _exporter.get_finished_spans()
        assert ATTR_WORKFLOW_NAME not in (spans[0].attributes or {})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_agent_sets_workflow_name(self) -> None:
        @traced(name="my_agent", span_kind=SpanKind.AGENT)
        async def a() -> str:
            return "ok"

        await a()

        spans = _exporter.get_finished_spans()
        assert spans[0].attributes[ATTR_WORKFLOW_NAME] == "my_agent"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_input_output_recorded(self) -> None:
        @traced(name="echo")
        async def echo(x: int) -> int:
            return x * 2

        await echo(5)

        spans = _exporter.get_finished_spans()
        attrs = spans[0].attributes or {}
        assert ATTR_ENTITY_INPUT in attrs
        assert ATTR_ENTITY_OUTPUT in attrs
        assert "5" in str(attrs[ATTR_ENTITY_INPUT])
        assert "10" in str(attrs[ATTR_ENTITY_OUTPUT])


# ========================================================================= #
#  Span naming                                                               #
# ========================================================================= #


class TestSpanNaming:
    def test_standalone_function_name(self) -> None:
        @traced(name="compute")
        def compute() -> int:
            return 1

        compute()
        assert _exporter.get_finished_spans()[0].name == "compute.task"

    def test_workflow_name(self) -> None:
        @traced(name="pipeline", span_kind=SpanKind.WORKFLOW)
        def pipeline() -> int:
            return 1

        pipeline()
        assert _exporter.get_finished_spans()[0].name == "pipeline.workflow"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_method_uses_instance_name(self) -> None:
        class MyTool:
            name = "search"
            tracing_enabled = True
            _span_kind = SpanKind.TOOL

            @traced(name="execute")
            async def execute(self) -> str:
                return "found"

        await MyTool().execute()

        # TOOL kind reads instance.name
        assert _exporter.get_finished_spans()[0].name == "search.execute"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_exec_id_suffix(self) -> None:
        class MyAgent:
            name = "writer"
            tracing_enabled = True
            _span_kind = SpanKind.AGENT

            @traced(name="run")
            async def run(self, exec_id: str = "") -> str:
                return "ok"

        await MyAgent().run(exec_id="abc123")

        assert _exporter.get_finished_spans()[0].name == "writer.run[abc123]"


# ========================================================================= #
#  Span nesting (parent-child hierarchy)                                     #
# ========================================================================= #


class TestSpanNesting:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_nested_spans_have_parent(self) -> None:
        @traced(name="outer", span_kind=SpanKind.WORKFLOW)
        async def outer() -> str:
            return await inner()

        @traced(name="inner")
        async def inner() -> str:
            return "done"

        await outer()

        spans = _exporter.get_finished_spans()
        assert len(spans) == 2

        inner_span = next(s for s in spans if s.name == "inner.task")
        outer_span = next(s for s in spans if s.name == "outer.workflow")

        assert inner_span.parent is not None
        assert inner_span.parent.span_id == outer_span.context.span_id


# ========================================================================= #
#  Helper functions                                                          #
# ========================================================================= #


class TestHelpers:
    def test_to_plain_pydantic(self) -> None:
        from pydantic import BaseModel

        class Item(BaseModel):
            x: int
            y: str

        assert _to_plain(Item(x=1, y="hello")) == {"x": 1, "y": "hello"}

    def test_to_plain_excludes_default_fields(self) -> None:
        result = _to_plain({"a": 1, "_hidden_params": "secret", "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_to_plain_nested(self) -> None:
        result = _to_plain({"items": [{"a": 1}, {"b": 2}]})
        assert result == {"items": [{"a": 1}, {"b": 2}]}

    def test_truncate_if_needed(self) -> None:
        with patch.dict(os.environ, {"OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT": "5"}):
            assert _truncate_if_needed("hello world") == "hello"
            assert _truncate_if_needed("hi") == "hi"

    def test_truncate_no_limit(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", None)
            assert _truncate_if_needed("hello world") == "hello world"


# ========================================================================= #
#  _should_send_prompts env var precedence                                   #
# ========================================================================= #


class TestShouldSendPrompts:
    def test_default_true(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _should_send_prompts() is True

    def test_grasp_env_var(self) -> None:
        with patch.dict(
            os.environ,
            {"GRASP_TRACE_CONTENT": "false"},
            clear=True,
        ):
            assert _should_send_prompts() is False

    def test_traceloop_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {"TRACELOOP_TRACE_CONTENT": "false"},
            clear=True,
        ):
            assert _should_send_prompts() is False

    def test_grasp_takes_precedence(self) -> None:
        with patch.dict(
            os.environ,
            {"GRASP_TRACE_CONTENT": "true", "TRACELOOP_TRACE_CONTENT": "false"},
        ):
            assert _should_send_prompts() is True


# ========================================================================= #
#  Telemetry setup functions                                                 #
# ========================================================================= #


class TestTelemetrySetup:
    def test_init_tracing_returns_existing_provider(self) -> None:
        """When a TracerProvider already exists, init_tracing returns it."""
        from grasp_agents.telemetry.setup import init_tracing

        provider = init_tracing(project_name="test")
        assert isinstance(provider, TracerProvider)

    def test_init_tracing_idempotent(self) -> None:
        from grasp_agents.telemetry.setup import init_tracing

        p1 = init_tracing(project_name="a")
        p2 = init_tracing(project_name="b")
        assert p1 is p2

    def test_add_exporter_attaches_to_provider(self) -> None:
        from grasp_agents.telemetry.setup import add_exporter

        extra_exporter = MemoryExporter()
        add_exporter(extra_exporter, provider=_provider, batch=False)

        @traced(name="probe")
        def probe() -> int:
            return 42

        probe()

        # The extra exporter should have captured the span too
        assert len(extra_exporter.get_finished_spans()) >= 1

    def test_add_exporter_raises_without_provider(self) -> None:
        from unittest.mock import patch as mock_patch

        from grasp_agents.telemetry.setup import add_exporter

        with mock_patch(
            "grasp_agents.telemetry.setup.trace.get_tracer_provider",
            return_value=trace.NoOpTracerProvider(),
        ):
            with pytest.raises(RuntimeError, match="No TracerProvider configured"):
                add_exporter(MemoryExporter())

