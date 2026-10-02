"""Tests for the tracing decorator system and telemetry setup."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.evals.online import SpanRecord, TraceItem, default_extractor
from grasp_agents.processors.parallel_processor import ParallelProcessor
from grasp_agents.processors.processor import Processor
from grasp_agents.session_context import SessionContext
from grasp_agents.telemetry import (
    InheritedAttributesSpanProcessor,
    SpanKind,
    SpanStart,
    attributes,
    derive_session_span_context,
    fit_json,
    inherited_span_attributes,
    set_run_span_attributes,
    traced,
)
from grasp_agents.telemetry.attributes import (
    ATTR_FAILED_ATTEMPTS,
    ATTR_INPUT_MIME_TYPE,
    ATTR_INPUT_VALUE,
    ATTR_OI_SPAN_KIND,
    ATTR_OUTPUT_VALUE,
    ATTR_SPAN_KIND,
)
from grasp_agents.telemetry.decorators import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    _clip,
    _inherited,
    _resolve_run_span_context,
    _should_send_prompts,
    _to_plain,
)
from grasp_agents.tools.agent_tool import AgentTool
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.events import Event, ProcPayloadOutEvent, ToolErrorInfo
from tests._helpers import (
    AddInput,
    AddTool,
    MockLLM,
    _text_response,
    _tool_call_response,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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
_provider.add_span_processor(InheritedAttributesSpanProcessor())
_provider.add_span_processor(SimpleSpanProcessor(_exporter))

# Reset the set-once guard so we can install our test provider.
# This is the same approach OTel's own test suite uses.
import opentelemetry.trace as _trace_mod

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
        assert spans[0].name == "add_one"
        assert spans[0].attributes[ATTR_SPAN_KIND] == "task"

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
        assert spans[0].attributes.get(ATTR_OUTPUT_VALUE) == '"b"'


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
        assert spans[0].name == "async_add"

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
        assert spans[0].name == "async_gen"

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
#  Objects describing their own spans                                        #
# ========================================================================= #


class TestSpanDescriptionHooks:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_instance_describes_name_kind_attributes_and_input(self) -> None:
        class Planner:
            tracing_enabled = True

            def _trace_span_start(
                self, entity: str, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> SpanStart:
                return SpanStart(
                    name=f"planner.{entity}",
                    kind=SpanKind.AGENT,
                    attributes={"planner.depth": 2},
                    input={"goal": args[0]},
                )

            @traced(name="run")
            async def run(self, goal: str) -> str:
                return goal.upper()

        await Planner().run("ship")

        (span,) = _exporter.get_finished_spans()
        attrs = span.attributes or {}
        assert span.name == "planner.run"
        assert attrs[ATTR_SPAN_KIND] == "agent"
        assert attrs[ATTR_OI_SPAN_KIND] == "AGENT"
        assert attrs["planner.depth"] == 2
        assert attrs[ATTR_INPUT_VALUE] == '{"goal": "ship"}'
        assert attrs[ATTR_OUTPUT_VALUE] == '"SHIP"'

    @pytest.mark.asyncio(loop_scope="function")
    async def test_without_hooks_the_decorator_describes_the_span(self) -> None:
        class Plain:
            @traced(name="do_work", span_kind=SpanKind.WORKFLOW)
            async def do_work(self, n: int) -> str:
                return "ok"

        await Plain().do_work(3)

        (span,) = _exporter.get_finished_spans()
        assert span.name == "do_work"
        assert (span.attributes or {})[ATTR_SPAN_KIND] == "workflow"
        assert (span.attributes or {})[ATTR_INPUT_VALUE] == '{"n": 3}'


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
#  tracing_enabled=False suppresses downstream auto-instrumentation          #
# ========================================================================= #


def _suppression_active() -> bool:
    """The switch OpenInference / OTel-contrib instrumentors check."""
    return bool(otel_context.get_value(_SUPPRESS_INSTRUMENTATION_KEY))


class TestDisabledInstrumentationSuppression:
    """
    ``tracing_enabled=False`` must go fully dark: not just skip the grasp
    span, but mark OTel auto-instrumentation suppressed for the call's
    duration — otherwise a disabled agent's provider-SDK calls (OpenInference
    google-genai / anthropic / openai instrumentors) still surface as orphan
    traces in the backend.
    """

    @pytest.mark.asyncio(loop_scope="function")
    async def test_disabled_call_suppresses_all_four_wrapper_shapes(self) -> None:
        observed: list[bool] = []

        class Disabled:
            tracing_enabled = False

            @traced(name="noop_async")
            async def run_async(self) -> None:
                observed.append(_suppression_active())

            @traced(name="noop_async_gen")
            async def run_async_gen(self) -> AsyncIterator[int]:
                observed.append(_suppression_active())
                yield 1

            @traced(name="noop_sync")
            def run_sync(self) -> None:
                observed.append(_suppression_active())

            @traced(name="noop_sync_gen")
            def run_sync_gen(self) -> Iterator[int]:
                observed.append(_suppression_active())
                yield 1

        d = Disabled()
        await d.run_async()
        async for _ in d.run_async_gen():
            pass
        d.run_sync()
        for _ in d.run_sync_gen():
            pass

        assert observed == [True, True, True, True]
        # Detached once each call ends — later calls are not affected.
        assert _suppression_active() is False
        assert len(_exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio(loop_scope="function")
    async def test_enabled_call_does_not_suppress(self) -> None:
        observed: list[bool] = []

        class Enabled:
            @traced(name="probe")
            async def run(self) -> None:
                observed.append(_suppression_active())

        await Enabled().run()

        assert observed == [False]
        assert len(_exporter.get_finished_spans()) == 1

    @pytest.mark.asyncio(loop_scope="function")
    async def test_nested_traced_call_inside_disabled_region_emits_no_span(
        self,
    ) -> None:
        class Inner:
            @traced(name="inner")
            async def run(self) -> str:
                return "ok"

        class Outer:
            tracing_enabled = False

            @traced(name="outer")
            async def run(self) -> str:
                return await Inner().run()

        assert await Outer().run() == "ok"
        assert len(_exporter.get_finished_spans()) == 0


# ========================================================================= #
#  Span attributes                                                           #
# ========================================================================= #


class TestSpanAttributes:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_openinference_span_kind_set(self) -> None:
        """Each grasp span kind maps to an OpenInference span kind."""

        @traced(name="wf", span_kind=SpanKind.WORKFLOW)
        async def wf() -> str:
            return "ok"

        @traced(name="ag", span_kind=SpanKind.AGENT)
        async def ag() -> str:
            return "ok"

        @traced(name="tl", span_kind=SpanKind.TOOL)
        async def tl() -> str:
            return "ok"

        @traced(name="tk", span_kind=SpanKind.TASK)
        async def tk() -> str:
            return "ok"

        await wf()
        await ag()
        await tl()
        await tk()

        by_name = {s.name: s.attributes or {} for s in _exporter.get_finished_spans()}
        assert by_name["wf"][ATTR_OI_SPAN_KIND] == "CHAIN"
        assert by_name["ag"][ATTR_OI_SPAN_KIND] == "AGENT"
        assert by_name["tl"][ATTR_OI_SPAN_KIND] == "TOOL"
        assert by_name["tk"][ATTR_OI_SPAN_KIND] == "CHAIN"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_input_output_recorded_as_json(self) -> None:
        @traced(name="echo")
        async def echo(x: int, *, label: str = "") -> int:
            return x * 2

        await echo(5, label="n")

        attrs = _exporter.get_finished_spans()[0].attributes or {}
        assert attrs[ATTR_INPUT_VALUE] == '{"x": 5, "label": "n"}'
        assert attrs[ATTR_INPUT_MIME_TYPE] == "application/json"
        assert attrs[ATTR_OUTPUT_VALUE] == "10"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_content_off_records_no_payloads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GRASP_TRACE_CONTENT", "false")

        @traced(name="echo")
        async def echo(x: int) -> int:
            return x

        await echo(5)

        attrs = _exporter.get_finished_spans()[0].attributes or {}
        assert ATTR_INPUT_VALUE not in attrs
        assert ATTR_OUTPUT_VALUE not in attrs
        assert attrs[ATTR_SPAN_KIND] == "task"


# ========================================================================= #
#  Span naming                                                               #
# ========================================================================= #


class TestSpanNaming:
    def test_function_named_by_decorator(self) -> None:
        @traced(name="compute")
        def compute() -> int:
            return 1

        compute()
        assert _exporter.get_finished_spans()[0].name == "compute"

    def test_function_named_by_qualname_by_default(self) -> None:
        @traced()
        def compute() -> int:
            return 1

        compute()
        assert _exporter.get_finished_spans()[0].name.endswith("compute")

    def test_class_decoration_names_class_and_method(self) -> None:
        @traced(method_name="work")
        class Worker:
            def work(self) -> int:
                return 1

        Worker().work()
        assert _exporter.get_finished_spans()[0].name == "Worker.work"


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

        inner_span = next(s for s in spans if s.name == "inner")
        outer_span = next(s for s in spans if s.name == "outer")

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

    def test_clip_tiny_limit_falls_back_to_head(self) -> None:
        # Limit too small to fit a head…tail marker → plain head clip.
        assert _clip("hello world", 5) == "hello"
        assert _clip("hi", 5) == "hi"

    def test_clip_keeps_head_and_tail(self) -> None:
        text = "H" * 40 + "X" * 200 + "T" * 40
        out = _clip(text, 60)
        assert len(out) <= 60
        assert out.startswith("H")
        assert out.endswith("T")
        assert "chars]" in out
        assert "X" * 20 not in out


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

    def test_init_phoenix_reentry_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A second init_phoenix against the same provider must not attach the
        span processors again — each duplicate doubles every exported span.
        """
        from grasp_agents.telemetry import phoenix as phoenix_mod
        from grasp_agents.telemetry.setup import init_tracing

        monkeypatch.setenv(
            "TELEMETRY_COLLECTOR_HTTP_ENDPOINT", "http://localhost:6006/v1/traces"
        )
        provider = init_tracing(project_name="test")
        try:
            with patch.object(provider, "add_span_processor") as add_mock:
                phoenix_mod.init_phoenix(
                    use_litellm_instr=False, use_llm_provider_instr=False
                )
                first_calls = add_mock.call_count
                phoenix_mod.init_phoenix(
                    use_litellm_instr=False, use_llm_provider_instr=False
                )
            assert first_calls == 2
            assert add_mock.call_count == first_calls
        finally:
            phoenix_mod._phoenix_attached.discard(provider)

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


# ========================================================================= #
#  Span I/O serialization must never fail the traced call                    #
# ========================================================================= #


class TestSerializationFailureContained:
    def test_circular_output_does_not_break_traced_call(self) -> None:
        @traced(name="circular")
        def make() -> Any:
            d: dict[str, Any] = {}
            d["self"] = d  # circular → recursion while serializing the span
            return d

        result = make()  # must return normally despite the unserializable value
        assert result["self"] is result

        spans = _exporter.get_finished_spans()
        assert len(spans) == 1
        # Span still finishes; the serialization failure was swallowed, so no
        # output attribute was set and the call did not error out.
        assert spans[0].attributes is not None
        assert spans[0].attributes.get(ATTR_OUTPUT_VALUE) is None
        assert spans[0].status.status_code != trace.StatusCode.ERROR


# ---------- @traced generators stream through ----------


class TestTracedGeneratorPassThrough:
    @pytest.mark.asyncio
    async def test_async_gen_yields_all_items(self) -> None:
        @traced(name="gen")
        async def gen() -> AsyncIterator[int]:
            for i in range(5):
                yield i

        assert [i async for i in gen()] == [0, 1, 2, 3, 4]

    def test_sync_gen_yields_all_items(self) -> None:
        @traced(name="gen")
        def gen():
            yield from range(5)

        assert list(gen()) == [0, 1, 2, 3, 4]


# ========================================================================= #
#  Caller-supplied run-span overrides (span_name / span_attributes)          #
#  — the seam ``run`` / ``run_stream`` use to attach domain attributes.      #
# ========================================================================= #


class TestCallerSpanOverrides:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_span_attributes_attached(self) -> None:
        @traced(name="run")
        async def run(**kwargs: Any) -> str:
            return "ok"

        await run(span_attributes={"goal.id": "g1", "lesson.n": 3})

        attrs = _exporter.get_finished_spans()[0].attributes or {}
        assert attrs["goal.id"] == "g1"
        assert attrs["lesson.n"] == 3

    @pytest.mark.asyncio(loop_scope="function")
    async def test_span_name_override(self) -> None:
        @traced(name="run")
        async def run(**kwargs: Any) -> str:
            return "ok"

        await run(span_name="pathway_generation.pg-42")

        assert _exporter.get_finished_spans()[0].name == "pathway_generation.pg-42"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_overrides_keeps_framework_defaults(self) -> None:
        @traced(name="run")
        async def run() -> str:
            return "ok"

        await run()

        span = _exporter.get_finished_spans()[0]
        assert span.name == "run"
        assert "goal.id" not in (span.attributes or {})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_set_run_span_attributes_mid_run(self) -> None:
        @traced(name="run")
        async def run() -> str:
            set_run_span_attributes(discovered="mid-run", count=2)
            return "ok"

        await run()

        attrs = _exporter.get_finished_spans()[0].attributes or {}
        assert attrs["discovered"] == "mid-run"
        assert attrs["count"] == 2


# ========================================================================= #
#  Session trace grouping (deterministic, backend-agnostic)                  #
# ========================================================================= #


_RUN_KINDS = frozenset({"agent", "workflow", "processor"})


def _processor_spans() -> list[ReadableSpan]:
    """Processor-run spans (one per run_stream), selected by span kind."""
    return [
        s
        for s in _exporter.get_finished_spans()
        if (s.attributes or {}).get(ATTR_SPAN_KIND) in _RUN_KINDS
    ]


def _session_trace_id(session_key: str) -> int:
    parent = derive_session_span_context(session_key)
    return trace.get_current_span(parent).get_span_context().trace_id


class TestSessionTraceDerivation:
    def test_deterministic_and_distinct(self) -> None:
        assert _session_trace_id("sess-A") == _session_trace_id("sess-A") != 0
        assert _session_trace_id("sess-A") != _session_trace_id("sess-B")

    def test_resolver_skips_default_and_missing_session(self) -> None:
        class Named:
            def _trace_session_info(self) -> tuple[str, bool] | None:
                return ("sess-A", True)

        class DefaultSession:
            def _trace_session_info(self) -> tuple[str, bool] | None:
                return None

        assert _resolve_run_span_context(object()) is None  # no hook
        assert _resolve_run_span_context(DefaultSession()) is None  # unnamed
        assert _resolve_run_span_context(Named()) is not None  # at a run root

    def test_under_a_caller_span_the_session_keeps_the_callers_trace(self) -> None:
        class Named:
            def _trace_session_info(self) -> tuple[str, bool] | None:
                return ("sess-A", True)

        # A caller's span (an HTTP request, an evaluation trial): the session
        # id is stamped, the run stays in the caller's trace.
        with trace.get_tracer("test").start_as_current_span("outer") as outer:
            context = _resolve_run_span_context(Named())
            assert context is not None
            assert trace.get_current_span(context) is outer
            assert _inherited(context)["session.id"] == "sess-A"
            # Runs nested in the session's run inherit it.
            token = otel_context.attach(context)
            try:
                assert _resolve_run_span_context(Named()) is None
            finally:
                otel_context.detach(token)

    def test_derived_session_traces_are_namespaced(self) -> None:
        def trace_id(namespace: str | None) -> int:
            parent = derive_session_span_context("s", namespace=namespace)
            return trace.get_current_span(parent).get_span_context().trace_id

        assert trace_id("staging/app") != trace_id("prod/app")
        assert trace_id(None) == _session_trace_id("s")


class TestSessionTraceGrouping:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_named_session_runs_share_one_trace_without_a_store(self) -> None:
        # No checkpoint_store wired — grouping is purely from the session_key.
        with SessionContext[None](session_key="sess-multi-turn"):
            agent = LLMAgent[str, str, None](
                name="chat",
                llm=MockLLM(
                    responses_queue=[_text_response("a1"), _text_response("a2")]
                ),
            )
            await agent.run(chat_inputs="turn 1")
            await agent.run(chat_inputs="turn 2")

        roots = _processor_spans()
        assert len(roots) == 2
        assert {s.context.trace_id for s in roots} == {
            _session_trace_id("sess-multi-turn")
        }

    @pytest.mark.asyncio(loop_scope="function")
    async def test_opt_out_makes_named_session_runs_independent(self) -> None:
        with SessionContext[None](
            session_key="sess-optout", session_trace_grouping=False
        ):
            agent = LLMAgent[str, str, None](
                name="chat",
                llm=MockLLM(
                    responses_queue=[_text_response("a1"), _text_response("a2")]
                ),
            )
            await agent.run(chat_inputs="turn 1")
            await agent.run(chat_inputs="turn 2")

        roots = _processor_spans()
        assert len(roots) == 2
        # Opted out → each run is its own trace root despite the session_key.
        assert roots[0].context.trace_id != roots[1].context.trace_id

    @pytest.mark.asyncio(loop_scope="function")
    async def test_default_session_runs_are_independent_traces(self) -> None:
        agent = LLMAgent[str, str, None](
            name="chat",
            llm=MockLLM(responses_queue=[_text_response("a1"), _text_response("a2")]),
        )
        await agent.run(chat_inputs="turn 1")
        await agent.run(chat_inputs="turn 2")

        roots = _processor_spans()
        assert len(roots) == 2
        # Unnamed session → each run is its own trace root (prior behavior).
        assert roots[0].context.trace_id != roots[1].context.trace_id

    @pytest.mark.asyncio(loop_scope="function")
    async def test_nested_run_stays_in_parent_session_trace(self) -> None:
        # Construct inside the block: the session binds at construction time.
        with SessionContext[None](session_key="sess-nested"):
            child = AgentTool[None](
                name="research",
                description="Research a topic",
                llm=MockLLM(responses_queue=[_text_response("child answer")]),
            )
            parent = LLMAgent[str, str, None](
                name="parent",
                llm=MockLLM(
                    responses_queue=[
                        _tool_call_response("research", '{"prompt": "x"}', "tc1"),
                        _text_response("done"),
                    ]
                ),
                tools=[child],
            )
            await parent.run(chat_inputs="go")

        roots = _processor_spans()
        # Parent (run root) + spawned child both emit a processor span, and ALL
        # live in the one session trace: the child nested under the parent's
        # tool span instead of detaching to its own session root.
        assert len(roots) >= 2
        assert {s.context.trace_id for s in roots} == {_session_trace_id("sess-nested")}


# ========================================================================= #
#  Session attributes (propagated to every span of a session's run)        #
# ========================================================================= #


def _session_attr(span: ReadableSpan, key: str = "gen_ai.conversation.id") -> Any:
    return (span.attributes or {}).get(key)


async def _run_chat(session_key: str | None = None, **ctx_kwargs: Any) -> None:
    llm = MockLLM(responses_queue=[_text_response("ok")])
    if session_key is None:
        agent = LLMAgent[str, str, None](name="chat", llm=llm)
        await agent.run(chat_inputs="hi")
        return
    with SessionContext[None](session_key=session_key, **ctx_kwargs):
        agent = LLMAgent[str, str, None](name="chat", llm=llm)
        await agent.run(chat_inputs="hi")


class TestSessionAttributes:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_stamped_on_run_and_child_spans(self) -> None:
        await _run_chat("sess-attr")
        spans = _exporter.get_finished_spans()
        by_kind = {(s.attributes or {}).get(ATTR_SPAN_KIND): s for s in spans}
        # The run root (agent) AND a child it created (generate) both carry
        # the session id — the processor propagates it down the whole run tree.
        assert "agent" in by_kind
        assert "generate" in by_kind
        for key in ("session.id", "gen_ai.conversation.id"):
            assert _session_attr(by_kind["agent"], key) == "sess-attr"
            assert _session_attr(by_kind["generate"], key) == "sess-attr"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_emitted_even_when_grouping_off(self) -> None:
        # Session attributes are independent of headless trace grouping.
        await _run_chat("sess-nogroup", session_trace_grouping=False)
        spans = _exporter.get_finished_spans()
        assert spans
        assert all(_session_attr(s) == "sess-nogroup" for s in spans)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_absent_for_default_session(self) -> None:
        await _run_chat(session_key=None)
        spans = _exporter.get_finished_spans()
        assert spans
        assert all(_session_attr(s) is None for s in spans)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_keys_configurable_via_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GRASP_SESSION_ID_ATTRIBUTES", "session.id, custom.session")
        await _run_chat("sess-cfg")
        spans = _exporter.get_finished_spans()
        root = next(s for s in spans if _session_attr(s, ATTR_SPAN_KIND) == "agent")
        attrs = root.attributes or {}
        assert attrs.get("session.id") == "sess-cfg"
        assert attrs.get("custom.session") == "sess-cfg"
        assert "gen_ai.conversation.id" not in attrs  # default replaced

    @pytest.mark.asyncio(loop_scope="function")
    async def test_opt_out_with_empty_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GRASP_SESSION_ID_ATTRIBUTES", "")
        await _run_chat("sess-empty")
        spans = _exporter.get_finished_spans()
        assert spans
        assert all(_session_attr(s) is None for s in spans)


# ========================================================================= #
#  Retry visibility (with_retry records every failed attempt)                #
# ========================================================================= #


class _FlakyProcessor(Processor[str, str, None]):
    """Fails its first ``fail_times`` attempts, then yields an output."""

    def __init__(
        self,
        name: str,
        *,
        fail_times: int,
        err_msg: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self._fail_times = fail_times
        self._err_msg = err_msg
        self.calls = 0

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise ValueError(self._err_msg or f"boom {self.calls}")
        for inp in in_args or []:
            yield ProcPayloadOutEvent(data=f"{inp}!", source=self.name, exec_id=exec_id)


def _span_by_kind(spans: Sequence[ReadableSpan], kind: str) -> ReadableSpan:
    return next(s for s in spans if (s.attributes or {}).get(ATTR_SPAN_KIND) == kind)


def _exception_events(span: ReadableSpan) -> list[Any]:
    return [e for e in (span.events or []) if e.name == "exception"]


class TestRetryExceptionRecording:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_recovered_retry_records_events_without_error_status(self) -> None:
        proc = _FlakyProcessor("flaky", fail_times=2, max_retries=2)
        out = await proc.run(in_args="hi")
        assert out.payloads == ["hi!"]

        span = _span_by_kind(_exporter.get_finished_spans(), "processor")
        events = _exception_events(span)

        # Each swallowed attempt is visible, in order, with its own cause.
        assert len(events) == 2
        assert [(e.attributes or {})["exception.type"] for e in events] == [
            "ValueError",
            "ValueError",
        ]
        assert [(e.attributes or {})["exception.message"] for e in events] == [
            "boom 1",
            "boom 2",
        ]
        assert (span.attributes or {})[ATTR_FAILED_ATTEMPTS] == 2

        # The run ultimately succeeded: flipping the shared processor span to
        # ERROR here would make recovered runs indistinguishable from failures.
        assert span.status.status_code is not trace.StatusCode.ERROR

    @pytest.mark.asyncio(loop_scope="function")
    async def test_exhausted_retries_record_cause_and_keep_error_status(self) -> None:
        proc = _FlakyProcessor("doomed", fail_times=99, max_retries=1)
        with pytest.raises(ProcRunError):
            await proc.run(in_args="hi")

        span = _span_by_kind(_exporter.get_finished_spans(), "processor")
        events = _exception_events(span)
        types_ = [(e.attributes or {})["exception.type"] for e in events]

        # Both attempts recorded by with_retry — including the terminal one,
        # whose underlying cause `@traced` would otherwise hide behind the
        # wrapping ProcRunError.
        assert types_.count("ValueError") == 2
        assert any("ProcRunError" in str(t) for t in types_)
        assert (span.attributes or {})[ATTR_FAILED_ATTEMPTS] == 2
        # Terminal failure still goes red — set by `@traced`, untouched here.
        assert span.status.status_code is trace.StatusCode.ERROR

    @pytest.mark.asyncio(loop_scope="function")
    async def test_disabled_processor_does_not_record_on_enclosing_span(self) -> None:
        proc = _FlakyProcessor(
            "quiet", fail_times=1, max_retries=1, tracing_enabled=False
        )

        @traced(name="outer")
        async def outer() -> None:
            await proc.run(in_args="hi")

        await outer()
        spans = _exporter.get_finished_spans()

        # A tracing-disabled processor emits no span of its own, so the current
        # span inside its retry loop belongs to the enclosing (recording)
        # parent — its retries must not be attributed there.
        assert all(
            (s.attributes or {}).get(ATTR_SPAN_KIND) != "processor" for s in spans
        )
        parent = next(s for s in spans if s.name == "outer")
        assert _exception_events(parent) == []
        assert ATTR_FAILED_ATTEMPTS not in (parent.attributes or {})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_agent_output_parse_failure_lands_on_processor_span(self) -> None:
        # The production shape (a failing output parser): the LLM call itself
        # succeeded, so its `generate` span closed green and the failure has no
        # child span of its own — the processor span is the only one still open
        # when `parse_output` raises.
        agent = LLMAgent[str, str, None](
            name="writer",
            llm=MockLLM(
                responses_queue=[_text_response("bad"), _text_response("good")]
            ),
            max_retries=1,
        )

        @agent.add_output_parser
        def _parse(final_answer: str, **_: Any) -> str:
            if final_answer == "bad":
                raise ValueError("resource block did not resolve")
            return final_answer

        await agent.run(chat_inputs="hi")

        spans = _exporter.get_finished_spans()
        proc_span = _span_by_kind(spans, "agent")
        (event,) = _exception_events(proc_span)
        assert (event.attributes or {})["exception.message"] == (
            "resource block did not resolve"
        )
        assert proc_span.status.status_code is not trace.StatusCode.ERROR

        # The model answered fine on both attempts, so no `generate` span is
        # red — without this change the wasted attempt is invisible in the trace.
        gen_spans = [
            s for s in spans if (s.attributes or {}).get(ATTR_SPAN_KIND) == "generate"
        ]
        assert len(gen_spans) == 2
        assert all(not _exception_events(s) for s in gen_spans)


class _ParentAbandoningChild(Processor[str, str, None]):
    """Fails once mid-iteration of a child stream (paused at a yield), then succeeds."""

    def __init__(
        self, name: str, *, child: Processor[str, str, None], **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)
        self._child = child
        self.calls = 0

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        self.calls += 1
        if self.calls == 1:
            # The child yields its first event and pauses; raising here
            # abandons it with its context line still on the whiteboard.
            async for _ in self._child.run_stream(in_args=["a", "b"]):
                raise ValueError("parent failed mid-iteration")
        for inp in in_args or []:
            yield ProcPayloadOutEvent(data=f"{inp}!", source=self.name, exec_id=exec_id)


class TestAbandonedChildStream:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_retry_events_land_on_the_exported_parent_span(self) -> None:
        child = _FlakyProcessor("child", fail_times=0)
        parent = _ParentAbandoningChild("parent", child=child, max_retries=1)

        out = await parent.run(in_args="hi")
        assert out.payloads == ["hi!"]

        exported = _exporter.get_finished_spans()
        parent_span = next(s for s in exported if s.name == "parent")
        events = _exception_events(parent_span)
        assert [(e.attributes or {})["exception.type"] for e in events] == [
            "ValueError"
        ]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_content_off_still_records_exception_details(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GRASP_TRACE_CONTENT", "false")
        proc = _FlakyProcessor(
            "gated", fail_times=1, max_retries=1, err_msg="whole lesson text here"
        )
        await proc.run(in_args="hi")

        span = _span_by_kind(_exporter.get_finished_spans(), "processor")
        (event,) = _exception_events(span)
        attrs = event.attributes or {}
        assert attrs["exception.type"] == "ValueError"
        assert "lesson text" in str(attrs["exception.message"])
        assert "lesson text" in str(attrs["exception.stacktrace"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_escaping_error_recorded_once_with_full_description(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GRASP_TRACE_CONTENT", "false")

        @traced(name="leaky")
        async def fail() -> None:
            raise ValueError("whole lesson text here")

        with pytest.raises(ValueError):
            await fail()

        (span,) = _exporter.get_finished_spans()
        assert span.status.status_code is trace.StatusCode.ERROR
        assert "lesson text" in (span.status.description or "")
        (event,) = _exception_events(span)
        assert "lesson text" in str((event.attributes or {})["exception.message"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_escaping_generator_error_recorded_once_with_full_description(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GRASP_TRACE_CONTENT", "false")

        @traced(name="leaky_gen")
        async def fail_gen() -> AsyncIterator[str]:
            yield "a"
            raise ValueError("whole lesson text here")

        with pytest.raises(ValueError):
            async for _ in fail_gen():
                pass

        (span,) = _exporter.get_finished_spans()
        assert span.status.status_code is trace.StatusCode.ERROR
        assert "lesson text" in (span.status.description or "")
        (event,) = _exception_events(span)
        assert "lesson text" in str((event.attributes or {})["exception.message"])


# ========================================================================= #
#  Framework span conventions                                                #
# ========================================================================= #


def _spans_of_kind(kind: str) -> list[ReadableSpan]:
    return [
        s
        for s in _exporter.get_finished_spans()
        if (s.attributes or {}).get(ATTR_SPAN_KIND) == kind
    ]


class TestFrameworkSpans:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_agent_run_span_identity_and_payloads(self) -> None:
        agent = LLMAgent[str, str, None](
            name="writer",
            llm=MockLLM(responses_queue=[_text_response("done")]),
            version="v2",
        )
        await agent.run(chat_inputs="hello")

        (run,) = _spans_of_kind("agent")
        attrs = run.attributes or {}
        assert run.name == "writer"
        assert attrs[ATTR_OI_SPAN_KIND] == "AGENT"
        assert attrs[attributes.ATTR_PROCESSOR_NAME] == "writer"
        assert attrs[attributes.ATTR_PROCESSOR_CLASS] == "LLMAgent"
        assert attrs[attributes.ATTR_PROCESSOR_PATH] == "writer"
        assert attrs[attributes.ATTR_PROCESSOR_VERSION] == "v2"
        assert str(attrs[attributes.ATTR_PROCESSOR_EXEC_ID]).endswith("_writer")
        assert attrs[attributes.ATTR_AGENT_MODEL] == "mock"
        assert attrs[ATTR_INPUT_VALUE] == '"hello"'
        assert attrs[ATTR_OUTPUT_VALUE] == '"done"'

    @pytest.mark.asyncio(loop_scope="function")
    async def test_model_call_span_carries_models_usage_and_output_items(
        self,
    ) -> None:
        agent = LLMAgent[str, str, None](
            name="writer", llm=MockLLM(responses_queue=[_text_response("done")])
        )
        await agent.run(chat_inputs="hello")

        (call,) = _spans_of_kind("generate")
        attrs = call.attributes or {}
        assert call.name == "writer.generate"
        assert attrs[attributes.ATTR_AGENT_NAME] == "writer"
        assert attrs[attributes.ATTR_LLM_REQUEST_MODEL] == "mock"
        assert attrs[attributes.ATTR_LLM_RESPONSE_MODEL] == "mock"
        assert attrs[attributes.ATTR_LLM_INPUT_TOKENS] == 10
        assert attrs[attributes.ATTR_LLM_OUTPUT_TOKENS] == 5
        assert ATTR_INPUT_VALUE not in attrs
        (item,) = json.loads(str(attrs[ATTR_OUTPUT_VALUE]))
        assert item["content"][0]["text"] == "done"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_tool_span_names_the_tool_and_its_caller(self) -> None:
        agent = LLMAgent[str, str, None](
            name="calc",
            llm=MockLLM(
                responses_queue=[
                    _tool_call_response("add", '{"a": 2, "b": 3}', "tc1"),
                    _text_response("5"),
                ]
            ),
            tools=[AddTool()],
        )
        await agent.run(chat_inputs="2+3")

        (tool,) = _spans_of_kind("tool")
        attrs = tool.attributes or {}
        assert tool.name == "add"
        assert attrs[ATTR_OI_SPAN_KIND] == "TOOL"
        assert attrs[attributes.ATTR_TOOL_NAME] == "add"
        assert attrs[attributes.ATTR_AGENT_NAME] == "calc"
        assert json.loads(str(attrs[ATTR_INPUT_VALUE])) == {"a": 2, "b": 3}
        assert attrs[ATTR_OUTPUT_VALUE] == "5"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_parallel_replicas_are_named_after_their_template(self) -> None:
        parallel = ParallelProcessor(_FlakyProcessor("echo", fail_times=0))
        await parallel.run(in_args=["a", "b"])

        runs = _spans_of_kind("processor")
        replicas = [s for s in runs if s.name == "echo"]
        assert sorted(
            (s.attributes or {})[attributes.ATTR_PROCESSOR_REPLICA] for s in replicas
        ) == [0, 1]
        assert {(s.attributes or {})[ATTR_INPUT_VALUE] for s in replicas} == {
            '"a"',
            '"b"',
        }
        (container,) = [s for s in runs if s.name == "echo_par"]
        assert (container.attributes or {})[attributes.ATTR_PROCESSOR_CLASS] == (
            "ParallelProcessor"
        )
        assert (container.attributes or {})[ATTR_OUTPUT_VALUE] == '["a!", "b!"]'

    @pytest.mark.asyncio(loop_scope="function")
    async def test_failed_run_records_cause_chain_and_root_type(self) -> None:
        proc = _FlakyProcessor("doomed", fail_times=99, err_msg="empty chunk")
        with pytest.raises(ProcRunError):
            await proc.run(in_args="hi")

        (span,) = _spans_of_kind("processor")
        assert span.status.status_code is trace.StatusCode.ERROR
        description = span.status.description or ""
        assert description.startswith("ProcRunError: ")
        assert "Caused by: ValueError: empty chunk" in description
        assert (span.attributes or {})[attributes.ATTR_ERROR_TYPE] == "ValueError"
        assert ATTR_OUTPUT_VALUE not in (span.attributes or {})


class TestInheritedSpanAttributes:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_stamped_on_every_span_started_in_the_block(self) -> None:
        agent = LLMAgent[str, str, None](
            name="writer",
            llm=MockLLM(responses_queue=[_text_response("a"), _text_response("b")]),
        )
        with inherited_span_attributes({"grasp.eval.run_id": "r1"}):
            with inherited_span_attributes({"grasp.eval.example_id": "e1"}):
                await agent.run(chat_inputs="one")
        inside = _exporter.get_finished_spans()
        _exporter.clear()
        await agent.run(chat_inputs="two")

        assert len(inside) >= 2
        for span in inside:
            attrs = span.attributes or {}
            assert attrs["grasp.eval.run_id"] == "r1"
            assert attrs["grasp.eval.example_id"] == "e1"
        assert all(
            "grasp.eval.run_id" not in (s.attributes or {})
            for s in _exporter.get_finished_spans()
        )


class TestFitJson:
    def test_short_value_is_unchanged(self) -> None:
        assert fit_json({"a": [1, 2]}, 100) == ('{"a": [1, 2]}', False)
        assert fit_json({"a": "x" * 500}, None) == (json.dumps({"a": "x" * 500}), False)

    def test_long_strings_are_cut_inside_valid_json(self) -> None:
        value = {"error": "chunk misaligned", "draft": "D" * 5000, "n": 3}
        text, truncated = fit_json(value, 400)
        parsed = json.loads(text)
        assert truncated
        assert len(text) <= 400
        assert parsed["error"] == "chunk misaligned"
        assert parsed["n"] == 3
        assert parsed["draft"].startswith("D")
        assert "chars]" in parsed["draft"]

    def test_long_lists_keep_head_and_tail(self) -> None:
        text, truncated = fit_json(list(range(1000)), 120)
        parsed = json.loads(text)
        assert truncated
        assert len(text) <= 120
        assert parsed[0] == 0
        assert parsed[-1] == 999
        assert any(isinstance(v, str) and "items]" in v for v in parsed)

    def test_short_lists_survive_when_a_long_string_is_cut(self) -> None:
        # A model's output items: the answer text is cut, no item is dropped.
        items = [{"type": "reasoning"}, {"text": "A" * 5000}, {"type": "call"}]
        text, truncated = fit_json(items, 1000)
        parsed = json.loads(text)
        assert truncated
        assert len(text) <= 1000
        assert [list(item) for item in parsed] == [["type"], ["text"], ["type"]]
        assert len(parsed[1]["text"]) > 800
        one_over = {"note": "x" * 100, "ids": [101, 102, 103]}
        text, _ = fit_json(one_over, len(json.dumps(one_over)) - 1)
        assert json.loads(text)["ids"] == [101, 102, 103]

    def test_wide_mappings_keep_their_head_and_tail_keys(self) -> None:
        value = {f"key_{i}": i for i in range(200)}
        text, truncated = fit_json(value, 80)
        parsed = json.loads(text)
        assert truncated
        assert len(text) <= 80
        assert parsed["key_0"] == 0
        assert parsed["key_199"] == 199
        assert parsed["…"] == f"[{200 - len(parsed) + 1} more keys]"

    def test_structure_too_large_becomes_a_json_string(self) -> None:
        value = [{"a": [{"b": i}]} for i in range(50)]
        text, truncated = fit_json(value, 20)
        assert truncated
        assert len(text) <= 20
        assert isinstance(json.loads(text), str)

    def test_non_finite_numbers_stay_valid_json(self) -> None:
        @traced(name="scores")
        def scores() -> list[float]:
            return [float("nan"), float("inf"), 1.5]

        scores()
        attrs = _exporter.get_finished_spans()[0].attributes or {}
        assert json.loads(str(attrs[ATTR_OUTPUT_VALUE])) == ["NaN", "Infinity", 1.5]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_span_payload_marked_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The provider's limit, however it was configured (env or code).
        monkeypatch.setattr(
            _provider.get_tracer("grasp_agents"),
            "_span_limits",
            SpanLimits(max_span_attribute_length=200),
        )

        @traced(name="big")
        async def big(text: str) -> dict[str, str]:
            return {"text": text}

        await big("x" * 1000)

        attrs = _exporter.get_finished_spans()[0].attributes or {}
        assert attrs[attributes.ATTR_INPUT_TRUNCATED] is True
        assert attrs[attributes.ATTR_OUTPUT_TRUNCATED] is True
        assert json.loads(str(attrs[ATTR_OUTPUT_VALUE]))["text"].startswith("x")


class TestOnlineEvaluationRoundTrip:
    """Production spans carry what online evaluations read back."""

    @staticmethod
    def _record(span: ReadableSpan) -> SpanRecord:
        context = span.get_span_context()
        return SpanRecord(
            trace_id=format(context.trace_id, "032x"),
            span_id=format(context.span_id, "016x"),
            name=span.name,
            start_time=datetime.fromtimestamp((span.start_time or 0) / 1e9, UTC),
            status="ERROR"
            if span.status.status_code is trace.StatusCode.ERROR
            else "UNSET",
            status_message=span.status.description,
            attributes=dict(span.attributes or {}),
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_agent_spans_extract_into_inputs_and_outputs(self) -> None:
        class Question(BaseModel):
            topic: str

        agent = LLMAgent[Question, str, None](
            name="tutor", llm=MockLLM(responses_queue=[_text_response("an answer")])
        )
        await agent.run(in_args=Question(topic="fractions"))

        (run,) = _spans_of_kind("agent")
        record = self._record(run)
        extracted = default_extractor(
            TraceItem(scope="span", id=record.span_id, spans=(record,))
        )
        assert extracted.input == {"topic": "fractions"}
        assert extracted.output == "an answer"
        assert extracted.error is None
        assert Question.model_validate(extracted.input) == Question(topic="fractions")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_failed_run_extracts_as_a_task_error(self) -> None:
        proc = _FlakyProcessor("doomed", fail_times=99, err_msg="empty chunk")
        with pytest.raises(ProcRunError):
            await proc.run(in_args="hi")

        (span,) = _spans_of_kind("processor")
        record = self._record(span)
        extracted = default_extractor(
            TraceItem(scope="span", id=record.span_id, spans=(record,))
        )
        assert extracted.input == "hi"
        assert extracted.error is not None
        assert extracted.error.type == "ValueError"
        assert "empty chunk" in extracted.error.message


def test_a_class_level_version_is_the_declared_version() -> None:
    class Versioned(_FlakyProcessor):
        version = "7"

    assert Versioned("v", fail_times=0).version == "7"
    assert Versioned("v", fail_times=0, version="8").version == "8"
    assert _FlakyProcessor("plain", fail_times=0).version is None


# ========================================================================= #
#  Review regressions                                                        #
# ========================================================================= #


class _Ask(BaseModel):
    topic: str
    secret: str


class TestTracingRegressions:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_traced_helper_of_an_agent_subclass_is_not_a_run(self) -> None:
        class Writer(LLMAgent[str, str, None]):
            @traced(name="score_draft")
            async def score_draft(self, draft: str) -> int:
                return len(draft)

        await Writer(name="writer", llm=MockLLM()).score_draft("my draft")

        (span,) = _exporter.get_finished_spans()
        attrs = span.attributes or {}
        assert span.name == "writer.score_draft"
        assert attrs[ATTR_SPAN_KIND] == "task"
        assert attributes.ATTR_PROCESSOR_NAME not in attrs
        assert attrs[ATTR_INPUT_VALUE] == '{"draft": "my draft"}'
        assert attrs[ATTR_OUTPUT_VALUE] == "8"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_traced_helper_of_a_tool_is_not_a_call(self) -> None:
        class Lookup(AddTool):
            @traced(name="fetch_page")
            async def fetch_page(self, url: str) -> str:
                return url

        await Lookup().fetch_page("https://x")

        (span,) = _exporter.get_finished_spans()
        assert span.name == "add.fetch_page"
        assert (span.attributes or {})[ATTR_SPAN_KIND] == "task"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_odd_hook_results_never_break_the_call(self) -> None:
        class Default:
            def _trace_span_start(self, *_: Any) -> None:
                return None

            @traced(name="run")
            async def run(self) -> str:
                return "ok"

        class Wrong:
            def _trace_span_start(self, *_: Any) -> str:
                return "not a start"

            def _trace_span_result(self, *_: Any) -> int:
                return 42

            @traced(name="run")
            async def run(self) -> str:
                return "ok"

        class PlainKind:
            def _trace_span_start(self, *_: Any) -> SpanStart:
                return SpanStart(name="plain", kind="agent")  # type: ignore[arg-type]

            @traced(name="run")
            async def run(self) -> str:
                return "ok"

        assert await Default().run() == "ok"
        assert await Wrong().run() == "ok"
        assert await PlainKind().run() == "ok"
        by_name = {s.name: s.attributes or {} for s in _exporter.get_finished_spans()}
        assert by_name["plain"][ATTR_SPAN_KIND] == "agent"
        assert by_name["run"][ATTR_SPAN_KIND] == "task"

    def test_a_payload_that_cannot_be_shown_does_not_fail_the_call(self) -> None:
        class Unshowable:
            def __str__(self) -> str:
                raise RuntimeError("no")

        @traced(name="show")
        def show(value: Any) -> Any:
            return value

        value = Unshowable()
        assert show(value) is value
        (span,) = _exporter.get_finished_spans()
        assert [e.name for e in span.events] == ["exception", "exception"]
        assert ATTR_INPUT_VALUE not in (span.attributes or {})

    def test_an_invalid_item_result_keeps_the_last_good_one(self) -> None:
        from grasp_agents.telemetry import SpanResult  # noqa: PLC0415

        class Stream:
            def _trace_span_result(self, entity: str, item: Any) -> Any:
                return SpanResult(output=item) if item == "good" else 42

            @traced(name="stream")
            def stream(self) -> Iterator[str]:
                yield "good"
                yield "ignored"

        assert list(Stream().stream()) == ["good", "ignored"]
        attrs = _exporter.get_finished_spans()[0].attributes or {}
        assert attrs[ATTR_OUTPUT_VALUE] == '"good"'

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_failed_tool_call_is_an_error(self) -> None:
        class Broken(AddTool):
            async def _run(self, inp: Any, **_: Any) -> int:
                raise KeyError("missing key")

        result = await Broken().run(AddInput(a=1, b=2))

        assert isinstance(result, ToolErrorInfo)
        (span,) = _spans_of_kind("tool")
        assert span.status.status_code is trace.StatusCode.ERROR
        assert (span.attributes or {})[attributes.ATTR_ERROR_TYPE] == "KeyError"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_tool_calls_record_every_input_field(self) -> None:
        class PathInput(BaseModel):
            path: str

        class Reader(BaseTool[PathInput, str, Any]):
            def __init__(self) -> None:
                super().__init__(name="read", description="Read a file.")

            async def _run(self, inp: PathInput, **_: Any) -> str:
                return inp.path

        await Reader()(path="/notes.md")

        (span,) = _spans_of_kind("tool")
        assert json.loads(str((span.attributes or {})[ATTR_INPUT_VALUE])) == {
            "path": "/notes.md"
        }

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_session_under_a_caller_span_keeps_the_callers_trace(
        self,
    ) -> None:
        with trace.get_tracer("app").start_as_current_span("request") as request:
            with SessionContext[None](session_key="sess-req"):
                agent = LLMAgent[str, str, None](
                    name="chat", llm=MockLLM(responses_queue=[_text_response("ok")])
                )
                await agent.run(chat_inputs="hi")

        (run,) = _spans_of_kind("agent")
        assert run.context.trace_id == request.get_span_context().trace_id
        assert (run.attributes or {})["session.id"] == "sess-req"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_sampled_out_caller_keeps_a_session_unsampled(self) -> None:
        parent = trace.NonRecordingSpan(
            trace.SpanContext(
                trace_id=7, span_id=7, is_remote=True, trace_flags=trace.TraceFlags(0)
            )
        )
        token = otel_context.attach(trace.set_span_in_context(parent))
        try:
            with SessionContext[None](session_key="sess-unsampled"):
                agent = LLMAgent[str, str, None](
                    name="chat", llm=MockLLM(responses_queue=[_text_response("ok")])
                )
                await agent.run(chat_inputs="hi")
        finally:
            otel_context.detach(token)

        assert _exporter.get_finished_spans() == []

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_wrapped_agents_masked_fields_stay_masked_on_its_tool(
        self,
    ) -> None:
        sub = LLMAgent[_Ask, str, None](
            name="sub",
            llm=MockLLM(responses_queue=[_text_response("done")]),
            tracing_exclude_input_fields={"secret"},
        )
        tool = sub.as_tool("ask_sub", "Ask the sub-agent.")
        await tool.run(_Ask(topic="x", secret="TOPSECRET"))

        for span in _exporter.get_finished_spans():
            assert "TOPSECRET" not in str((span.attributes or {}).get(ATTR_INPUT_VALUE))

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_cancelled_call_is_marked(self) -> None:
        @traced(name="slow")
        async def slow() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(slow())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        (span,) = _exporter.get_finished_spans()
        assert (span.attributes or {})[attributes.ATTR_CANCELLED] is True
        assert span.status.status_code is not trace.StatusCode.ERROR

    def test_a_function_whose_argument_has_its_name_records_it(self) -> None:
        @traced(name="split")
        def split(text: str) -> list[str]:
            return text.split()

        split("a b")
        attrs = _exporter.get_finished_spans()[0].attributes or {}
        assert attrs[ATTR_INPUT_VALUE] == '{"text": "a b"}'

    def test_init_tracing_marks_an_existing_provider_once(self) -> None:
        from grasp_agents.telemetry import setup

        provider = TracerProvider()
        with patch.object(setup.trace, "get_tracer_provider", return_value=provider):
            assert setup.init_tracing() is provider
            setup.init_tracing()
        processors = provider._active_span_processor._span_processors  # type: ignore[attr-defined]
        assert (
            sum(isinstance(p, InheritedAttributesSpanProcessor) for p in processors)
            == 1
        )
