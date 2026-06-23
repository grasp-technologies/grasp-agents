"""Tests for the tracing decorator system and telemetry setup."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.run_context import RunContext
from grasp_agents.telemetry import (
    SessionSpanProcessor,
    SpanKind,
    derive_session_span_context,
    set_run_span_attributes,
    traced,
)
from grasp_agents.telemetry.decorators import (
    ATTR_ENTITY_INPUT,
    ATTR_ENTITY_NAME,
    ATTR_ENTITY_OUTPUT,
    ATTR_OI_SPAN_KIND,
    ATTR_SPAN_KIND,
    ATTR_WORKFLOW_NAME,
    _resolve_run_span_context,
    _resolve_span_kind,
    _should_send_prompts,
    _to_plain,
    _truncate_if_needed,
)
from grasp_agents.tools.agent_tool import AgentTool
from tests._helpers import MockLLM, _text_response, _tool_call_response

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
_provider.add_span_processor(SessionSpanProcessor())  # stamps session attrs
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

        spans = _exporter.get_finished_spans()
        by_name = {s.attributes[ATTR_ENTITY_NAME]: s for s in spans}  # type: ignore[index]
        assert by_name["wf"].attributes[ATTR_OI_SPAN_KIND] == "CHAIN"
        assert by_name["ag"].attributes[ATTR_OI_SPAN_KIND] == "AGENT"
        assert by_name["tl"].attributes[ATTR_OI_SPAN_KIND] == "TOOL"
        assert by_name["tk"].attributes[ATTR_OI_SPAN_KIND] == "CHAIN"

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

    def test_truncate_tiny_limit_falls_back_to_head_clip(self) -> None:
        # Limit too small to fit a head…tail marker → plain head clip.
        with patch.dict(os.environ, {"OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT": "5"}):
            assert _truncate_if_needed("hello world") == "hello"
            assert _truncate_if_needed("hi") == "hi"

    def test_truncate_keeps_head_and_tail(self) -> None:
        with patch.dict(os.environ, {"OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT": "60"}):
            text = "H" * 40 + "X" * 200 + "T" * 40
            out = _truncate_if_needed(text)
            assert len(out) <= 60  # never exceeds the limit
            assert out.startswith("H")  # head kept
            assert out.endswith("T")  # tail kept
            assert "chars]" in out  # head…tail marker present
            assert "X" * 20 not in out  # middle dropped

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
        assert spans[0].attributes.get(ATTR_ENTITY_OUTPUT) is None
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
        assert span.name == "run.task"
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


def _processor_spans() -> list[ReadableSpan]:
    """Run-root spans (one per run_stream), selected by entity name."""
    return [
        s
        for s in _exporter.get_finished_spans()
        if (s.attributes or {}).get(ATTR_ENTITY_NAME) == "processor"
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

    def test_resolver_skips_when_nested_under_recording_span(self) -> None:
        class Named:
            def _trace_session_info(self) -> tuple[str, bool] | None:
                return ("sess-A", True)

        # A live ancestor span means we are nested — inherit the enclosing run's
        # context instead of starting a fresh session root.
        with trace.get_tracer("test").start_as_current_span("outer"):
            assert _resolve_run_span_context(Named()) is None


class TestSessionTraceGrouping:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_named_session_runs_share_one_trace_without_a_store(self) -> None:
        # No checkpoint_store wired — grouping is purely from the session_key.
        with RunContext[None](session_key="sess-multi-turn"):
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
        with RunContext[None](
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
        with RunContext[None](session_key="sess-nested"):
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
#  Session attributes (SessionSpanProcessor — propagated to every span)      #
# ========================================================================= #


def _session_attr(span: ReadableSpan, key: str = "gen_ai.conversation.id") -> Any:
    return (span.attributes or {}).get(key)


async def _run_chat(session_key: str | None = None, **ctx_kwargs: Any) -> None:
    llm = MockLLM(responses_queue=[_text_response("ok")])
    if session_key is None:
        agent = LLMAgent[str, str, None](name="chat", llm=llm)
        await agent.run(chat_inputs="hi")
        return
    with RunContext[None](session_key=session_key, **ctx_kwargs):
        agent = LLMAgent[str, str, None](name="chat", llm=llm)
        await agent.run(chat_inputs="hi")


class TestSessionAttributes:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_stamped_on_run_and_child_spans(self) -> None:
        await _run_chat("sess-attr")
        spans = _exporter.get_finished_spans()
        by_entity = {(s.attributes or {}).get(ATTR_ENTITY_NAME): s for s in spans}
        # The run root (processor) AND a child it created (generate) both carry
        # the session id — the processor propagates it down the whole run tree.
        assert "processor" in by_entity
        assert "generate" in by_entity
        assert _session_attr(by_entity["processor"]) == "sess-attr"
        assert _session_attr(by_entity["generate"]) == "sess-attr"

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
        root = next(
            s for s in spans if _session_attr(s, ATTR_ENTITY_NAME) == "processor"
        )
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
