"""
Telemetry setup helpers.

Provides TracerProvider initialization, LLM auto-instrumentation,
and convenience functions for common OTel exporters.
"""

import threading
from logging import getLogger
from typing import Any
from weakref import WeakSet

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span

from .decorators import stamp_inherited_attributes

logger = getLogger(__name__)

_init_lock = threading.Lock()


class InheritedAttributesSpanProcessor(SpanProcessor):
    """
    Stamp inherited attributes onto every span when it starts: the run's
    session id (under each key in ``GRASP_SESSION_ID_ATTRIBUTES``, default
    ``session.id`` and ``gen_ai.conversation.id``) and attributes set with
    :func:`~grasp_agents.telemetry.inherited_span_attributes`.

    :func:`init_tracing` installs it; add it to a hand-built ``TracerProvider``
    so ALL spans carry them — including provider-instrumentation spans — for
    backends that group or filter by them per span.
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        stamp_inherited_attributes(span, parent_context)


def init_tracing(project_name: str = "grasp-agents") -> TracerProvider:
    """
    Set up a basic TracerProvider with the given service name.

    This makes grasp-agents tracing decorators emit real spans. By default
    no exporter is attached -- add one via add_exporter() or init_phoenix().
    An already configured provider is kept (and given an
    :class:`InheritedAttributesSpanProcessor` if it has none from here).

    Returns the TracerProvider so callers can attach exporters/processors.
    """
    with _init_lock:
        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            _add_inherited_attributes(existing)
            return existing

        provider = TracerProvider(
            resource=Resource.create(
                {
                    SERVICE_NAME: project_name,
                    "openinference.project.name": project_name,
                }
            ),
        )
        _add_inherited_attributes(provider)
        trace.set_tracer_provider(provider)
        logger.info("Initialized TracerProvider for %s", project_name)
        return provider


_stamping: WeakSet[TracerProvider] = WeakSet()


def _add_inherited_attributes(provider: TracerProvider) -> None:
    # Propagates the session id and inherited attributes onto every span.
    if provider not in _stamping:
        provider.add_span_processor(InheritedAttributesSpanProcessor())
        _stamping.add(provider)


def add_exporter(
    exporter: SpanExporter,
    provider: TracerProvider | None = None,
    batch: bool = True,
) -> None:
    """
    Add a span exporter to the TracerProvider.

    Args:
        exporter: Any OTel-compatible SpanExporter.
        provider: TracerProvider to attach to. Uses the global one if None.
        batch: Use BatchSpanProcessor (True) or SimpleSpanProcessor (False).

    """
    if provider is None:
        existing = trace.get_tracer_provider()
        if not isinstance(existing, TracerProvider):
            msg = "No TracerProvider configured. Call init_tracing() first."
            raise RuntimeError(msg)
        provider = existing

    processor = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)


def add_otlp_http_exporter(
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    provider: TracerProvider | None = None,
    batch: bool = True,
) -> None:
    """
    Add an OTLP/HTTP exporter. Works with Jaeger, Tempo, Datadog, Langfuse, etc.

    Requires: pip install opentelemetry-exporter-otlp-proto-http
    """
    # Deferred: needs the optional otlp-proto-http exporter package.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
        OTLPSpanExporter,
    )

    exporter_kwargs: dict[str, Any] = {}
    if endpoint is not None:
        exporter_kwargs["endpoint"] = endpoint
    if headers is not None:
        exporter_kwargs["headers"] = headers
    add_exporter(OTLPSpanExporter(**exporter_kwargs), provider=provider, batch=batch)
