"""
Telemetry setup helpers.

Provides TracerProvider initialization, LLM auto-instrumentation,
and convenience functions for common OTel exporters.
"""

import threading
from logging import getLogger
from typing import Any

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

from .decorators import stamp_session_attributes

logger = getLogger(__name__)

_init_lock = threading.Lock()
_initialized = False


class SessionSpanProcessor(SpanProcessor):
    """
    Stamp the active session id onto every span as configured attribute(s).

    Reads the session id a run root placed on the OTel context and writes it to
    each key in ``GRASP_SESSION_ID_ATTRIBUTES`` (default ``gen_ai.conversation.id``).
    :func:`init_tracing` installs it automatically; add it to a hand-built
    ``TracerProvider`` to propagate the session id to ALL spans — including
    provider-instrumentation spans — for backends that group or filter by it per
    span (Langfuse, Datadog, …). No-op when no session is active.
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        stamp_session_attributes(span, parent_context)


def init_tracing(project_name: str = "grasp-agents") -> TracerProvider:
    """
    Set up a basic TracerProvider with the given service name.

    This makes grasp-agents tracing decorators emit real spans. By default
    no exporter is attached -- add one via add_exporter() or init_phoenix().

    Returns the TracerProvider so callers can attach exporters/processors.
    """
    global _initialized
    with _init_lock:
        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            return existing

        provider = TracerProvider(
            resource=Resource.create(
                {
                    SERVICE_NAME: project_name,
                    "openinference.project.name": project_name,
                }
            ),
        )
        # Propagates the run's session id onto every span (see the processor).
        provider.add_span_processor(SessionSpanProcessor())
        trace.set_tracer_provider(provider)
        _initialized = True
        logger.info("Initialized TracerProvider for %s", project_name)
        return provider


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
