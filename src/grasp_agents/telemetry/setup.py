"""
Telemetry setup helpers.

Provides TracerProvider initialization, LLM auto-instrumentation,
and convenience functions for common OTel exporters.
"""

import threading
from logging import getLogger
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

logger = getLogger(__name__)

_init_lock = threading.Lock()
_initialized = False


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
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )

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
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    exporter_kwargs: dict[str, Any] = {}
    if endpoint is not None:
        exporter_kwargs["endpoint"] = endpoint
    if headers is not None:
        exporter_kwargs["headers"] = headers
    add_exporter(OTLPSpanExporter(**exporter_kwargs), provider=provider, batch=batch)
