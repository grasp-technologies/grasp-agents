"""Tracing evaluation runs into Phoenix, so trials carry trace ids."""

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

TRACE_PROJECT_ENV = "GRASP_EVALS_TRACE_PROJECT"
_PROJECT_ATTRIBUTE = "openinference.project.name"


def trace_project() -> str | None:
    """
    The Phoenix project the active tracer provider sends spans to, or
    ``None`` when no recording provider is set up.
    """
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        return None
    name = provider.resource.attributes.get(_PROJECT_ATTRIBUTE)
    return str(name) if name else None


def start_tracing(project: str, *, endpoint: str | None = None) -> str:
    """
    Trace trials into the Phoenix project ``project``: install a tracer
    provider for it and attach the Phoenix exporter, sending to ``endpoint``,
    else ``$TELEMETRY_COLLECTOR_HTTP_ENDPOINT``, else
    ``$PHOENIX_BASE_URL/v1/traces``. When tracing is already set up (e.g. by
    the spec module), its project is kept; one without a Phoenix project is
    an error. Returns the project spans go to.
    """
    existing = trace_project()
    if existing is not None:
        if existing != project:
            logger.warning(
                "Tracing is already set up for project %r; %r is not used",
                existing,
                project,
            )
        return existing
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        raise ValueError(  # ruff: ignore[type-check-without-type-error]
            "Tracing is already set up without a Phoenix project (resource "
            f"attribute {_PROJECT_ATTRIBUTE!r}), so spans would not reach "
            f"{project!r}: set the attribute where tracing is set up instead"
        )
    from grasp_agents.telemetry import (  # ruff: ignore[import-outside-top-level]
        init_tracing,
    )
    from grasp_agents.telemetry.phoenix import (  # ruff: ignore[import-outside-top-level]
        init_phoenix,
    )

    target = endpoint or os.environ.get("TELEMETRY_COLLECTOR_HTTP_ENDPOINT")
    if not target:
        base_url = os.environ.get("PHOENIX_BASE_URL")
        if not base_url:
            raise ValueError(
                "Tracing needs a collector: set TELEMETRY_COLLECTOR_HTTP_ENDPOINT "
                "or PHOENIX_BASE_URL"
            )
        target = f"{base_url.rstrip('/')}/v1/traces"
    init_tracing(project_name=project)
    init_phoenix(project_name=project, collector_endpoint=target)
    return project


def flush_traces() -> None:
    """Export the spans still buffered (before the process exits)."""
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush()
