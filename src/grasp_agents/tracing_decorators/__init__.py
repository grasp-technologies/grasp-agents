from collections.abc import Callable

from opentelemetry.semconv_ai import TraceloopSpanKindValues

from .base import F, T, entity_class, entity_method


def task(
    name: str | None = None,
    version: int | None = None,
    method_name: str | None = None,
    tlp_span_kind: TraceloopSpanKindValues = TraceloopSpanKindValues.TASK,
) -> Callable[[F], F] | Callable[[T], T]:
    if method_name is None:
        return entity_method(name=name, version=version, tlp_span_kind=tlp_span_kind)
    return entity_class(
        name=name,
        version=version,
        method_name=method_name,
        tlp_span_kind=tlp_span_kind,
    )


def workflow(
    name: str | None = None,
    version: int | None = None,
    method_name: str | None = None,
    tlp_span_kind: TraceloopSpanKindValues = TraceloopSpanKindValues.WORKFLOW,
) -> Callable[[F], F] | Callable[[T], T]:
    if method_name is None:
        return entity_method(name=name, version=version, tlp_span_kind=tlp_span_kind)
    return entity_class(
        name=name,
        version=version,
        method_name=method_name,
        tlp_span_kind=tlp_span_kind,
    )


def agent(
    name: str | None = None,
    version: int | None = None,
    method_name: str | None = None,
) -> Callable[[F], F] | Callable[[T], T]:
    return workflow(
        name=name,
        version=version,
        method_name=method_name,
        tlp_span_kind=TraceloopSpanKindValues.AGENT,
    )


def tool(
    name: str | None = None,
    version: int | None = None,
    method_name: str | None = None,
) -> Callable[[F], F] | Callable[[T], T]:
    return task(
        name=name,
        version=version,
        method_name=method_name,
        tlp_span_kind=TraceloopSpanKindValues.TOOL,
    )
