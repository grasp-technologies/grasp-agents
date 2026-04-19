"""
Tracing decorators using raw OpenTelemetry API.

Follows the OTel library instrumentation pattern: depends only on opentelemetry-api.
If no TracerProvider is configured by the application, all spans are no-ops.
"""

import inspect
import json
import os
import traceback
from collections.abc import Callable
from enum import Enum
from functools import wraps
from logging import getLogger
from typing import Any, TypeVar, cast, overload

from opentelemetry import trace
from pydantic import BaseModel

logger = getLogger(__name__)

# ---------------------------------------------------------------------------
# Span kind values
# ---------------------------------------------------------------------------


class SpanKind(str, Enum):
    WORKFLOW = "workflow"
    TASK = "task"
    AGENT = "agent"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Span attribute keys
# Align with gen_ai.* where applicable; use grasp.* for framework-specific.
# ---------------------------------------------------------------------------

ATTR_SPAN_KIND = "grasp.span.kind"
ATTR_ENTITY_NAME = "grasp.entity.name"
ATTR_ENTITY_VERSION = "grasp.entity.version"
ATTR_ENTITY_INPUT = "grasp.entity.input"
ATTR_ENTITY_OUTPUT = "grasp.entity.output"
ATTR_WORKFLOW_NAME = "grasp.workflow.name"

# OpenInference compatibility — Phoenix uses this to show span type icons
ATTR_OI_SPAN_KIND = "openinference.span.kind"
_GRASP_TO_OI_KIND: dict[str, str] = {
    "workflow": "CHAIN",
    "task": "CHAIN",
    "agent": "AGENT",
    "tool": "TOOL",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDE_FIELDS = {"_hidden_params", "completions"}
_TRACER_NAME = "grasp_agents"

T = TypeVar("T", bound=type)
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_plain(obj: Any, exclude_fields: set[str] | None = None) -> Any:
    """Recursively convert objects to JSON-serializable primitives."""
    all_exclude = DEFAULT_EXCLUDE_FIELDS.union(exclude_fields or set())
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump(exclude=all_exclude)
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {
            str(k): _to_plain(v, exclude_fields)
            for k, v in cast("dict[Any, Any]", obj).items()
            if str(k) not in all_exclude
        }
    if isinstance(obj, (tuple, list, set)):
        return [
            _to_plain(v, exclude_fields)
            for v in cast("list[Any] | tuple[Any, ...] | set[Any]", obj)
        ]
    return obj


def _truncate_if_needed(json_str: str) -> str:
    limit_str = os.getenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT")
    if limit_str:
        try:
            limit = int(limit_str)
            if limit > 0 and len(json_str) > limit:
                return json_str[:limit]
        except ValueError:
            pass
    return json_str


def _should_send_prompts() -> bool:
    val = os.getenv("GRASP_TRACE_CONTENT") or os.getenv("TRACELOOP_TRACE_CONTENT")
    return (val or "true").lower() == "true"


def _tracing_enabled(instance: Any | None = None) -> bool:
    if instance is None:
        return True
    return bool(getattr(instance, "tracing_enabled", True))


def _exclude_fields_from_instance(instance: Any | None = None) -> set[str] | None:
    if instance is None:
        return None
    fields: set[str] | None = getattr(instance, "tracing_exclude_input_fields", None)
    return set(fields) if fields else None


# ---------------------------------------------------------------------------
# Span construction
# ---------------------------------------------------------------------------


def _get_span_name(
    entity_name: str,
    span_kind: SpanKind,
    instance: Any | None = None,
    kwargs: dict[str, Any] | None = None,
) -> str:
    instance_name = None
    if instance is not None:
        if span_kind in {SpanKind.WORKFLOW, SpanKind.AGENT, SpanKind.TOOL}:
            instance_name = getattr(instance, "name", None)
        elif span_kind == SpanKind.TASK and entity_name == "generate":
            instance_name = getattr(instance, "agent_name", None)

    if instance_name:
        exec_id = (kwargs or {}).get("exec_id")
        suffix = f"[{exec_id}]" if exec_id else ""
        return f"{instance_name}.{entity_name}{suffix}"
    return f"{entity_name}.{span_kind.value}"


def _set_span_attributes(
    span: trace.Span,
    entity_name: str,
    span_kind: SpanKind,
    version: int | None,
) -> None:
    span.set_attribute(ATTR_SPAN_KIND, span_kind.value)
    span.set_attribute(ATTR_ENTITY_NAME, entity_name)
    oi_kind = _GRASP_TO_OI_KIND.get(span_kind.value)
    if oi_kind:
        span.set_attribute(ATTR_OI_SPAN_KIND, oi_kind)
    if span_kind in {SpanKind.WORKFLOW, SpanKind.AGENT}:
        span.set_attribute(ATTR_WORKFLOW_NAME, entity_name)
    if version:
        span.set_attribute(ATTR_ENTITY_VERSION, version)


def _handle_span_input(
    span: trace.Span,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    exclude_fields: set[str] | None = None,
) -> None:
    if not span.is_recording():
        return
    try:
        if _should_send_prompts():
            json_input = json.dumps(
                {
                    "args": _to_plain(list(args), exclude_fields=exclude_fields),
                    "kwargs": _to_plain(kwargs, exclude_fields=exclude_fields),
                },
                default=str,
                indent=2,
            )
            span.set_attribute(ATTR_ENTITY_INPUT, _truncate_if_needed(json_input))
    except TypeError as e:
        span.record_exception(e)


def _handle_span_output(
    span: trace.Span,
    res: Any,
    exclude_fields: set[str] | None = None,
) -> None:
    if not span.is_recording():
        return
    try:
        if _should_send_prompts():
            json_output = json.dumps(
                _to_plain(res, exclude_fields=exclude_fields),
                default=str,
                indent=2,
            )
            span.set_attribute(ATTR_ENTITY_OUTPUT, _truncate_if_needed(json_output))
    except TypeError as e:
        span.record_exception(e)


def _resolve_span_kind(instance: Any | None, default: SpanKind) -> SpanKind:
    """Resolve span kind from instance ``_span_kind`` attribute, else use default."""
    if instance is not None:
        kind = getattr(instance, "_span_kind", None)
        if isinstance(kind, SpanKind):
            return kind
    return default


def _is_bound_method(func: Callable[..., Any], self_candidate: Any) -> bool:
    return (inspect.ismethod(func) and (func.__self__ is self_candidate)) or hasattr(
        self_candidate, func.__name__
    )


def _is_async(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


def _camel_to_snake(name: str) -> str:
    import re

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# ---------------------------------------------------------------------------
# Internal decorator factories
# ---------------------------------------------------------------------------


def _entity_method(
    name: str | None = None,
    version: int | None = None,
    span_kind: SpanKind = SpanKind.TASK,
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        is_async = _is_async(fn)
        entity_name = name or fn.__qualname__

        if is_async:
            if inspect.isasyncgenfunction(fn):

                @wraps(fn)
                async def async_gen_wrap(*args: Any, **kwargs: Any) -> Any:
                    is_bound = _is_bound_method(fn, args[0] if args else False)
                    instance = args[0] if is_bound else None
                    input_args = args[1:] if is_bound else args
                    exclude_fields = _exclude_fields_from_instance(instance)

                    if not _tracing_enabled(instance):
                        async for item in fn(*args, **kwargs):
                            yield item
                        return

                    resolved_kind = _resolve_span_kind(instance, span_kind)
                    span_name = _get_span_name(
                        entity_name, resolved_kind, instance=instance, kwargs=kwargs
                    )
                    tracer = trace.get_tracer(_TRACER_NAME)

                    with tracer.start_as_current_span(span_name) as span:
                        _set_span_attributes(span, entity_name, resolved_kind, version)
                        _handle_span_input(span, input_args, kwargs, exclude_fields)
                        items: list[Any] = []

                        try:
                            async for item in fn(*args, **kwargs):
                                items.append(item)
                                yield item
                        except Exception as e:
                            span.set_status(
                                trace.Status(trace.StatusCode.ERROR, str(e))
                            )
                            span.record_exception(
                                e, attributes={"tb": traceback.format_exc()}
                            )
                            raise
                        finally:
                            if items:
                                _handle_span_output(span, items[-1])

                return cast("F", async_gen_wrap)

            @wraps(fn)
            async def async_wrap(*args: Any, **kwargs: Any) -> Any:
                is_bound = _is_bound_method(fn, args[0] if args else False)
                instance = args[0] if is_bound else None
                input_args = args[1:] if is_bound else args
                exclude_fields = _exclude_fields_from_instance(instance)

                if not _tracing_enabled(instance):
                    return await fn(*args, **kwargs)

                resolved_kind = _resolve_span_kind(instance, span_kind)
                span_name = _get_span_name(
                    entity_name, resolved_kind, instance=instance, kwargs=kwargs
                )
                tracer = trace.get_tracer(_TRACER_NAME)

                with tracer.start_as_current_span(span_name) as span:
                    _set_span_attributes(span, entity_name, resolved_kind, version)
                    _handle_span_input(span, input_args, kwargs, exclude_fields)
                    try:
                        res = await fn(*args, **kwargs)
                        _handle_span_output(span, res)
                        return res
                    except Exception as e:
                        span.set_status(
                            trace.Status(trace.StatusCode.ERROR, str(e))
                        )
                        span.record_exception(
                            e, attributes={"tb": traceback.format_exc()}
                        )
                        raise

            return cast("F", async_wrap)

        # --- Sync paths ---

        if inspect.isgeneratorfunction(fn):

            @wraps(fn)
            def sync_gen_wrap(*args: Any, **kwargs: Any) -> Any:
                is_bound = _is_bound_method(fn, args[0] if args else False)
                instance = args[0] if is_bound else None
                input_args = args[1:] if is_bound else args
                exclude_fields = _exclude_fields_from_instance(instance)

                if not _tracing_enabled(instance):
                    yield from fn(*args, **kwargs)
                    return

                resolved_kind = _resolve_span_kind(instance, span_kind)
                span_name = _get_span_name(
                    entity_name, resolved_kind, instance=instance, kwargs=kwargs
                )
                tracer = trace.get_tracer(_TRACER_NAME)

                with tracer.start_as_current_span(span_name) as span:
                    _set_span_attributes(span, entity_name, resolved_kind, version)
                    _handle_span_input(span, input_args, kwargs, exclude_fields)
                    items: list[Any] = []

                    try:
                        for item in fn(*args, **kwargs):
                            items.append(item)
                            yield item
                    except Exception as e:
                        span.set_status(
                            trace.Status(trace.StatusCode.ERROR, str(e))
                        )
                        span.record_exception(
                            e, attributes={"tb": traceback.format_exc()}
                        )
                        raise
                    finally:
                        if items:
                            _handle_span_output(span, items[-1])

            return cast("F", sync_gen_wrap)

        @wraps(fn)
        def sync_wrap(*args: Any, **kwargs: Any) -> Any:
            is_bound = _is_bound_method(fn, args[0] if args else False)
            instance = args[0] if is_bound else None
            input_args = args[1:] if is_bound else args
            exclude_fields = _exclude_fields_from_instance(instance)

            if not _tracing_enabled(instance):
                return fn(*args, **kwargs)

            resolved_kind = _resolve_span_kind(instance, span_kind)
            span_name = _get_span_name(
                entity_name, resolved_kind, instance=instance, kwargs=kwargs
            )
            tracer = trace.get_tracer(_TRACER_NAME)

            with tracer.start_as_current_span(span_name) as span:
                _set_span_attributes(span, entity_name, resolved_kind, version)
                _handle_span_input(span, input_args, kwargs, exclude_fields)
                try:
                    res = fn(*args, **kwargs)
                    _handle_span_output(span, res)
                    return res
                except Exception as e:
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, str(e))
                    )
                    span.record_exception(
                        e, attributes={"tb": traceback.format_exc()}
                    )
                    raise

        return cast("F", sync_wrap)

    return decorate


def _entity_class(
    name: str | None,
    version: int | None,
    method_name: str,
    span_kind: SpanKind = SpanKind.TASK,
) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        task_name = name or _camel_to_snake(cls.__qualname__)
        method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            _entity_method(name=task_name, version=version, span_kind=span_kind)(
                method
            ),
        )
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@overload
def traced(
    name: str | None = ...,
    version: int | None = ...,
    span_kind: SpanKind = ...,
) -> Callable[[F], F]: ...


@overload
def traced(
    name: str | None = ...,
    version: int | None = ...,
    span_kind: SpanKind = ...,
    *,
    method_name: str,
) -> Callable[[T], T]: ...


def traced(
    name: str | None = None,
    version: int | None = None,
    span_kind: SpanKind = SpanKind.TASK,
    method_name: str | None = None,
) -> Callable[[F], F] | Callable[[T], T]:
    """
    Trace a function or class method with an OTel span.

    Span kind is resolved at call time: ``instance._span_kind`` (if present)
    takes precedence over the *span_kind* argument.  Use the decorator
    parameter for methods decorated directly; use the class attribute for
    inherited methods that need different kinds in subclasses.
    """
    if method_name is None:
        return _entity_method(name=name, version=version, span_kind=span_kind)
    return _entity_class(
        name=name,
        version=version,
        method_name=method_name,
        span_kind=span_kind,
    )
