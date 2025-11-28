import contextlib
import inspect
import json
import os
import traceback
import types
import warnings
from collections.abc import Callable
from functools import wraps
from logging import getLogger
from typing import Any, TypeVar, cast

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.trace.status import Status, StatusCode
from pydantic import BaseModel
from traceloop.sdk.telemetry import Telemetry
from traceloop.sdk.tracing import set_workflow_name
from traceloop.sdk.tracing.tracing import (
    TracerWrapper,
    get_chained_entity_path,
    set_entity_path,
)
from traceloop.sdk.utils import camel_to_snake  # type: ignore[import]
from traceloop.sdk.utils.json_encoder import JSONEncoder

logger = getLogger(__name__)

DEFAULT_EXCLUDE_FIELDS = {"_hidden_params", "completions"}

_CTX_DISABLE_TRACING_KEY = "override_disable_tracing"

T = TypeVar("T", bound=type)
F = TypeVar("F", bound=Callable[..., Any])


def _is_tracing_globally_disabled() -> bool:
    try:
        return bool(context_api.get_value(_CTX_DISABLE_TRACING_KEY))
    except Exception:
        return False


def _to_plain(obj: Any, exclude_fields: set[str] | None = None) -> Any:
    """
    Recursively convert objects to JSON-serializable primitives.

    - Pydantic BaseModel -> dict via model_dump()
    - dict/list/tuple/set -> recurse (sets become lists)
    - other objects -> returned as-is (left to JSONEncoder)
    """
    if isinstance(obj, BaseModel):
        try:
            # TODO: remove this temporary hack
            return obj.model_dump(
                exclude=DEFAULT_EXCLUDE_FIELDS.union(exclude_fields or set())
            )
        except Exception:
            return str(obj)

    if isinstance(obj, dict):
        items_dict = cast("dict[Any, Any]", obj)
        result: dict[str, Any] = {}
        for k, v in items_dict.items():
            result[str(k)] = _to_plain(v)
        return result

    if isinstance(obj, (tuple, list, set)):
        items = cast("tuple[Any, ...] | list[Any] | set[Any]", obj)
        lst: list[Any] = []
        for v in items:
            lst.append(_to_plain(v))
        return lst

    return obj


def _truncate_json_if_needed(json_str: str) -> str:
    """
    Truncate JSON string if it exceeds OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT;
    truncation may yield an invalid JSON string, which is expected for logging purposes.
    """
    limit_str = os.getenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT")

    if limit_str:
        try:
            limit = int(limit_str)
            if limit > 0 and len(json_str) > limit:
                return json_str[:limit]
        except ValueError:
            pass

    return json_str


def _should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


# Quiet wrapper that suppresses prints and warnings from TracerWrapper.verify_initialized
def _tracing_initialized_quietly() -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with (
                open(os.devnull, "w") as devnull,
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                return TracerWrapper.verify_initialized()
        except Exception:
            return False


def _is_async_method(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


def _get_span_name(
    entity_name: str,
    tlp_span_kind: TraceloopSpanKindValues,
    instance: type | None = None,
    kwargs: dict[str, Any] | None = None,
):
    if tlp_span_kind in {
        TraceloopSpanKindValues.WORKFLOW,
        TraceloopSpanKindValues.AGENT,
    }:
        set_workflow_name(entity_name)

    instance_name = None

    if instance is not None:
        if tlp_span_kind in {
            TraceloopSpanKindValues.WORKFLOW,
            TraceloopSpanKindValues.AGENT,
            TraceloopSpanKindValues.TOOL,
        }:
            instance_name = getattr(instance, "name", None)

        elif (
            tlp_span_kind == TraceloopSpanKindValues.TASK and entity_name == "generate"
        ):
            instance_name = getattr(instance, "agent_name", None)

    if instance_name:
        call_id = (kwargs or {}).get("call_id")
        span_name = f"{instance_name}.{entity_name}" + (
            f"[{call_id}]" if call_id else ""
        )
    else:
        span_name = f"{entity_name}.{tlp_span_kind.value}"

    return span_name


def _set_span_attributes(
    span: trace.Span,
    entity_name: str,
    tlp_span_kind: TraceloopSpanKindValues,
    version: int | None,
):
    if tlp_span_kind in {
        TraceloopSpanKindValues.TASK,
        TraceloopSpanKindValues.TOOL,
    }:
        entity_path = get_chained_entity_path(entity_name)
        set_entity_path(entity_path)

    span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, tlp_span_kind.value)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
    if version:
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_VERSION, version)


def _handle_span_input(
    span: trace.Span,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    cls: type[JSONEncoder] | None = None,
    exclude_fields: set[str] | None = None,
):
    try:
        if _should_send_prompts():
            json_input = json.dumps(
                {
                    "args": _to_plain(list(args), exclude_fields=exclude_fields),
                    "kwargs": _to_plain(kwargs, exclude_fields=exclude_fields),
                },
                cls=cls,
                indent=2,
            )
            truncated_json = _truncate_json_if_needed(json_input)
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, truncated_json)

    except TypeError as e:
        Telemetry().log_exception(e)


def _handle_span_output(
    span: trace.Span,
    res: Any,
    cls: type[JSONEncoder] | None = None,
    exclude_fields: set[str] | None = None,
):
    """Handles entity output logging in JSON for both sync and async functions"""
    try:
        if _should_send_prompts():
            json_output = json.dumps(
                _to_plain(res, exclude_fields=exclude_fields), cls=cls, indent=2
            )
            truncated_json = _truncate_json_if_needed(json_output)
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, truncated_json)

    except TypeError as e:
        Telemetry().log_exception(e)


def is_bound_method(func: Callable[..., Any], self_candidate: Any) -> bool:
    return (inspect.ismethod(func) and (func.__self__ is self_candidate)) or hasattr(
        self_candidate, func.__name__
    )


@contextlib.contextmanager
def _suppress_tracing_globally():
    token = context_api.attach(context_api.set_value(_CTX_DISABLE_TRACING_KEY, True))
    try:
        with suppress_instrumentation():
            yield
    finally:
        context_api.detach(token)


def tracing_enabled(
    instance: type | None = None,
) -> bool:
    if _is_tracing_globally_disabled():
        return False
    if instance is None:
        return True
    flag = getattr(instance, "tracing_enabled", True)
    return bool(flag)


def entity_method(
    name: str | None = None,
    version: int | None = None,
    tlp_span_kind: TraceloopSpanKindValues = TraceloopSpanKindValues.TASK,
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        is_async = _is_async_method(fn)
        entity_name = name or fn.__qualname__

        if is_async:
            if inspect.isasyncgenfunction(fn):

                @wraps(fn)
                async def async_gen_wrap(*args: Any, **kwargs: Any) -> Any:
                    is_bound = is_bound_method(fn, args[0] if args else False)
                    instance = args[0] if is_bound else None
                    is_enabled = tracing_enabled(instance)
                    if not (is_enabled and _tracing_initialized_quietly()):
                        with _suppress_tracing_globally():
                            async for item in fn(*args, **kwargs):
                                yield item
                            return
                    span_name = _get_span_name(
                        entity_name, tlp_span_kind, instance=instance, kwargs=kwargs
                    )
                    tracer = trace.get_tracer(__name__)

                    with tracer.start_as_current_span(span_name) as span:
                        _set_span_attributes(span, entity_name, tlp_span_kind, version)
                        _handle_span_input(span, args, kwargs, cls=JSONEncoder)
                        items: list[Any] = []

                        try:
                            async for item in fn(*args, **kwargs):
                                items.append(item)
                                yield item

                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(
                                e,
                                attributes={"tb": traceback.format_exc()},
                            )
                            raise

                        finally:
                            if items:
                                _handle_span_output(span, items[-1], cls=JSONEncoder)

                return cast("F", async_gen_wrap)

            @wraps(fn)
            async def async_wrap(*args: Any, **kwargs: Any) -> Any:
                is_bound = is_bound_method(fn, args[0] if args else False)
                instance = args[0] if is_bound else None
                is_enabled = tracing_enabled(instance)
                if not (is_enabled and _tracing_initialized_quietly()):
                    with _suppress_tracing_globally():
                        return await fn(*args, **kwargs)
                span_name = _get_span_name(
                    entity_name,
                    tlp_span_kind,
                    instance=instance,
                    kwargs=kwargs,
                )

                tracer = trace.get_tracer(__name__)

                with tracer.start_as_current_span(span_name) as span:
                    _set_span_attributes(span, entity_name, tlp_span_kind, version)
                    _handle_span_input(span, args, kwargs, cls=JSONEncoder)

                    try:
                        res = await fn(*args, **kwargs)
                        _handle_span_output(span, res, cls=JSONEncoder)
                        return res

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(
                            e, attributes={"tb": traceback.format_exc()}
                        )
                        raise

            return cast("F", async_wrap)

        @wraps(fn)
        def sync_wrap(*args: Any, **kwargs: Any) -> Any:
            is_bound = is_bound_method(fn, args[0] if args else False)
            instance = args[0] if is_bound else None
            is_enabled = tracing_enabled(instance)
            if not (is_enabled and _tracing_initialized_quietly()):
                with _suppress_tracing_globally():
                    return fn(*args, **kwargs)
            span_name = _get_span_name(
                entity_name,
                tlp_span_kind,
                instance=instance,
                kwargs=kwargs,
            )

            tracer = trace.get_tracer(__name__)

            with tracer.start_as_current_span(span_name) as span:
                _set_span_attributes(span, entity_name, tlp_span_kind, version)
                _handle_span_input(span, args, kwargs, cls=JSONEncoder)
                items: list[Any] = []

                try:
                    res = fn(*args, **kwargs)
                    if isinstance(res, types.GeneratorType):
                        for item in res:  # type: ignore[union-attr]
                            items.append(item)
                            yield item
                    else:
                        _handle_span_output(span, res, cls=JSONEncoder)
                        return res

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e, attributes={"tb": traceback.format_exc()})
                    raise

                finally:
                    if items:
                        _handle_span_output(span, items[-1], cls=JSONEncoder)

        return cast("F", sync_wrap)

    return decorate


def entity_class(
    name: str | None,
    version: int | None,
    method_name: str,
    tlp_span_kind: TraceloopSpanKindValues = TraceloopSpanKindValues.TASK,
) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        task_name = name or camel_to_snake(cls.__qualname__)
        method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            entity_method(name=task_name, version=version, tlp_span_kind=tlp_span_kind)(
                method
            ),
        )
        return cls

    return decorator
