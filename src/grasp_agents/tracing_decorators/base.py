import contextlib
import inspect
import json
import os
import types
import warnings
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import (
    Any,
    ParamSpec,
    TypeVar,
    cast,
)

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.trace.status import Status, StatusCode
from pydantic import BaseModel
from traceloop.sdk.telemetry import Telemetry
from traceloop.sdk.tracing import get_tracer, set_workflow_name
from traceloop.sdk.tracing.tracing import (
    TracerWrapper,
    get_chained_entity_path,
    set_entity_path,
)
from traceloop.sdk.utils import camel_to_snake
from traceloop.sdk.utils.json_encoder import JSONEncoder

P = ParamSpec("P")

R = TypeVar("R")
F = TypeVar("F", bound=Callable[P, R | Awaitable[R]])


def _to_plain(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable primitives.

    - Pydantic BaseModel -> dict via model_dump()
    - dict/list/tuple/set -> recurse (sets become lists)
    - other objects -> returned as-is (left to JSONEncoder)
    """
    if isinstance(obj, BaseModel):
        try:
            # TODO: remove this temporary hack
            return obj.model_dump(exclude={"_hidden_params", "completions"})
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


# Async Decorators - Deprecated


def aentity_method(
    name: str | None = None,
    version: int | None = None,
    tlp_span_kind: TraceloopSpanKindValues | None = TraceloopSpanKindValues.TASK,
):
    warnings.warn(
        "DeprecationWarning: The @aentity_method function will be removed in a future version. "
        "Please migrate to @entity_method for both sync and async operations.",
        DeprecationWarning,
        stacklevel=2,
    )

    return entity_method(
        name=name,
        version=version,
        tlp_span_kind=tlp_span_kind,
    )


def aentity_class(
    name: str | None,
    version: int | None,
    method_name: str,
    tlp_span_kind: TraceloopSpanKindValues | None = TraceloopSpanKindValues.TASK,
):
    warnings.warn(
        "DeprecationWarning: The @aentity_class function will be removed in a future version. "
        "Please migrate to @entity_class for both sync and async operations.",
        DeprecationWarning,
        stacklevel=2,
    )

    return entity_class(
        name=name,
        version=version,
        method_name=method_name,
        tlp_span_kind=tlp_span_kind,
    )


def _handle_generator(span, res):
    # for some reason the SPAN_KEY is not being set in the context of the generator, so we re-set it
    context_api.attach(trace.set_span_in_context(span))
    try:
        for item in res:
            # span.add_event(
            #     name=getattr(item, "type", "unknown"),
            #     attributes=_to_plain(item),
            # )
            yield item
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        span.end()
        raise
    # finally:
    #     span.end()
    # Note: we don't detach the context here as this fails in some situations
    # https://github.com/open-telemetry/opentelemetry-python/issues/2606
    # This is not a problem since the context will be detached automatically during garbage collection


async def _ahandle_generator(span, ctx_token, res):
    try:
        async for part in res:
            # span.add_event(
            #     name=getattr(part, "type", "unknown"),
            #     attributes=_to_plain(part),
            # )
            yield part
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        span.end()
        raise
    # finally:
    #     span.end()
    #     context_api.detach(ctx_token)


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


# Unified Decorators : handles both sync and async operations


def _is_async_method(fn):
    # check if co-routine function or async generator( example : using async & yield)
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


def _setup_span(entity_name, tlp_span_kind, version):
    """Sets up the OpenTelemetry span and context"""
    if tlp_span_kind in [
        TraceloopSpanKindValues.WORKFLOW,
        TraceloopSpanKindValues.AGENT,
    ]:
        set_workflow_name(entity_name)
    span_name = f"{entity_name}.{tlp_span_kind.value}"

    with get_tracer() as tracer:
        span = tracer.start_span(span_name)
        ctx = trace.set_span_in_context(span)
        ctx_token = context_api.attach(ctx)

        if tlp_span_kind in [
            TraceloopSpanKindValues.TASK,
            TraceloopSpanKindValues.TOOL,
        ]:
            entity_path = get_chained_entity_path(entity_name)
            set_entity_path(entity_path)

        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, tlp_span_kind.value)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        if version:
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_VERSION, version)

    return span, ctx, ctx_token


def _handle_span_input(span, args, kwargs, cls=None):
    """Handles entity input logging in JSON for both sync and async functions"""
    try:
        if _should_send_prompts():
            json_input = json.dumps(
                {"args": _to_plain(list(args)), "kwargs": _to_plain(kwargs)},
                **({"cls": cls} if cls else {}),
                indent=2,
            )
            truncated_json = _truncate_json_if_needed(json_input)
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                truncated_json,
            )
    except TypeError as e:
        Telemetry().log_exception(e)


def _handle_span_output(span, res, cls=None):
    """Handles entity output logging in JSON for both sync and async functions"""
    try:
        if _should_send_prompts():
            json_output = json.dumps(
                _to_plain(res), **({"cls": cls} if cls else {}), indent=2
            )
            truncated_json = _truncate_json_if_needed(json_output)
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                truncated_json,
            )
    except TypeError as e:
        Telemetry().log_exception(e)


def _cleanup_span(span, ctx_token):
    """End the span process and detach the context token"""
    span.end()
    context_api.detach(ctx_token)


def entity_method(
    name: str | None = None,
    version: int | None = None,
    tlp_span_kind: TraceloopSpanKindValues | None = TraceloopSpanKindValues.TASK,
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        is_async = _is_async_method(fn)
        entity_name = name or fn.__qualname__
        if is_async:
            if inspect.isasyncgenfunction(fn):

                @wraps(fn)
                async def async_gen_wrap(*args: Any, **kwargs: Any) -> Any:
                    if not _tracing_initialized_quietly():
                        async for item in fn(*args, **kwargs):
                            yield item
                        return

                    span, ctx, ctx_token = _setup_span(
                        entity_name, tlp_span_kind, version
                    )
                    _handle_span_input(span, args, kwargs, cls=JSONEncoder)
                    items = []
                    async for item in _ahandle_generator(
                        span, ctx_token, fn(*args, **kwargs)
                    ):
                        items.append(item)
                        yield item
                    if items:
                        _handle_span_output(span, items[-1], cls=JSONEncoder)
                    span.end()
                    context_api.detach(ctx_token)

                return cast("F", async_gen_wrap)

            @wraps(fn)
            async def async_wrap(*args: Any, **kwargs: Any) -> Any:
                if not _tracing_initialized_quietly():
                    return await fn(*args, **kwargs)

                span, ctx, ctx_token = _setup_span(entity_name, tlp_span_kind, version)
                _handle_span_input(span, args, kwargs, cls=JSONEncoder)
                try:
                    res = await fn(*args, **kwargs)
                    _handle_span_output(span, res, cls=JSONEncoder)
                    return res
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    _cleanup_span(span, ctx_token)

            return cast("F", async_wrap)

        @wraps(fn)
        def sync_wrap(*args: Any, **kwargs: Any) -> Any:
            if not _tracing_initialized_quietly():
                return fn(*args, **kwargs)

            span, ctx, ctx_token = _setup_span(entity_name, tlp_span_kind, version)
            _handle_span_input(span, args, kwargs, cls=JSONEncoder)
            try:
                res = fn(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                _cleanup_span(span, ctx_token)
                raise

            # span will be ended in the generator
            if isinstance(res, types.GeneratorType):
                items = []
                for item in _handle_generator(span, res):
                    items.append(item)
                    yield item
                if items:
                    _handle_span_output(span, items[-1], cls=JSONEncoder)
                span.end()

            _handle_span_output(span, res, cls=JSONEncoder)
            _cleanup_span(span, ctx_token)
            return res

        return cast("F", sync_wrap)

    return decorate


def entity_class(
    name: str | None,
    version: int | None,
    method_name: str,
    tlp_span_kind: TraceloopSpanKindValues | None = TraceloopSpanKindValues.TASK,
):
    def decorator(cls):
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
