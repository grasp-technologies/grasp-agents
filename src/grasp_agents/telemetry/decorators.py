"""
Tracing decorators using raw OpenTelemetry API.

Follows the OTel library instrumentation pattern: depends only on opentelemetry-api.
If no TracerProvider is configured by the application, all spans are no-ops.
"""

import hashlib
import inspect
import json
import os
import re
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import StrEnum
from functools import wraps
from logging import getLogger
from typing import Any, cast, overload

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,  # noqa: PLC2701 # pyright: ignore[reportPrivateUsage]
    Context,
)
from opentelemetry.trace.propagation import set_span_in_context
from pydantic import BaseModel

logger = getLogger(__name__)

# The plain-string twin of the context-api key above, read by older OTel
# contrib instrumentations (see opentelemetry.instrumentation.utils).
_SUPPRESS_INSTRUMENTATION_KEY_PLAIN = "suppress_instrumentation"

# ---------------------------------------------------------------------------
# Span kind values
# ---------------------------------------------------------------------------


class SpanKind(StrEnum):
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

DEFAULT_EXCLUDE_FIELDS = {"_hidden_params", "responses"}
_TRACER_NAME = "grasp_agents"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_plain(obj: Any, exclude_fields: set[str] | None = None) -> Any:
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
    if not limit_str:
        return json_str
    try:
        limit = int(limit_str)
    except ValueError:
        return json_str
    if limit <= 0 or len(json_str) <= limit:
        return json_str
    # Keep the head AND tail, dropping the middle: the start and end of a
    # prompt/response carry the most signal (the bulk is usually repetitive
    # context). Fitted within `limit` so the SDK's own head-only cap never fires.
    marker = f" …[{len(json_str) - limit} chars]… "
    keep = limit - len(marker)
    if keep <= 0:
        # Limit too small to fit a head…tail marker — fall back to a head clip.
        return json_str[:limit]
    head, tail = keep // 2, keep - keep // 2
    # Removing head+tail (not just the over-limit slice) raises the true omitted
    # count; recompute the marker so the result still fits within `limit`.
    marker = f" …[{len(json_str) - head - tail} chars]… "
    keep = max(0, limit - len(marker))
    head, tail = keep // 2, keep - keep // 2
    return (json_str[:head] + marker + json_str[len(json_str) - tail :])[:limit]


def _should_send_prompts() -> bool:
    val = os.getenv("GRASP_TRACE_CONTENT") or os.getenv("TRACELOOP_TRACE_CONTENT")
    return (val or "true").lower() == "true"


@contextmanager
def _suppressed_instrumentation() -> Generator[None, None, None]:
    """
    Mark downstream auto-instrumentation suppressed for the duration.

    A ``tracing_enabled=False`` component must go fully dark: skipping its own
    span still leaves provider-SDK auto-instrumentation (the OpenInference
    openai / anthropic / google-genai instrumentors) emitting orphan spans for
    the LLM calls the component makes. Instrumentors check the OTel context
    keys set here — ``create_key`` appends a uuid, so the context-api key must
    be the imported object, not a recreation. Mirrors
    ``opentelemetry.instrumentation.utils.suppress_instrumentation`` without
    depending on that package.
    """
    ctx = otel_context.get_current()
    for key in (_SUPPRESS_INSTRUMENTATION_KEY, _SUPPRESS_INSTRUMENTATION_KEY_PLAIN):
        ctx = otel_context.set_value(key, value=True, context=ctx)
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


def _tracing_enabled(instance: Any | None = None) -> bool:
    # Inside a suppressed region (a disabled ancestor), nested grasp spans go
    # dark too, matching the suppressed provider instrumentation.
    if otel_context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return False
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


def _apply_caller_span_overrides(span: trace.Span, kwargs: dict[str, Any]) -> None:
    # ``run`` / ``run_stream`` accept ``span_name`` / ``span_attributes`` so a
    # caller can rename the run span and attach domain attributes (e.g.
    # ``goal.id``). Applied after the framework attributes so the caller wins.
    # Absent on any other traced call → no-op.
    if not span.is_recording():
        return
    name = kwargs.get("span_name")
    if name:
        span.update_name(name)
    attributes = kwargs.get("span_attributes")
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)


def set_run_span_attributes(**attributes: str | float | bool) -> None:
    """
    Attach attributes to the currently-active run span.

    For attributes discovered mid-run (inside a hook or tool); attributes known
    at call time are better passed to ``run`` / ``run_stream`` via
    ``span_attributes=``. No-op when tracing is off or no span is recording.
    """
    span = trace.get_current_span()
    if span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


# ---------------------------------------------------------------------------
# Session trace grouping + session attributes
# ---------------------------------------------------------------------------


def derive_session_span_context(session_key: str) -> Context:
    """
    Deterministic remote-parent context for a session.

    Hashes ``session_key`` into a stable ``trace_id`` + root ``span_id`` so
    every run of one session — across turns and across processes, with or
    without a checkpoint store — lands in a single trace, parented to a common
    session root. The root span itself is never emitted (it is a remote parent),
    so the grouping is expressed purely in OTel primitives and renders in any
    backend. Pass the result as a span's ``context=`` to correlate your own
    work with a grasp-agents session.
    """
    digest = hashlib.sha256(session_key.encode("utf-8")).digest()
    # trace_id is 128-bit, span_id 64-bit; both must be non-zero (OTel treats a
    # zero id as invalid). A SHA-256 slice is non-zero in practice — guard anyway.
    trace_id = int.from_bytes(digest[:16], "big") or 1
    span_id = int.from_bytes(digest[16:24], "big") or 1
    span_context = trace.SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
    )
    return set_span_in_context(trace.NonRecordingSpan(span_context))


_DEFAULT_SESSION_ID_ATTRIBUTES = ("gen_ai.conversation.id",)


def _session_id_attribute_keys() -> tuple[str, ...]:
    """
    Span-attribute keys to stamp with the session id.

    Defaults to ``gen_ai.conversation.id`` (the OpenTelemetry GenAI convention).
    Override via the ``GRASP_SESSION_ID_ATTRIBUTES`` env var — comma-separated,
    e.g. ``gen_ai.conversation.id,session.id`` (set it empty to stamp none).
    Which keys a backend reads is a deployment concern, so it is configured at
    the deployment level (env) rather than per session.
    """
    raw = os.getenv("GRASP_SESSION_ID_ATTRIBUTES")
    if raw is None:
        return _DEFAULT_SESSION_ID_ATTRIBUTES
    return tuple(k.strip() for k in raw.split(",") if k.strip())


# Carries the active session id down the OTel context so SessionSpanProcessor
# can stamp it onto every descendant span. A context value, NOT baggage: it
# never rides outbound request headers, so a possibly-identifying session id is
# not leaked to LLM providers / downstream services.
_SESSION_ID_CTX_KEY = otel_context.create_key("grasp.session_id")


def _resolve_run_span_context(instance: Any | None) -> Context | None:
    """
    Resolve the context for a run-ROOT span: session parent + session id.

    A run root (``Processor`` / ``Runner``) exposes ``_trace_session_info`` — its
    session id plus whether to group every run of the session into one trace.
    Both apply only when this span would actually *be* a trace root (no span is
    already recording); a nested run (a sub-agent under a runner, an
    agent-as-tool, a generate/tool span) inherits the enclosing run's context
    and is left untouched. The returned context carries:

    * a derived session-root parent — so every run of the session shares one
      trace — when grouping is on; otherwise no parent override;
    * the session id as a context value, which :class:`SessionSpanProcessor`
      reads to stamp the configured attribute(s) onto this span and every
      descendant, attributing the whole run tree (incl. provider spans) to the
      session.

    ``None`` when nested or there is no named session. Telemetry must never fail
    the call: any error falls back to ambient parenting.
    """
    if instance is None:
        return None
    get_info = getattr(instance, "_trace_session_info", None)
    if get_info is None:
        return None
    try:
        # Only the outermost run span adopts the session; a recording ancestor
        # means we are nested and must inherit the enclosing run's context.
        if trace.get_current_span().is_recording():
            return None
        info = get_info()
        if info is None:
            return None
        session_id, group = info
        ctx = derive_session_span_context(session_id) if group else None
        return otel_context.set_value(_SESSION_ID_CTX_KEY, session_id, ctx)
    except Exception:
        logger.debug("session span resolution failed", exc_info=True)
        return None


def stamp_session_attributes(
    span: trace.Span, parent_context: Context | None = None
) -> None:
    """
    Stamp the active session id onto ``span`` as the configured attribute(s).

    Reads the session id a run root placed on the OTel context (resolved from
    ``parent_context``, else the current context) and writes it to each key in
    ``GRASP_SESSION_ID_ATTRIBUTES`` (default ``gen_ai.conversation.id``). Called
    by :class:`grasp_agents.telemetry.SessionSpanProcessor` for every span — so
    the whole run tree, including provider-instrumentation spans, is attributed
    to the session. No-op when no session is active or the span is not recording.
    """
    if not span.is_recording():
        return
    session_id = otel_context.get_value(_SESSION_ID_CTX_KEY, parent_context)
    if not session_id:
        return
    for key in _session_id_attribute_keys():
        span.set_attribute(key, str(session_id))


@contextmanager
def _run_span(
    span_name: str, instance: Any | None
) -> Generator[trace.Span, None, None]:
    """
    Open a span, attaching the run-root's session context for its duration.

    At a run root the attached context carries the session parent + id, so the
    span parents into the shared session trace (when grouping is on) and
    :class:`SessionSpanProcessor` stamps the session attribute(s) on this span
    AND every descendant — the attach (not just ``context=``) is what lets the
    id reach child spans, since ``start_as_current_span`` re-bases the active
    context on the current one. A nested span attaches nothing and inherits the
    enclosing run's context.
    """
    parent_context = _resolve_run_span_context(instance)
    token = otel_context.attach(parent_context) if parent_context is not None else None
    try:
        with trace.get_tracer(_TRACER_NAME).start_as_current_span(span_name) as span:
            yield span
    finally:
        if token is not None:
            otel_context.detach(token)


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
    except (TypeError, ValueError, RecursionError) as e:
        # Telemetry serialization must never fail the traced call. json.dumps
        # raises ValueError on circular refs and RecursionError on deeply
        # nested payloads; _to_plain can raise either too.
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
    except (TypeError, ValueError, RecursionError) as e:
        # Telemetry serialization must never fail the traced call. json.dumps
        # raises ValueError on circular refs and RecursionError on deeply
        # nested payloads; _to_plain can raise either too.
        span.record_exception(e)


def _resolve_span_kind(instance: Any | None, default: SpanKind) -> SpanKind:
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
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# ---------------------------------------------------------------------------
# Internal decorator factories
# ---------------------------------------------------------------------------


def _entity_method[F: Callable[..., Any]](
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
                        with _suppressed_instrumentation():
                            async for item in fn(*args, **kwargs):
                                yield item
                        return

                    resolved_kind = _resolve_span_kind(instance, span_kind)
                    span_name = _get_span_name(
                        entity_name, resolved_kind, instance=instance, kwargs=kwargs
                    )
                    with _run_span(span_name, instance) as span:
                        _set_span_attributes(span, entity_name, resolved_kind, version)
                        _apply_caller_span_overrides(span, kwargs)
                        _handle_span_input(span, input_args, kwargs, exclude_fields)
                        # Track only the LAST item — buffering every yielded
                        # event multiplies memory by the tracing nesting depth.
                        last_item: Any = None
                        has_items = False

                        try:
                            async for item in fn(*args, **kwargs):
                                last_item = item
                                has_items = True
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
                            if has_items:
                                _handle_span_output(span, last_item)

                return cast("F", async_gen_wrap)

            @wraps(fn)
            async def async_wrap(*args: Any, **kwargs: Any) -> Any:
                is_bound = _is_bound_method(fn, args[0] if args else False)
                instance = args[0] if is_bound else None
                input_args = args[1:] if is_bound else args
                exclude_fields = _exclude_fields_from_instance(instance)

                if not _tracing_enabled(instance):
                    with _suppressed_instrumentation():
                        return await fn(*args, **kwargs)

                resolved_kind = _resolve_span_kind(instance, span_kind)
                span_name = _get_span_name(
                    entity_name, resolved_kind, instance=instance, kwargs=kwargs
                )
                with _run_span(span_name, instance) as span:
                    _set_span_attributes(span, entity_name, resolved_kind, version)
                    _apply_caller_span_overrides(span, kwargs)
                    _handle_span_input(span, input_args, kwargs, exclude_fields)
                    try:
                        res = await fn(*args, **kwargs)
                        _handle_span_output(span, res)
                        return res
                    except Exception as e:
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
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
                    with _suppressed_instrumentation():
                        yield from fn(*args, **kwargs)
                    return

                resolved_kind = _resolve_span_kind(instance, span_kind)
                span_name = _get_span_name(
                    entity_name, resolved_kind, instance=instance, kwargs=kwargs
                )
                with _run_span(span_name, instance) as span:
                    _set_span_attributes(span, entity_name, resolved_kind, version)
                    _apply_caller_span_overrides(span, kwargs)
                    _handle_span_input(span, input_args, kwargs, exclude_fields)
                    # Track only the LAST item — see the async generator path.
                    last_item: Any = None
                    has_items = False

                    try:
                        for item in fn(*args, **kwargs):
                            last_item = item
                            has_items = True
                            yield item
                    except Exception as e:
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        span.record_exception(
                            e, attributes={"tb": traceback.format_exc()}
                        )
                        raise
                    finally:
                        if has_items:
                            _handle_span_output(span, last_item)

            return cast("F", sync_gen_wrap)

        @wraps(fn)
        def sync_wrap(*args: Any, **kwargs: Any) -> Any:
            is_bound = _is_bound_method(fn, args[0] if args else False)
            instance = args[0] if is_bound else None
            input_args = args[1:] if is_bound else args
            exclude_fields = _exclude_fields_from_instance(instance)

            if not _tracing_enabled(instance):
                with _suppressed_instrumentation():
                    return fn(*args, **kwargs)

            resolved_kind = _resolve_span_kind(instance, span_kind)
            span_name = _get_span_name(
                entity_name, resolved_kind, instance=instance, kwargs=kwargs
            )
            with _run_span(span_name, instance) as span:
                _set_span_attributes(span, entity_name, resolved_kind, version)
                _apply_caller_span_overrides(span, kwargs)
                _handle_span_input(span, input_args, kwargs, exclude_fields)
                try:
                    res = fn(*args, **kwargs)
                    _handle_span_output(span, res)
                    return res
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e, attributes={"tb": traceback.format_exc()})
                    raise

        return cast("F", sync_wrap)

    return decorate


def _entity_class[T: type](
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
def traced[F: Callable[..., Any]](
    name: str | None = ...,
    version: int | None = ...,
    span_kind: SpanKind = ...,
) -> Callable[[F], F]: ...


@overload
def traced[T: type](
    name: str | None = ...,
    version: int | None = ...,
    span_kind: SpanKind = ...,
    *,
    method_name: str,
) -> Callable[[T], T]: ...


def traced[F: Callable[..., Any], T: type](
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
