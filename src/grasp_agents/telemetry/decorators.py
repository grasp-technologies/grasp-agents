"""
Tracing decorators using raw OpenTelemetry API.

Follows the OTel library instrumentation pattern: depends only on opentelemetry-api.
If no TracerProvider is configured by the application, all spans are no-ops.
Span names and attributes are listed in :mod:`grasp_agents.telemetry.attributes`.
"""

import asyncio
import hashlib
import inspect
import json
import math
import os
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property, wraps
from logging import getLogger
from typing import Any, Final, cast, overload

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,  # noqa: PLC2701 # pyright: ignore[reportPrivateUsage]
    Context,
)
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

from grasp_agents.utils.errors import format_error_chain, root_cause

from .attributes import (
    ATTR_CANCELLED,
    ATTR_CONVERSATION_ID,
    ATTR_ERROR_TYPE,
    ATTR_INPUT_MIME_TYPE,
    ATTR_INPUT_TRUNCATED,
    ATTR_INPUT_VALUE,
    ATTR_OI_SPAN_KIND,
    ATTR_OUTPUT_MIME_TYPE,
    ATTR_OUTPUT_TRUNCATED,
    ATTR_OUTPUT_VALUE,
    ATTR_SESSION_ID,
    ATTR_SPAN_KIND,
    JSON_MIME_TYPE,
    OPENINFERENCE_SPAN_KINDS,
    SpanKind,
)

logger = getLogger(__name__)

# The plain-string twin of the context-api key above, read by older OTel
# contrib instrumentations (see opentelemetry.instrumentation.utils).
_SUPPRESS_INSTRUMENTATION_KEY_PLAIN = "suppress_instrumentation"

DEFAULT_EXCLUDE_FIELDS = {"_hidden_params", "responses", "encrypted_content"}
_TRACER_NAME = "grasp_agents"


class _NoPayload:
    def __repr__(self) -> str:
        return "NO_PAYLOAD"


NO_PAYLOAD: Final[Any] = _NoPayload()
"""Marks a span that records no ``input.value`` / ``output.value``."""


class _CallArguments:
    def __repr__(self) -> str:
        return "CALL_ARGUMENTS"


CALL_ARGUMENTS: Final[Any] = _CallArguments()
"""As a ``SpanStart.input``: record the call's arguments, by parameter name."""


@dataclass(frozen=True)
class SpanStart:
    """How a traced object describes the span of one of its calls."""

    name: str
    kind: SpanKind
    attributes: Mapping[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )
    # Recorded as ``input.value`` (JSON) when content tracing is on.
    input: Any = NO_PAYLOAD


@dataclass(frozen=True)
class SpanResult:
    """What a call's result (or a yielded item) adds to its span."""

    # Recorded as ``output.value`` (JSON) when content tracing is on.
    output: Any = NO_PAYLOAD
    attributes: Mapping[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def _to_plain(obj: Any, exclude_fields: set[str] | None = None) -> Any:
    all_exclude = DEFAULT_EXCLUDE_FIELDS.union(exclude_fields or set())
    if isinstance(obj, BaseModel):
        try:
            return _to_plain(obj.model_dump(exclude=all_exclude), exclude_fields)
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
    if isinstance(obj, float) and not math.isfinite(obj):
        # NaN and infinities are not JSON.
        return "NaN" if math.isnan(obj) else ("Infinity" if obj > 0 else "-Infinity")
    return obj


_LIMIT_ENV = (
    "OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT",
    "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT",
)


def _attribute_limit(span: trace.Span | None = None) -> int | None:
    """The length the SDK cuts the span's attribute values to, if any."""
    limits = getattr(span, "_limits", None)
    if limits is not None:
        limit = getattr(limits, "max_span_attribute_length", None)
        return limit if isinstance(limit, int) and limit > 0 else None
    for name in _LIMIT_ENV:
        raw = os.getenv(name)
        if raw:
            try:
                limit = int(raw)
            except ValueError:
                return None
            return limit if limit > 0 else None
    return None


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _clip(text: str, limit: int) -> str:
    """``text`` within ``limit`` characters, keeping its head and tail."""
    if len(text) <= limit:
        return text
    marker = f"…[{len(text) - limit} chars]…"
    keep = limit - len(marker)
    if keep <= 0:
        return text[:limit]
    # The marker counts what is dropped, which grows once it takes room too.
    marker = f"…[{len(text) - keep} chars]…"
    keep = max(0, limit - len(marker))
    head, tail = keep - keep // 2, keep // 2
    return text[:head] + marker + text[len(text) - tail :]


@dataclass(frozen=True)
class _Cut:
    """The head and tail of a longer list or mapping, and how large it was."""

    value: list[Any] | dict[str, Any]
    total: int


def _ends[T](items: list[T], keep: int) -> list[T]:
    if len(items) <= keep:
        return items
    tail = keep // 2
    return [*items[: keep - tail], *items[len(items) - tail :]]


def _cut(value: Any, max_chars: int, max_items: int) -> Any:
    # Containers keep their head and tail and remember their size, so markers
    # added by ``_shrunk`` later count what the original held.
    if isinstance(value, str):
        return _clip(value, max_chars)
    if isinstance(value, list):
        items = cast("list[Any]", value)
        kept = [_cut(v, max_chars, max_items) for v in _ends(items, max_items)]
        return kept if len(items) <= max_items else _Cut(kept, len(items))
    if isinstance(value, dict):
        entries = list(cast("dict[str, Any]", value).items())
        cut = {k: _cut(v, max_chars, max_items) for k, v in _ends(entries, max_items)}
        return cut if len(entries) <= max_items else _Cut(cut, len(entries))
    return value


def _shrunk(value: Any, max_chars: int, max_items: int) -> Any:
    """``value`` with strings cut to ``max_chars`` and containers to ``max_items``."""
    total: int | None = None
    if isinstance(value, _Cut):
        value, total = value.value, value.total
    if isinstance(value, str):
        return _clip(value, max_chars)
    if isinstance(value, list):
        items = cast("list[Any]", value)
        total = total or len(items)
        if total > max_items:
            tail = max_items // 2
            marker = f"…[{total - max_items} items]…"
            items = [*items[: max_items - tail], marker, *items[len(items) - tail :]]
        return [_shrunk(v, max_chars, max_items) for v in items]
    if isinstance(value, dict):
        entries = list(cast("dict[str, Any]", value).items())
        total = total or len(entries)
        if total > max_items:
            tail = max_items // 2
            entries = [
                *entries[: max_items - tail],
                ("…", f"[{total - max_items} more keys]"),
                *entries[len(entries) - tail :],
            ]
        return {k: _shrunk(v, max_chars, max_items) for k, v in entries}
    return value


def _extent(value: Any) -> tuple[int, int]:
    """The longest string and the largest container inside a JSON value."""
    if isinstance(value, _Cut):
        value = value.value
    if isinstance(value, str):
        return len(value), 0
    if isinstance(value, list | dict):
        children = cast(
            "list[Any]",
            list(cast("dict[str, Any]", value).values())
            if isinstance(value, dict)
            else value,
        )
        extents = [_extent(v) for v in children]
        return max((e[0] for e in extents), default=0), max(
            [len(children), *(e[1] for e in extents)]
        )
    return 0, 0


_MIN_CHARS = 16
_MIN_ITEMS = 2


def _largest_fitting(low: int, high: int, fits: Callable[[int], bool]) -> int:
    """The largest ``n`` in ``[low, high]`` with ``fits(n)``, given ``fits(low)``."""
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low


def fit_json(value: Any, limit: int | None) -> tuple[str, bool]:
    """
    ``value`` as JSON text of at most ``limit`` characters, and whether it
    had to be shortened. Long strings are cut in the middle first (keeping
    head and tail, with a marker), then — only if that is not enough — long
    lists and mappings, so the text stays valid JSON with as much content as
    fits; a value whose structure alone is too large becomes a JSON string
    holding the cut text.
    """
    text = _dumps(value)
    if limit is None or len(text) <= limit:
        return text, False
    # Nothing longer than the limit, and no container with more entries than
    # half of it, can be kept whole: cut those first, so the search below
    # handles only what can fit.
    native = _cut(json.loads(text), limit, max(_MIN_ITEMS, limit // 2))
    longest, widest = _extent(native)

    def size(max_chars: int, max_items: int) -> int:
        return len(_dumps(_shrunk(native, max_chars, max_items)))

    if size(_MIN_CHARS, _MIN_ITEMS) <= limit:
        whole = size(_MIN_CHARS, widest)
        if whole <= limit:
            items = widest
        else:
            # Size grows about linearly with the entries kept: search below
            # twice the proportional share, so candidates stay small.
            def fits(n: int) -> bool:
                return size(_MIN_CHARS, n) <= limit

            high = max(_MIN_ITEMS, min(widest, 2 * widest * limit // whole))
            items = _largest_fitting(_MIN_ITEMS, widest if fits(high) else high, fits)
        chars = _largest_fitting(
            _MIN_CHARS, max(_MIN_CHARS, longest), lambda n: size(n, items) <= limit
        )
        return _dumps(_shrunk(native, chars, items)), True
    budget = limit
    while budget > 0:
        quoted = _dumps(_clip(text, budget))
        if len(quoted) <= limit:
            return quoted, True
        budget -= len(quoted) - limit
    return _dumps(""), True


def _should_send_prompts() -> bool:
    val = os.getenv("GRASP_TRACE_CONTENT") or os.getenv("TRACELOOP_TRACE_CONTENT")
    return (val or "true").lower() == "true"


def record_span_payload(
    span: trace.Span,
    payload: Any,
    *,
    output: bool,
    exclude_fields: set[str] | None = None,
) -> None:
    """
    Record ``payload`` as the span's ``input.value`` (or ``output.value``):
    JSON, shortened to the span's attribute length limit without breaking it.
    No-op when content tracing is off or ``payload`` is :data:`NO_PAYLOAD`.
    """
    if payload is NO_PAYLOAD or not _should_send_prompts() or not span.is_recording():
        return
    if output:
        value_key, mime_key = ATTR_OUTPUT_VALUE, ATTR_OUTPUT_MIME_TYPE
        truncated_key = ATTR_OUTPUT_TRUNCATED
    else:
        value_key, mime_key = ATTR_INPUT_VALUE, ATTR_INPUT_MIME_TYPE
        truncated_key = ATTR_INPUT_TRUNCATED
    limit = _attribute_limit(span)
    try:
        text, truncated = fit_json(_to_plain(payload, exclude_fields), limit)
    except Exception as e:
        # Telemetry must never fail the traced call (unserializable or
        # circular payloads, a failing ``__str__``).
        span.record_exception(e)
        return
    if limit is not None and len(text) > limit:
        return
    span.set_attribute(value_key, text)
    span.set_attribute(mime_key, JSON_MIME_TYPE)
    if truncated:
        span.set_attribute(truncated_key, value=True)


# ---------------------------------------------------------------------------
# Instrumentation suppression
# ---------------------------------------------------------------------------


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
# Run-span helpers
# ---------------------------------------------------------------------------


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


def capture_run_span(instance: Any) -> trace.Span | None:
    """
    Snapshot the current run span while the ambient context is trustworthy.

    Call at the START of a retry loop, before any child stream runs: an
    abandoned child leaves the ambient current-span stale, so a failure-time
    lookup can point at a span that is never exported. Returns None when
    tracing is off for ``instance`` — the ambient span then belongs to an
    enclosing parent, and recording there would misattribute the failure.
    """
    if not _tracing_enabled(instance):
        return None
    span = trace.get_current_span()
    if not span.is_recording():
        return None
    return span


# ---------------------------------------------------------------------------
# Inherited attributes: session id + scoped attributes
# ---------------------------------------------------------------------------


def _trace_namespace() -> str:
    # The project (or service) the global tracer provider exports as.
    resource = getattr(trace.get_tracer_provider(), "resource", None)
    attributes = cast("Mapping[str, Any]", getattr(resource, "attributes", None) or {})
    value = attributes.get("openinference.project.name") or attributes.get(
        "service.name"
    )
    return str(value) if value else ""


def derive_session_span_context(
    session_key: str, namespace: str | None = None
) -> Context:
    """
    Deterministic remote-parent context for a session.

    Hashes ``session_key`` into a stable ``trace_id`` + root ``span_id`` so
    every run of one session — across turns and across processes, with or
    without a checkpoint store — lands in a single trace, parented to a common
    session root. The root span itself is never emitted (it is a remote parent),
    so the grouping is expressed purely in OTel primitives and renders in any
    backend. ``namespace`` (default: the tracer provider's project or service
    name) keeps equal session keys of different projects apart — a trace
    stays in the backend project that first received it. Pass the result as a
    span's ``context=`` to correlate your own work with a grasp-agents session.
    """
    space = _trace_namespace() if namespace is None else namespace
    key = f"{space}\x00{session_key}" if space else session_key
    digest = hashlib.sha256(key.encode("utf-8")).digest()
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


_DEFAULT_SESSION_ID_ATTRIBUTES = (ATTR_SESSION_ID, ATTR_CONVERSATION_ID)


def _session_id_attribute_keys() -> tuple[str, ...]:
    """
    Span-attribute keys to stamp with the session id.

    Defaults to ``session.id`` (OpenInference: Phoenix's Sessions view) and
    ``gen_ai.conversation.id`` (the OpenTelemetry GenAI convention). Override
    via the ``GRASP_SESSION_ID_ATTRIBUTES`` env var — comma-separated (set it
    empty to stamp none). Which keys a backend reads is a deployment concern,
    so it is configured at the deployment level (env) rather than per session.
    """
    raw = os.getenv("GRASP_SESSION_ID_ATTRIBUTES")
    if raw is None:
        return _DEFAULT_SESSION_ID_ATTRIBUTES
    return tuple(k.strip() for k in raw.split(",") if k.strip())


# Attributes every span started in this context receives (see
# InheritedAttributesSpanProcessor). A context value, NOT baggage: it never
# rides outbound request headers, so a possibly-identifying session id is not
# leaked to LLM providers / downstream services.
_INHERITED_ATTRIBUTES_KEY = otel_context.create_key("grasp.inherited_attributes")
# The session the enclosing grasp run belongs to.
_SESSION_KEY = otel_context.create_key("grasp.session")


def _inherited(context: Context | None = None) -> dict[str, AttributeValue]:
    value = otel_context.get_value(_INHERITED_ATTRIBUTES_KEY, context)
    return dict(cast("Mapping[str, AttributeValue]", value)) if value else {}


@contextmanager
def inherited_span_attributes(
    attributes: Mapping[str, AttributeValue | None],
) -> Generator[None]:
    """
    Stamp ``attributes`` (those not ``None``) onto every span started inside
    the block — this process's spans only; they are never propagated to other
    services. grasp-agents spans always get them; other instrumentation's
    spans need :class:`~grasp_agents.telemetry.InheritedAttributesSpanProcessor`,
    which :func:`~grasp_agents.telemetry.init_tracing` installs.
    """
    merged = {
        **_inherited(),
        **{k: v for k, v in attributes.items() if v is not None},
    }
    token = otel_context.attach(
        otel_context.set_value(_INHERITED_ATTRIBUTES_KEY, merged)
    )
    try:
        yield
    finally:
        otel_context.detach(token)


def _resolve_run_span_context(instance: Any | None) -> Context | None:
    """
    The context for a session's outermost run: its session id, and its
    trace when sessions are grouped.

    A run root (``Processor`` / ``Runner``) exposes ``_trace_session_info`` — its
    session id plus whether to group every run of the session into one trace.
    The outermost grasp run of a session (no session is active yet) stamps the
    session id on its span and every descendant, whatever the caller's spans
    (an HTTP server span, an evaluation trial). It is parented to the derived
    session root only when it has no parent of its own, so a caller's trace —
    and its sampling decision — is kept. Nested runs (a sub-agent, an
    agent-as-tool) inherit the enclosing run's context.

    ``None`` when nested or there is no named session. Telemetry must never fail
    the call: any error falls back to ambient parenting.
    """
    if instance is None:
        return None
    get_info = getattr(instance, "_trace_session_info", None)
    if get_info is None:
        return None
    try:
        if otel_context.get_value(_SESSION_KEY) is not None:
            return None
        info = get_info()
        return None if info is None else _session_context(*info)
    except Exception:
        logger.debug("session span resolution failed", exc_info=True)
        return None


def _session_context(session_id: str, group: bool) -> Context:
    orphan = not trace.get_current_span().get_span_context().is_valid
    context = derive_session_span_context(session_id) if group and orphan else None
    inherited = {
        **_inherited(),
        **dict.fromkeys(_session_id_attribute_keys(), str(session_id)),
    }
    context = otel_context.set_value(_SESSION_KEY, str(session_id), context)
    return otel_context.set_value(_INHERITED_ATTRIBUTES_KEY, inherited, context)


def stamp_inherited_attributes(
    span: trace.Span, parent_context: Context | None = None
) -> None:
    """
    Stamp the attributes inherited from ``parent_context`` (else the current
    context) onto ``span``: the run's session id and anything set with
    :func:`inherited_span_attributes`. Called by
    :class:`grasp_agents.telemetry.InheritedAttributesSpanProcessor` for every
    span, so the whole run tree — provider-instrumentation spans included —
    carries them. No-op when nothing is inherited or the span is not recording.
    """
    if not span.is_recording():
        return
    for key, value in _inherited(parent_context).items():
        span.set_attribute(key, value)


@contextmanager
def _run_span(
    span_name: str, instance: Any | None
) -> Generator[trace.Span, None, None]:
    """
    Open a span, attaching the session context for its duration.

    At a session's outermost run the attached context carries the session id
    (and its trace, when grouped), so this span and every descendant get the
    session attribute(s) — the attach (not just ``context=``) is what lets the
    id reach child spans, since ``start_as_current_span`` re-bases the active
    context on the current one.
    """
    parent_context = _resolve_run_span_context(instance)
    token = otel_context.attach(parent_context) if parent_context is not None else None
    try:
        with trace.get_tracer(_TRACER_NAME).start_as_current_span(
            span_name,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            stamp_inherited_attributes(span)
            yield span
    finally:
        if token is not None:
            otel_context.detach(token)


# ---------------------------------------------------------------------------
# One traced call
# ---------------------------------------------------------------------------


def _is_method(signature: inspect.Signature | None) -> bool:
    if signature is None:
        return False
    first = next(iter(signature.parameters), None)
    return first in {"self", "cls"}


def _display_name(obj: Any) -> str:
    # ``Class.method`` without the enclosing function of a local definition.
    return str(obj.__qualname__).rsplit(".<locals>.", 1)[-1]


def _is_async(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


def _call_arguments(
    signature: inspect.Signature | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    bound: bool,
) -> Any:
    if signature is not None:
        try:
            arguments = dict(signature.bind_partial(*args, **kwargs).arguments)
        except TypeError:
            pass
        else:
            if bound and arguments:
                arguments.pop(next(iter(arguments)))
            return arguments
    return {"args": list(args[1:] if bound else args), "kwargs": kwargs}


class _TracedCall:
    """Describes, observes and closes the span of one traced call."""

    def __init__(
        self,
        *,
        entity_name: str,
        span_kind: SpanKind,
        signature: inspect.Signature | None,
        method: bool,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._bound = method and bool(args)
        self.instance: Any = args[0] if self._bound else None
        self.entity_name = entity_name
        self.kwargs = kwargs
        self.result: SpanResult | None = None
        self._span_kind = span_kind
        self._signature = signature
        self._args = args
        self._observe: Callable[[str, Any], SpanResult | None] | None = getattr(
            self.instance, "_trace_span_result", None
        )

    @property
    def enabled(self) -> bool:
        return _tracing_enabled(self.instance)

    @cached_property
    def start(self) -> SpanStart:
        start = self._described() or SpanStart(
            name=self.entity_name, kind=self._span_kind, input=CALL_ARGUMENTS
        )
        if start.input is CALL_ARGUMENTS:
            arguments = _call_arguments(
                self._signature, self._args, self.kwargs, bound=self._bound
            )
            start = replace(start, input=arguments)
        return start

    def _described(self) -> SpanStart | None:
        describe = getattr(self.instance, "_trace_span_start", None)
        if describe is None:
            return None
        try:
            start = describe(self.entity_name, self._args[1:], self.kwargs)
            if not isinstance(start, SpanStart):
                return None
            # A kind given as a plain string.
            return replace(start, kind=SpanKind(start.kind))
        except Exception:
            logger.debug("span description failed", exc_info=True)
            return None

    def begin(self, span: trace.Span) -> None:
        if not span.is_recording():
            return
        try:
            start = self.start
            span.set_attribute(ATTR_SPAN_KIND, start.kind.value)
            span.set_attribute(
                ATTR_OI_SPAN_KIND, OPENINFERENCE_SPAN_KINDS.get(start.kind, "CHAIN")
            )
            for key, value in start.attributes.items():
                span.set_attribute(key, value)
            _apply_caller_span_overrides(span, self.kwargs)
            record_span_payload(
                span,
                start.input,
                output=False,
                exclude_fields=_exclude_fields_from_instance(self.instance),
            )
        except Exception:
            logger.debug("span start failed", exc_info=True)

    def observe(self, item: Any) -> None:
        """Note a yielded item, or the return value."""
        if self._observe is None:
            self.result = SpanResult(output=item)
            return
        try:
            result = self._observe(self.entity_name, item)
        except Exception:
            logger.debug("span result failed", exc_info=True)
            return
        if isinstance(result, SpanResult):
            self.result = result

    def end(self, span: trace.Span) -> None:
        result = self.result
        if result is None or not span.is_recording():
            return
        try:
            for key, value in result.attributes.items():
                span.set_attribute(key, value)
            record_span_payload(span, result.output, output=True)
        except Exception:
            logger.debug("span end failed", exc_info=True)


def record_span_error(span: trace.Span, error: BaseException) -> None:
    """Mark ``span`` failed: the cause chain, the root cause's type, the event."""
    span.set_status(trace.Status(trace.StatusCode.ERROR, format_error_chain(error)))
    span.set_attribute(ATTR_ERROR_TYPE, type(root_cause(error)).__name__)
    span.record_exception(error)


# ---------------------------------------------------------------------------
# Decorator factories
# ---------------------------------------------------------------------------


def _entity_method[F: Callable[..., Any]](
    name: str | None = None,
    span_kind: SpanKind = SpanKind.TASK,
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        entity_name = name or _display_name(fn)
        try:
            signature: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):
            signature = None
        method = _is_method(signature)

        def traced_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> _TracedCall:
            return _TracedCall(
                entity_name=entity_name,
                span_kind=span_kind,
                signature=signature,
                method=method,
                args=args,
                kwargs=kwargs,
            )

        if inspect.isasyncgenfunction(fn):

            @wraps(fn)
            async def async_gen_wrap(*args: Any, **kwargs: Any) -> Any:
                call = traced_call(args, kwargs)
                if not call.enabled:
                    with _suppressed_instrumentation():
                        async for item in fn(*args, **kwargs):
                            yield item
                    return
                with _run_span(call.start.name, call.instance) as span:
                    call.begin(span)
                    try:
                        async for item in fn(*args, **kwargs):
                            call.observe(item)
                            yield item
                    except asyncio.CancelledError:
                        span.set_attribute(ATTR_CANCELLED, value=True)
                        raise
                    except Exception as e:
                        record_span_error(span, e)
                        raise
                    finally:
                        call.end(span)

            return cast("F", async_gen_wrap)

        if _is_async(fn):

            @wraps(fn)
            async def async_wrap(*args: Any, **kwargs: Any) -> Any:
                call = traced_call(args, kwargs)
                if not call.enabled:
                    with _suppressed_instrumentation():
                        return await fn(*args, **kwargs)
                with _run_span(call.start.name, call.instance) as span:
                    call.begin(span)
                    try:
                        res = await fn(*args, **kwargs)
                    except asyncio.CancelledError:
                        span.set_attribute(ATTR_CANCELLED, value=True)
                        raise
                    except Exception as e:
                        record_span_error(span, e)
                        raise
                    call.observe(res)
                    call.end(span)
                    return res

            return cast("F", async_wrap)

        if inspect.isgeneratorfunction(fn):

            @wraps(fn)
            def sync_gen_wrap(*args: Any, **kwargs: Any) -> Any:
                call = traced_call(args, kwargs)
                if not call.enabled:
                    with _suppressed_instrumentation():
                        yield from fn(*args, **kwargs)
                    return
                with _run_span(call.start.name, call.instance) as span:
                    call.begin(span)
                    try:
                        for item in fn(*args, **kwargs):
                            call.observe(item)
                            yield item
                    except Exception as e:
                        record_span_error(span, e)
                        raise
                    finally:
                        call.end(span)

            return cast("F", sync_gen_wrap)

        @wraps(fn)
        def sync_wrap(*args: Any, **kwargs: Any) -> Any:
            call = traced_call(args, kwargs)
            if not call.enabled:
                with _suppressed_instrumentation():
                    return fn(*args, **kwargs)
            with _run_span(call.start.name, call.instance) as span:
                call.begin(span)
                try:
                    res = fn(*args, **kwargs)
                except Exception as e:
                    record_span_error(span, e)
                    raise
                call.observe(res)
                call.end(span)
                return res

        return cast("F", sync_wrap)

    return decorate


def _entity_class[T: type](
    name: str | None,
    method_name: str,
    span_kind: SpanKind = SpanKind.TASK,
) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            _entity_method(
                name=name or f"{_display_name(cls)}.{method_name}",
                span_kind=span_kind,
            )(method),
        )
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@overload
def traced[F: Callable[..., Any]](
    name: str | None = ...,
    span_kind: SpanKind = ...,
) -> Callable[[F], F]: ...


@overload
def traced[T: type](
    name: str | None = ...,
    span_kind: SpanKind = ...,
    *,
    method_name: str,
) -> Callable[[T], T]: ...


def traced[F: Callable[..., Any], T: type](
    name: str | None = None,
    span_kind: SpanKind = SpanKind.TASK,
    method_name: str | None = None,
) -> Callable[[F], F] | Callable[[T], T]:
    """
    Trace a function or class method with an OTel span named ``name`` (default:
    the function's qualified name), recording its arguments and result.

    A traced method's object may describe its spans itself: ``_trace_span_start
    (entity, args, kwargs) -> SpanStart | None`` (name, kind, attributes, input;
    ``None`` keeps the default description) and ``_trace_span_result(entity,
    item) -> SpanResult | None``, called with the return value or with each
    yielded item (the last ``SpanResult`` is recorded). ``entity`` is the span
    name the decorator would use (its ``name``, else the qualified name).
    """
    if method_name is None:
        return _entity_method(name=name, span_kind=span_kind)
    return _entity_class(name=name, method_name=method_name, span_kind=span_kind)
