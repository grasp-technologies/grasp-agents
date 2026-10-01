"""Human annotations made in the Phoenix UI, read back as labels."""

from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any, cast
from urllib.parse import quote

from pydantic import BaseModel, Field

from .client import PhoenixClient, PhoenixError

# Trace ids per request (the server accepts up to 1000; the SDK pages span
# ids by 100).
_TRACE_BATCH = 100
# More spans than any batch has: the SDK pages through all of them.
_ALL_SPANS = 10**9


class PhoenixProjectNotFoundError(LookupError):
    """The Phoenix project does not exist."""


class PhoenixAnnotation(BaseModel):
    """One human annotation on a trace, or on one of its spans."""

    trace_id: str
    span_id: str | None = None
    name: str
    label: str | None = None
    score: float | None = None
    explanation: str | None = None
    user_id: str | None = None
    updated_at: datetime


class HumanAnnotations(BaseModel):
    """Human annotations found on a set of traces."""

    # Trace id → annotation name → the newest human annotation.
    by_trace: dict[str, dict[str, PhoenixAnnotation]] = Field(
        default_factory=dict[str, dict[str, PhoenixAnnotation]]
    )
    # Traces the project does not have (another project's, or deleted).
    missing_traces: list[str] = Field(default_factory=list[str])


def _batches(items: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _annotation(
    raw: dict[str, Any], *, trace_id: str, span_id: str | None
) -> PhoenixAnnotation:
    result = cast("dict[str, Any]", raw.get("result") or {})
    return PhoenixAnnotation(
        trace_id=trace_id,
        span_id=span_id,
        name=str(raw["name"]),
        label=result.get("label"),
        score=result.get("score"),
        explanation=result.get("explanation"),
        user_id=raw.get("user_id"),
        updated_at=raw["updated_at"],
    )


async def _trace_annotations(
    client: PhoenixClient, project: str, trace_ids: list[str], names: Sequence[str]
) -> list[PhoenixAnnotation]:
    found: list[PhoenixAnnotation] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {
            "trace_ids": trace_ids,
            "include_annotation_names": list(names),
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        try:
            body = await client.request_json(
                "GET", f"v1/projects/{project}/trace_annotations", params=params
            )
        except PhoenixError as exc:
            if exc.status == 404:
                # None of these traces exist (any more): nothing to read.
                return found
            raise
        for raw in cast("list[dict[str, Any]]", body.get("data") or []):
            if raw.get("annotator_kind") == "HUMAN":
                found.append(
                    _annotation(raw, trace_id=str(raw["trace_id"]), span_id=None)
                )
        cursor = body.get("next_cursor")
        if not cursor:
            return found


async def _span_annotations(
    client: PhoenixClient,
    project: str,
    trace_ids: list[str],
    names: Sequence[str],
    seen: set[str],
) -> list[PhoenixAnnotation]:
    try:
        spans = await client.call(
            client.sdk.spans.get_spans(
                project_identifier=project,
                trace_ids=trace_ids,
                limit=_ALL_SPANS,
                timeout=int(client.timeout_s),
            )
        )
    except PhoenixError as exc:
        if exc.status == 404:
            return []
        raise
    trace_of = {
        span["context"]["span_id"]: span["context"]["trace_id"] for span in spans
    }
    seen.update(trace_of.values())
    if not trace_of:
        return []
    try:
        annotations = await client.call(
            client.sdk.spans.get_span_annotations(
                span_ids=list(trace_of),
                project_identifier=project,
                include_annotation_names=list(names),
                timeout=int(client.timeout_s),
            )
        )
    except PhoenixError as exc:
        if exc.status == 404:
            return []
        raise
    found: list[PhoenixAnnotation] = []
    for raw in cast("list[dict[str, Any]]", annotations):
        span_id = str(raw["span_id"])
        if raw.get("annotator_kind") == "HUMAN" and span_id in trace_of:
            found.append(_annotation(raw, trace_id=trace_of[span_id], span_id=span_id))
    return found


async def human_annotations(
    client: PhoenixClient,
    project: str,
    trace_ids: Sequence[str],
    names: Sequence[str],
) -> HumanAnnotations:
    """
    The newest human annotation named one of ``names`` on each trace — made
    on the trace itself or on any of its spans — and the traces the project
    does not have. Raises :class:`PhoenixProjectNotFoundError` for an unknown
    project.
    """
    await client.check_server()
    # Percent-encoded once; the SDK and httpx leave existing escapes alone.
    path_project = quote(project, safe="")
    try:
        await client.request_json("GET", f"v1/projects/{path_project}")
    except PhoenixError as exc:
        if exc.status == 404:
            raise PhoenixProjectNotFoundError(
                f"No Phoenix project {project!r} at {client.base_url}"
            ) from exc
        raise
    unique = sorted(set(trace_ids))
    result = HumanAnnotations()
    seen: set[str] = set()
    for batch in _batches(unique, _TRACE_BATCH):
        for annotation in [
            *await _trace_annotations(client, path_project, batch, names),
            *await _span_annotations(client, path_project, batch, names, seen),
        ]:
            by_name = result.by_trace.setdefault(annotation.trace_id, {})
            known = by_name.get(annotation.name)
            if known is None or annotation.updated_at > known.updated_at:
                by_name[annotation.name] = annotation
    result.missing_traces = [t for t in unique if t not in seen]
    return result
