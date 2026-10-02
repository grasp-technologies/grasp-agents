"""Production traces in Phoenix: spans for online evaluations, scores back."""

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast
from urllib.parse import quote

from grasp_agents.evals.online import (
    AnnotationsRejectedError,
    SpanRecord,
    TraceAnnotation,
    TraceQuery,
    flatten_attributes,
)

from .annotations import PhoenixProjectNotFoundError
from .client import PhoenixClient, PhoenixError

# More spans than any query returns: the SDK pages through all of them.
_ALL = 10**9
_ANNOTATION_BATCH = 100
_ANNOTATION_ROUTES = {
    "span": ("v1/span_annotations", "span_id"),
    "trace": ("v1/trace_annotations", "trace_id"),
    "session": ("v1/session_annotations", "session_id"),
}
# Our spans leave a successful status unset.
_STATUS_CODES = {"ok": ["OK", "UNSET"], "error": ["ERROR"]}
# Annotation names Phoenix refuses.
_RESERVED_NAMES = frozenset({"note"})

logger = logging.getLogger(__name__)


def span_record(raw: dict[str, Any]) -> SpanRecord:
    """A span as Phoenix's REST API returns it."""
    context = cast("dict[str, Any]", raw.get("context") or {})
    return SpanRecord(
        trace_id=str(context["trace_id"]),
        span_id=str(context["span_id"]),
        parent_id=raw.get("parent_id") or None,
        name=str(raw.get("name") or ""),
        start_time=raw["start_time"],
        end_time=raw.get("end_time"),
        status=str(raw.get("status_code") or "UNSET"),
        status_message=raw.get("status_message") or None,
        attributes=flatten_attributes(
            cast("dict[str, Any]", raw.get("attributes") or {})
        ),
    )


def _annotation_body(annotation: TraceAnnotation, id_key: str) -> dict[str, Any]:
    result = {
        key: value
        for key, value in (
            ("label", annotation.label),
            ("score", annotation.score),
            ("explanation", annotation.explanation),
        )
        if value is not None
    }
    # Phoenix stores session ids without surrounding whitespace.
    target = (
        annotation.target_id.strip()
        if annotation.scope == "session"
        else annotation.target_id
    )
    return {
        "name": annotation.name,
        "annotator_kind": annotation.annotator,
        id_key: target,
        "result": result,
        "metadata": annotation.metadata,
        "identifier": annotation.identifier,
    }


class PhoenixTraceSource:
    """The projects of a Phoenix server as a trace source for online evaluations."""

    def __init__(self, client: PhoenixClient) -> None:
        self.client = client
        self._known: set[str] = set()

    @property
    def location(self) -> str:
        return f"phoenix:{self.client.base_url}"

    async def _project_path(self, project: str) -> str:
        # Percent-encoded once; the SDK and httpx leave existing escapes alone.
        path = quote(project, safe="")
        if project in self._known:
            return path
        await self.client.check_server()
        try:
            await self.client.request_json("GET", f"v1/projects/{path}")
        except PhoenixError as exc:
            if exc.status == 404:
                raise PhoenixProjectNotFoundError(
                    f"No Phoenix project {project!r} at {self.client.base_url}"
                ) from exc
            raise
        self._known.add(project)
        return path

    async def spans(
        self,
        query: TraceQuery,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        trace_ids: Sequence[str] | None = None,
    ) -> list[SpanRecord]:
        project = await self._project_path(query.project)
        found = await self.client.call(
            self.client.sdk.spans.get_spans(
                project_identifier=project,
                start_time=start,
                end_time=end,
                trace_ids=list(trace_ids) if trace_ids else None,
                attributes=query.span_attributes or None,
                status_code=_STATUS_CODES[query.status] if query.status else None,
                limit=_ALL,
                timeout=int(self.client.timeout_s),
            )
        )
        return [span_record(cast("dict[str, Any]", raw)) for raw in found]

    async def annotate(
        self, project: str, annotations: Sequence[TraceAnnotation]
    ) -> int:
        del project
        await self.client.check_server()
        written = 0
        for scope, (route, id_key) in _ANNOTATION_ROUTES.items():
            bodies: list[dict[str, Any]] = []
            for annotation in annotations:
                if annotation.scope != scope:
                    continue
                if annotation.name in _RESERVED_NAMES:
                    logger.warning(
                        "Phoenix reserves the annotation name %r: not written",
                        annotation.name,
                    )
                    continue
                bodies.append(_annotation_body(annotation, id_key))
            for offset in range(0, len(bodies), _ANNOTATION_BATCH):
                written += await self._write(
                    route, bodies[offset : offset + _ANNOTATION_BATCH]
                )
        return written

    async def _write(self, route: str, batch: list[dict[str, Any]]) -> int:
        try:
            response = await self.client.request_json(
                "POST", route, params={"sync": "true"}, json={"data": batch}
            )
        except PhoenixError as exc:
            if exc.status == 404 and len(batch) > 1:
                # Phoenix stores none of a batch with a missing target: write
                # them one by one, leaving out those that are gone.
                return sum([await self._write(route, [body]) for body in batch])
            if exc.status == 404:
                return 0
            if 400 <= exc.status < 500 and exc.status != 429:
                raise AnnotationsRejectedError(str(exc)) from exc
            raise
        stored = cast("list[Any]", response.get("data") or [])
        if len(stored) != len(batch):
            raise PhoenixError(
                200, "POST", route, f"stored {len(stored)} of {len(batch)} annotations"
            )
        return len(stored)
