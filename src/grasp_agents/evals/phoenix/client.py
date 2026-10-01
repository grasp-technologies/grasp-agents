"""Minimal async client for the parts of Phoenix's REST API that evals use."""

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Self, cast

import httpx

logger = logging.getLogger(__name__)

BASE_URL_ENV = "PHOENIX_BASE_URL"
API_KEY_ENV = "PHOENIX_API_KEY"

type ServerVersion = tuple[int, int, int]
type AnnotatorKind = Literal["LLM", "CODE", "HUMAN"]

# Server releases that changed what the client may send.
EXTERNAL_EXAMPLE_IDS: ServerVersion = (15, 0, 0)

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class PhoenixError(RuntimeError):
    def __init__(self, status: int, method: str, path: str, detail: str) -> None:
        super().__init__(f"Phoenix {method} {path} failed ({status}): {detail}")
        self.status = status
        self.detail = detail


class PhoenixConflictError(PhoenixError):
    pass


def _parse_version(text: str) -> ServerVersion:
    parts = [
        int("".join(c for c in p if c.isdigit()) or 0) for p in text.split(".")[:3]
    ]
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


class PhoenixClient:
    """
    Async client for a (self-hosted) Phoenix server.

    Defaults come from ``PHOENIX_BASE_URL`` and ``PHOENIX_API_KEY``. Transient
    failures (429/5xx, timeouts) are retried; anything else raises
    :class:`PhoenixError`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        retries: int = 3,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved = base_url or os.environ.get(BASE_URL_ENV)
        if not resolved:
            raise ValueError(
                f"Phoenix base URL not given and ${BASE_URL_ENV} is not set"
            )
        self.base_url = resolved.rstrip("/")
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout_s)
        self._headers = headers
        self._retries = retries
        self._version: ServerVersion | None = None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # --- Transport ---

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            try:
                response = await self._http.request(
                    method, url, params=params, json=json, headers=self._headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self._retries:
                    raise
                logger.warning("Phoenix %s %s failed (%s); retrying", method, path, exc)
            else:
                if response.status_code < 400:
                    return response
                if (
                    response.status_code not in _RETRY_STATUSES
                    or attempt >= self._retries
                ):
                    detail = response.text[:500]
                    error_cls = (
                        PhoenixConflictError
                        if response.status_code == 409
                        else PhoenixError
                    )
                    raise error_cls(response.status_code, method, path, detail)
                logger.warning(
                    "Phoenix %s %s returned %s; retrying",
                    method,
                    path,
                    response.status_code,
                )
            attempt += 1
            await asyncio.sleep(min(2.0**attempt, 10.0))

    async def _json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
    ) -> dict[str, Any]:
        response = await self._request(method, path, params=params, json=json)
        return cast("dict[str, Any]", response.json())

    # --- Server ---

    async def server_version(self) -> ServerVersion:
        if self._version is None:
            response = await self._request("GET", "/arize_phoenix_version")
            self._version = _parse_version(response.text.strip())
        return self._version

    async def supports(self, minimum: ServerVersion) -> bool:
        return await self.server_version() >= minimum

    # --- Datasets ---

    async def find_dataset(self, name: str) -> dict[str, Any] | None:
        body = await self._json(
            "GET", "/v1/datasets", params={"name": name, "limit": 1}
        )
        found = cast("list[dict[str, Any]]", body.get("data") or [])
        return found[0] if found else None

    async def latest_version_id(self, dataset_id: str) -> str | None:
        body = await self._json(
            "GET", f"/v1/datasets/{dataset_id}/versions", params={"limit": 1}
        )
        versions = cast("list[dict[str, Any]]", body.get("data") or [])
        return str(versions[0]["version_id"]) if versions else None

    async def dataset_examples(
        self, dataset_id: str, version_id: str | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """``(version_id, examples)`` of a dataset version (latest when omitted)."""
        params = {"version_id": version_id} if version_id else None
        body = await self._json(
            "GET", f"/v1/datasets/{dataset_id}/examples", params=params
        )
        data = cast("dict[str, Any]", body["data"])
        return str(data["version_id"]), cast("list[dict[str, Any]]", data["examples"])

    async def upload_dataset(
        self,
        *,
        action: Literal["create", "append", "update"],
        name: str,
        inputs: Sequence[Mapping[str, Any]],
        outputs: Sequence[Mapping[str, Any]],
        metadata: Sequence[Mapping[str, Any]],
        description: str | None = None,
        splits: Sequence[Sequence[str] | None] | None = None,
        example_ids: Sequence[str] | None = None,
    ) -> tuple[str, str]:
        """Upload examples synchronously; returns ``(dataset_id, version_id)``."""
        payload: dict[str, Any] = {
            "action": action,
            "name": name,
            "inputs": list(inputs),
            "outputs": list(outputs),
            "metadata": list(metadata),
        }
        if description:
            payload["description"] = description
        if splits is not None:
            payload["splits"] = [list(s) if s else None for s in splits]
        if example_ids is not None:
            payload["example_ids"] = list(example_ids)
        body = await self._json(
            "POST", "/v1/datasets/upload", params={"sync": "true"}, json=payload
        )
        data = cast("dict[str, Any]", body["data"])
        return str(data["dataset_id"]), str(data["version_id"])

    # --- Experiments ---

    async def create_experiment(
        self,
        dataset_id: str,
        *,
        version_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        repetitions: int = 1,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"repetitions": repetitions}
        if version_id:
            payload["version_id"] = version_id
        if name:
            payload["name"] = name
        if description:
            payload["description"] = description
        if metadata:
            payload["metadata"] = dict(metadata)
        body = await self._json(
            "POST", f"/v1/datasets/{dataset_id}/experiments", json=payload
        )
        return cast("dict[str, Any]", body["data"])

    async def create_run(
        self,
        experiment_id: str,
        *,
        dataset_example_id: str,
        output: Any,
        repetition_number: int,
        start_time: datetime,
        end_time: datetime,
        trace_id: str | None = None,
        error: str | None = None,
    ) -> str | None:
        """
        Log one externally executed run. Returns its id, or ``None`` when a
        successful run already exists for this (example, repetition).
        """
        payload = {
            "dataset_example_id": dataset_example_id,
            "output": output,
            "repetition_number": repetition_number,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "trace_id": trace_id,
            "error": error,
        }
        try:
            body = await self._json(
                "POST", f"/v1/experiments/{experiment_id}/runs", json=payload
            )
        except PhoenixConflictError:
            return None
        return str(cast("dict[str, Any]", body["data"])["id"])

    async def list_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        body = await self._json("GET", f"/v1/experiments/{experiment_id}/runs")
        return cast("list[dict[str, Any]]", body.get("data") or [])

    async def upsert_evaluation(
        self,
        *,
        experiment_run_id: str,
        name: str,
        annotator_kind: AnnotatorKind,
        start_time: datetime,
        end_time: datetime,
        score: float | None = None,
        label: str | None = None,
        explanation: str | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "experiment_run_id": experiment_run_id,
            "name": name,
            "annotator_kind": annotator_kind,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "trace_id": trace_id,
        }
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = {
                "score": score,
                "label": label,
                "explanation": explanation,
            }
        if metadata:
            payload["metadata"] = dict(metadata)
        body = await self._json("POST", "/v1/experiment_evaluations", json=payload)
        return str(cast("dict[str, Any]", body["data"])["id"])

    # --- Annotations ---

    async def log_span_annotations(
        self, annotations: Sequence[Mapping[str, Any]]
    ) -> None:
        """Upsert span annotations, keyed by name, span and ``identifier``."""
        await self._json(
            "POST",
            "/v1/span_annotations",
            params={"sync": "true"},
            json={"data": list(annotations)},
        )
