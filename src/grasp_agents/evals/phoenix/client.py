"""
Connection to a Phoenix server: the official ``phoenix.client`` SDK on an
HTTP client that only retries requests that are safe to repeat.
"""

import asyncio
import email.utils
import logging
import math
import os
import re
import time
from collections.abc import Awaitable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Self, cast, override
from urllib.parse import quote

import httpx

if TYPE_CHECKING:
    from phoenix.client import AsyncClient
    from phoenix.client.resources.datasets import Dataset as PhoenixDataset

logger = logging.getLogger(__name__)

BASE_URL_ENV = "PHOENIX_BASE_URL"
API_KEY_ENV = "PHOENIX_API_KEY"

type ServerVersion = tuple[int, int, int]
type AnnotatorKind = Literal["LLM", "CODE", "HUMAN"]

MIN_SERVER_VERSION: ServerVersion = (20, 0, 0)

_IDEMPOTENT = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
# POSTs the server makes idempotent: a repeated experiment run is refused
# (409) once it succeeded and overwritten otherwise; an evaluation is an
# upsert on (run, name).
_IDEMPOTENT_POSTS = (
    re.compile(r"/v1/experiments/[^/]+/runs$"),
    re.compile(r"/v1/experiment_evaluations$"),
)
_GATEWAY_ERRORS = frozenset({502, 503, 504})


def _repeatable(request: httpx.Request) -> bool:
    if request.method in _IDEMPOTENT:
        return True
    path = request.url.path
    return request.method == "POST" and any(p.search(path) for p in _IDEMPOTENT_POSTS)


_MAX_DELAY_S = 60.0
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)\S*")


class PhoenixError(RuntimeError):
    """A failed Phoenix request; ``status`` is 0 when no response came."""

    def __init__(self, status: int, method: str, path: str, detail: str) -> None:
        code = f" ({status})" if status else ""
        super().__init__(f"Phoenix {method} {path} failed{code}: {detail}")
        self.status = status
        self.method = method
        self.path = path
        self.detail = detail


class PhoenixCompatibilityError(RuntimeError):
    pass


class PhoenixUnavailableError(PhoenixError):
    """
    The server answered with a server error (5xx) after retries, or no
    response came.
    """


def phoenix_error(status: int, method: str, path: str, detail: str) -> PhoenixError:
    """A :class:`PhoenixError`, :class:`PhoenixUnavailableError` for a 5xx."""
    cls = PhoenixUnavailableError if status >= 500 else PhoenixError
    return cls(status, method, path, detail)


def normalize_base_url(url: str) -> str:
    """Canonical spelling of a server URL (lowercase scheme/host, no trailing /)."""
    parsed = httpx.URL(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.host:
        raise ValueError(f"Not an http(s) server URL: {url!r}")
    if parsed.userinfo:
        raise ValueError(
            "Put Phoenix credentials in PHOENIX_API_KEY, not in the server URL"
        )
    host = parsed.host.lower()
    if ":" in host:  # IPv6
        host = f"[{host}]"
    port = parsed.port
    default = {"http": 80, "https": 443}.get(parsed.scheme.lower())
    netloc = host + (f":{port}" if port and port != default else "")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{netloc}{path}"


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "").strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        seconds = when.timestamp() - time.time()
    if not math.isfinite(seconds):
        return None
    return min(max(0.0, seconds), _MAX_DELAY_S)


class _RetryingClient(httpx.AsyncClient):
    """
    Retries a request only when repeating it cannot duplicate a write: it
    never reached the server, the server rate-limited it (429), or it is
    idempotent (by method, or a POST the server makes idempotent) and failed
    on a gateway error, a timeout or a dropped connection.
    """

    def __init__(self, *, retries: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._retries = retries

    @override
    async def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        auth: Any = httpx.USE_CLIENT_DEFAULT,
        follow_redirects: Any = httpx.USE_CLIENT_DEFAULT,
    ) -> httpx.Response:
        attempt = 0
        while True:
            delay = min(2.0**attempt, 10.0)
            try:
                response = await super().send(
                    request,
                    stream=stream,
                    auth=auth,
                    follow_redirects=follow_redirects,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if attempt >= self._retries:
                    raise
            except (httpx.TimeoutException, httpx.NetworkError):
                if not _repeatable(request) or attempt >= self._retries:
                    raise
            else:
                status = response.status_code
                retryable = status == 429 or (
                    status in _GATEWAY_ERRORS and _repeatable(request)
                )
                if not retryable or attempt >= self._retries:
                    return response
                requested = _retry_after(response)
                delay = requested if requested is not None else delay
                await response.aclose()
            attempt += 1
            logger.warning(
                "Phoenix %s %s failed; retrying in %.1fs",
                request.method,
                request.url.path,
                delay,
            )
            await asyncio.sleep(delay)


def _phoenix_error(
    exc: httpx.HTTPStatusError, detail: str | None = None
) -> PhoenixError:
    return phoenix_error(
        exc.response.status_code,
        exc.request.method,
        exc.request.url.path,
        detail or exc.response.text[:500],
    )


def _same_server(url: str, base_url: str) -> bool:
    try:
        return normalize_base_url(url) == base_url
    except ValueError:
        return False


class PhoenixClient:
    """
    A Phoenix server (20.0 or later). Defaults come from ``PHOENIX_BASE_URL``
    and ``PHOENIX_API_KEY``; that key is never sent to another server.
    Proxies are taken from the environment (``HTTPS_PROXY`` etc.) unless a
    ``transport`` is given.

    ``sdk`` is the official async client (``phoenix.client.AsyncClient``) on
    this connection; failed requests surface as :class:`PhoenixError`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        env_url = os.environ.get(BASE_URL_ENV)
        resolved = base_url or env_url
        if not resolved:
            raise ValueError(
                f"Phoenix base URL not given and ${BASE_URL_ENV} is not set"
            )
        try:
            import phoenix.client as phoenix_client  # ruff: ignore[import-outside-top-level]
        except ImportError as exc:
            raise ImportError(
                "Phoenix support needs the phoenix extra: "
                "pip install 'grasp-agents[phoenix]'"
            ) from exc
        self.base_url = normalize_base_url(resolved)
        self.timeout_s = timeout_s
        key = api_key
        if key is None and (not env_url or _same_server(env_url, self.base_url)):
            key = os.environ.get(API_KEY_ENV)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.http = _RetryingClient(
            retries=retries,
            base_url=f"{self.base_url}/",
            headers=headers,
            timeout=timeout_s,
            transport=transport,
            follow_redirects=False,
        )
        self.sdk: AsyncClient = phoenix_client.AsyncClient(http_client=self.http)
        self._version: ServerVersion | None = None

    async def aclose(self) -> None:
        await self.http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def call[T](self, request: Awaitable[T]) -> T:
        """Await an SDK call, reporting HTTP failures as :class:`PhoenixError`."""
        try:
            return await request
        except httpx.HTTPStatusError as exc:
            raise _phoenix_error(exc) from exc
        except Exception as exc:
            # The SDK wraps some failures (e.g. ``DatasetUploadError``).
            if isinstance(exc.__cause__, httpx.HTTPStatusError):
                raise _phoenix_error(exc.__cause__, str(exc)) from exc
            raise

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
    ) -> dict[str, Any]:
        response = await self.http.request(method, path, params=params, json=json)
        if not response.is_success:
            raise phoenix_error(response.status_code, method, path, response.text[:500])
        return cast("dict[str, Any]", response.json())

    async def graphql(
        self, query: str, variables: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """The ``data`` of a GraphQL request (errors raise :class:`PhoenixError`)."""
        body = await self.request_json(
            "POST", "graphql", json={"query": query, "variables": dict(variables or {})}
        )
        errors = cast("list[dict[str, Any]]", body.get("errors") or [])
        if errors:
            detail = "; ".join(str(e.get("message")) for e in errors)
            raise PhoenixError(200, "POST", "graphql", detail)
        return cast("dict[str, Any]", body.get("data") or {})

    async def delete_project(self, name: str) -> bool:
        """Delete a project and its traces; ``False`` when there is none."""
        response = await self.http.delete(f"v1/projects/{quote(name, safe='')}")
        if response.status_code == 404:
            return False
        if not response.is_success:
            raise phoenix_error(
                response.status_code, "DELETE", "v1/projects", response.text[:500]
            )
        return True

    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment with its project; ``False`` when there is none."""
        response = await self.http.delete(
            f"v1/experiments/{quote(experiment_id, safe='')}",
            params={"delete_project": "true"},
        )
        if response.status_code == 404:
            return False
        if not response.is_success:
            raise phoenix_error(
                response.status_code, "DELETE", "v1/experiments", response.text[:500]
            )
        return True

    async def delete_dataset(self, dataset_id: str) -> None:
        """
        Delete a dataset with its experiments and their projects (Phoenix's REST
        delete leaves the projects behind; its GraphQL mutation does not).
        """
        await self.graphql(
            "mutation($id: ID!) { deleteDataset(input: {datasetId: $id}) "
            "{ dataset { id } } }",
            {"id": dataset_id},
        )

    async def retention(self, project: str) -> dict[str, Any] | None:
        """
        The trace retention policy of a project (``{name, rule}``), or
        ``None`` when there is no such project.
        """
        data = await self.graphql(
            "query($name: String!) { getProjectByName(name: $name) "
            "{ traceRetentionPolicy { name rule { __typename "
            "... on TraceRetentionRuleMaxDays { maxDays } "
            "... on TraceRetentionRuleMaxCount { maxCount } "
            "... on TraceRetentionRuleMaxDaysOrCount { maxDays maxCount } } } } }",
            {"name": project},
        )
        found = cast("dict[str, Any] | None", data.get("getProjectByName"))
        if found is None:
            return None
        return cast("dict[str, Any]", found["traceRetentionPolicy"])

    async def trace_project(self, trace_id: str) -> str | None:
        """The project holding a trace (by its OTel id), or ``None``."""
        data = await self.graphql(
            "query($id: String!) { getTraceByOtelId(traceId: $id) "
            "{ project { name } } }",
            {"id": trace_id.lower()},
        )
        found = cast("dict[str, Any] | None", data.get("getTraceByOtelId"))
        return None if found is None else str(found["project"]["name"])

    async def server_version(self) -> ServerVersion:
        if self._version is None:
            response = await self.http.get("arize_phoenix_version")
            text = response.text.strip()
            if response.status_code >= 500:
                raise PhoenixUnavailableError(
                    response.status_code,
                    "GET",
                    "arize_phoenix_version",
                    f"{self.base_url} is unavailable: {text[:80]!r}",
                )
            match = _VERSION_RE.fullmatch(text) if response.is_success else None
            if match is None:
                raise PhoenixCompatibilityError(
                    f"{self.base_url} did not answer like a Phoenix server "
                    f"(HTTP {response.status_code}: {text[:80]!r})"
                )
            major, minor, patch = (int(g) for g in match.groups())
            self._version = (major, minor, patch)
        return self._version

    async def check_server(self) -> None:
        version = await self.server_version()
        if version < MIN_SERVER_VERSION:
            raise PhoenixCompatibilityError(
                f"Phoenix {'.'.join(map(str, version))} at {self.base_url} is older "
                f"than {'.'.join(map(str, MIN_SERVER_VERSION))}; upgrade the server"
            )

    async def upsert_dataset(
        self,
        *,
        name: str,
        examples: list[dict[str, Any]],
        description: str | None = None,
    ) -> "PhoenixDataset":
        """
        Create dataset ``name``, or make it hold exactly ``examples`` (a new
        version when anything changed; examples carrying an ``id`` keep their
        identity).
        """
        create = self.sdk.datasets.create_dataset  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return await self.call(
            create(
                name=name,
                examples=examples,
                dataset_description=description,
                timeout=int(self.timeout_s),
            )
        )

    async def find_dataset(self, name: str) -> dict[str, Any] | None:
        body = await self.request_json("GET", "v1/datasets", params={"name": name})
        found = cast("list[dict[str, Any]]", body.get("data") or [])
        return found[0] if found else None

    async def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        body = await self.request_json(
            "GET", f"v1/experiments/{quote(experiment_id, safe='')}"
        )
        return cast("dict[str, Any]", body.get("data") or {})

    async def update_experiment(
        self,
        experiment_id: str,
        *,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Set an experiment's description and metadata (which is replaced)."""
        body: dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = dict(metadata)
        if body:
            await self.request_json(
                "PATCH", f"v1/experiments/{quote(experiment_id, safe='')}", json=body
            )
