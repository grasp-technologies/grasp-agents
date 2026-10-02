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
_GATEWAY_ERRORS = frozenset({502, 503, 504})
_MAX_DELAY_S = 60.0
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)\S*")


class PhoenixError(RuntimeError):
    def __init__(self, status: int, method: str, path: str, detail: str) -> None:
        super().__init__(f"Phoenix {method} {path} failed ({status}): {detail}")
        self.status = status
        self.detail = detail


class PhoenixCompatibilityError(RuntimeError):
    pass


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
    idempotent and failed on a gateway error or timeout.
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
                if request.method not in _IDEMPOTENT or attempt >= self._retries:
                    raise
            else:
                status = response.status_code
                retryable = status == 429 or (
                    status in _GATEWAY_ERRORS and request.method in _IDEMPOTENT
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
    return PhoenixError(
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
            import phoenix.client as phoenix_client  # noqa: PLC0415
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
            raise PhoenixError(response.status_code, method, path, response.text[:500])
        if "json" not in response.headers.get("content-type", ""):
            # Unknown routes fall through to Phoenix's web app (HTML, 200).
            raise PhoenixError(
                response.status_code,
                method,
                path,
                f"expected JSON, got {response.headers.get('content-type')!r}",
            )
        return cast("dict[str, Any]", response.json())

    async def server_version(self) -> ServerVersion:
        if self._version is None:
            response = await self.http.get("arize_phoenix_version")
            text = response.text.strip()
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

    async def update_experiment_metadata(
        self, experiment_id: str, metadata: Mapping[str, Any]
    ) -> None:
        await self.request_json(
            "PATCH",
            f"v1/experiments/{experiment_id}",
            json={"metadata": dict(metadata)},
        )
