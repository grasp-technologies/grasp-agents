import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, TypeAdapter

from grasp_agents.evals import Dataset, Example, Score, Trial
from grasp_agents.evals.phoenix import (
    PhoenixClient,
    PhoenixCompatibilityError,
    PhoenixError,
    from_phoenix_record,
    normalize_base_url,
    phoenix_source,
    to_phoenix_record,
)
from grasp_agents.evals.phoenix.client import (
    _retry_after,  # pyright: ignore[reportPrivateUsage]
)
from grasp_agents.evals.phoenix.sync import (
    GRASP_KEY,
    _digest,  # pyright: ignore[reportPrivateUsage]
    _is_example_node_id,  # pyright: ignore[reportPrivateUsage]
    parse_phoenix_source,
    record_id,
    record_node_id,
)


class Question(BaseModel):
    text: str


def _round_trip(
    example: Example[Any, Any], input_type: Any, reference_type: Any
) -> Example[Any, Any]:
    record = to_phoenix_record(example)
    payload = {
        "id": record.id,
        "node_id": "RGF0YXNldEV4YW1wbGU6MQ==",
        "input": record.input,
        "output": record.output,
        "metadata": record.metadata,
    }
    return from_phoenix_record(
        payload,
        input_adapter=TypeAdapter(input_type),
        reference_adapter=TypeAdapter(reference_type),
    )


def test_object_inputs_are_stored_as_is() -> None:
    example = Example(
        id="q1",
        input=Question(text="2+2"),
        reference={"answer": 4},
        metadata={"topic": "math"},
        splits=["dev"],
    )
    record = to_phoenix_record(example)
    assert record.input == {"text": "2+2"}
    assert record.output == {"answer": 4}
    assert record.metadata["topic"] == "math"
    assert record.metadata[GRASP_KEY] == {"splits": ["dev"]}
    back = _round_trip(example, Question, dict[str, int])
    assert back == example


def test_scalars_are_wrapped_and_unwrapped() -> None:
    example = Example[str, str](id="s", input="hello", reference="HELLO")
    record = to_phoenix_record(example)
    assert record.input == {"value": "hello"}
    assert record.output == {"value": "HELLO"}
    assert _round_trip(example, str, str) == example


def test_missing_reference_survives() -> None:
    example = Example[int, int](id="n", input=3)
    record = to_phoenix_record(example)
    assert record.output == {}
    back = _round_trip(example, int, int)
    assert back.reference is None
    assert back == example


def test_foreign_records_use_phoenix_ids() -> None:
    record = {
        "id": "ext-7",
        "node_id": "RGF0YXNldEV4YW1wbGU6Nw==",
        "input": {"question": "why?"},
        "output": {"answer": "because"},
        "metadata": {"source": "ui"},
    }
    example = from_phoenix_record(
        record, input_adapter=TypeAdapter(Any), reference_adapter=TypeAdapter(Any)
    )
    assert example.id == "ext-7"
    assert example.input == {"question": "why?"}
    assert example.reference == {"answer": "because"}
    assert record_id(record) == "ext-7"
    assert record_node_id(record) == "RGF0YXNldEV4YW1wbGU6Nw=="


def test_content_identity_is_stable() -> None:
    a = to_phoenix_record(
        Example[str, None](id="x", input="i", metadata={"b": 1, "a": 2})
    )
    b = to_phoenix_record(
        Example[str, None](id="x", input="i", metadata={"a": 2, "b": 1})
    )
    assert a.content() == b.content()


# --- Connection and identity ---


def _client(handler: Any, *, retries: int = 3) -> PhoenixClient:
    return PhoenixClient(
        "http://phoenix.test", transport=httpx.MockTransport(handler), retries=retries
    )


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr("grasp_agents.evals.phoenix.client.asyncio.sleep", instant)


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_backoff")
async def test_only_safe_requests_are_retried() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(503)

    async with _client(handler) as client:
        assert (await client.http.get("v1/datasets")).status_code == 503
        assert calls == ["GET"] * 4
        calls.clear()
        # The server may have committed a POST before failing: never repeat it.
        assert (
            await client.http.post("v1/datasets/upload", json={})
        ).status_code == 503
        assert calls == ["POST"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_backoff")
async def test_requests_that_never_arrived_or_were_rate_limited_are_retried() -> None:
    attempts = {"connect": 0, "limited": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("connect") and attempts["connect"] < 2:
            attempts["connect"] += 1
            raise httpx.ConnectError("refused", request=request)
        if request.url.path.endswith("limited") and attempts["limited"] < 1:
            attempts["limited"] += 1
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={})

    async with _client(handler) as client:
        assert (await client.http.post("connect", json={})).status_code == 200
        assert (await client.http.post("limited", json={})).status_code == 200
    assert attempts == {"connect": 2, "limited": 1}


@pytest.mark.asyncio
async def test_old_or_foreign_servers_are_refused() -> None:
    def old(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="12.18.0")

    def foreign(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>login</html>")

    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://elsewhere"})

    for handler, message in [
        (old, "older than"),
        (foreign, "Phoenix"),
        (redirect, "302"),
    ]:
        async with _client(handler) as client:
            with pytest.raises(PhoenixCompatibilityError, match=message):
                await client.check_server()


def test_server_identity_is_normalized() -> None:
    assert (
        normalize_base_url("HTTP://Phoenix.Example.com:80/")
        == "http://phoenix.example.com"
    )
    assert (
        normalize_base_url("https://host:8443/phoenix/") == "https://host:8443/phoenix"
    )
    assert normalize_base_url("http://[::1]:6006/") == "http://[::1]:6006"
    for wrong in ("localhost:6006", "http://user:secret@host"):
        with pytest.raises(ValueError):
            normalize_base_url(wrong)
    source = phoenix_source("https://host/phoenix", "RGF0YXNldDox")
    assert parse_phoenix_source(source) == ("https://host/phoenix", "RGF0YXNldDox")
    assert parse_phoenix_source("data/file.jsonl") is None


def test_phoenix_example_ids_are_recognized() -> None:
    assert _is_example_node_id("RGF0YXNldEV4YW1wbGU6Nw==")  # DatasetExample:7
    assert not _is_example_node_id("RGF0YXNldDox")  # Dataset:1
    assert not _is_example_node_id("q1")


def test_metadata_is_added_only_when_needed() -> None:
    plain = to_phoenix_record(Example(id="q", input={"text": "a"}, reference={"x": 1}))
    assert GRASP_KEY not in plain.metadata
    empty = to_phoenix_record(Example(id="e", input={"text": "a"}, reference={}))
    assert empty.metadata[GRASP_KEY] == {"empty_reference": True}
    back = from_phoenix_record(
        {
            "id": "e",
            "input": empty.input,
            "output": empty.output,
            "metadata": empty.metadata,
        },
        input_adapter=TypeAdapter(Any),
        reference_adapter=TypeAdapter(Any),
    )
    assert back.reference == {}


def test_pulled_examples_hash_like_local_ones() -> None:
    local = Example(id="q1", input=Question(text="2+2"), reference={"answer": 4})
    assert (
        _round_trip(local, Question, dict[str, int]).content_hash == local.content_hash
    )


def test_trial_digests_change_with_scores() -> None:
    trial = Trial(
        example_id="a",
        example_hash="h",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        duration_s=0.1,
        output=1,
        scores=[Score(name="ok", value=True)],
    )
    rescored = trial.model_copy(update={"scores": [Score(name="ok", value=False)]})
    assert _digest(trial) != _digest(rescored)


class Draft(BaseModel):
    text: str
    lang: str = "en"


def test_examples_are_pushed_as_stored() -> None:
    (loaded,) = Dataset.from_records(
        [{"id": "q", "input": {"text": "a", "lang": "en"}}], input_type=Draft
    )
    record = to_phoenix_record(loaded)
    assert record.input == {"text": "a", "lang": "en"}  # the default stays
    pulled = _round_trip(loaded, Draft, Any)
    assert pulled.content_hash == loaded.content_hash
    assert to_phoenix_record(pulled).content() == record.content()


def test_retry_after_is_read_defensively() -> None:
    def wait(value: str) -> float | None:
        return _retry_after(httpx.Response(429, headers={"retry-after": value}))

    assert wait("1.5") == pytest.approx(1.5)
    assert wait("-5") == pytest.approx(0.0)
    assert wait("soon") is None
    assert wait("86400") == pytest.approx(60.0)


@pytest.mark.asyncio
async def test_the_api_key_stays_with_its_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOENIX_BASE_URL", "http://team.test")
    monkeypatch.setenv("PHOENIX_API_KEY", "team-secret")
    async with PhoenixClient() as default, PhoenixClient("http://TEAM.test/") as same:
        assert default.http.headers["authorization"] == "Bearer team-secret"
        assert same.http.headers["authorization"] == "Bearer team-secret"
    async with PhoenixClient("http://other.test") as other:
        assert "authorization" not in other.http.headers
    async with PhoenixClient("http://other.test", api_key="k") as explicit:
        assert explicit.http.headers["authorization"] == "Bearer k"


@pytest.mark.asyncio
async def test_environment_proxies_are_used(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def proxy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        seen.append((await reader.readline()).decode().strip())
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\n20.2.1"
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(proxy, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    for name in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{port}")
    for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    try:
        async with PhoenixClient("http://phoenix.internal.test", retries=0) as client:
            assert await client.server_version() == (20, 2, 1)
    finally:
        server.close()
        await server.wait_closed()
    assert seen == ["GET http://phoenix.internal.test/arize_phoenix_version HTTP/1.1"]


@pytest.mark.asyncio
async def test_wrapped_sdk_failures_are_phoenix_errors() -> None:
    async def upload() -> None:
        request = httpx.Request("POST", "http://phoenix.test/v1/datasets/upload")
        response = httpx.Response(422, text="bad rows", request=request)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError("Dataset upload failed: bad rows") from exc

    async with _client(lambda _: httpx.Response(200)) as client:
        with pytest.raises(PhoenixError, match="422"):
            await client.call(upload())
