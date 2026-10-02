"""
Against real Phoenix servers (20.0 or later). Set ``GRASP_EVALS_PHOENIX_URLS``
to a comma-separated list of base URLs and run
``pytest -m integration tests/evals/test_phoenix_live.py``.
"""

import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from grasp_agents.evals import (
    Dataset,
    EvalContext,
    Example,
    FunctionTask,
    LocalRunStore,
    evaluate,
    evaluator,
)
from grasp_agents.evals.phoenix import (
    DatasetPushError,
    PhoenixClient,
    PhoenixError,
    StaleDatasetError,
    phoenix_source,
    pull_dataset,
    push_dataset,
    push_run,
)

_URLS = [u for u in os.environ.get("GRASP_EVALS_PHOENIX_URLS", "").split(",") if u]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _URLS, reason="GRASP_EVALS_PHOENIX_URLS not set"),
]


class Problem(BaseModel):
    a: int
    b: int


def _dataset(
    name: str, n: int = 4, *, sealed_from: int | None = None
) -> Dataset[Problem, int]:
    return Dataset(
        [
            Example(
                id=f"p{i}",
                input=Problem(a=i, b=1),
                reference=i + 1,
                splits=["test"]
                if sealed_from is not None and i >= sealed_from
                else ["dev"],
            )
            for i in range(n)
        ],
        name=name,
    )


async def add(problem: Problem) -> int:
    return problem.a + problem.b if problem.a != 2 else -1


@evaluator(version="3")
def exact(ctx: EvalContext[Problem, int, int]) -> bool:
    return ctx.output == ctx.reference


@evaluator(name="judge", version="1", annotator="LLM")
def judge(ctx: EvalContext[Problem, int, int]) -> dict[str, float | str]:
    return {
        "closeness": 1.0 / (1 + abs(ctx.output - (ctx.reference or 0))),
        "tone": "ok",
    }


class _Offline(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(params=_URLS or ["unset"])
def base_url(request: pytest.FixtureRequest) -> str:
    return str(request.param)


async def _task_runs(client: PhoenixClient, experiment_id: str) -> list[Any]:
    experiment = await client.sdk.experiments.get_experiment(
        experiment_id=experiment_id
    )
    return list(experiment["task_runs"])


@pytest.mark.asyncio
async def test_dataset_round_trip_and_cache(base_url: str, tmp_path: Path) -> None:
    name = _name("roundtrip")
    async with PhoenixClient(base_url) as client:
        pushed = await push_dataset(client, _dataset(name))
        pulled = await pull_dataset(
            client, name, input_type=Problem, reference_type=int, cache_dir=tmp_path
        )
    assert pushed.created == 4
    assert pulled.version == pushed.version_id
    assert pulled.source == phoenix_source(client.base_url, pushed.dataset_id)
    assert pulled.ids == _dataset(name).ids
    assert pulled.fingerprint == _dataset(name).fingerprint
    assert pulled["p1"].input == Problem(a=1, b=1)
    assert pulled["p1"].splits == ["dev"]

    async with PhoenixClient(base_url, transport=_Offline(), retries=0) as offline:
        cached = await pull_dataset(
            offline,
            name,
            version=pushed.version_id,
            input_type=Problem,
            reference_type=int,
            cache_dir=tmp_path,
        )
    assert cached.fingerprint == pulled.fingerprint


@pytest.mark.asyncio
async def test_dataset_push_guards(base_url: str, tmp_path: Path) -> None:
    name = _name("guards")
    async with PhoenixClient(base_url) as client:
        await push_dataset(client, _dataset(name, 3))
        pulled = await pull_dataset(
            client, name, input_type=Problem, reference_type=int, cache_dir=tmp_path
        )
        before = {
            e["id"]: e["node_id"]
            for e in (
                await client.sdk.datasets.get_dataset(dataset=name, timeout=30)
            ).examples
        }

        unchanged = await push_dataset(client, pulled)
        assert unchanged.version_id == pulled.version
        assert unchanged.unchanged == 3

        edited = Dataset(
            [pulled["p0"].model_copy(update={"reference": 99}), *pulled.examples[1:]],
            name=name,
        )
        with pytest.raises(StaleDatasetError):
            await push_dataset(client, edited)
        updated = await push_dataset(client, edited, base_version=pulled.version)
        assert updated.updated == 1
        latest = await pull_dataset(
            client, name, input_type=Problem, reference_type=int, cache_dir=tmp_path
        )
        assert latest["p0"].reference == 99
        assert latest.version != pulled.version
        after = {
            e["id"]: e["node_id"]
            for e in (
                await client.sdk.datasets.get_dataset(dataset=name, timeout=30)
            ).examples
        }
        assert after == before  # rows keep their identity across versions

        with pytest.raises(DatasetPushError, match="subset"):
            await push_dataset(client, latest.split("dev").head(1))
        shrunk = Dataset(latest.examples[:2], name=name)
        with pytest.raises(DatasetPushError, match="delete 1"):
            await push_dataset(client, shrunk, base_version=latest.version)
        removed = await push_dataset(
            client, shrunk, base_version=latest.version, allow_deletes=True
        )
        assert removed.deleted == 1


@pytest.mark.asyncio
async def test_phoenix_born_datasets_round_trip(base_url: str, tmp_path: Path) -> None:
    name = _name("born")
    async with PhoenixClient(base_url) as client:
        await client.upsert_dataset(
            name=name,
            examples=[
                {
                    "input": {"q": "why?"},
                    "output": {"a": "because"},
                    "metadata": {"src": "ui"},
                },
                {"input": {"q": "how?"}, "output": {"a": "so"}, "metadata": {}},
            ],
        )
        pulled = await pull_dataset(client, name, cache_dir=tmp_path)
        unchanged = await push_dataset(client, pulled)
        assert unchanged.version_id == pulled.version
        copy = await push_dataset(client, pulled, name=_name("copy"), force=True)
        assert copy.created == 2


@pytest.mark.asyncio
async def test_push_run_is_idempotent_and_resumable(
    base_url: str, tmp_path: Path
) -> None:
    store = LocalRunStore(tmp_path / "evals")
    dataset = _dataset(_name("runs"))
    run = await evaluate(
        FunctionTask(add), dataset, [exact, judge], repetitions=2, store=store
    )
    async with PhoenixClient(base_url) as client:
        link = await push_run(client, run, store=store)
        assert link.experiment_id is not None
        assert len(link.logged_trials) == 8
        assert len(await _task_runs(client, link.experiment_id)) == 8

        # A repeated push adds nothing.
        again = await push_run(client, store.load(run.id), store=store)
        assert again.experiment_id == link.experiment_id
        assert len(await _task_runs(client, link.experiment_id)) == 8

        # Progress lost entirely (e.g. a lost response): the experiment is found
        # again by run id and runs already logged are not duplicated.
        reloaded = store.load(run.id)
        assert reloaded.phoenix is not None
        reloaded.phoenix.experiment_id = None
        reloaded.phoenix.logged_trials = {}
        relinked = await push_run(client, reloaded, store=store)
        assert relinked.experiment_id == link.experiment_id
        assert len(await _task_runs(client, link.experiment_id)) == 8
        experiments = await client.sdk.experiments.list(dataset_id=link.dataset_id)
        assert [e["id"] for e in experiments] == [link.experiment_id]
        assert experiments[0]["metadata"]["grasp_run_id"] == run.id
        assert experiments[0]["metadata"]["evaluators"] == {"exact": "3", "judge": "1"}


@pytest.mark.asyncio
async def test_changed_trials_are_logged_again(base_url: str, tmp_path: Path) -> None:
    store = LocalRunStore(tmp_path / "evals")

    async def flaky(problem: Problem) -> int:
        if problem.a == 1:
            raise RuntimeError("transient")
        return problem.a + problem.b

    run = await evaluate(
        FunctionTask(flaky), _dataset(_name("resync")), [exact], store=store
    )
    async with PhoenixClient(base_url) as client:
        link = await push_run(client, run, store=store)
        assert link.experiment_id is not None
        failed = [
            r for r in await _task_runs(client, link.experiment_id) if r.get("error")
        ]
        assert len(failed) == 1

        # The failed trial succeeds later (as after a resume): pushing again
        # replaces the failed Phoenix run and refreshes the metadata.
        stored = store.load(run.id)
        trial = stored.trial("p1")
        assert trial is not None
        trial.error = None
        trial.output = 2
        stored.counts.task_errors = 0
        await push_run(client, stored, store=store)
        runs = await _task_runs(client, link.experiment_id)
        assert len(runs) == 4
        assert not [r for r in runs if r.get("error")]
        experiments = await client.sdk.experiments.list(dataset_id=link.dataset_id)
        assert experiments[0]["metadata"]["counts"]["task_errors"] == 0


@pytest.mark.asyncio
async def test_sealed_trials_are_withheld(base_url: str, tmp_path: Path) -> None:
    store = LocalRunStore(tmp_path / "evals")
    run = await evaluate(
        FunctionTask(add),
        _dataset(_name("sealed"), sealed_from=2),
        [exact],
        sealed_splits=["test"],
        store=store,
    )
    async with PhoenixClient(base_url) as client:
        link = await push_run(client, run, store=store)
        assert link.experiment_id is not None
        runs = await _task_runs(client, link.experiment_id)
        assert len(runs) == 2
        remote = await client.sdk.datasets.get_dataset(
            dataset=link.dataset_id, timeout=30
        )
        assert sorted(e["id"] for e in remote.examples) == ["p0", "p1"]
        experiments = await client.sdk.experiments.list(dataset_id=link.dataset_id)
        assert experiments[0]["metadata"]["sealed_trials_withheld"] == 2


@pytest.mark.asyncio
async def test_run_on_pulled_dataset_reuses_its_version(
    base_url: str, tmp_path: Path
) -> None:
    name = _name("pulled")
    store = LocalRunStore(tmp_path / "evals")
    async with PhoenixClient(base_url) as client:
        pushed = await push_dataset(client, _dataset(name))
        pulled = await pull_dataset(
            client, name, input_type=Problem, reference_type=int, cache_dir=tmp_path
        )
        run = await evaluate(
            FunctionTask(add), pulled.split("dev"), [exact], store=store
        )
        assert run.dataset.source == phoenix_source(client.base_url, pushed.dataset_id)
        link = await push_run(client, run, store=store)
    assert link.dataset_id == pushed.dataset_id
    assert link.dataset_version_id == pushed.version_id


@pytest.mark.asyncio
async def test_a_foreign_dataset_of_the_same_name_is_not_touched(
    base_url: str, tmp_path: Path
) -> None:
    name = _name("foreign")
    store = LocalRunStore(tmp_path / "evals")
    async with PhoenixClient(base_url) as client:
        await client.upsert_dataset(
            name=name,
            examples=[{"input": {"q": "unrelated"}, "output": {}, "metadata": {}}],
        )
        run = await evaluate(FunctionTask(add), _dataset(name), [exact], store=store)
        with pytest.raises(DatasetPushError, match="does not hold"):
            await push_run(client, run, store=store)
        remote = await client.sdk.datasets.get_dataset(dataset=name, timeout=30)
        assert len(remote.examples) == 1


@pytest.mark.asyncio
async def test_a_fully_sealed_run_needs_a_shared_dataset(
    base_url: str, tmp_path: Path
) -> None:
    store = LocalRunStore(tmp_path / "evals")
    name = _name("allsealed")
    held_out = await evaluate(
        FunctionTask(add),
        _dataset(name, sealed_from=0),
        [exact],
        sealed_splits=["test"],
        store=store,
    )
    async with PhoenixClient(base_url) as client:
        with pytest.raises(DatasetPushError, match="sealed split"):
            await push_run(client, held_out, store=store)


class Draft(BaseModel):
    text: str
    key_issue: str | None = None


@evaluator(version="1")
def nonempty(ctx: EvalContext[Draft, str, None]) -> bool:
    return bool(ctx.output)


async def echo(draft: Draft) -> str:
    return draft.text


@pytest.mark.asyncio
async def test_runs_attach_to_a_dataset_pushed_from_its_file(
    base_url: str, tmp_path: Path
) -> None:
    name = _name("fromfile")
    source = tmp_path / f"{name}.jsonl"
    # Defaulted fields written out, as hand-written files often have them.
    source.write_text(
        '{"id": "a", "input": {"text": "x", "key_issue": null}}\n'
        '{"id": "b", "input": {"text": "y"}}\n',
        encoding="utf-8",
    )
    dataset = Dataset.load(source, input_type=Draft)
    store = LocalRunStore(tmp_path / "evals")
    async with PhoenixClient(base_url) as client:
        await push_dataset(client, dataset)
        run = await evaluate(FunctionTask(echo), dataset, [nonempty], store=store)
        link = await push_run(client, run, store=store)
        # A run read back from disk matches the same dataset.
        other = await evaluate(FunctionTask(echo), dataset, [nonempty], store=store)
        other_link = await push_run(client, store.load(other.id), store=store)
        assert other_link.dataset_id == link.dataset_id
        pulled = await pull_dataset(
            client, name, input_type=Draft, cache_dir=tmp_path / "cache"
        )
        assert pulled.fingerprint == dataset.fingerprint
        unchanged = await push_dataset(client, pulled)
        assert unchanged.unchanged == 2
        assert unchanged.version_id == pulled.version


class _FailRuns(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/runs"):
            return httpx.Response(500, text="boom", request=request)
        return await self.inner.handle_async_request(request)


@pytest.mark.asyncio
async def test_a_failed_trial_push_reports_the_phoenix_error(
    base_url: str, tmp_path: Path
) -> None:
    store = LocalRunStore(tmp_path / "evals")
    run = await evaluate(
        FunctionTask(add), _dataset(_name("fails")), [exact], store=store
    )
    async with PhoenixClient(base_url, transport=_FailRuns(), retries=0) as client:
        with pytest.raises(PhoenixError, match="500"):
            await push_run(client, run, store=store)
