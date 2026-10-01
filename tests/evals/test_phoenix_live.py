"""
Against real Phoenix servers. Set ``GRASP_EVALS_PHOENIX_URLS`` to a
comma-separated list of base URLs (e.g. a 12.x and a 20.x server) and run
``pytest -m integration tests/evals/test_phoenix_live.py``.
"""

import os
import uuid
from pathlib import Path

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
    PhoenixClient,
    PhoenixCompatibilityError,
    StaleDatasetError,
    pull_dataset,
    push_dataset,
    push_run,
)
from grasp_agents.evals.phoenix.client import EXTERNAL_EXAMPLE_IDS

_URLS = [u for u in os.environ.get("GRASP_EVALS_PHOENIX_URLS", "").split(",") if u]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _URLS, reason="GRASP_EVALS_PHOENIX_URLS not set"),
]


class Problem(BaseModel):
    a: int
    b: int


def _dataset(name: str, n: int = 4) -> Dataset[Problem, int]:
    return Dataset(
        [
            Example(
                id=f"p{i}", input=Problem(a=i, b=1), reference=i + 1, splits=["dev"]
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


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(params=_URLS or ["unset"])
def base_url(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.mark.asyncio
async def test_dataset_round_trip_and_cache(base_url: str, tmp_path: Path) -> None:
    name = _name("roundtrip")
    async with PhoenixClient(base_url) as client:
        dataset_id, version_id = await push_dataset(client, _dataset(name))
        pulled = await pull_dataset(
            client, name, input_type=Problem, reference_type=int, cache_dir=tmp_path
        )
    assert pulled.version == version_id
    assert pulled.source == f"phoenix:{dataset_id}"
    assert pulled.ids == _dataset(name).ids
    assert pulled.fingerprint == _dataset(name).fingerprint
    assert pulled["p1"].input == Problem(a=1, b=1)
    assert pulled["p1"].splits == ["dev"]

    async with PhoenixClient("http://127.0.0.1:9") as offline:
        cached = await pull_dataset(
            offline,
            name,
            version=version_id,
            input_type=Problem,
            reference_type=int,
            cache_dir=tmp_path,
        )
    assert cached.fingerprint == pulled.fingerprint


@pytest.mark.asyncio
async def test_pushing_changes_respects_server_capabilities(
    base_url: str, tmp_path: Path
) -> None:
    name = _name("changes")
    async with PhoenixClient(base_url) as client:
        await push_dataset(client, _dataset(name, 3))
        pulled = await pull_dataset(
            client, name, input_type=Problem, reference_type=int, cache_dir=tmp_path
        )
        grown = Dataset(
            [*pulled.examples, Example(id="p9", input=Problem(a=9, b=1), reference=10)],
            name=name,
        )
        edited = Dataset(
            [pulled["p0"].model_copy(update={"reference": 99}), *pulled.examples[1:]],
            name=name,
        )
        if await client.supports(EXTERNAL_EXAMPLE_IDS):
            with pytest.raises(StaleDatasetError):
                await push_dataset(client, edited)
            await push_dataset(client, edited, base_version=pulled.version)
            latest = await pull_dataset(
                client, name, input_type=Problem, reference_type=int, cache_dir=tmp_path
            )
            assert latest["p0"].reference == 99
            assert latest.version != pulled.version
        else:
            _, appended = await push_dataset(client, grown)
            latest = await pull_dataset(
                client, name, input_type=Problem, reference_type=int, cache_dir=tmp_path
            )
            assert latest.version == appended
            assert sorted(latest.ids) == ["p0", "p1", "p2", "p9"]
            with pytest.raises(PhoenixCompatibilityError):
                await push_dataset(client, edited)


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
        runs = await client.list_runs(link.experiment_id)
        assert len(runs) == 8

        # A repeated push adds nothing.
        again = await push_run(client, store.load(run.id), store=store)
        assert again.experiment_id == link.experiment_id
        assert len(await client.list_runs(link.experiment_id)) == 8

        # A push interrupted before recording progress resumes via the 409 path.
        reloaded = store.load(run.id)
        assert reloaded.phoenix is not None
        reloaded.phoenix.logged_trials = reloaded.phoenix.logged_trials[:3]
        await push_run(client, reloaded, store=store)
        assert len(await client.list_runs(link.experiment_id)) == 8
        experiment = await client._json("GET", f"/v1/experiments/{link.experiment_id}")
        data = experiment["data"]
        assert data["successful_run_count"] == 8
        assert data["metadata"]["grasp_run_id"] == run.id
        assert data["metadata"]["evaluators"] == {"exact": "3", "judge": "1"}


@pytest.mark.asyncio
async def test_run_on_pulled_dataset_reuses_its_version(
    base_url: str, tmp_path: Path
) -> None:
    name = _name("pulled")
    store = LocalRunStore(tmp_path / "evals")
    async with PhoenixClient(base_url) as client:
        dataset_id, version_id = await push_dataset(client, _dataset(name))
        pulled = await pull_dataset(
            client, name, input_type=Problem, reference_type=int, cache_dir=tmp_path
        )
        run = await evaluate(
            FunctionTask(add), pulled.split("dev"), [exact], store=store
        )
        assert run.dataset.source == f"phoenix:{dataset_id}"
        link = await push_run(client, run, store=store)
    assert link.dataset_id == dataset_id
    assert link.dataset_version_id == version_id
