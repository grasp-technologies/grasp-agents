import asyncio
import logging
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path

import pytest

from grasp_agents.evals import (
    Dataset,
    Example,
    LocalRunStore,
    RunNotFoundError,
    ScoreContext,
    evaluate,
    scorer,
)
from grasp_agents.evals.store import CACHE_DIR_ENV, default_cache_dir


def _numbers(n: int = 3) -> Dataset[int, int]:
    return Dataset(
        [Example(id=f"n{i}", input=i, reference=i * 2) for i in range(n)],
        name="numbers",
    )


async def double(x: int) -> int:
    return x * 2


@scorer
def correct(ctx: ScoreContext[int, int, int]) -> bool:
    return ctx.output == ctx.reference


@pytest.fixture
def store(tmp_path: Path) -> LocalRunStore:
    return LocalRunStore(tmp_path / "evals")


@pytest.mark.asyncio
async def test_an_interrupted_write_does_not_lose_the_run(store: LocalRunStore) -> None:
    run = await evaluate(double, _numbers(), [correct], store=store)
    trials = store.run_dir(run.id) / "trials.jsonl"
    with trials.open("a", encoding="utf-8") as fh:
        fh.write('{"example_id": "n9", "repet')  # a crash mid-append
    reloaded = store.load(run.id)
    assert len(reloaded.trials) == 3

    trial = reloaded.trials[0].model_copy(update={"duration_s": 9.0})
    store.append_trial(run.id, trial)
    lines = trials.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all(line.endswith("}") for line in lines)
    assert store.load(run.id).trial("n0").duration_s == pytest.approx(9.0)  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_corrupt_middle_line_is_an_error(store: LocalRunStore) -> None:
    run = await evaluate(double, _numbers(), [correct], store=store)
    trials = store.run_dir(run.id) / "trials.jsonl"
    lines = trials.read_text(encoding="utf-8").splitlines()
    trials.write_text("\n".join([lines[0], "{broken", *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match=r"trials\.jsonl:2: corrupt record"):
        store.load(run.id)


@pytest.mark.asyncio
async def test_a_renamed_run_directory_is_refused(store: LocalRunStore) -> None:
    run = await evaluate(double, _numbers(), [correct], store=store)
    copy = store.runs_dir / f"{run.id}-copy"
    shutil.copytree(store.run_dir(run.id), copy)
    with pytest.raises(RunNotFoundError, match="named after its run id"):
        store.load(copy.name)


@pytest.mark.asyncio
async def test_latest_matches_the_evaluation_attribute(store: LocalRunStore) -> None:
    first = await evaluate(
        double, _numbers(), [correct], store=store, evaluation="specs.py:doubling_v1"
    )
    await evaluate(
        double, _numbers(), [correct], store=store, evaluation="specs.py:doubling_v2"
    )
    assert store.resolve("latest:doubling_v1") == first.id


@pytest.mark.asyncio
async def test_a_complete_record_missing_its_newline_is_kept(
    store: LocalRunStore,
) -> None:
    run = await evaluate(double, _numbers(), [correct], store=store)
    trials = store.run_dir(run.id) / "trials.jsonl"
    # A crash between writing a record and its newline.
    trials.write_text(trials.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    reloaded = store.load(run.id)
    assert len(reloaded.trials) == 3
    store.append_trial(run.id, reloaded.trials[0])
    assert len(trials.read_text(encoding="utf-8").splitlines()) == 4
    assert len(store.load(run.id).trials) == 3


# --- Where things are kept ---


def test_pulled_datasets_are_cached_per_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == tmp_path / "xdg" / "grasp-evals"
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "mine"))
    assert default_cache_dir() == tmp_path / "mine"


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_a_default_store_inside_a_repository_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git = shutil.which("git")
    assert git is not None
    await asyncio.to_thread(
        subprocess.run,
        [git, "init", "-q", str(repo)],
        check=True,
    )
    monkeypatch.chdir(repo)
    monkeypatch.delenv("GRASP_EVALS_DIR", raising=False)
    caplog.set_level(logging.WARNING, logger="grasp_agents.evals.store")
    await evaluate(double, _numbers(1), [correct])
    assert any("inside the git repository" in m for m in caplog.messages)
    caplog.clear()
    shutil.rmtree(repo / ".evals")
    (repo / ".gitignore").write_text(".evals/\n")
    await evaluate(double, _numbers(1), [correct])
    assert not any("inside the git repository" in m for m in caplog.messages)
