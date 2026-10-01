from pathlib import Path
from typing import Any

import pytest

from grasp_agents.evals import (
    Dataset,
    EvalContext,
    Example,
    FunctionPairwiseJudge,
    FunctionTask,
    LocalRunStore,
    Measure,
    PairwiseContext,
    PairwiseVerdict,
    compare,
    evaluate,
    evaluator,
    pairwise,
    render_comparison_markdown,
)

type Ctx = EvalContext[int, int, int]


def _dataset(n: int = 20) -> Dataset[int, int]:
    return Dataset([Example(id=f"x{i}", input=i, reference=i * 2) for i in range(n)])


@evaluator
def correct(ctx: Ctx) -> bool:
    return ctx.output == ctx.reference


async def good(x: int) -> int:
    return x * 2


async def flawed(x: int) -> int:
    return x * 2 if x % 2 == 0 else x * 2 + 1


@pytest.fixture
def store(tmp_path: Path) -> LocalRunStore:
    return LocalRunStore(tmp_path)


class TestCompare:
    @pytest.mark.asyncio
    async def test_paired_binary_comparison(self, store: LocalRunStore) -> None:
        base = await evaluate(
            FunctionTask(flawed, name="solver"), _dataset(), [correct], store=store
        )
        candidate = await evaluate(
            FunctionTask(good, name="solver"), _dataset(), [correct], store=store
        )
        result = compare(base, candidate)
        target = result.target("correct")
        assert target is not None
        assert result.n_paired_examples == 20
        assert target.test == "mcnemar"
        assert target.diff == pytest.approx(0.5)
        assert target.improved == 10
        assert target.regressed == 0
        assert target.p_value is not None
        assert target.p_value < 0.01
        assert target.significant
        assert not target.top_regressions
        assert len(target.top_improvements) == 5
        assert result.target(str(Measure.DURATION)) is not None
        text = render_comparison_markdown(result)
        assert "correct" in text
        assert "✱" in text

    @pytest.mark.asyncio
    async def test_changed_examples_are_not_paired(self, store: LocalRunStore) -> None:
        base = await evaluate(good, _dataset(4), [correct], store=store)
        changed = Dataset(
            [*_dataset(3).examples, Example(id="x3", input=3, reference=7)]
        )
        candidate = await evaluate(good, changed, [correct], store=store)
        result = compare(base, candidate)
        assert result.n_paired_examples == 3
        assert result.n_changed_examples == 1
        assert any("different dataset content" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_identical_configuration_warns_about_noise(
        self, store: LocalRunStore
    ) -> None:
        a = await evaluate(good, _dataset(3), [correct], store=store)
        b = await evaluate(good, _dataset(3), [correct], store=store)
        assert any("noise" in w for w in compare(a, b).warnings)

    @pytest.mark.asyncio
    async def test_sealed_examples_never_listed(self, store: LocalRunStore) -> None:
        dataset = Dataset(
            [
                Example(id=f"x{i}", input=i, reference=i * 2, splits=["test"])
                for i in range(6)
            ]
        )
        base = await evaluate(
            flawed, dataset, [correct], sealed_splits=["test"], store=store
        )
        candidate = await evaluate(
            good, dataset, [correct], sealed_splits=["test"], store=store
        )
        target = compare(base, candidate).target("correct")
        assert target is not None
        assert target.improved == 3
        assert target.top_improvements == []


def _prefer_longer(ctx: PairwiseContext[int, str, Any]) -> PairwiseVerdict:
    if len(ctx.first) == len(ctx.second):
        return PairwiseVerdict(winner="tie")
    return PairwiseVerdict(
        winner="first" if len(ctx.first) > len(ctx.second) else "second",
        explanation="longer is better",
    )


def _always_first(ctx: PairwiseContext[int, str, Any]) -> PairwiseVerdict:
    return PairwiseVerdict(winner="first")


class TestPairwise:
    @pytest.mark.asyncio
    async def test_order_swapped_win_rate(self, store: LocalRunStore) -> None:
        async def short(x: int) -> str:
            return "a"

        async def long(x: int) -> str:
            return "a" * (2 if x < 8 else 1)

        base = await evaluate(FunctionTask(short), _dataset(10), store=store)
        candidate = await evaluate(FunctionTask(long), _dataset(10), store=store)
        run = await pairwise(
            base.id,
            candidate.id,
            FunctionPairwiseJudge(_prefer_longer, name="length"),
            store=store,
        )
        assert run.kind == "pairwise"
        assert run.parent_run_id is None
        assert run.counts.trials_done == 10
        win = run.metric("win_rate(length)")
        assert win is not None
        assert win.value == pytest.approx((8 + 0.5 * 2) / 10)
        assert win.details["candidate"] == 8
        assert win.details["tie"] == 2
        consistency = run.metric("pass_rate(length.position_consistent)")
        assert consistency is not None
        assert consistency.value == pytest.approx(1.0)
        assert store.load(run.id).trials[0].output == {"base": "a", "candidate": "aa"}

    @pytest.mark.asyncio
    async def test_position_bias_is_exposed(self, store: LocalRunStore) -> None:
        async def same(x: int) -> str:
            return "same"

        base = await evaluate(FunctionTask(same), _dataset(5), store=store)
        candidate = await evaluate(FunctionTask(same), _dataset(5), store=store)
        run = await pairwise(
            base,
            candidate,
            FunctionPairwiseJudge(_always_first, name="biased"),
            store=store,
        )
        consistency = run.metric("pass_rate(biased.position_consistent)")
        assert consistency is not None
        assert consistency.value == pytest.approx(0.0)
        win = run.metric("win_rate(biased)")
        assert win is not None
        assert win.value == pytest.approx(0.5)
        assert win.details["inconsistent"] == 5

    @pytest.mark.asyncio
    async def test_failed_trials_are_not_paired(self, store: LocalRunStore) -> None:
        async def broken(x: int) -> str:
            if x == 0:
                raise ValueError("no")
            return "ok"

        async def fine(x: int) -> str:
            return "ok"

        base = await evaluate(FunctionTask(broken), _dataset(3), store=store)
        candidate = await evaluate(FunctionTask(fine), _dataset(3), store=store)
        run = await pairwise(
            base,
            candidate,
            FunctionPairwiseJudge(_prefer_longer, name="j"),
            store=store,
        )
        assert run.counts.trials_done == 2


@pytest.mark.asyncio
async def test_pairwise_validates_stored_types(store: LocalRunStore) -> None:
    from pydantic import BaseModel

    class Item(BaseModel):
        word: str

    def by_length(ctx: PairwiseContext[Item, str, Any]) -> PairwiseVerdict:
        assert isinstance(ctx.input, Item)  # stored JSON re-validated
        longer = len(ctx.first) - len(ctx.second)
        return PairwiseVerdict(
            winner="tie" if not longer else ("first" if longer > 0 else "second")
        )

    dataset = Dataset([Example(id=w, input=Item(word=w)) for w in ["a", "bb", "ccc"]])

    async def echo(item: Item) -> str:
        return item.word

    async def double(item: Item) -> str:
        return item.word * 2

    base = await evaluate(FunctionTask(echo), dataset, store=store)
    candidate = await evaluate(FunctionTask(double), dataset, store=store)
    run = await pairwise(
        base.id,
        candidate.id,
        FunctionPairwiseJudge(by_length, name="len"),
        input_type=Item,
        store=store,
    )
    assert run.counts.evaluator_failures == 0
    assert run.metric("win_rate(len)").value == pytest.approx(1.0)  # type: ignore[union-attr]
