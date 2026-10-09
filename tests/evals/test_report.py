import pytest

from grasp_agents.evals import (
    Dataset,
    EvalContext,
    Example,
    evaluate,
    render_run_markdown,
    run_summary,
    scorer,
)
from grasp_agents.evals.report import trial_summary


@scorer
def correct(ctx: EvalContext[int, int, int]) -> bool:
    return ctx.output == ctx.reference


async def wrong_on_odd(x: int) -> int:
    return x * 2 if x % 2 == 0 else -1


def _data() -> Dataset[int, int]:
    return Dataset(
        [
            Example(
                id=f"x{i}",
                input=i,
                reference=i * 2,
                splits=["test"] if i >= 4 else ["dev"],
                metadata={"band": "held" if i >= 4 else f"b{i % 2}"},
            )
            for i in range(6)
        ],
        name="numbers",
    )


@pytest.mark.asyncio
async def test_reports_never_list_sealed_examples() -> None:
    run = await evaluate(
        wrong_on_odd,
        _data(),
        [correct],
        sealed_splits=["test"],
        group_by=["band"],
        persist=False,
    )
    text = render_run_markdown(run)
    assert "x1" in text  # a failing dev example is listed
    assert "x5" not in text  # a failing sealed one is not
    # A group made only of a few held-out examples would be close to
    # per-example results.
    assert "band=held" not in text
    summary = run_summary(run)
    assert "band=held" not in summary["metrics"]["pass_rate(correct)"].get("groups", {})


@pytest.mark.asyncio
async def test_sealed_trials_expose_nothing_identifying() -> None:
    run = await evaluate(
        wrong_on_odd, _data(), [correct], sealed_splits=["test"], persist=False
    )
    sealed = next(t for t in run.trials if t.sealed)
    assert trial_summary(sealed) == {
        "sealed": True,
        "repetition": 0,
        "ok": True,
    }
    shown = trial_summary(next(t for t in run.trials if not t.sealed))
    assert shown["example_id"] == "x0"
    assert "scores" in shown


@pytest.mark.asyncio
async def test_headers_name_the_confidence_level() -> None:
    run = await evaluate(wrong_on_odd, _data(), [correct], persist=False)
    assert "95% CI" in render_run_markdown(run)


@pytest.mark.asyncio
async def test_groups_with_a_few_sealed_members_are_hidden() -> None:
    data = Dataset(
        [
            Example(
                id=f"x{i}",
                input=i,
                reference=i * 2,
                splits=["test"] if i == 5 else ["dev"],
                metadata={"band": "mixed" if i >= 3 else "visible"},
            )
            for i in range(6)
        ],
        name="numbers",
    )
    run = await evaluate(
        wrong_on_odd,
        data,
        [correct],
        sealed_splits=["test"],
        group_by=["band"],
        persist=False,
    )
    # band=mixed minus its visible members would reveal x5's result.
    groups = run_summary(run)["metrics"]["pass_rate(correct)"].get("groups", {})
    assert "band=mixed" not in groups
    assert "band=visible" in groups
    assert "band=mixed" not in render_run_markdown(run)
