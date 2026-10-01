"""
``grasp-evals`` — run, inspect and compare evaluations from the shell.

Designed for coding agents as much as for people: every command accepts
``--json`` (one JSON document on stdout; progress goes to stderr), runs are
addressed by id, unique id prefix, ``latest`` or ``latest:<name>``, and the
exit code is a gate (1 when a run is invalid or incomplete, a ``--fail-under``
threshold is missed, or ``--fail-on-regression`` finds a significant
regression against ``--baseline``; 2 for usage errors).
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.markdown import Markdown

from ._execution import TrialProgress
from .compare import Comparison, compare
from .dataset import Dataset, DatasetError, example_json_schema
from .evaluation import Evaluation, list_evaluations, load_evaluation, load_object
from .pairwise import PairwiseJudge, pairwise
from .report import (
    render_comparison_markdown,
    render_run_markdown,
    run_summary,
    trial_summary,
)
from .store import LocalRunStore, RunNotFoundError, store_for_ref
from .types import EvaluationRun, RunStatus

EXIT_OK = 0
EXIT_GATE = 1
EXIT_USAGE = 2


class CLIError(Exception):
    pass


def _print_json(data: Any) -> None:
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")


def _print_markdown(text: str) -> None:
    Console().print(Markdown(text))


def _progress_printer(quiet: bool) -> Any:
    if quiet:
        return None

    def report(progress: TrialProgress) -> None:
        trial = progress.trial
        outcome = (
            "ok" if trial.ok else f"error {trial.error.type if trial.error else ''}"
        )
        if trial.evaluator_failures:
            outcome += f" ({len(trial.evaluator_failures)} evaluator failures)"
        sys.stderr.write(
            f"[{progress.done}/{progress.total}] {trial.example_id}#{trial.repetition} "
            f"{outcome} {trial.duration_s:.1f}s · ${progress.cost_usd:.4f}\n"
        )
        sys.stderr.flush()

    return report


def _parse_thresholds(values: Sequence[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in values:
        name, sep, raw = item.rpartition("=")
        if not sep or not name:
            raise CLIError(f"--fail-under expects NAME=VALUE, got {item!r}")
        try:
            thresholds[name] = float(raw)
        except ValueError as exc:
            raise CLIError(f"--fail-under value must be a number: {item!r}") from exc
    return thresholds


def _gate_messages(
    run: EvaluationRun,
    thresholds: dict[str, float],
    comparison: Comparison | None,
    fail_on_regression: bool,
) -> list[str]:
    failures: list[str] = []
    if run.status != RunStatus.COMPLETED:
        failures.append(f"run status is {run.status}")
    if run.invalid_reason:
        failures.append(f"run is invalid: {run.invalid_reason}")
    for name, minimum in thresholds.items():
        metric = run.metric(name)
        if metric is None:
            failures.append(f"metric {name!r} not found")
        elif metric.value is None or metric.value < minimum:
            failures.append(f"{name} = {metric.value} < {minimum}")
    if fail_on_regression and comparison is not None:
        for target in comparison.targets:
            regressed = (
                target.ci_high is not None and target.ci_high < 0
                if target.higher_is_better
                else target.ci_low is not None and target.ci_low > 0
            )
            if regressed:
                failures.append(f"significant regression in {target.target}")
    return failures


# --- Commands ---


async def _cmd_run(args: argparse.Namespace) -> int:
    evaluation = load_evaluation(args.spec)
    store = LocalRunStore(args.root)
    thresholds = _parse_thresholds(args.fail_under)
    ids = [i for i in (args.ids or "").split(",") if i]
    run = await evaluation.run(
        dataset=args.dataset,
        split=args.split,
        ids=ids or None,
        sample=args.sample,
        seed=args.seed,
        limit=args.limit,
        repetitions=args.repetitions,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
        max_cost_usd=args.max_cost,
        score=not args.no_score,
        name=args.name,
        store=store,
        resume=args.resume,
        tags=args.tag,
        progress=_progress_printer(args.quiet),
    )
    comparison: Comparison | None = None
    if args.baseline:
        base_store, base_id = store_for_ref(args.baseline, args.root)
        comparison = compare(base_store.load(base_id), run)
    pushed: str | None = None
    if args.push:
        pushed = await _push(run, store, args.base_url)
    failures = _gate_messages(run, thresholds, comparison, args.fail_on_regression)
    if args.json:
        payload = run_summary(run)
        payload["path"] = str(store.run_dir(run.id))
        payload["phoenix_url"] = pushed
        payload["gate_failures"] = failures
        if comparison is not None:
            payload["comparison"] = comparison.model_dump()
        _print_json(payload)
    else:
        _print_markdown(render_run_markdown(run))
        if comparison is not None:
            _print_markdown(render_comparison_markdown(comparison))
        for failure in failures:
            sys.stderr.write(f"gate: {failure}\n")
        sys.stderr.write(f"run stored in {store.run_dir(run.id)}\n")
        if pushed:
            sys.stderr.write(f"pushed to Phoenix: {pushed}\n")
    return EXIT_GATE if failures else EXIT_OK


async def _push(run: EvaluationRun, store: LocalRunStore, base_url: str | None) -> str:
    from .phoenix import (  # noqa: PLC0415
        PhoenixClient,
        experiment_url,
        push_run,
    )

    async with PhoenixClient(base_url) as client:
        link = await push_run(client, run, store=store)
    return experiment_url(link) or link.base_url


async def _cmd_push(args: argparse.Namespace) -> int:
    store, run_id = store_for_ref(args.run, args.root)
    run = store.load(run_id)
    url = await _push(run, store, args.base_url)
    if args.json:
        _print_json({"id": run.id, "phoenix_url": url})
    else:
        sys.stdout.write(f"{url}\n")
    return EXIT_OK


async def _cmd_rescore(args: argparse.Namespace) -> int:
    store, run_id = store_for_ref(args.run, args.root)
    parent = store.load(run_id)
    spec = args.spec or parent.evaluation
    if not spec:
        raise CLIError(
            f"Run {run_id} was not started from an Evaluation; pass --spec MODULE:ATTR"
        )
    evaluation = load_evaluation(spec)
    child = await evaluation.rescore(
        parent, store=store, progress=_progress_printer(args.quiet)
    )
    if args.json:
        _print_json({**run_summary(child), "path": str(store.run_dir(child.id))})
    else:
        _print_markdown(render_run_markdown(child))
    return EXIT_OK


def _cmd_show(args: argparse.Namespace) -> int:
    store, run_id = store_for_ref(args.run, args.root)
    run = store.load(run_id)
    if args.example:
        trials = [t for t in run.trials if t.example_id == args.example]
        if not trials:
            raise CLIError(f"No trials for example {args.example!r} in {run_id}")
        if any(t.sealed for t in trials):
            raise CLIError(
                f"Example {args.example!r} is in a sealed split; "
                "only aggregates are shown"
            )
        example = run.example(args.example)
        events = store.load_events(run_id)
        detail: dict[str, Any] = {
            "example": None if example is None else example.model_dump(mode="json"),
            "trials": [trial_summary(t) for t in trials],
            "transcripts": {
                f"{t.example_id}#{t.repetition}": [
                    {"type": e.type, "source": e.source} for e in events.get(t.key, [])
                ]
                for t in trials
            },
        }
        _print_json(detail)
        return EXIT_OK
    selected = run.trials
    if args.failures:
        selected = [t for t in run.trials if not t.ok or t.evaluator_failures]
    if args.json:
        payload = run_summary(run)
        if args.trials or args.failures:
            payload["trials"] = [
                trial_summary(t, include_output=args.outputs) for t in selected
            ]
        _print_json(payload)
        return EXIT_OK
    _print_markdown(render_run_markdown(run, max_rows=50 if args.failures else 10))
    if args.trials:
        for trial in selected:
            _print_json(trial_summary(trial, include_output=args.outputs))
    return EXIT_OK


async def _cmd_pairwise(args: argparse.Namespace) -> int:
    judge = load_object(args.judge)
    if not isinstance(judge, PairwiseJudge):
        raise CLIError(f"{args.judge} is not a PairwiseJudge")
    base_store, base_id = store_for_ref(args.base, args.root)
    candidate_store, candidate_id = store_for_ref(args.candidate, args.root)
    base = base_store.load(base_id)
    spec = args.spec or base.evaluation
    types: dict[str, Any] = {}
    if spec:
        evaluation = load_evaluation(spec)
        types = {
            "input_type": evaluation.resolved_input_type,
            "reference_type": evaluation.reference_type,
            "output_type": evaluation.build_task().output_type,
        }
    run = await pairwise(
        base,
        candidate_store.load(candidate_id),
        cast("PairwiseJudge[Any, Any, Any]", judge),
        both_orders=not args.one_order,
        **types,
        store=LocalRunStore(args.root),
        progress=_progress_printer(args.quiet),
    )
    if args.json:
        _print_json(run_summary(run))
    else:
        _print_markdown(render_run_markdown(run))
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    base_store, base_id = store_for_ref(args.base, args.root)
    candidate_store, candidate_id = store_for_ref(args.candidate, args.root)
    comparison = compare(
        base_store.load(base_id), candidate_store.load(candidate_id), top=args.top
    )
    if args.json:
        _print_json(comparison.model_dump())
    else:
        _print_markdown(render_comparison_markdown(comparison, max_rows=args.top))
    return EXIT_OK


def _cmd_runs(args: argparse.Namespace) -> int:
    store = LocalRunStore(args.root)
    runs = store.list_runs(name=args.name, limit=args.limit)
    if args.json:
        _print_json(
            [
                {
                    "id": r.id,
                    "name": r.name,
                    "kind": r.kind,
                    "status": str(r.status),
                    "invalid_reason": r.invalid_reason,
                    "created_at": r.created_at.isoformat(),
                    "examples": r.dataset.selected_size,
                    "trials": r.counts.trials_done,
                    "metrics": {m.name: m.value for m in r.metrics},
                }
                for r in runs
            ]
        )
        return EXIT_OK
    for r in runs:
        first = r.metrics[0] if r.metrics else None
        headline = (
            f"{first.name}={first.value:.3g}"
            if first is not None and first.value is not None
            else ""
        )
        flag = " INVALID" if r.invalid_reason else ""
        sys.stdout.write(
            f"{r.id}  {r.status}{flag}  {r.counts.trials_done} trials  {headline}\n"
        )
    return EXIT_OK


def _cmd_list(args: argparse.Namespace) -> int:
    found = list_evaluations(args.module)
    rows = [
        {
            "spec": f"{args.module}:{attr}",
            "name": evaluation.name,
            "description": evaluation.description,
            "evaluators": [e.name for e in evaluation.evaluators],
        }
        for attr, evaluation in found.items()
    ]
    if args.json:
        _print_json(rows)
    else:
        for row in rows:
            sys.stdout.write(f"{row['spec']}  {row['description'] or ''}\n")
    return EXIT_OK


def _evaluation_schema(evaluation: Evaluation) -> dict[str, Any]:
    return example_json_schema(
        evaluation.resolved_input_type, evaluation.reference_type
    )


async def _cmd_datasets(args: argparse.Namespace) -> int:
    if args.datasets_command == "schema":
        _print_json(_evaluation_schema(load_evaluation(args.spec)))
        return EXIT_OK
    if args.datasets_command == "validate":
        evaluation = load_evaluation(args.spec)
        try:
            dataset = await evaluation.load_dataset(args.dataset)
        except DatasetError as exc:
            _print_json({"valid": False, "errors": [str(exc)]})
            return EXIT_GATE
        problems = evaluation.check_dataset(dataset)
        _print_json(
            {
                "valid": not problems,
                "name": dataset.name,
                "examples": len(dataset),
                "fingerprint": dataset.fingerprint,
                "problems": [p.model_dump() for p in problems],
            }
        )
        return EXIT_GATE if problems else EXIT_OK
    if args.datasets_command in {"pull", "push"}:
        return await _cmd_phoenix_datasets(args)
    dataset = Dataset.load(args.path)
    splits = {s: len(dataset.split(s)) for s in dataset.splits}
    info = {
        "name": dataset.name,
        "examples": len(dataset),
        "fingerprint": dataset.fingerprint,
        "splits": splits,
        "ids": dataset.ids[: args.limit],
    }
    _print_json(info)
    return EXIT_OK


async def _cmd_phoenix_datasets(args: argparse.Namespace) -> int:
    from .phoenix import (  # noqa: PLC0415
        PhoenixClient,
        pull_dataset,
        push_dataset,
    )

    async with PhoenixClient(args.base_url) as client:
        if args.datasets_command == "pull":
            name, _, version = args.name.partition("@")
            dataset = await pull_dataset(
                client,
                name,
                version=version or None,
                cache_dir=(args.root / "datasets") if args.root else None,
            )
            if args.output:
                dataset.save(args.output)
            _print_json(
                {
                    "name": dataset.name,
                    "version": dataset.version,
                    "source": dataset.source,
                    "examples": len(dataset),
                    "fingerprint": dataset.fingerprint,
                    "saved_to": args.output,
                }
            )
            return EXIT_OK
        dataset = Dataset.load(args.path)
        if args.base_version:
            dataset.version = args.base_version
        dataset_id, version_id = await push_dataset(
            client,
            dataset,
            name=args.name,
            base_version=args.base_version,
            force=args.force,
        )
        _print_json(
            {
                "name": args.name or dataset.name,
                "dataset_id": dataset_id,
                "version_id": version_id,
                "examples": len(dataset),
            }
        )
        return EXIT_OK


# --- Parser ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grasp-evals", description="Run, inspect and compare evaluations."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="evals directory (default: $GRASP_EVALS_DIR or ./.evals)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an Evaluation (module:attr or file.py:attr)")
    run.add_argument("spec")
    run.add_argument("--dataset", help="override the evaluation's dataset file")
    run.add_argument("--split")
    run.add_argument("--ids", help="comma-separated example ids")
    run.add_argument("--sample", type=int, help="random subset of N examples")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--limit", type=int, help="first N examples (after other filters)")
    run.add_argument("-r", "--repetitions", type=int)
    run.add_argument("-c", "--concurrency", type=int)
    run.add_argument("--timeout", type=float, help="per-trial timeout in seconds")
    run.add_argument("--max-cost", type=float, help="stop scheduling after USD spent")
    run.add_argument(
        "--no-score", action="store_true", help="execute only; rescore later"
    )
    run.add_argument("--resume", help="continue an interrupted run")
    run.add_argument("--baseline", help="compare against this run afterwards")
    run.add_argument(
        "--fail-under",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="exit 1 if a metric's value is below VALUE",
    )
    run.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="with --baseline: exit 1 on a significant regression",
    )
    run.add_argument("--push", action="store_true", help="mirror the run to Phoenix")
    run.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    run.add_argument("--name")
    run.add_argument("--tag", action="append", default=[])
    run.add_argument("--json", action="store_true")
    run.add_argument("-q", "--quiet", action="store_true", help="no progress lines")

    push = sub.add_parser("push", help="mirror a finished run to Phoenix")
    push.add_argument("run")
    push.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    push.add_argument("--json", action="store_true")

    rescore = sub.add_parser("rescore", help="score a run's outputs again (child run)")
    rescore.add_argument("run")
    rescore.add_argument("--spec", help="Evaluation to take evaluators from")
    rescore.add_argument("--json", action="store_true")
    rescore.add_argument("-q", "--quiet", action="store_true")

    show = sub.add_parser("show", help="show a run")
    show.add_argument("run")
    show.add_argument("--trials", action="store_true", help="list every trial")
    show.add_argument("--failures", action="store_true", help="only failed trials")
    show.add_argument("--outputs", action="store_true", help="include outputs")
    show.add_argument("--example", help="full detail for one example")
    show.add_argument("--json", action="store_true")

    cmp = sub.add_parser("compare", help="paired comparison of two runs")
    cmp.add_argument("base")
    cmp.add_argument("candidate")
    cmp.add_argument("--top", type=int, default=5)
    cmp.add_argument("--json", action="store_true")

    pair = sub.add_parser("pairwise", help="A/B two runs with an order-swapped judge")
    pair.add_argument("base")
    pair.add_argument("candidate")
    pair.add_argument("--judge", required=True, help="PairwiseJudge as MODULE:ATTR")
    pair.add_argument("--one-order", action="store_true", help="skip the swapped order")
    pair.add_argument(
        "--spec", help="Evaluation whose types to validate with (default: the base's)"
    )
    pair.add_argument("--json", action="store_true")
    pair.add_argument("-q", "--quiet", action="store_true")

    runs = sub.add_parser("runs", help="list runs, newest first")
    runs.add_argument("--name")
    runs.add_argument("--limit", type=int, default=20)
    runs.add_argument("--json", action="store_true")

    listing = sub.add_parser("list", help="list the Evaluations defined in a module")
    listing.add_argument("module")
    listing.add_argument("--json", action="store_true")

    datasets = sub.add_parser("datasets", help="dataset utilities")
    dsub = datasets.add_subparsers(dest="datasets_command", required=True)
    schema = dsub.add_parser("schema", help="JSON schema of an evaluation's examples")
    schema.add_argument("spec")
    validate = dsub.add_parser("validate", help="type-check and run dataset checks")
    validate.add_argument("spec")
    validate.add_argument("--dataset", help="file to validate instead of the default")
    info = dsub.add_parser("show", help="summarize a dataset file")
    info.add_argument("path")
    info.add_argument("--limit", type=int, default=20)
    pull = dsub.add_parser("pull", help="fetch a Phoenix dataset version")
    pull.add_argument("name", help="NAME or NAME@VERSION_ID")
    pull.add_argument("-o", "--output", help="also save it to this file")
    pull.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    push_ds = dsub.add_parser("push", help="make a Phoenix dataset match a file")
    push_ds.add_argument("path")
    push_ds.add_argument("--name", help="Phoenix dataset name (default: the file's)")
    push_ds.add_argument(
        "--base-version", help="the Phoenix version this file was pulled from"
    )
    push_ds.add_argument(
        "--force", action="store_true", help="replace even if Phoenix moved on"
    )
    push_ds.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    return parser


def _dispatch(args: argparse.Namespace) -> int:
    match args.command:
        case "run":
            return asyncio.run(_cmd_run(args))
        case "rescore":
            return asyncio.run(_cmd_rescore(args))
        case "push":
            return asyncio.run(_cmd_push(args))
        case "show":
            return _cmd_show(args)
        case "compare":
            return _cmd_compare(args)
        case "pairwise":
            return asyncio.run(_cmd_pairwise(args))
        case "runs":
            return _cmd_runs(args)
        case "list":
            return _cmd_list(args)
        case "datasets":
            return asyncio.run(_cmd_datasets(args))
        case _:
            raise CLIError(f"Unknown command {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (CLIError, RunNotFoundError, DatasetError, LookupError) as exc:
        sys.stderr.write(f"grasp-evals: {exc}\n")
        return EXIT_USAGE


def console_main() -> None:
    raise SystemExit(main())
