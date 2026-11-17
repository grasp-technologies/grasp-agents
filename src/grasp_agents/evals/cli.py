"""
``grasp-evals`` — run, inspect and compare evaluations from the shell.

Designed for coding agents as much as for people: every command accepts
``--json`` (one JSON document on stdout, also for errors), progress goes to
stderr (``--progress json``: one JSON object per finished trial), runs are
addressed by id, unique id prefix, run directory,
``latest`` or ``latest:<name>`` (a run name or the spec attribute), and the
exit code says what happened:

- 0: done;
- 1: a gate failed — the run is invalid or incomplete, a ``--fail-under``
  threshold is missed, ``--fail-on-regression`` found a significant regression
  against ``--baseline``, a judge is not validated, or a dataset is invalid;
- 2: usage error (bad arguments, spec, dataset file or run reference);
- 3: an unexpected error, or Phoenix failed;
- 130 / 128 + signal number: interrupted (Ctrl-C) or stopped (SIGTERM,
  SIGHUP); the run is finalized as ``cancelled`` and can be resumed.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import traceback
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast, override

from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown

from ._execution import ProgressCallback, TrialProgress
from .compare import Comparison, compare, regressions
from .dataset import Dataset, DatasetError, example_json_schema
from .evaluation import (
    PHOENIX_PREFIX,
    Evaluation,
    SpecError,
    list_evaluations,
    load_evaluation,
    load_object,
    parse_phoenix_ref,
)
from .pairwise import PairwiseJudge, pairwise
from .report import (
    render_comparison_markdown,
    render_run_markdown,
    run_summary,
    trial_summary,
)
from .runner import ResumeError, SealedSelectionError, annotate_run, close_run
from .store import LocalRunStore, RunNotFoundError, store_for_ref
from .tracing import TRACE_PROJECT_ENV, flush_traces, start_tracing
from .types import EvaluationRun, RunStatus, Trial

if TYPE_CHECKING:
    from .labeling import LabelImport
from .validation import UnvalidatedJudgeError

EXIT_OK = 0
EXIT_GATE = 1
EXIT_USAGE = 2
EXIT_ERROR = 3

_PREVIEW = 300


class CLIError(Exception):
    def __init__(self, message: str, *, usage: str | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class _Parser(argparse.ArgumentParser):
    @override
    def error(self, message: str) -> NoReturn:
        # Reported by ``main`` (as a JSON document under ``--json``).
        raise CLIError(message, usage=self.format_usage())


class Stopped(Exception):
    """The process was asked to stop (SIGTERM, SIGHUP)."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"stopped by {signal.Signals(signum).name}")
        self.signum = signum


_STOP_SIGNALS = tuple(
    s for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)) if s
)


def _run_async(command: Coroutine[Any, Any, int]) -> int:
    """
    Run a command on an event loop where SIGTERM and SIGHUP cancel it as
    Ctrl-C does, so a stopped process still finalizes its run.
    """

    async def main() -> int:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        received: list[int] = []

        def stop(signum: int) -> None:
            received.append(signum)
            if task is not None:
                task.cancel()

        handled: list[int] = []
        for signum in _STOP_SIGNALS:
            try:
                loop.add_signal_handler(signum, stop, signum)
            except (NotImplementedError, RuntimeError, ValueError):
                continue  # no loop signal handlers here (Windows, a thread)
            handled.append(signum)
        try:
            return await command
        except asyncio.CancelledError:
            if received:
                raise Stopped(received[0]) from None
            raise
        finally:
            for signum in handled:
                loop.remove_signal_handler(signum)

    return asyncio.run(main())


def _print_json(data: Any) -> None:
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")


def _print_markdown(text: str) -> None:
    # Rendered for a terminal; plain Markdown when piped or captured, so
    # nothing is truncated to the default width.
    if sys.stdout.isatty():
        Console().print(Markdown(text))
    else:
        sys.stdout.write(text)


def _progress_printer(mode: str) -> ProgressCallback | None:
    if mode == "none":
        return None

    def report(progress: TrialProgress) -> None:
        trial = progress.trial
        if mode == "json":
            # Sealed trials are redacted as in ``show``.
            event = {
                "event": "trial",
                "run_id": progress.run_id,
                "done": progress.done,
                "total": progress.total,
                "task_errors": progress.task_errors,
                "cost_usd": progress.cost_usd,
                "trial": trial_summary(trial, include_output=False),
            }
            sys.stderr.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
            sys.stderr.flush()
            return
        outcome = (
            "ok" if trial.ok else f"error {trial.error.type if trial.error else ''}"
        )
        if trial.scorer_failures:
            outcome += f" ({len(trial.scorer_failures)} scorer failures)"
        label = "(sealed)" if trial.sealed else trial.example_id
        sys.stderr.write(
            f"[{progress.done}/{progress.total}] {label}#{trial.repetition} "
            f"{outcome} {trial.duration_s:.1f}s · ${progress.cost_usd:.4f}\n"
        )
        sys.stderr.flush()

    return report


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


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


def _check_threshold_names(
    evaluation: Evaluation, thresholds: dict[str, float]
) -> None:
    metrics = evaluation.metrics
    if metrics is None or callable(metrics) or not thresholds:
        return
    known = [m.name for m in metrics] + [
        f"corrected_pass_rate({score})" for score in evaluation.validation_gates
    ]
    unknown = [n for n in thresholds if n not in known]
    if unknown:
        raise CLIError(f"--fail-under names unknown metrics {unknown}; known: {known}")


def _gate_messages(
    run: EvaluationRun,
    thresholds: dict[str, float],
    comparison: Comparison | None,
    gate_targets: Sequence[str] | None,
    min_effect: float,
) -> list[str]:
    failures: list[str] = []
    if run.status != RunStatus.COMPLETED:
        failures.append(f"run status is {run.status}")
    if run.invalid_reason:
        failures.append(f"run is invalid: {run.invalid_reason}")
    for name, minimum in thresholds.items():
        metric = run.metric(name)
        if metric is None:
            known = ", ".join(m.name for m in run.metrics)
            failures.append(f"metric {name!r} not found (metrics: {known})")
        elif metric.value is None or metric.value < minimum:
            failures.append(f"{name} = {metric.value} < {minimum}")
    if comparison is not None:
        failures.extend(
            regressions(comparison, targets=gate_targets, min_effect=min_effect)
        )
    return failures


def _resolve_run(ref: str, root: Path | None) -> tuple[LocalRunStore, EvaluationRun]:
    store, run_id = store_for_ref(ref, root)
    return store, store.load(run_id)


# --- Commands ---


async def _cmd_run(args: argparse.Namespace) -> int:
    if args.fail_on_regression and not args.baseline:
        raise CLIError("--fail-on-regression needs --baseline")
    if (args.gate or args.min_effect) and not args.fail_on_regression:
        raise CLIError("--gate and --min-effect need --fail-on-regression")
    evaluation = load_evaluation(args.spec)
    thresholds = _parse_thresholds(args.fail_under)
    _check_threshold_names(evaluation, thresholds)
    store = LocalRunStore(args.root)
    baseline: EvaluationRun | None = None
    if args.baseline:
        # Before running: ``latest`` must not resolve to the run being made.
        _, baseline = _resolve_run(args.baseline, args.root)
    resume: EvaluationRun | None = None
    if args.resume:
        store, resume = _resolve_run(args.resume, args.root)
    traced = args.trace or os.environ.get(TRACE_PROJECT_ENV)
    if traced:
        project = start_tracing(traced)
        sys.stderr.write(f"tracing trials into Phoenix project {project!r}\n")
    ids = [i for i in (args.ids or "").split(",") if i]
    run = await evaluation.run(
        dataset=args.dataset,
        split=args.split,
        ids=ids or None,
        sample=args.sample,
        seed=args.seed,
        limit=args.limit,
        include_sealed=args.include_sealed,
        repetitions=args.repetitions,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
        max_cost_usd=args.max_cost,
        score=not args.no_score,
        name=args.name,
        store=store,
        resume=resume,
        force=args.force,
        tags=args.tag,
        progress=_progress_printer(args.progress),
        phoenix_url=args.base_url,
        allow_unvalidated=args.allow_unvalidated,
    )
    if traced:
        flush_traces()
    comparison = compare(baseline, run) if baseline is not None else None
    pushed: str | None = None
    push_error: str | None = None
    push_exit = EXIT_OK
    if args.push:
        _warn_untraced(run)
        try:
            pushed = await _push(
                run, store, args.base_url, create_dataset=args.create_dataset
            )
        except Exception as exc:
            push_error = f"{type(exc).__name__}: {exc}"
            push_exit = _exit_code(exc)
    failures = _gate_messages(
        run,
        thresholds,
        comparison if args.fail_on_regression else None,
        args.gate or None,
        args.min_effect,
    )
    if args.json:
        payload = run_summary(run)
        payload["path"] = str(store.run_dir(run.id))
        payload["phoenix_url"] = pushed
        if push_error is not None:
            payload["push_error"] = push_error
        payload["gate_failures"] = failures
        if comparison is not None:
            payload["comparison"] = comparison.model_dump()
        _print_json(payload)
    else:
        _print_markdown(render_run_markdown(run))
        if comparison is not None:
            _print_markdown(render_comparison_markdown(comparison))
        sys.stderr.write(f"run stored in {store.run_dir(run.id)}\n")
        if pushed:
            sys.stderr.write(f"pushed to Phoenix: {pushed}\n")
    for failure in failures:
        sys.stderr.write(f"gate: {failure}\n")
    if push_error is not None:
        sys.stderr.write(f"push failed: {push_error}\n")
        return push_exit
    return EXIT_GATE if failures else EXIT_OK


async def _push(
    run: EvaluationRun,
    store: LocalRunStore,
    base_url: str | None,
    *,
    dataset: str | None = None,
    create_dataset: bool = False,
) -> str:
    from .phoenix import (  # ruff: ignore[import-outside-top-level]
        PhoenixClient,
        experiment_url,
        push_run,
    )

    async with PhoenixClient(base_url) as client:
        link = await push_run(
            client, run, store=store, dataset=dataset, create_dataset=create_dataset
        )
    if link.trace_retention is not None:
        sys.stderr.write(
            f"traces in Phoenix project {link.trace_project!r}: "
            f"{link.trace_retention}\n"
        )
    return experiment_url(link) or link.base_url


def _warn_untraced(run: EvaluationRun) -> None:
    shared = [t for t in run.trials if not t.sealed]
    if shared and not any(t.trace_id for t in shared):
        sys.stderr.write(
            f"note: no trial of {run.id} has a trace id, so its experiment links no "
            "traces (run with --trace PROJECT)\n"
        )


async def _cmd_push(args: argparse.Namespace) -> int:
    store, run = _resolve_run(args.run, args.root)
    _warn_untraced(run)
    url = await _push(
        run,
        store,
        args.base_url,
        dataset=args.dataset,
        create_dataset=args.create_dataset,
    )
    if args.json:
        _print_json({"id": run.id, "phoenix_url": url})
    else:
        sys.stdout.write(f"{url}\n")
    return EXIT_OK


async def _cmd_rescore(args: argparse.Namespace) -> int:
    store, parent = _resolve_run(args.run, args.root)
    spec = args.spec or parent.evaluation
    if not spec:
        raise CLIError(
            f"Run {parent.id} was not started from an Evaluation; "
            "pass --spec MODULE:ATTR"
        )
    evaluation = load_evaluation(spec)
    ids = [i for i in (args.ids or "").split(",") if i]
    child = await evaluation.rescore(
        parent,
        split=args.split,
        ids=ids or None,
        dataset=args.dataset,
        rerun=args.rerun,
        store=store,
        progress=_progress_printer(args.progress),
        allow_unvalidated=args.allow_unvalidated,
        phoenix_url=args.base_url,
    )
    unpaired = child.metadata.get("unpaired") if args.dataset else None
    if unpaired and (unpaired["missing"] or unpaired["input_changed"]):
        sys.stderr.write(
            f"not rescored: {len(unpaired['missing'])} examples missing from the "
            f"dataset, {len(unpaired['input_changed'])} with a changed input "
            "(see metadata.unpaired)\n"
        )
    if args.json:
        _print_json({**run_summary(child), "path": str(store.run_dir(child.id))})
    else:
        _print_markdown(render_run_markdown(child))
    return EXIT_OK


async def _cmd_import(args: argparse.Namespace) -> int:
    evaluation = load_evaluation(args.spec)
    store = LocalRunStore(args.root)
    ids = [i for i in (args.ids or "").split(",") if i]
    run = await evaluation.import_outputs(
        args.outputs,
        task=args.task,
        task_version=args.task_version,
        dataset=args.dataset,
        split=args.split,
        ids=ids or None,
        sample=args.sample,
        seed=args.seed,
        limit=args.limit,
        include_sealed=args.include_sealed,
        repetitions=args.repetitions,
        name=args.name,
        store=store,
        tags=args.tag,
        progress=_progress_printer(args.progress),
        phoenix_url=args.base_url,
        allow_unvalidated=args.allow_unvalidated,
    )
    if args.json:
        _print_json({**run_summary(run), "path": str(store.run_dir(run.id))})
    else:
        _print_markdown(render_run_markdown(run))
        sys.stderr.write(f"run stored in {store.run_dir(run.id)}\n")
    return EXIT_OK


async def _cmd_experiments(args: argparse.Namespace) -> int:
    from .phoenix import (  # ruff: ignore[import-outside-top-level]
        PhoenixClient,
        delete_experiment,
    )

    store, run = _resolve_run(args.run, args.root)
    experiment_id = run.phoenix.experiment_id if run.phoenix else None
    base_url = args.base_url or (run.phoenix.base_url if run.phoenix else None)
    async with PhoenixClient(base_url) as client:
        deleted = await delete_experiment(client, run, store=store)
    _print_json({"id": run.id, "experiment_id": experiment_id, "deleted": deleted})
    return EXIT_OK


def _cmd_annotate(args: argparse.Namespace) -> int:
    store, run = _resolve_run(args.run, args.root)
    links: dict[str, str] = {}
    for item in args.link:
        name, sep, url = item.partition("=")
        if not sep or not name or not url:
            raise CLIError(f"--link expects NAME=URL, got {item!r}")
        links[name] = url
    annotated = annotate_run(
        run, links=links, notes=args.note, remove_links=args.remove_link, store=store
    )
    payload = {"id": annotated.id, "links": annotated.links, "notes": annotated.notes}
    _print_json(payload)
    if annotated.phoenix is not None and (links or args.note or args.remove_link):
        sys.stderr.write("push the run again to show them on its experiment\n")
    return EXIT_OK


def _cmd_close(args: argparse.Namespace) -> int:
    store, run = _resolve_run(args.run, args.root)
    spec = args.spec or run.evaluation
    scorers: Sequence[Any] = ()
    metrics: Any = None
    if spec:
        evaluation = load_evaluation(spec)
        scorers = evaluation.scorers
        metrics, _ = evaluation.check_judges(store, allow_unvalidated=True)
    closed = close_run(
        run, scorers=scorers, metrics=metrics, store=store, force=args.force
    )
    if args.json:
        _print_json({**run_summary(closed), "path": str(store.run_dir(closed.id))})
    else:
        _print_markdown(render_run_markdown(closed))
    return EXIT_OK


def _failing(trial: Trial) -> bool:
    if not trial.ok or trial.scorer_failures:
        return True
    return any(s.value is False or not s.scored for s in trial.scores)


def _event_view(event: Any) -> dict[str, Any]:
    data = json.dumps(getattr(event, "data", None), default=str, ensure_ascii=False)
    preview = data if len(data) <= _PREVIEW else data[: _PREVIEW - 1] + "…"
    return {"type": event.type, "source": event.source, "data": preview}


def _cmd_show(args: argparse.Namespace) -> int:
    store, run = _resolve_run(args.run, args.root)
    if args.example:
        trials = [t for t in run.trials if t.example_id == args.example]
        if not trials:
            raise CLIError(f"No trials for example {args.example!r} in {run.id}")
        if any(t.sealed for t in trials):
            raise CLIError(
                f"Example {args.example!r} is in a sealed split; "
                "only aggregates are shown"
            )
        example = run.example(args.example)
        events = store.load_events(run.id)
        detail: dict[str, Any] = {
            "example": None if example is None else example.model_dump(mode="json"),
            "trials": [trial_summary(t) for t in trials],
            "transcripts": {
                f"{t.example_id}#{t.repetition}": [
                    _event_view(e) for e in events.get(t.key, [])
                ]
                for t in trials
            },
        }
        _print_json(detail)
        return EXIT_OK
    selected = run.trials
    if args.failures:
        selected = [t for t in run.trials if _failing(t)]
    if args.json:
        payload = run_summary(run)
        if args.trials or args.failures:
            payload["trials"] = [
                trial_summary(t, include_output=args.outputs) for t in selected
            ]
        _print_json(payload)
        return EXIT_OK
    _print_markdown(render_run_markdown(run, max_rows=50 if args.failures else 10))
    if args.trials or args.failures:
        for trial in selected:
            _print_json(trial_summary(trial, include_output=args.outputs))
    return EXIT_OK


async def _cmd_pairwise(args: argparse.Namespace) -> int:
    judge = load_object(args.judge)
    if not isinstance(judge, PairwiseJudge):
        raise CLIError(f"{args.judge} is not a PairwiseJudge")
    base_store, base = _resolve_run(args.base, args.root)
    _, candidate = _resolve_run(args.candidate, args.root)
    spec = args.spec or base.evaluation
    types: dict[str, Any] = {}
    if spec:
        evaluation = load_evaluation(spec)
        types = {
            "input_type": evaluation.resolved_input_type,
            "reference_type": evaluation.reference_type,
            "output_type": evaluation.resolved_output_type,
        }
    run = await pairwise(
        base,
        candidate,
        cast("PairwiseJudge[Any, Any, Any]", judge),
        both_orders=not args.one_order,
        **types,
        store=base_store,
        progress=_progress_printer(args.progress),
    )
    if args.json:
        _print_json({**run_summary(run), "path": str(base_store.run_dir(run.id))})
    else:
        _print_markdown(render_run_markdown(run))
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    if (args.gate or args.min_effect) and not args.fail_on_regression:
        raise CLIError("--gate and --min-effect need --fail-on-regression")
    _, base = _resolve_run(args.base, args.root)
    _, candidate = _resolve_run(args.candidate, args.root)
    comparison = compare(base, candidate, top=args.top)
    failures = (
        regressions(comparison, targets=args.gate or None, min_effect=args.min_effect)
        if args.fail_on_regression
        else []
    )
    if args.json:
        _print_json({**comparison.model_dump(), "gate_failures": failures})
    else:
        _print_markdown(render_comparison_markdown(comparison, max_rows=args.top))
    for failure in failures:
        sys.stderr.write(f"gate: {failure}\n")
    return EXIT_GATE if failures else EXIT_OK


def _cmd_runs(args: argparse.Namespace) -> int:
    store = LocalRunStore(args.root)
    runs = store.list_runs(name=args.name, limit=args.limit)
    rows = [
        {
            "id": r.id,
            "name": r.name,
            "kind": r.kind,
            "evaluation": r.evaluation,
            "task_version": r.task.version,
            "parent_run_id": r.parent_run_id,
            "status": str(r.status),
            "invalid_reason": r.invalid_reason,
            "created_at": r.created_at.isoformat(),
            "examples": r.dataset.selected_size,
            "selection": r.dataset.selection,
            "trials": r.counts.trials_done,
            "metrics": {m.name: m.value for m in r.metrics},
        }
        for r in runs
    ]
    if args.json:
        _print_json(rows)
        return EXIT_OK
    for row, r in zip(rows, runs, strict=True):
        first = r.metrics[0] if r.metrics else None
        headline = (
            f"{first.name}={first.value:.3g}"
            if first is not None and first.value is not None
            else ""
        )
        flag = " INVALID" if r.invalid_reason else ""
        version = f"@{row['task_version']}" if row["task_version"] else ""
        spec = (r.evaluation or "").rpartition(":")[2] or r.name
        parent = f" ← {r.parent_run_id}" if r.parent_run_id else ""
        sys.stdout.write(
            f"{r.id}  {r.kind:<10} {spec}{version}  {r.status}{flag}  "
            f"{r.counts.trials_done} trials  {headline}{parent}\n"
        )
    return EXIT_OK


def _cmd_list(args: argparse.Namespace) -> int:
    found = list_evaluations(args.module)
    rows = [
        {
            "spec": f"{args.module}:{attr}",
            "name": evaluation.name,
            "description": evaluation.description,
            "scorers": [e.name for e in evaluation.scorers],
        }
        for attr, evaluation in found.items()
    ]
    if args.json:
        _print_json(rows)
    else:
        for row in rows:
            sys.stdout.write(f"{row['spec']}  {row['description'] or ''}\n")
    return EXIT_OK


async def _cmd_datasets(args: argparse.Namespace) -> int:
    if args.datasets_command == "schema":
        evaluation = load_evaluation(args.spec)
        _print_json(
            example_json_schema(
                evaluation.resolved_input_type, evaluation.reference_type
            )
        )
        return EXIT_OK
    if args.datasets_command == "validate":
        evaluation = load_evaluation(args.spec)
        try:
            dataset = await evaluation.load_dataset(
                args.dataset,
                phoenix_url=args.base_url,
                cache_dir=(args.root / "datasets") if args.root else None,
            )
        except DatasetError as exc:
            _print_json({"valid": False, "errors": [str(exc)]})
            return EXIT_GATE
        problems = await evaluation.dataset_problems(
            dataset,
            phoenix_url=args.base_url,
            cache_dir=(args.root / "datasets") if args.root else None,
        )
        _print_json(
            {
                "valid": not problems,
                "name": dataset.name,
                "examples": len(dataset),
                "fingerprint": dataset.fingerprint,
                "splits": {s: len(dataset.split(s)) for s in dataset.splits},
                "problems": [p.model_dump() for p in problems],
            }
        )
        return EXIT_GATE if problems else EXIT_OK
    if args.datasets_command in {"pull", "push", "delete"}:
        return await _cmd_phoenix_datasets(args)
    dataset = Dataset.load(args.path)
    _print_json(
        {
            "name": dataset.name,
            "examples": len(dataset),
            "fingerprint": dataset.fingerprint,
            "splits": {s: len(dataset.split(s)) for s in dataset.splits},
            "ids": dataset.ids[: args.limit],
        }
    )
    return EXIT_OK


async def _cmd_phoenix_datasets(args: argparse.Namespace) -> int:
    from .phoenix import (  # ruff: ignore[import-outside-top-level]
        PhoenixClient,
        pull_dataset,
        push_dataset,
    )

    ref = parse_phoenix_ref(args.name) if args.datasets_command == "pull" else None
    server = ref.server(args.base_url) if ref is not None else args.base_url
    if args.datasets_command == "delete":
        return await _delete_dataset(args)
    async with PhoenixClient(server) as client:
        if ref is not None:
            dataset = await pull_dataset(
                client,
                ref.name,
                version=ref.version,
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
        result = await push_dataset(
            client,
            dataset,
            name=args.name,
            base_version=args.base_version,
            force=args.force,
            allow_deletes=args.allow_deletes,
        )
        _print_json(result.model_dump())
        return EXIT_OK


async def _delete_dataset(args: argparse.Namespace) -> int:
    from .phoenix import (  # ruff: ignore[import-outside-top-level]
        PhoenixClient,
        delete_dataset,
    )

    if not args.yes:
        raise CLIError(
            f"Deleting Phoenix dataset {args.name!r} deletes its experiments too; "
            "pass --yes"
        )
    async with PhoenixClient(args.base_url) as client:
        result = await delete_dataset(client, args.name)
    _print_json(result.model_dump())
    return EXIT_OK


async def _merge_labels(
    args: argparse.Namespace, records: Sequence[Any]
) -> "LabelImport":
    from .labeling import (  # ruff: ignore[import-outside-top-level]
        merge_labels,
        merge_labels_into_phoenix,
    )

    if args.into.startswith(PHOENIX_PREFIX):
        return await merge_labels_into_phoenix(
            records,
            args.into,
            labeler=getattr(args, "labeler", None),
            replace=args.replace,
            phoenix_url=args.base_url,
        )
    return merge_labels(
        records, args.into, labeler=getattr(args, "labeler", None), replace=args.replace
    )


def _report_merge(result: "LabelImport") -> None:
    _print_json(result.model_dump())
    for conflict in result.conflicts:
        sys.stderr.write(f"conflict: {json.dumps(conflict, default=str)}\n")
    if result.unlabeled and not (result.added or result.updated):
        sys.stderr.write(
            f"{result.unlabeled} records had no label yet: fill in "
            "their reference first\n"
        )
    if result.ref and (result.added or result.updated):
        sys.stderr.write(f"pin the new labels version: {result.ref}\n")


async def _excluded_records(
    sources: Sequence[str], base_url: str | None
) -> list[dict[str, Any]]:
    from .labeling import label_records  # ruff: ignore[import-outside-top-level]

    return [
        r
        for source in sources
        for r in await label_records(source, phoenix_url=base_url)
    ]


def _cmd_labels(args: argparse.Namespace) -> int:
    from .labeling import (  # ruff: ignore[import-outside-top-level]
        known_splits,
        label_requests,
        merge_labels,
        sample_for_labeling,
        write_records,
    )

    if args.labels_command == "sample":
        output = Path(args.output)
        if any(Path(p).resolve() == output.resolve() for p in args.exclude):
            raise CLIError(
                f"-o {output} is also excluded: write the new requests elsewhere"
            )
        if output.exists() and not args.force:
            raise CLIError(f"{output} exists; pass --force to replace it")
        _, run = _resolve_run(args.run, args.root)
        on_phoenix = [s for s in args.exclude if s.startswith(PHOENIX_PREFIX)]
        earlier = [
            r
            for path in args.exclude
            if not path.startswith(PHOENIX_PREFIX)
            for r in label_requests(path)
        ]
        if on_phoenix:
            earlier += asyncio.run(_excluded_records(on_phoenix, args.base_url))
        records = sample_for_labeling(
            run,
            args.n,
            score=args.score,
            against=args.against,
            strata=args.strata,
            exclude={str(r["id"]) for r in earlier},
            test_share=args.test_share,
            splits=known_splits(earlier),
            seed=args.seed,
        )
        path = write_records(records, output)
        splits: dict[str, int] = {}
        for record in records:
            for split in record["splits"]:
                splits[split] = splits.get(split, 0) + 1
        scored = args.score is None or args.score in run.score_names()
        _print_json(
            {
                "path": str(path),
                "records": len(records),
                "splits": splits,
                "run_id": run.id,
                "score": args.score,
                "score_in_run": scored,
            }
        )
        if not scored:
            sys.stderr.write(
                f"note: no trial of {run.id} has a score named {args.score!r}, so the "
                "sample is not spread over its verdicts\n"
            )
        return EXIT_OK
    if args.labels_command == "import":
        records = [r for path in args.paths for r in label_requests(path)]
        result = (
            asyncio.run(_merge_labels(args, records))
            if args.into.startswith(PHOENIX_PREFIX)
            else merge_labels(
                records, args.into, labeler=args.labeler, replace=args.replace
            )
        )
        _report_merge(result)
        return EXIT_OK
    if args.labels_command == "pull":
        return _run_async(_pull_labels(args))
    return _show_labels(Path(args.path))


async def _pull_labels(args: argparse.Namespace) -> int:
    from .labeling import (  # ruff: ignore[import-outside-top-level]
        annotation_value,
        label_requests,
    )
    from .phoenix import PhoenixClient  # ruff: ignore[import-outside-top-level]
    from .phoenix.annotations import (  # ruff: ignore[import-outside-top-level]
        PhoenixProjectNotFoundError,
        human_annotations,
    )

    records = label_requests(args.path)
    traced = [r for r in records if r.get("metadata", {}).get("trace_id")]
    if not traced:
        raise CLIError(
            f"No record in {args.path} carries a trace id (metadata.trace_id): "
            "only trials traced into Phoenix can be labeled there"
        )
    names = {args.name or r.get("metadata", {}).get("score") for r in traced}
    if None in names:
        raise CLIError("Pass --name: some records do not say which score they label")
    async with PhoenixClient(args.base_url) as client:
        try:
            found = await human_annotations(
                client,
                args.project,
                [r["metadata"]["trace_id"] for r in traced],
                sorted(cast("set[str]", names)),
            )
        except PhoenixProjectNotFoundError as exc:
            raise CLIError(str(exc)) from exc
    if len(found.missing_traces) == len({r["metadata"]["trace_id"] for r in traced}):
        raise CLIError(
            f"None of the {len(traced)} traced records' traces are in project "
            f"{args.project!r}: is it the project the run was traced into?"
        )
    labeled: list[dict[str, Any]] = []
    for record in traced:
        name = args.name or record["metadata"]["score"]
        annotation = found.by_trace.get(record["metadata"]["trace_id"], {}).get(name)
        if annotation is None:
            continue
        value = annotation_value(
            annotation.label,
            annotation.score,
            true_labels=args.true,
            false_labels=args.false,
        )
        if value is None:
            continue
        metadata = {
            **record.get("metadata", {}),
            "score": name,
            "labeler": annotation.user_id or "phoenix",
            "labeled_at": annotation.updated_at.isoformat(),
        }
        if annotation.explanation:
            metadata["label_note"] = annotation.explanation
        labeled.append({**record, "reference": value, "metadata": metadata})
    result = await _merge_labels(args, labeled)
    _print_json(
        {
            **result.model_dump(),
            "requested": len(records),
            "traced": len(traced),
            "annotated": len(labeled),
            "missing_traces": len(found.missing_traces),
        }
    )
    for conflict in result.conflicts:
        sys.stderr.write(f"conflict: {json.dumps(conflict, default=str)}\n")
    if result.ref and (result.added or result.updated):
        sys.stderr.write(f"pin the new labels version: {result.ref}\n")
    return EXIT_OK


def _show_labels(path: Path) -> int:
    dataset = Dataset.load(path)
    splits: dict[str, int] = {}
    labels: dict[str, dict[str, int]] = {}
    unlabeled = 0
    for example in dataset:
        for split in example.splits or ["(none)"]:
            splits[split] = splits.get(split, 0) + 1
        reference = example.record.reference
        if reference is None:
            unlabeled += 1
            continue
        named = (
            cast("dict[str, Any]", reference)
            if isinstance(reference, dict)
            else {str(example.metadata.get("score") or "label"): reference}
        )
        for score, value in named.items():
            counts = labels.setdefault(score, {})
            key = json.dumps(value)
            counts[key] = counts.get(key, 0) + 1
    _print_json(
        {
            "path": str(path),
            "records": len(dataset),
            "unlabeled": unlabeled,
            "splits": splits,
            "labels": labels,
        }
    )
    return EXIT_OK


# --- Parser ---


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="one JSON document on stdout"
    )


def _add_progress(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--progress",
        choices=["text", "json", "none"],
        default="text",
        help="progress on stderr: lines, one JSON object per trial, or nothing",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="progress",
        action="store_const",
        const="none",
        help="no progress (--progress none)",
    )


def _add_gate(parser: argparse.ArgumentParser, *, needs: str) -> None:
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help=f"{needs}exit 1 on a significant regression of a score or the error rate",
    )
    parser.add_argument(
        "--gate",
        action="append",
        default=[],
        metavar="TARGET",
        help="gate these targets instead of the scores and error (e.g. duration_s)",
    )
    parser.add_argument(
        "--min-effect",
        type=float,
        default=0.0,
        help="ignore regressions smaller than this (in the target's units)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
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
    run.add_argument(
        "--dataset",
        help="override the evaluation's dataset (file or phoenix:[URL/]NAME[@VER])",
    )
    run.add_argument("--split")
    run.add_argument("--ids", help="comma-separated example ids")
    run.add_argument(
        "--sample", type=_non_negative_int, help="random subset of N examples"
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--limit", type=_non_negative_int, help="first N examples (after other filters)"
    )
    run.add_argument(
        "--include-sealed",
        action="store_true",
        help="also run sealed splits (left out unless --split names one)",
    )
    run.add_argument("-r", "--repetitions", type=_positive_int)
    run.add_argument("-c", "--concurrency", type=_positive_int)
    run.add_argument("--timeout", type=float, help="per-trial timeout in seconds")
    run.add_argument("--max-cost", type=float, help="stop scheduling after USD spent")
    run.add_argument(
        "--no-score", action="store_true", help="execute only; score on --resume"
    )
    run.add_argument(
        "--resume",
        help="continue a run (a completed one is retried as a new run)",
    )
    run.add_argument(
        "--force", action="store_true", help="resume even if the code changed"
    )
    run.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="score with judges whose validation gate does not pass",
    )
    run.add_argument("--baseline", help="compare against this run afterwards")
    run.add_argument(
        "--fail-under",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="exit 1 if a metric's value is below VALUE",
    )
    _add_gate(run, needs="with --baseline: ")
    run.add_argument(
        "--trace",
        metavar="PROJECT",
        help=(
            "trace trials into this Phoenix project (default: "
            f"${TRACE_PROJECT_ENV}); needs TELEMETRY_COLLECTOR_HTTP_ENDPOINT "
            "or PHOENIX_BASE_URL"
        ),
    )
    run.add_argument("--push", action="store_true", help="mirror the run to Phoenix")
    run.add_argument(
        "--create-dataset",
        action="store_true",
        help="with --push: create the Phoenix dataset when it is missing",
    )
    run.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    run.add_argument("--name")
    run.add_argument("--tag", action="append", default=[])
    _add_json(run)
    _add_progress(run)

    push = sub.add_parser("push", help="mirror a finished run to Phoenix")
    push.add_argument("run")
    push.add_argument(
        "--dataset",
        help="attach to this Phoenix dataset (it must hold the run's examples)",
    )
    push.add_argument(
        "--create-dataset",
        action="store_true",
        help="create the Phoenix dataset when it is missing",
    )
    push.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    _add_json(push)

    imp_run = sub.add_parser(
        "import", help="score outputs produced elsewhere as a run (JSONL file)"
    )
    imp_run.add_argument("spec")
    imp_run.add_argument(
        "outputs", help="JSONL: example_id, repetition, output, error, cost_usd, ..."
    )
    imp_run.add_argument(
        "--task", help="what produced the outputs (default: the file's stem)"
    )
    imp_run.add_argument("--task-version")
    imp_run.add_argument("--dataset", help="override the evaluation's dataset")
    imp_run.add_argument("--split")
    imp_run.add_argument("--ids", help="comma-separated example ids")
    imp_run.add_argument("--sample", type=_non_negative_int)
    imp_run.add_argument("--seed", type=int, default=0)
    imp_run.add_argument("--limit", type=_non_negative_int)
    imp_run.add_argument("--include-sealed", action="store_true")
    imp_run.add_argument(
        "-r", "--repetitions", type=_positive_int, help="default: the most in the file"
    )
    imp_run.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="score with judges whose validation gate does not pass",
    )
    imp_run.add_argument("--base-url", help="Phoenix URL for phoenix: datasets")
    imp_run.add_argument("--name")
    imp_run.add_argument("--tag", action="append", default=[])
    _add_json(imp_run)
    _add_progress(imp_run)

    annotate = sub.add_parser(
        "annotate", help="add links and notes to a run (pushed with it; JSON output)"
    )
    annotate.add_argument("run")
    annotate.add_argument(
        "--link",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="add or replace a link (repeatable)",
    )
    annotate.add_argument(
        "--note", action="append", default=[], help="add a note (repeatable)"
    )
    annotate.add_argument("--remove-link", action="append", default=[], metavar="NAME")
    _add_json(annotate)

    close = sub.add_parser(
        "close", help="finalize a run whose process is gone (partial or completed)"
    )
    close.add_argument("run")
    close.add_argument(
        "--spec", help="Evaluation to take metrics from (default: the run's)"
    )
    close.add_argument(
        "--force",
        action="store_true",
        help="close even if its process may still be running",
    )
    _add_json(close)

    rescore = sub.add_parser("rescore", help="score a run's outputs again (child run)")
    rescore.add_argument("run")
    rescore.add_argument("--spec", help="Evaluation to take scorers from")
    rescore.add_argument("--split", help="only this split's trials")
    rescore.add_argument("--ids", help="only these examples' trials (comma-separated)")
    rescore.add_argument(
        "--dataset",
        help="score against this dataset version instead (file or phoenix:...)",
    )
    rescore.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    rescore.add_argument(
        "--rerun", action="store_true", help="re-run unchanged scorers too"
    )
    rescore.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="score with judges whose validation gate does not pass",
    )
    _add_json(rescore)
    _add_progress(rescore)

    show = sub.add_parser("show", help="show a run")
    show.add_argument("run")
    show.add_argument("--trials", action="store_true", help="list every trial")
    show.add_argument(
        "--failures",
        action="store_true",
        help="only trials with errors, failed or unscored scores",
    )
    show.add_argument("--outputs", action="store_true", help="include outputs")
    show.add_argument("--example", help="full detail for one example")
    _add_json(show)

    cmp = sub.add_parser("compare", help="paired comparison of two runs")
    cmp.add_argument("base")
    cmp.add_argument("candidate")
    cmp.add_argument("--top", type=_positive_int, default=5)
    _add_gate(cmp, needs="")
    _add_json(cmp)

    pair = sub.add_parser("pairwise", help="A/B two runs with an order-swapped judge")
    pair.add_argument("base")
    pair.add_argument("candidate")
    pair.add_argument("--judge", required=True, help="PairwiseJudge as MODULE:ATTR")
    pair.add_argument("--one-order", action="store_true", help="skip the swapped order")
    pair.add_argument(
        "--spec", help="Evaluation whose types to validate with (default: the base's)"
    )
    _add_json(pair)
    _add_progress(pair)

    runs = sub.add_parser("runs", help="list runs, newest first")
    runs.add_argument("--name", help="run name or evaluation spec attribute")
    runs.add_argument("--limit", type=_positive_int, default=20)
    _add_json(runs)

    listing = sub.add_parser("list", help="list the Evaluations defined in a module")
    listing.add_argument("module")
    _add_json(listing)

    datasets = sub.add_parser("datasets", help="dataset utilities (JSON output)")
    dsub = datasets.add_subparsers(dest="datasets_command", required=True)
    schema = dsub.add_parser("schema", help="JSON schema of an evaluation's examples")
    schema.add_argument("spec")
    _add_json(schema)
    validate = dsub.add_parser("validate", help="type-check and run dataset checks")
    validate.add_argument("spec")
    validate.add_argument(
        "--dataset", help="dataset to validate instead of the default"
    )
    validate.add_argument("--base-url", help="Phoenix URL for phoenix:NAME datasets")
    _add_json(validate)
    info = dsub.add_parser("show", help="summarize a dataset file")
    info.add_argument("path")
    info.add_argument("--limit", type=_positive_int, default=20)
    _add_json(info)
    pull = dsub.add_parser("pull", help="fetch a Phoenix dataset version")
    pull.add_argument(
        "name", help="[https://HOST/]NAME[@VERSION_ID] (a server named here wins)"
    )
    pull.add_argument("-o", "--output", help="also save it to this file")
    pull.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    _add_json(pull)
    push_ds = dsub.add_parser("push", help="make a Phoenix dataset match a file")
    push_ds.add_argument("path")
    push_ds.add_argument("--name", help="Phoenix dataset name (default: the file's)")
    push_ds.add_argument(
        "--base-version", help="the Phoenix version this file was pulled from"
    )
    push_ds.add_argument(
        "--force", action="store_true", help="replace even if Phoenix moved on"
    )
    push_ds.add_argument(
        "--allow-deletes",
        action="store_true",
        help="delete Phoenix examples the file does not have",
    )
    push_ds.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    _add_json(push_ds)

    delete_ds = dsub.add_parser(
        "delete", help="delete a Phoenix dataset with its experiments and projects"
    )
    delete_ds.add_argument("name")
    delete_ds.add_argument("--yes", action="store_true", help="confirm the deletion")
    delete_ds.add_argument(
        "--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)"
    )
    _add_json(delete_ds)

    experiments = sub.add_parser("experiments", help="Phoenix experiments of runs")
    esub = experiments.add_subparsers(dest="experiments_command", required=True)
    edelete = esub.add_parser(
        "delete", help="delete a run's experiment and its project (JSON output)"
    )
    edelete.add_argument("run")
    edelete.add_argument("--base-url", help="Phoenix URL (default: the run's)")
    _add_json(edelete)

    labels = sub.add_parser(
        "labels", help="collect labels for judge validation (JSON output)"
    )
    lsub = labels.add_subparsers(dest="labels_command", required=True)
    sample = lsub.add_parser(
        "sample", help="choose a run's outputs to label (writes a to-label file)"
    )
    sample.add_argument("run")
    sample.add_argument("-o", "--output", required=True, help="to-label JSONL file")
    sample.add_argument("-n", type=_positive_int, default=50, help="how many")
    sample.add_argument("--score", help="the judge score to be labeled")
    sample.add_argument(
        "--against", help="another score: outputs where the two disagree come first"
    )
    sample.add_argument("--strata", help="metadata key to spread the sample over")
    sample.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="FILE|phoenix:REF",
        help="skip outputs already in this labels or to-label file, or Phoenix dataset",
    )
    sample.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    sample.add_argument(
        "--test-share",
        type=float,
        default=0.4,
        help="share of examples in the sealed test split",
    )
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument(
        "--force", action="store_true", help="replace the output file if it exists"
    )
    _add_json(sample)
    imp = lsub.add_parser("import", help="merge filled-in to-label files")
    imp.add_argument("paths", nargs="+")
    imp.add_argument(
        "--into",
        required=True,
        help="labels dataset file, or phoenix:[URL/]NAME[@VER] (pushes a new version)",
    )
    imp.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    imp.add_argument("--labeler", help="who labeled them (unless a record says)")
    imp.add_argument(
        "--replace", action="store_true", help="replace conflicting labels"
    )
    _add_json(imp)
    lpull = lsub.add_parser(
        "pull", help="labels from human annotations in Phoenix (by trace id)"
    )
    lpull.add_argument("path", help="to-label file whose records carry trace ids")
    lpull.add_argument("--project", required=True, help="Phoenix project")
    lpull.add_argument("--name", help="annotation name (default: the records' score)")
    lpull.add_argument(
        "--true",
        action="append",
        default=[],
        metavar="LABEL",
        help="an annotation label that means pass (true/yes/pass already do)",
    )
    lpull.add_argument(
        "--false",
        action="append",
        default=[],
        metavar="LABEL",
        help="an annotation label that means fail (false/no/fail already do)",
    )
    lpull.add_argument(
        "--into",
        required=True,
        help="labels dataset file, or phoenix:[URL/]NAME[@VER] (pushes a new version)",
    )
    lpull.add_argument(
        "--replace", action="store_true", help="replace conflicting labels"
    )
    lpull.add_argument("--base-url", help="Phoenix URL (default: $PHOENIX_BASE_URL)")
    _add_json(lpull)
    lshow = lsub.add_parser("show", help="count labels by split and value")
    lshow.add_argument("path")
    _add_json(lshow)
    return parser


def _dispatch(args: argparse.Namespace) -> int:
    match args.command:
        case "run":
            return _run_async(_cmd_run(args))
        case "rescore":
            return _run_async(_cmd_rescore(args))
        case "close":
            return _cmd_close(args)
        case "annotate":
            return _cmd_annotate(args)
        case "experiments":
            return _run_async(_cmd_experiments(args))
        case "import":
            return _run_async(_cmd_import(args))
        case "push":
            return _run_async(_cmd_push(args))
        case "show":
            return _cmd_show(args)
        case "compare":
            return _cmd_compare(args)
        case "pairwise":
            return _run_async(_cmd_pairwise(args))
        case "runs":
            return _cmd_runs(args)
        case "list":
            return _cmd_list(args)
        case "datasets":
            return _run_async(_cmd_datasets(args))
        case "labels":
            return _cmd_labels(args)
        case _:
            raise CLIError(f"Unknown command {args.command!r}")


# What a caller can fix by changing the command, spec or files. Other errors
# are reported with a traceback.
_USAGE_ERRORS: tuple[type[BaseException], ...] = (
    CLIError,
    SpecError,
    RunNotFoundError,
    DatasetError,
    ResumeError,
    SealedSelectionError,
    ImportError,
    SyntaxError,
    FileNotFoundError,
    ValueError,
)


def _exit_code(exc: BaseException) -> int:
    from .phoenix import (  # ruff: ignore[import-outside-top-level]
        DatasetPushError,
        PhoenixError,
        StaleDatasetError,
    )

    if isinstance(exc, UnvalidatedJudgeError):
        return EXIT_GATE
    if isinstance(exc, StaleDatasetError | DatasetPushError):
        return EXIT_USAGE
    if isinstance(exc, PhoenixError | ValidationError):
        return EXIT_ERROR
    if isinstance(exc, _USAGE_ERRORS):
        return EXIT_USAGE
    return EXIT_ERROR


def _report_error(exc: BaseException, code: int, *, as_json: bool) -> None:
    message = str(exc) or type(exc).__name__
    if as_json:
        _print_json(
            {"error": {"type": type(exc).__name__, "message": message, "exit": code}}
        )
    sys.stderr.write(f"grasp-evals: {message}\n")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(arguments)
    except CLIError as exc:
        if exc.usage and "--json" not in arguments:
            sys.stderr.write(exc.usage)
        _report_error(exc, EXIT_USAGE, as_json="--json" in arguments)
        return EXIT_USAGE
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        sys.stderr.write("grasp-evals: interrupted\n")
        return 130
    except Stopped as exc:
        sys.stderr.write(f"grasp-evals: {exc}\n")
        return 128 + exc.signum
    except Exception as exc:
        code = _exit_code(exc)
        _report_error(exc, code, as_json=getattr(args, "json", False))
        if code == EXIT_ERROR:
            traceback.print_exception(exc, file=sys.stderr)
        return code


def console_main() -> None:
    raise SystemExit(main())
