"""Human- and agent-readable views of runs and comparisons."""

from collections.abc import Iterable
from operator import itemgetter
from typing import Any

from .compare import Comparison, TargetComparison
from .types import EvaluationRun, MetricResult, Score, Trial

_MAX_TEXT = 300


def _num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000 or magnitude < 0.001:
        return f"{value:.{digits}g}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "-"
    return f"[{_num(low)}, {_num(high)}]"


def _clip(text: str | None, limit: int = _MAX_TEXT) -> str:
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _metric_rows(metric: MetricResult, label: str | None = None) -> list[str]:
    name = label or metric.name
    value = _num(metric.value)
    if metric.value is None and metric.details.get("shares"):
        shares: dict[str, float] = metric.details["shares"]
        value = ", ".join(f"{k}: {v:.0%}" for k, v in list(shares.items())[:6])
    cells = [
        _cell(name),
        _cell(value),
        _ci(metric.ci_low, metric.ci_high),
        str(metric.n),
        str(metric.n_missing),
        str(metric.n_na),
    ]
    rows = ["| " + " | ".join(cells) + " |"]
    for group, result in metric.groups.items():
        rows.extend(_metric_rows(result, label=f"  ↳ {group}"))
    return rows


def _lowest(
    trials: Iterable[Trial], name: str, limit: int
) -> list[tuple[str, float, int, Score]]:
    """Weakest examples: ``(example_id, mean value, repetitions, worst score)``."""
    by_example: dict[str, list[Score]] = {}
    for trial in trials:
        if trial.sealed:
            continue
        score = trial.score(name)
        if score is not None and score.as_float() is not None:
            by_example.setdefault(trial.example_id, []).append(score)
    rows: list[tuple[str, float, int, Score]] = []
    for example_id, scores in by_example.items():
        values = [s.as_float() or 0.0 for s in scores]
        worst = min(scores, key=lambda s: s.as_float() or 0.0)
        rows.append((example_id, sum(values) / len(values), len(values), worst))
    rows.sort(key=itemgetter(1))
    return rows[:limit]


def render_run_markdown(run: EvaluationRun, *, max_rows: int = 10) -> str:
    """Summary of a run: identity, metrics, failures and the weakest examples."""
    lines = [f"# {run.name} — {run.status}"]
    if run.invalid_reason:
        lines.append(f"\n> **INVALID:** {run.invalid_reason}")
    lines.extend(
        [
            "",
            f"- **Run:** `{run.id}` ({run.kind}) · {run.created_at:%Y-%m-%d %H:%M} UTC",
        ]
    )
    if run.parent_run_id:
        lines.append(f"- **Parent run:** `{run.parent_run_id}`")
    if run.description:
        lines.append(f"- **Question:** {run.description}")
    task_version = f"@{run.task.version}" if run.task.version else ""
    lines.append(f"- **Task:** {run.task.name}{task_version} (`{run.task.kind}`)")
    if run.provenance.observed_models:
        models = "; ".join(
            f"{agent}: {', '.join(m)}"
            for agent, m in run.provenance.observed_models.items()
        )
        lines.append(f"- **Models:** {models}")
    ds = run.dataset
    version = f"@{ds.version}" if ds.version else ""
    selection = f" — {', '.join(ds.selection)}" if ds.selection else ""
    lines.append(
        f"- **Dataset:** {ds.name}{version} `{ds.fingerprint}` · "
        f"{ds.selected_size}/{ds.size} examples{selection}"
    )
    if run.evaluators:
        evaluators = ", ".join(
            f"{e.name}@{e.version}" + (f" ({e.annotator})" if e.annotator else "")
            for e in run.evaluators
        )
        lines.append(f"- **Evaluators:** {evaluators}")
    prov = run.provenance
    if prov.git_commit:
        dirty = " +uncommitted changes" if prov.git_dirty else ""
        lines.append(f"- **Code:** `{prov.git_commit[:12]}` ({prov.git_branch}){dirty}")
    counts = run.counts
    lines.append(
        f"- **Trials:** {counts.trials_done}/{counts.trials_expected} "
        f"({run.config.repetitions} per example) · task errors {counts.task_errors} · "
        f"evaluator failures {counts.evaluator_failures} · unscored {counts.unscored}"
    )
    usage = run.usage
    if not usage.is_empty:
        cost = f" · ${usage.cost_usd:.4f}" if usage.cost_usd is not None else ""
        lines.append(
            f"- **Usage:** {usage.input_tokens} in / {usage.output_tokens} out "
            f"tokens{cost}"
        )
    sealed = sum(1 for t in run.trials if t.sealed)
    if sealed:
        lines.append(
            f"- **Sealed:** {sealed} trials in held-out splits are reported in "
            "aggregate only"
        )

    if run.metrics:
        lines += [
            "",
            "## Metrics",
            "",
            "| metric | value | 95% CI | n | missing | n/a |",
            "|---|---|---|---|---|---|",
        ]
        for metric in run.metrics:
            lines.extend(_metric_rows(metric))

    errors = [t for t in run.trials if t.error is not None]
    if errors:
        lines += ["", f"## Task errors ({len(errors)})", ""]
        by_type: dict[str, int] = {}
        for trial in errors:
            assert trial.error is not None
            by_type[trial.error.type] = by_type.get(trial.error.type, 0) + 1
        lines.append(", ".join(f"{k} x {v}" for k, v in by_type.items()))
        lines += ["", "| example | rep | error |", "|---|---|---|"]
        for trial in [t for t in errors if not t.sealed][:max_rows]:
            assert trial.error is not None
            lines.append(
                f"| `{trial.example_id}` | {trial.repetition} | "
                f"{_cell(_clip(trial.error.message))} |"
            )

    failures = [(t, f) for t in run.trials for f in t.evaluator_failures]
    if failures:
        lines += [
            "",
            f"## Evaluator failures ({len(failures)})",
            "",
            "| example | rep | evaluator | error |",
            "|---|---|---|---|",
        ]
        for trial, failure in [x for x in failures if not x[0].sealed][:max_rows]:
            lines.append(
                f"| `{trial.example_id}` | {trial.repetition} | {failure.evaluator} | "
                f"{_cell(_clip(failure.error.message))} |"
            )

    for name in run.score_names():
        lowest = _lowest(run.trials, name, 5)
        if not lowest or all(mean >= 1.0 for _, mean, _, _ in lowest):
            continue
        lines += [
            "",
            f"## Lowest `{name}`",
            "",
            "| example | mean | reps | explanation (worst repetition) |",
            "|---|---|---|---|",
        ]
        for example_id, mean, reps, worst in lowest:
            explanation = _cell(_clip(worst.explanation or worst.reason))
            row = [f"`{example_id}`", _num(mean), str(reps), explanation]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _target_row(t: TargetComparison) -> str:
    arrow = "↑" if t.higher_is_better else "↓"
    flag = " ✱" if t.significant else ""
    cells = [
        f"{arrow} {_cell(t.target)}",
        _num(t.base_mean),
        _num(t.candidate_mean),
        f"{_num(t.diff)}{flag}",
        _ci(t.ci_low, t.ci_high),
        _num(t.p_value),
        _num(t.mde),
        str(t.n_pairs),
        f"+{t.improved}/-{t.regressed}",
    ]
    return "| " + " | ".join(cells) + " |"


def render_comparison_markdown(comparison: Comparison, *, max_rows: int = 5) -> str:
    c = comparison
    lines = [
        f"# {c.candidate_name} vs {c.base_name}",
        "",
        f"- **Base:** `{c.base_run}` · **Candidate:** `{c.candidate_run}`",
        (
            f"- **Paired examples:** {c.n_paired_examples} (changed content "
            f"{c.n_changed_examples}, base-only {c.n_base_only}, candidate-only "
            f"{c.n_candidate_only})"
        ),
    ]
    for warning in c.warnings:
        lines.append(f"- ⚠ {warning}")
    lines += [
        "",
        (
            "Δ = candidate - base on paired examples; ✱ = 95% CI excludes 0; "
            "MDE = smallest true Δ detectable at 80% power."
        ),
        "",
        ("| target | base | candidate | Δ | 95% CI | p | MDE | n | +/- |"),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_target_row(t) for t in c.targets)
    for target in c.targets:
        if not target.top_regressions or target.kind != "score":
            continue
        lines += [
            "",
            f"## Regressions in `{target.target}`",
            "",
            "| example | base | candidate | candidate explanation |",
            "|---|---|---|---|",
        ]
        for delta in target.top_regressions[:max_rows]:
            explanation = _cell(_clip(delta.candidate_explanation))
            lines.append(
                f"| `{delta.example_id}` | {_num(delta.base)} | "
                f"{_num(delta.candidate)} | {explanation} |"
            )
    return "\n".join(lines) + "\n"


def _metric_summary(metric: MetricResult) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "value": metric.value,
        "ci": None if metric.ci_low is None else [metric.ci_low, metric.ci_high],
        "n": metric.n,
        "missing": metric.n_missing,
        "na": metric.n_na,
    }
    if metric.details:
        summary["details"] = metric.details
    if metric.groups:
        summary["groups"] = {k: _metric_summary(v) for k, v in metric.groups.items()}
    return summary


def run_summary(run: EvaluationRun) -> dict[str, Any]:
    """Compact machine-readable summary (what ``--json`` prints)."""
    return {
        "id": run.id,
        "name": run.name,
        "kind": run.kind,
        "status": str(run.status),
        "invalid_reason": run.invalid_reason,
        "parent_run_id": run.parent_run_id,
        "dataset": {
            "name": run.dataset.name,
            "version": run.dataset.version,
            "fingerprint": run.dataset.fingerprint,
            "selected": run.dataset.selected_size,
            "size": run.dataset.size,
            "selection": run.dataset.selection,
        },
        "task": {"name": run.task.name, "version": run.task.version},
        "evaluators": {e.name: e.version for e in run.evaluators},
        "counts": run.counts.model_dump(),
        "cost_usd": run.usage.cost_usd,
        "metrics": {m.name: _metric_summary(m) for m in run.metrics},
        "config_hash": run.config_hash,
        "git_commit": run.provenance.git_commit,
        "git_dirty": run.provenance.git_dirty,
    }


def trial_summary(trial: Trial, *, include_output: bool = True) -> dict[str, Any]:
    """One trial for ``show --json``; sealed trials never expose outputs."""
    data: dict[str, Any] = {
        "example_id": trial.example_id,
        "repetition": trial.repetition,
        "ok": trial.ok,
        "duration_s": trial.duration_s,
        "cost_usd": trial.total_usage.cost_usd,
        "trace_id": trial.trace_id,
        "sealed": trial.sealed,
    }
    if trial.sealed:
        return data
    data["scores"] = {
        s.name: {"value": s.value, "explanation": s.explanation, "reason": s.reason}
        for s in trial.scores
    }
    if trial.error is not None:
        data["error"] = {"type": trial.error.type, "message": trial.error.message}
    if trial.evaluator_failures:
        data["evaluator_failures"] = {
            f.evaluator: f.error.message for f in trial.evaluator_failures
        }
    if include_output:
        data["output"] = trial.output
    return data
