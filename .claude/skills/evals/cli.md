# `grasp-evals` CLI reference

`grasp-evals` is the same as `python -m grasp_agents.evals`. Code: `evals/cli.py`.

## Invocation

```bash
grasp-evals [--root DIR] <command> ...      # --root goes BEFORE the command
```

- **Store:** `--root`, else `$GRASP_EVALS_DIR`, else `./.evals` in the current directory.
  `grasp-evals run SPEC --root DIR` fails with "unrecognized arguments".
- **Spec forms:** `pkg.module:attr` (portable) or `path/to/file.py:attr` (recorded by
  absolute path).
  - `attr` may name a zero-argument function that returns an `Evaluation`.
  - Without `:attr`, the module must define exactly one `Evaluation`.
  - `grasp-evals list MODULE` lists them.
- **Spec imports:** a spec file inside an importable package is imported under its module name.
  Otherwise its directory is added to `sys.path`, so sibling modules import.
- **Run references (`RUN`):** a full id, a unique id prefix, a run directory path
  (`.../runs/<id>`, which may be outside the store), `latest`, or `latest:<name>`.
  `<name>` matches the run name or the spec attribute. Both `latest` forms pick the
  newest run of *any* kind, including `retry`, `rescore` and `pairwise` children.

## Commands

| Command | Does | Notes |
|---|---|---|
| `run SPEC` | Runs an `Evaluation` | Selection, overrides, gates and `--push` below |
| `show RUN` | Markdown summary; `--json`; `--trials`; `--failures`; `--outputs`; `--example ID` | `--failures` lists trials with a task error, a scorer failure, a `False` score or an unscored score. `--example` always prints JSON and refuses sealed examples |
| `compare BASE CAND` | Paired comparison; `--top N`; gate flags | Exits 1 only with `--fail-on-regression` |
| `rescore RUN` | Child run with re-scored stored outputs | `--spec` defaults to the run's recorded spec; `--rerun`; `--allow-unvalidated` |
| `pairwise BASE CAND --judge MOD:ATTR` | Order-swapped A/B judging | `--one-order`; types come from `--spec` or the base run's spec; stored in the base run's store |
| `runs` | Newest first; `--name`, `--limit` (default 20) | One line per run: id, kind, spec attribute@task version, status, first metric |
| `list MODULE` | The `Evaluation`s defined in a module | |
| `datasets schema SPEC` | JSON Schema of one record, exactly what the loader accepts | Unknown keys are rejected (`additionalProperties: false`) |
| `datasets validate SPEC [--dataset P]` | Types plus `dataset_checks` | Exit 1 on problems |
| `datasets show PATH` | Count, fingerprint, splits, ids | No spec needed |
| `datasets pull NAME[@VER] [-o FILE]` | Pull from Phoenix (cached by version) | |
| `datasets push PATH [--name --base-version --force --allow-deletes]` | Make a Phoenix dataset match a file | Uploads every example, sealed ones included |
| `labels sample RUN -o FILE` | Writes a to-label JSONL | See [judges.md](judges.md#labeling-loop) |
| `labels import FILES --into LABELS.jsonl` | Merges filled-in labels | |
| `labels pull FILE --project P --into LABELS.jsonl` | Reads human annotations from Phoenix | Needs traced trials |
| `labels show FILE` | Counts labels by split and value | |
| `push RUN` | Mirrors a finished run to Phoenix as an experiment | |

`datasets` and `labels` always print JSON.

## `run` options

| Option | Meaning |
|---|---|
| `--dataset PATH\|phoenix:NAME[@VER]` | Overrides the spec's dataset |
| `--split S`, `--ids a,b`, `--sample N --seed S`, `--limit N` | Applied in this order: split → ids → sample → limit. An empty selection is an error |
| `-r`, `-c`, `--timeout`, `--max-cost` | Override the spec's repetitions, concurrency, per-trial task timeout and budget |
| `--no-score` | Runs the task only. The validation gate is skipped; scores come later with `--resume` (into a `retry` child) |
| `--resume RUN [--force]` | Continue (see SKILL.md pitfall 11). Pass the same selection and `-r` as the original run |
| `--baseline RUN` | Compares with that run afterwards. `latest` resolves *before* the new run exists |
| `--fail-under METRIC=V` | Repeatable. Exit 1 when the metric's value is below V or missing |
| `--fail-on-regression [--gate T]... [--min-effect X]` | Needs `--baseline`. `--gate` replaces the default targets (scores and `error`); `--min-effect` is in the target's units |
| `--allow-unvalidated` | Run even though a validation gate fails; the failures are recorded in `metadata.unvalidated_judges` |
| `--push [--base-url URL]` | Mirror the run to Phoenix after it finishes |
| `--name`, `--tag` | Run name override; extra tags |
| `--json`, `--progress text\|json\|none`, `-q` | See Output |

These settings live only in the spec and have no CLI override: `max_error_rate`,
`scorer_timeout_s`, `group_by`, `cluster_by`, `sealed_splits`, `dataset_checks` and
`validation_gates`.

## Output

- **`--json`** prints one JSON document on stdout, errors included:
  `{"error": {"type", "message", "exit"}}`. Anything else that prints while the
  command runs (the code under evaluation, a dependency) goes to stderr, so stdout
  always parses. The same holds for `datasets` and `labels`, which always print JSON.
- **Without `--json`**, `run` and `show` print Markdown (plain text when stdout is not a
  terminal), and the run directory is written to stderr.
- **Progress** goes to stderr. `--progress json` prints one object per finished trial:
  `{"event": "trial", "run_id", "done", "total", "task_errors", "cost_usd", "trial": {...}}`.
  Sealed trials show only `{"sealed": true, "repetition", "ok"}`.
- **The `run --json` payload** has these keys:
  - `id, name, kind, evaluation, status, invalid_reason, parent_run_id`
  - `dataset{name, version, fingerprint, selected, size, selection}`
  - `task{name, version}`, `scorers{name: version}`
  - `counts{examples, trials_expected, trials_done, task_errors, scorer_failures, unscored}`, `cost_usd`
  - `metrics{<metric name>: {value, ci, confidence, n, missing, na, details?, groups?}}`
  - `config_hash, git_commit, git_dirty, path, phoenix_url, gate_failures`
  - `comparison` (with `--baseline`) and `push_error`
- **In scripts**, capture the id with
  `ID=$(grasp-evals run SPEC --json -q | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')`.

## Exit codes

| Code | When |
|---|---|
| 0 | Done, and nothing below applies |
| 1 | **`run`:** the status is not `completed` (e.g. `partial` because the budget ran out); the run is invalid (no trial ran, every trial failed, `max_error_rate` exceeded); a `--fail-under` metric is below its value or missing; `--fail-on-regression` found a regression or an untestable or missing target; or a judge is unvalidated (`UnvalidatedJudgeError`). **`compare --fail-on-regression`:** the same regression rules. **`datasets validate`:** type errors or `dataset_checks` problems |
| 2 | Usage errors: bad arguments, spec, run reference or dataset file; `dataset_checks` problems found by `run`; `ResumeError`; `SealedSelectionError`; any other `ValueError`; a missing `PHOENIX_BASE_URL`; a stale or mismatched Phoenix dataset |
| 3 | An unexpected error (with a traceback), a Phoenix server error, or a pydantic `ValidationError` |
| 130 | Interrupted. The run is finalized as `cancelled` and can be resumed in place |

With `run --push`, a push failure's code replaces the gate result. The run is still
stored and reported, and `push_error` is set in the JSON.

## Gating recipes

```bash
# CI: candidate vs a pinned baseline, on the default targets (scores and error).
grasp-evals run evals/grading.py:grader_llm --split dev --json -q \
  --baseline "$BASE_ID" --fail-on-regression --fail-under 'pass_rate(agrees)=0.8'

# Latency regression only (resources are not gated unless named).
grasp-evals compare "$BASE_ID" "$CAND_ID" --fail-on-regression --gate duration_s --min-effect 0.5
```

The regression gate fails when:
- nothing pairs;
- a gated target has no paired values, or has pairs but no test (e.g. a single
  example, or all examples in one `cluster_by` cluster);
- a significant regression of at least `--min-effect`;
- without `--gate`: a score of the baseline is missing from the candidate (its scorer
  failed on every trial, or the score was renamed);
- a `--gate` target is not in the comparison.

See [metrics.md](metrics.md#regression-gate).

## On disk

```
<store>/runs/<run_id>/
  run.json            header: status, config, provenance, metrics, counts, Phoenix link
  examples.jsonl      the evaluated examples, as stored (the run is self-contained)
  trials.jsonl        append-only; a later line with the same (example, repetition) wins
  transcripts.jsonl   event streams (processor tasks with capture_events=True)
  report.md           the rendered summary
<store>/datasets/phoenix/...   pulled Phoenix dataset versions (cached by version)
```

- A run id looks like `20261103T120000-<name-slug>-<4 hex>`. A run directory must keep
  its id as its name (copies under another name are refused).
- A crash-truncated last line is skipped and trimmed before the next append. A corrupt
  middle line is an error.
- Provenance records:
  - the git commit, branch, dirty flag and diff hash of the repository that contains
    the *current directory*;
  - a hash of the task's and scorers' source files;
  - the Python and grasp-agents versions;
  - the models observed per agent.

## Phoenix

- **Setup.** Requires `pip install 'grasp-agents[phoenix]'` and a Phoenix server 20.0
  or later. Configure it with `PHOENIX_BASE_URL` (or `--base-url`) and
  `PHOENIX_API_KEY`. When `PHOENIX_BASE_URL` is set, the environment key is never sent
  to a different `--base-url`.
- **`datasets push`:**
  - refuses a derived subset unless `--force`;
  - refuses if Phoenix moved past the version you pulled (`--base-version`, or the
    pulled file's version) unless `--force`;
  - deletes Phoenix rows only with `--allow-deletes`;
  - creates no version when the copy is unchanged.
- **`push RUN`:**
  - creates one experiment per run (a rescore is a separate experiment);
  - attaches to a Phoenix dataset of the same name only if that dataset holds exactly
    the run's non-sealed examples, creates the dataset if it is missing, and otherwise
    refuses, so push the dataset file first;
  - withholds sealed trials (their count goes into the experiment metadata);
  - on a repeat push, logs only trials that changed.
- **Trace ids.** Trials get a `trace_id` only when an OpenTelemetry tracer provider is active during
  the run. For example, call `grasp_agents.telemetry.init_tracing(project_name=...)` and
  `grasp_agents.telemetry.phoenix.init_phoenix(...)` in the spec module;
  `init_phoenix` reads `TELEMETRY_COLLECTOR_HTTP_ENDPOINT`. `labels pull` needs these
  trace ids.
- **Without Phoenix**, everything works except `push`, `datasets pull/push`,
  `labels pull` and `phoenix:` datasets.
