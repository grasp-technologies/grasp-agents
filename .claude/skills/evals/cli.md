# `grasp-evals` CLI reference

`grasp-evals` is the same as `python -m grasp_agents.evals`. Code: `evals/cli.py`.

## Invocation

```bash
grasp-evals [--root DIR] <command> ...      # --root goes BEFORE the command
```

- **Store:** `--root`, else `$GRASP_EVALS_DIR`, else `./.evals` in the current directory
  (a warning says so when that is inside a git repository that does not ignore it).
  Pulled datasets go to the user cache, never under the working directory.
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
| `compare BASE CAND` | Paired comparison; `--top N`; `--where KEY=VALUE`; `--group-by KEY`; gate flags | Exits 1 only with `--fail-on-regression`. Weighted scores are compared pooled |
| `rescore RUN` | Child run with re-scored stored outputs | `--spec` defaults to the run's recorded spec; `--rerun`; `--allow-unvalidated`; `--split S` / `--ids a,b` for a child of those trials only (sealed splits whole); `--dataset FILE\|phoenix:...` to score against a newer dataset version |
| `import SPEC FILE` | Scores outputs produced elsewhere as a run (`kind=import`) | JSONL of `{example_id, repetition?, output \| error, duration_s?, cost_usd?, usage?, models?, measurements?, trace_id?, started_at?}`. `--task` (default: the file's stem) and `--task-version` identify what produced them. Selection flags as for `run` (sealed splits left out; an included one needs an output or `error` line for each of its examples and repetitions); outputs outside the selection are ignored, unknown ids and repeated (id, repetition) are errors; `-r` defaults to the most repetitions in the file |
| `experiments delete RUN` | Deletes the run's Phoenix experiment and its project | JSON output |
| `datasets delete NAME --yes` | Deletes a Phoenix dataset with its experiments and their projects | Refused without `--yes` |
| `annotate RUN [--link NAME=URL]... [--note TEXT]... [--remove-link NAME]...` | Adds links and notes to a run's header | Shown by `show`; pushed into the experiment metadata (`links`, `notes`) on the next push. Refused while the run is `running` |
| `close RUN` | Finalizes a run left `running` by a killed process: `partial` (or `completed`) from its stored trials | Refused while its process runs on this host, or when that cannot be told (another host, a run recorded before processes were), unless `--force`. Usage counts what its stored trials cost it (a retry's new attempts, a rescore's re-run scorers). Metrics from `--spec`, else the run's spec |
| `pairwise BASE CAND --judge MOD:ATTR` | Order-swapped A/B judging | `--one-order`; types come from `--spec` or the base run's spec; stored in the base run's store |
| `runs` | Newest first; `--name`, `--limit` (default 20) | One line per run: id, kind, spec attribute@task version, status, first metric |
| `list MODULE` | The `Evaluation`s defined in a module | |
| `datasets schema SPEC` | JSON Schema of one record, exactly what the loader accepts | Unknown keys are rejected (`additionalProperties: false`) |
| `datasets validate SPEC [--dataset P]` | Types plus `dataset_checks` | Exit 1 on problems |
| `datasets show PATH` | Count, fingerprint, splits, ids | No spec needed |
| `datasets pull [URL/]NAME[@VER] [-o FILE]` | Pull from Phoenix (cached by server and version) | A server in the reference wins over `$PHOENIX_BASE_URL`; a different `--base-url` is an error |
| `datasets push PATH [--name --base-version --force --allow-deletes]` | Make a Phoenix dataset match a file | Uploads every example, sealed ones included |
| `labels sample RUN -o FILE [--exclude FILE\|phoenix:REF]...` | Writes a to-label JSONL | See [judges.md](judges.md#labeling-loop) |
| `labels import FILES --into LABELS.jsonl\|phoenix:[URL/]NAME[@VER]` | Merges filled-in labels | Into Phoenix: pushes a new version based on the given one and prints the reference to pin |
| `labels pull FILE --project P --into LABELS.jsonl` | Reads human annotations from Phoenix | Needs traced trials |
| `labels show FILE` | Counts labels by split and value | |
| `push RUN [--dataset NAME --create-dataset]` | Mirrors a finished run to Phoenix as an experiment | See Phoenix below |

`datasets` and `labels` always print JSON.

## `run` options

| Option | Meaning |
|---|---|
| `--dataset PATH\|phoenix:[URL/]NAME[@VER]` | Overrides the spec's dataset. A server in the reference wins over `$PHOENIX_BASE_URL` |
| `--split S`, `--ids a,b`, `--sample N --seed S`, `--limit N` | Applied in this order: split → (sealed splits left out) → ids → sample → limit. An empty selection is an error |
| `--include-sealed` | Keep the sealed splits, which are otherwise left out unless `--split` names one |
| `-r`, `-c`, `--timeout`, `--max-cost` | Override the spec's repetitions, concurrency, per-trial task timeout and budget |
| `--no-score` | Runs the task only. The validation gate is skipped; scores come later with `--resume` (into a `retry` child) |
| `--resume RUN [--force]` | Continue (see SKILL.md pitfall 12). Pass the same selection and `-r` as the original run |
| `--baseline RUN` | Compares with that run afterwards. `latest` resolves *before* the new run exists |
| `--fail-under METRIC=V` | Repeatable. Exit 1 when the metric's value is below V or missing |
| `--fail-on-regression [--gate T]... [--min-effect X]` | Needs `--baseline`. `--gate` replaces the default targets (scores and `error`); `--min-effect` is in the target's units |
| `--allow-unvalidated` | Run even though a validation gate fails; the failures are recorded in `metadata.unvalidated_judges` |
| `--trace PROJECT` | Trace trials into this Phoenix project (default `$GRASP_EVALS_TRACE_PROJECT`); see Phoenix below |
| `--push [--base-url URL] [--create-dataset]` | Mirror the run to Phoenix after it finishes (creating a missing Phoenix dataset only with `--create-dataset`) |
| `--name`, `--tag` | Run name override; extra tags |
| `--json`, `--progress text\|json\|none`, `-q` | See Output |

These settings live only in the spec and have no CLI override: `max_error_rate`,
`scorer_timeout_s`, `group_by`, `cluster_by`, `sealed_splits`, `dataset_checks` and
`validation_gates`.

## Output

- **`--json`** prints one JSON document on stdout, errors included:
  `{"error": {"type", "message", "exit"}}`.
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
| 143, 129 | Stopped by SIGTERM or SIGHUP (a stopped shell, a container shutdown): finalized as `cancelled`, like Ctrl-C |

With `run --push`, a push failure's code replaces the gate result. The run is still
stored and reported, and `push_error` is set in the JSON.

## Gating recipes

```bash
# CI: candidate vs a pinned baseline. --gate replaces the default targets.
grasp-evals run evals/grading.py:grader_llm --split dev --json -q \
  --baseline "$BASE_ID" --fail-on-regression --gate agrees --gate error \
  --fail-under 'pass_rate(agrees)=0.8'

# Latency regression only (resources are not gated unless named).
grasp-evals compare "$BASE_ID" "$CAND_ID" --fail-on-regression --gate duration_s --min-effect 0.5
```

The regression gate fails in three cases:
- nothing pairs;
- a gated target has pairs but no test (e.g. a single example, or all examples in one
  `cluster_by` cluster);
- a significant regression of at least `--min-effect`.

See [metrics.md](metrics.md#regression-gate).

## On disk

```
<store>/runs/<run_id>/
  run.json            header: status, config, provenance, metrics, counts, Phoenix link
  examples.jsonl      the evaluated examples, as stored (the run is self-contained)
  trials.jsonl        append-only; a later line with the same (example, repetition) wins
  transcripts.jsonl   event streams (processor tasks with capture_events=True)
  report.md           the rendered summary
~/.cache/grasp-evals/phoenix/...   pulled Phoenix dataset versions, per server and version
                                   ($GRASP_EVALS_CACHE_DIR, else $XDG_CACHE_HOME/grasp-evals)
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
- **Dataset names.** With `GRASP_EVALS_DATASET_PREFIX` set (e.g. `staging/`), `push`
  and `datasets push` refuse any Phoenix dataset name outside it.
- **Failures.** A 5xx from the server after retries is `PhoenixUnavailableError` (exit
  3), not a compatibility error. GET/PUT/DELETE and the trial posts Phoenix
  deduplicates (a trial's run, its evaluations) are retried on 502–504, timeouts and
  dropped connections; other POSTs are not. A trial that still fails names the run,
  the example id and the repetition.
- **NUL characters.** Phoenix stores rows in PostgreSQL, which rejects `\u0000`. Pushes
  replace NULs in every string (keys included) with U+FFFD; `run.phoenix.altered_trials`
  lists the trials that had any. Datasets pushed with NULs pull back without them, so
  their content hashes differ from the local file's.
- **`datasets push`:**
  - refuses a derived subset unless `--force`;
  - refuses if Phoenix moved past the version you pulled (`--base-version`, or the
    pulled file's version) unless `--force`;
  - deletes Phoenix rows only with `--allow-deletes`;
  - creates no version when the copy is unchanged.
- **`push RUN`:**
  - creates one experiment per run (a rescore is a separate experiment);
  - attaches to the dataset the run's examples were pulled from on this server;
    otherwise to the Phoenix dataset named `--dataset NAME` (default: the run's
    dataset name), only if it holds exactly the run's non-sealed examples;
  - never creates a dataset unless `--create-dataset` (also on `run --push`): the
    error names the dataset it would have created. Push the dataset file first, or
    attach the run to the shared dataset with `--dataset`;
  - withholds sealed trials (their count goes into the experiment metadata);
  - scopes the experiment to the run's split when it selected exactly one
    (`--split dev`); otherwise the experiment covers every example;
  - on a repeat push, logs only trials that changed;
  - refreshes the experiment's description (the question, status and headline
    metrics) and metadata on every push, merging into the metadata on Phoenix so
    keys added there survive;
  - notes on stderr when no trial has a trace id (the experiment links no traces);
  - deletes the empty `Experiment-<id>` project Phoenix creates with each experiment
    (trials' traces stay in their tracing project, where the experiment's trace
    links find them);
  - reports how long Phoenix keeps the linked traces (the tracing project's retention
    policy, in `run.phoenix.trace_retention`) and warns when it deletes them: the
    experiment's trace links stop working then.
- **Deleting.** `experiments delete RUN` deletes a run's experiment with its project and
  forgets the link (a later push makes a new one). `datasets delete NAME --yes` deletes
  a dataset with its experiments and their projects (Phoenix's own REST delete leaves
  the projects behind).
- **Trace ids.** Trials get a `trace_id` only when an OpenTelemetry tracer provider is
  active during the run; otherwise nothing reaches Phoenix and nothing warns at run time
  (`push` notes it). `run --trace PROJECT` (or `$GRASP_EVALS_TRACE_PROJECT`) sets
  tracing up before the first trial, sending to `TELEMETRY_COLLECTOR_HTTP_ENDPOINT`,
  else `$PHOENIX_BASE_URL/v1/traces`, and flushes at the end. Tracing set up by the spec
  module (`init_tracing` + `init_phoenix`) wins over `--trace`. Runs record the project
  their traces went to (`trace_project`). `labels pull` needs these trace ids.
- **Without Phoenix**, everything works except `push`, `datasets pull/push`,
  `labels pull` and `phoenix:` datasets.
