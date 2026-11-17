# Evaluations

`grasp_agents.evals` measures a processor (or any function) against a
dataset: run it on every example, score each output with scorers, aggregate
with metrics, and compare runs. Every run is an append-only record on disk that
never changes once completed; rescoring, retrying and pairwise judging create
new runs from stored outputs. Judges (LLM or otherwise) are scorers too, and
can be validated against human labels before an evaluation is allowed to use
them. The `grasp-evals` CLI drives the same API from the shell.

## Key types

| Name | Module | Role |
|---|---|---|
| `Example[InT, RefT]` | `evals/types.py` | One case: `id`, `input`, `reference`, `metadata`, `splits` |
| `Dataset` | `evals/dataset.py` | Load (`.jsonl`, `.json`, `.yaml`), select, split, sample, check, save |
| `Task`, `ProcessorTask`, `FunctionTask` | `evals/task.py` | The system under test |
| `Scorer`, `@scorer`, `ScoreContext`, `Score` | `evals/scorer.py`, `evals/types.py` | Per-trial judgments |
| `Metric` (`PassRate`, `PassAtK`, `PassHatK`, `Mean`, `Percentile`, `ErrorRate`, `CohenKappa`, …) | `evals/metrics.py` | Aggregates with intervals |
| `evaluate`, `rescore`, `close_run` | `evals/runner.py` | Run / re-score / finalize a run whose process died |
| `import_outputs`, `ImportedTrial`, `read_outputs` | `evals/imported.py` | Score outputs produced elsewhere as a run |
| `Evaluation` | `evals/evaluation.py` | An importable, CLI-runnable definition |
| `EvaluationRun`, `Trial` | `evals/types.py` | The run record |
| `RunStore`, `LocalRunStore` | `evals/store.py` | Persistence |
| `compare` | `evals/compare.py` | Paired comparison of two runs |
| `pairwise`, `PairwiseJudge`, `FunctionPairwiseJudge` | `evals/pairwise.py` | A/B judging, both orders |
| `ProcessorScorer`, `judge_validation`, `judge_probes`, `ValidationGate`, `Perturbation` | `evals/judge.py`, `evals/validation.py` | Judges and their validation |
| `sample_for_labeling`, `import_labels`, `judged_outputs` | `evals/labeling.py` | Label collection |
| `PhoenixRef`, `parse_phoenix_ref`, `load_phoenix_dataset` | `evals/evaluation.py` | Phoenix dataset references |

Nothing here is exported from the root package; import from `grasp_agents.evals`.

## Data

A dataset file holds one example per line (JSONL) or a list (JSON/YAML):

```json
{"id": "bio-1", "input": {"question": "...", "student_answer": "..."},
 "reference": {"verdict": "correct"}, "metadata": {"difficulty": "easy"}, "splits": ["dev"]}
```

Give curated examples explicit, stable ids: runs pair and resume by id. A
missing `id` defaults to a hash of the input. Each example also carries a
`content_hash` of its stored record; two runs pair an example only when its
content is unchanged. Metadata keys that nothing reads (provenance notes,
descriptions) can be declared descriptive — `Evaluation(descriptive_metadata=
["source", "check"])`, `Dataset(..., descriptive_metadata=...)` or
`dataset.with_descriptive_metadata(...)` — and are then left out of the
content hash; runs record them, and `compare` and `pairwise` leave the keys
either run declared out when pairing, also against runs made before the
declaration. `Dataset.load(path, input_type=, reference_type=)`
validates records against your types. `dataset_checks` on an `Evaluation` are
integrity functions (`example -> problem message | None`) run before every run.

## Tasks

`evaluate` accepts a `Task`, any `Processor`, or a function — async, or plain,
which `FunctionTask` runs in a worker thread so it does not block other trials. A
`ProcessorTask(processor_or_factory, version=, ctx_factory=, input_mode=,
input_fn=, output_fn=, output_type=)` runs each trial in its own
`SessionContext`, with that session ambient for the whole trial:

- a template processor is copied per trial and rebound to the trial's session
  (agents start with an empty transcript); a zero-argument factory is called
  inside the trial's session;
- `ctx_factory(example)` builds the session (e.g. seeded `state`);
- the output is the run's single payload, or whatever `output_fn(packet, ctx)`
  derives (e.g. a value from `ctx.state`).

For a `ProcessorTask`, each trial records duration, usage and cost per agent,
the models used, the event stream (without token deltas) and, when traced, the
trace id. A `FunctionTask` records duration and the trace id; it records usage
and models only if the function fills its `TrialContext`, and it never records
events.

## Scorers

```python
from grasp_agents.evals import Score, ScoreContext, scorer

@scorer(version="1")
def agrees_with_teacher(ctx: ScoreContext[Submission, Grade, TeacherGrade]) -> Score:
    return Score(name="agrees_with_teacher", value=ctx.output.verdict == ctx.reference.verdict)
```

A scorer returns a `Score`, a list of them, a scalar (`bool` pass/fail, number,
or `str` label, named after the scorer), a mapping of names to values, or
`None` (not applicable, nothing recorded). `Score(value=None, reason=...)`
means *unscored*: it is excluded from metrics and counted separately, never
turned into a number. `ScoreContext` gives `example`, `output`, `trial`, the
task's `events`, the trial's `session` (for state-mutating tasks) and
`record_usage(...)` for a scorer's own model cost. Sync scorers run in a worker
thread. Bump `version` whenever a scorer's semantics change: runs are
comparable only under the same scorer versions.

A scorer's identity (`describe()`) also covers its code — its function or
class without docstrings, comments or layout, plus the user-code helpers,
classes and module-level constants it refers to, followed a few levels deep —
and anything listed in `identity=[...]`: strings (a prompt, a model id),
`Path`s (a prompt or rubric file, or a directory), functions. A change to any
of them makes `rescore` run the scorer again, voids judge validations and is
flagged by `compare`. `lower_is_better=True` (or a set of score names) marks
scores whose lower values are better, such as a defect share;
`Mean(..., higher_is_better=False)` does the same from a metric, also for
recorded measurements. Runs record each score's direction in
`EvaluationRun.directions`.

## Running

```python
from grasp_agents.evals import Dataset, PassRate, ProcessorTask, evaluate

run = await evaluate(
    ProcessorTask(grader, version="v2"),
    Dataset.load("data/short_answers.jsonl", input_type=Submission, reference_type=TeacherGrade).split("dev"),
    scorers=[agrees_with_teacher],
    metrics=[PassRate("agrees_with_teacher")],
    repetitions=3,
)
print(run.metric("pass_rate(agrees_with_teacher)"))
```

Options include `concurrency`, `timeout_s`, `scorer_timeout_s`,
`max_cost_usd` (stop scheduling once spent; the run ends `partial`),
`max_error_rate` (mark the run invalid), `group_by` (per-group metrics),
`cluster_by` (clustered intervals), and `sealed_splits`. With `metrics=None`,
defaults are picked from the scores produced.

An `Evaluation(name=, task=, dataset=, scorers=, metrics=, input_type=,
reference_type=, repetitions=, sealed_splits=, dataset_checks=,
validation_gates=, ...)` packages all of this. `task` may be a factory and
`dataset` a path, a Phoenix reference or a loader, so
importing a definition builds no clients. Run it with `await ev.run(split=...)`
or `grasp-evals run module_or_file.py:attr`.

**Sealed splits.** A split listed in `sealed_splits` is left out of a run
unless the selection names it (`--split test`, `dataset.split("test")`) or the
run opts in (`--include-sealed`, `include_sealed=True`); the run report and
`run_summary` (`dataset.sealed_left_out`) say which sealed splits were left
out. When a sealed split is run, reports, `show`, `compare` and Phoenix pushes
give only aggregates, and the run must include all of it (no `--ids`,
`--limit` or `--sample` inside it). This keeps an agent iterating through the
CLI from fitting to held-out examples. It is not access control: the dataset
file and run records contain the examples.

## Runs and storage

Runs are stored by `LocalRunStore` under `$GRASP_EVALS_DIR` (default
`./.evals`):

```
runs/<run_id>/run.json           header: status, provenance, config, metrics
runs/<run_id>/examples.jsonl     the examples evaluated
runs/<run_id>/trials.jsonl       one trial per line, appended as trials finish
runs/<run_id>/transcripts.jsonl  per-trial event streams
runs/<run_id>/report.md
```

The default `./.evals` is relative to the current directory; creating it
inside a git repository that does not ignore it logs a warning. Pulled
Phoenix datasets are cached outside it, per user: `$GRASP_EVALS_CACHE_DIR`,
else `$XDG_CACHE_HOME/grasp-evals`, else `~/.cache/grasp-evals`.

`EvaluationRun.kind` is `evaluation`, `rescore`, `pairwise`, `retry` or
`import`. A run
records provenance (git commit/branch/dirty state, source hashes, Python and
package versions, observed models). `evaluate(..., resume=run)` continues an
unfinished run in place, or retries a completed run's failed trials in a new
`retry` child (a completed run with nothing to redo is returned as is). A
resume must use the same task identity, scorers, selection and repetitions;
that cannot be forced. It is also refused when the git commit or diff of the
repository containing the current directory changed (any file, not only the
task's) or when the task's or scorers' source files changed; `force=True`
overrides those two checks. A model that the run never used stops the resume.
`rescore(run, scorers)` scores stored outputs again in a child run; unchanged
scorers keep their scores. `split=` and `ids=` make a child of those
examples' trials only; `dataset=` scores the outputs against another version
of the dataset, pairing examples by id where the input is unchanged (the
child records that version and its splits, and lists unpaired examples in
`metadata["unpaired"]`). Runs are addressed by id, unique id prefix, run
directory, `latest` or `latest:<name>`.

A run records the process executing it (`EvaluationRun.process`: host, pid).
Ctrl-C, SIGTERM and SIGHUP end a CLI run `cancelled`. A run whose process was
killed outright stays `running`; `close_run(run)` (`grasp-evals close RUN`)
finalizes it from its stored trials as `partial` (or `completed`), which
`rescore`, `push` and `resume` accept. It refuses while the process still runs
on this host, or when that cannot be told (another host, an older run),
unless forced.

**Outputs produced elsewhere** (another agent, a person, a vendor) become a
run with `import_outputs(records, dataset, scorers, task=...)`,
`Evaluation.import_outputs(path)` or `grasp-evals import SPEC FILE`. Each
`ImportedTrial` (one JSONL line) carries `example_id`, `repetition`, `output`
or `error`, and optionally `duration_s`, `cost_usd`, `usage`, `models`,
`measurements`, `trace_id` and `started_at`. The run (`kind="import"`) is
scored as a rescore scores stored outputs, on the selection a run would make,
so it compares, pairs and pushes like any run; its task is identified only by
the `task` name, version and config you give.

## Comparing

`compare(base, candidate)` pairs the examples both runs share with unchanged
content, reduces repetitions per example, and reports per target the mean
paired difference with a confidence interval and a p-value. Exact McNemar is
used only when every paired per-example value is 0 or 1 and nothing is
clustered; with repetitions > 1 or `cluster_by` it is a paired (or clustered)
t-test. `significant` reads the Holm-adjusted p-value. Task errors count as
failures of pass/fail scores. Each score is read in the direction the runs
recorded (a score they disagree on is left out). Default targets are the
numeric scores present in both runs; a score of the base run that the
candidate lacks is listed in `missing_scores` and fails the default
regression gate.

`pairwise(base, candidate, judge)` asks a `PairwiseJudge` which of two outputs
is better. The judge sees them only as `first` and `second`; with
`both_orders=True` (default) every pair is judged in both orders to expose
position bias. If only one arm failed, the other wins without a judge call.

## Judges and validation

`ProcessorScorer(processor, name=, version=, to_input=, to_scores=)` turns any
processor, typically an `LLMAgent` with a structured verdict, into a scorer.

A judge is validated like any other system:

- `judge_validation(judge, labels, ...)` returns an `Evaluation` that applies
  the judge to labeled outputs and reports accuracy, Cohen's κ, TPR, TNR and
  the confusion matrix per split, with intervals clustered on the original
  example. Labels are `JudgedOutput` records whose `reference` is the correct
  verdict. With `against=<the judged dataset>`, a label whose example is gone
  or whose input or reference changed since it was labeled stops the run
  (`Evaluation.judged_dataset`).
- `judge_probes(judge, items, perturbations)` checks sensitivity and invariance
  without labels: each `Perturbation(name, apply, expect="lower" | "changed" |
  "same")` changes an output and states what must happen to the verdict.
- `ValidationGate(min_accuracy=, min_kappa=, min_tpr=, min_tnr=,
  max_missing=0.0, labels=, bound="ci_low")` on
  `Evaluation.validation_gates={score_name: gate}` blocks the whole run (and
  `Evaluation.rescore`) before any trial, unless the
  newest validation run of the judge, exactly as it is now, on the sealed split
  of its labels, meets the thresholds (lower interval bound by default). A
  changed judge (version, prompt, model, settings, code) needs a new validation.
  Gated scores also report a pass rate corrected for the judge's measured error
  rates (`CorrectedPassRate`). A judge's prompt file, or a prompt or model it
  reads from elsewhere, belongs in its `identity=[...]` so a change to it
  needs a new validation too. The check raises `UnvalidatedJudgeError`
  (`--allow-unvalidated` overrides; a `--no-score` run skips the check).
- `sample_for_labeling(run, n, score=, against=)` picks outputs worth labeling
  (disagreements with a cheaper check first, never sealed trials);
  `import_labels(paths, into)` merges filled-in files with labeler and time.
  On the CLI, `labels sample --exclude` and `labels import --into` also take
  a Phoenix labels dataset (`phoenix:[URL/]NAME[@VER]`): the import pushes a
  new version based on the given one and prints the reference to pin.

## The `grasp-evals` CLI

`grasp-evals` (also `python -m grasp_agents.evals`):

| Command | Does |
|---|---|
| `run SPEC [--split --ids --sample --limit --include-sealed -r -c --timeout --max-cost --resume --baseline --fail-under M=V --fail-on-regression --trace PROJECT --push]` | run an `Evaluation` |
| `rescore RUN [--spec SPEC --split --ids --dataset --rerun --allow-unvalidated]` | score stored outputs again (`--spec` defaults to the run's recorded spec) |
| `close RUN [--spec SPEC --force]` | finalize a run whose process is gone |
| `annotate RUN [--link NAME=URL --note TEXT --remove-link NAME]` | add links and notes to a run |
| `import SPEC FILE [--task NAME --task-version V --split --ids ...]` | score outputs produced elsewhere as a run |
| `show RUN [--failures --trials --outputs --example ID]` | inspect a run |
| `compare BASE CANDIDATE [--fail-on-regression --gate T --min-effect X]` | paired comparison |
| `pairwise BASE CANDIDATE --judge MODULE:ATTR [--one-order]` | A/B judging |
| `runs`, `list MODULE` | list runs / evaluations |
| `datasets {schema, validate, show, pull, push, delete}` | dataset utilities |
| `experiments delete RUN` | delete a run's Phoenix experiment |
| `labels {sample, import, pull, show}` | label collection |
| `push RUN [--dataset NAME --create-dataset]` | mirror a finished run to Phoenix |

Every command accepts `--json` (one JSON document on stdout, errors included);
progress goes to stderr (`--progress json` for one object per trial).

| Exit code | Meaning |
|---|---|
| 0 | done |
| 1 | a gate failed: invalid or incomplete run, missed `--fail-under`, significant regression, unvalidated judge; `datasets validate` found problems |
| 2 | usage error: bad arguments, spec, dataset file or run reference; dataset problems found by `run`; resume refused |
| 3 | unexpected error, or Phoenix failed |
| 130, 143, 129 | interrupted (Ctrl-C), or stopped by SIGTERM or SIGHUP: the run is finalized `cancelled` |

## Phoenix

With the `phoenix` extra and `PHOENIX_BASE_URL` (plus `PHOENIX_API_KEY`) set,
datasets can be pushed to and pulled from Phoenix, runs are mirrored as
experiments (`push`; sealed trials are withheld, and a run on one split gets
an experiment over that split), and `labels pull` reads human annotations of
traced trials back as labels.

A dataset reference is `phoenix:[https://HOST/]NAME[@VERSION]`, or a
`PhoenixRef(name, version, base_url)`. A version id only means something on
the server that issued it, so a pinned reference should name its server: the
server a reference names wins over `PHOENIX_BASE_URL` (and conflicts with a
different `--base-url`). `PHOENIX_API_KEY` is still sent only to
`PHOENIX_BASE_URL`'s server, or to any server when that is unset. Pinned
versions are cached per server and pulled once. `load_phoenix_dataset` and
`parse_phoenix_ref` are exported for loaders of your own.

Every push refreshes the experiment's description (the question, status
and headline metrics) and metadata, merging into what Phoenix holds, so keys
added there survive; a run's own `links` and `notes` (`annotate_run`,
`grasp-evals annotate`) go into that metadata. Push deletes the empty
project Phoenix creates for each experiment: trials' traces stay in their
tracing project, and push reports that project's retention policy
(`run.phoenix.trace_retention`), warning when it will delete the traces the
experiment links to. `grasp-evals experiments delete RUN` and `datasets
delete NAME --yes` (`delete_experiment`, `delete_dataset`) remove
experiments and datasets with their projects.

`push` attaches a run to the dataset its examples were pulled from, or to a
Phoenix dataset (`--dataset NAME`, default the run's dataset name) that holds
exactly its examples; it creates a missing dataset only with
`--create-dataset`, and `GRASP_EVALS_DATASET_PREFIX` keeps push and
`datasets push` inside one namespace. NUL characters, which Phoenix's
PostgreSQL store rejects, are replaced with U+FFFD in everything pushed
(`run.phoenix.altered_trials` lists the trials affected). Trial posts that
Phoenix deduplicates are retried on gateway errors, and a failure names the
trial; a 5xx from the server is `PhoenixUnavailableError`. Phoenix 20.0 or
later is required; older servers raise
`PhoenixCompatibilityError`. Trials get trace ids only while tracing is on:
`grasp-evals run --trace PROJECT` (or `$GRASP_EVALS_TRACE_PROJECT`;
`grasp_agents.evals.tracing.start_tracing` in Python) sets it up before the
first trial, sending to `TELEMETRY_COLLECTOR_HTTP_ENDPOINT` or
`$PHOENIX_BASE_URL/v1/traces`; tracing the spec module set up itself
(`init_tracing` + `init_phoenix`, see [Observability](observability.md)) wins.
Runs record the project in `trace_project`, and `push` notes a run without
trace ids.

## Contracts

1. A completed run's trials and results are never modified; retries and
   rescoring create child runs (`EvaluationRun.parent_run_id`). `push` only adds
   the Phoenix link and digests to `run.json`, and `annotate_run` only its
   `links` and `notes`.
2. Examples pair across runs only by id with unchanged content (declared
   descriptive metadata keys aside).
3. Unscored values are never counted as numbers in metrics.
4. A run that includes a sealed split includes all of it.
5. A gated score is never computed by a judge that did not pass its gate,
   unless explicitly allowed.

## Common pitfalls

- Default metrics give `bool` scores a pass rate (task errors count as
  failures) but numeric scores a mean (task errors are dropped as missing).
  Return `bool` for pass/fail, and always read `error_rate`.
- A run is invalid only when no trial ran, every trial failed, or
  `max_error_rate` is exceeded; set `max_error_rate` to fail on errors.
- A run without `--split` leaves sealed splits out; a judge validation counts
  for its gate only once it ran the sealed split (`--split test`).
- In an `Evaluation`, pass `task=FunctionTask(fn)`: a bare function is treated
  as a zero-argument factory.
- Bare `rescore()` and `pairwise()` re-validate stored outputs only with
  `output_type=`; otherwise scorers receive plain JSON.
- A `ProcessorScorer` whose judge returns a structured verdict needs
  `to_scores=`. `judge_validation` expects a `test` split unless
  `sealed_splits=()`.

The `evals` skill in `.claude/skills/evals/` lists these and more, with fixes.

## Limitations

- Only offline evaluation. Outputs produced elsewhere can be imported
  (`import_outputs`), but traces are not read back into runs.
- LLM calls are not recorded for replay; rerunning a task calls the models
  again.
- Trials share the host: no per-trial sandbox or environment is created unless
  your `ctx_factory` builds one.

## See also

- [Observability](observability.md), [Orchestration](orchestration.md).
- Code: `evals/`.
- Examples: `src/grasp_agents/examples/evals/` (`grader_evals.py`,
  `walkthrough.sh`, `README.md`),
  `src/grasp_agents/examples/notebooks/evals_workflow.ipynb`.
