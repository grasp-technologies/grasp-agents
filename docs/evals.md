# Evaluations

`grasp_agents.evals` measures a processor (or any async function) against a
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
| `evaluate`, `rescore` | `evals/runner.py` | Run / re-score |
| `Evaluation` | `evals/evaluation.py` | An importable, CLI-runnable definition |
| `EvaluationRun`, `Trial` | `evals/types.py` | The run record |
| `RunStore`, `LocalRunStore` | `evals/store.py` | Persistence |
| `compare` | `evals/compare.py` | Paired comparison of two runs |
| `pairwise`, `PairwiseJudge`, `FunctionPairwiseJudge` | `evals/pairwise.py` | A/B judging, both orders |
| `ProcessorScorer`, `judge_validation`, `judge_probes`, `ValidationGate`, `Perturbation` | `evals/judge.py`, `evals/validation.py` | Judges and their validation |
| `sample_for_labeling`, `import_labels`, `judged_outputs` | `evals/labeling.py` | Label collection |

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
content is unchanged. `Dataset.load(path, input_type=, reference_type=)`
validates records against your types. `dataset_checks` on an `Evaluation` are
integrity functions (`example -> problem message | None`) run before every run.

## Tasks

`evaluate` accepts a `Task`, any `Processor`, or an async function. A
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
`dataset` a path, a `phoenix:NAME[@VERSION]` reference or a loader, so
importing a definition builds no clients. Run it with `await ev.run(split=...)`
or `grasp-evals run module_or_file.py:attr`.

**Sealed splits.** For a split listed in `sealed_splits`, reports, `show`,
`compare` and Phoenix pushes give only aggregates, and a run touching the split
must include all of it (no `--ids`, `--limit` or `--sample` inside it). This
keeps an agent iterating through the CLI from fitting to held-out examples. It
is not access control: the dataset file and run records contain the examples.

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

`EvaluationRun.kind` is `evaluation`, `rescore`, `pairwise` or `retry`. A run
records provenance (git commit/branch/dirty state, source hashes, Python and
package versions, observed models). `evaluate(..., resume=run)` continues an
unfinished run in place, or retries a completed run's failed trials in a new
`retry` child (a completed run with nothing to redo is returned as is). A
resume must use the same task identity, scorers, selection and repetitions;
that cannot be forced. It is also refused when the git commit or diff of the
repository containing the current directory changed (any file, not only the
task's) or when the task's or scorers' source files changed; `force=True`
overrides those two checks. A model that the run never used stops the resume. `rescore(run, scorers)` scores stored outputs again in a child run;
unchanged scorers keep their scores. Runs are addressed by id, unique id prefix,
run directory, `latest` or `latest:<name>`.

## Comparing

`compare(base, candidate)` pairs the examples both runs share with unchanged
content, reduces repetitions per example, and reports per target the mean
paired difference with a confidence interval and a p-value. Exact McNemar is
used only when every paired per-example value is 0 or 1 and nothing is
clustered; with repetitions > 1 or `cluster_by` it is a paired (or clustered)
t-test. `significant` reads the Holm-adjusted p-value. Task errors count as
failures of pass/fail scores. Default targets are the numeric scores present
in both runs, so a score that is missing from the candidate is not compared
unless named.

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
  verdict.
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
  rates (`CorrectedPassRate`). The check raises `UnvalidatedJudgeError`
  (`--allow-unvalidated` overrides; a `--no-score` run skips the check).
- `sample_for_labeling(run, n, score=, against=)` picks outputs worth labeling
  (disagreements with a cheaper check first, never sealed trials);
  `import_labels(paths, into)` merges filled-in files with labeler and time.

## The `grasp-evals` CLI

`grasp-evals` (also `python -m grasp_agents.evals`):

| Command | Does |
|---|---|
| `run SPEC [--split --ids --sample --limit -r -c --timeout --max-cost --resume --baseline --fail-under M=V --fail-on-regression --push]` | run an `Evaluation` |
| `rescore RUN [--spec SPEC --rerun --allow-unvalidated]` | score stored outputs again (`--spec` defaults to the run's recorded spec) |
| `show RUN [--failures --trials --outputs --example ID]` | inspect a run |
| `compare BASE CANDIDATE [--fail-on-regression --gate T --min-effect X]` | paired comparison |
| `pairwise BASE CANDIDATE --judge MODULE:ATTR [--one-order]` | A/B judging |
| `runs`, `list MODULE` | list runs / evaluations |
| `datasets {schema, validate, show, pull, push}` | dataset utilities |
| `labels {sample, import, pull, show}` | label collection |
| `push RUN` | mirror a finished run to Phoenix |

Every command accepts `--json` (one JSON document on stdout, errors included);
progress goes to stderr (`--progress json` for one object per trial).

| Exit code | Meaning |
|---|---|
| 0 | done |
| 1 | a gate failed: invalid or incomplete run, missed `--fail-under`, significant regression, unvalidated judge; `datasets validate` found problems |
| 2 | usage error: bad arguments, spec, dataset file or run reference; dataset problems found by `run`; resume refused |
| 3 | unexpected error, or Phoenix failed |

## Phoenix

With the `phoenix` extra and `PHOENIX_BASE_URL` (plus `PHOENIX_API_KEY`) set,
datasets can be pushed to and pulled from Phoenix (`phoenix:NAME[@VERSION]`
dataset references), runs are mirrored as experiments (`push`; sealed trials
are withheld, and a run on one split gets an experiment over that split), and `labels pull` reads human annotations of traced trials back
as labels. Phoenix 20.0 or later is required; older servers raise
`PhoenixCompatibilityError`. To give trials trace ids, initialize tracing in
the spec module (`init_tracing` + `init_phoenix`; see
[Observability](observability.md)).

## Contracts

1. A completed run's trials and results are never modified; retries and
   rescoring create child runs (`EvaluationRun.parent_run_id`). `push` only adds
   the Phoenix link and digests to `run.json`.
2. Examples pair across runs only by id with an unchanged `content_hash`.
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
- A run without `--split` includes sealed splits in its headline metrics.
- In an `Evaluation`, pass `task=FunctionTask(fn)`: a bare function is treated
  as a zero-argument factory.
- Bare `rescore()` and `pairwise()` re-validate stored outputs only with
  `output_type=`; otherwise scorers receive plain JSON.
- A `ProcessorScorer` whose judge returns a structured verdict needs
  `to_scores=`. `judge_validation` expects a `test` split unless
  `sealed_splits=()`.

The `evals` skill in `.claude/skills/evals/` lists these and more, with fixes.

## Limitations

- Only offline evaluation of runs started by the framework; traces produced
  elsewhere are not scored.
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
