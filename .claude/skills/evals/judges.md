# Judges, validation, labels and pairwise A/B

Code: `evals/judge.py`, `evals/validation.py`, `evals/labeling.py`, `evals/pairwise.py`,
`evals/phoenix/annotations.py`. A judge is a scorer, so everything in
[SKILL.md](SKILL.md) applies to it as well.

## An LLM judge as a scorer

```python
from pydantic import BaseModel

from grasp_agents import LLMAgent
from grasp_agents.evals import JudgedOutput, ProcessorScorer, Score


class Verdict(BaseModel):
    specific: bool
    explanation: str


type Judged = JudgedOutput[Submission, Grade, TeacherGrade]


def build_judge() -> LLMAgent[Judged, Verdict, None]:
    return LLMAgent[Judged, Verdict, None](
        name="feedback_judge", llm=make_llm(), sys_prompt="Is the feedback specific?"
    )


def to_scores(verdict: Verdict) -> Score:  # required for a structured verdict
    return Score(name="specific", value=verdict.specific, explanation=verdict.explanation)


judge = ProcessorScorer(
    build_judge,
    name="specific",
    version="1",
    config={"model": "<model id>", "prompt": "v1"},  # a factory judge has no fingerprint
    to_scores=to_scores,
)
```

| Fact | Detail |
|---|---|
| Input | By default a `JudgedOutput(input, output, reference, metadata)` built from the trial. `to_input(ctx)` replaces it |
| Verdict | `to_scores(verdict)` must return what a scorer returns. Without it, the processor's output must already be a `Score`, scalar, list or mapping; a pydantic model fails every call |
| Isolation | Each call runs a fresh copy (or a factory product) in its own session |
| Usage | Goes to `trial.scorer_usage[<judge name>]` and `run.scorer_usage`, not `trial.usage` or `run.task_usage`. It counts toward `cost_usd`, the budget and `run.usage`. Its tokens appear in the `scorer_*_tokens` measures, never in the task's `input_tokens`/`output_tokens`/`total_tokens`. Models land in `trial.models` |
| Failure | A raise, or a verdict that does not parse, is a scorer failure. Rescore retries it |
| Identity | `describe().fingerprint` covers a template's models, settings, prompts, output schema and tools. `describe().source` covers the code of `to_input`, `to_scores`, custom classes, hooks and tool functions. Any change makes rescore re-run the judge and voids earlier validations. A factory-built judge has no fingerprint, so put its model and prompt version in `version` or `config` |
| Load | All scorers of a trial run concurrently, so judge calls in flight = trials in flight (`concurrency`) × judges. A pairwise judge makes 2 calls per pair |
| Timeouts | None, unless the `Evaluation` sets `scorer_timeout_s` |

## Judge validation (agreement with labels)

A labels dataset is a regular dataset whose inputs are `JudgedOutput` records and whose
reference is the correct verdict:

```json
{"id": "c10e1eacfa4e", "input": {"input": {"question": "..."}, "output": {"verdict": "correct", "feedback": "Good job!"}, "reference": {"verdict": "correct"}}, "reference": {"specific": true}, "metadata": {"example_id": "q1", "score": "specific"}, "splits": ["dev"]}
```

```python
from grasp_agents.evals import judge_validation

judge_v1_validation = judge_validation(  # an Evaluation: define it in a spec module
    judge,
    LABELS,  # a path, "phoenix:NAME[@VER]", a Dataset or a loader
    input_type=Submission,  # the judged evaluation's types, so the judge sees typed values
    output_type=Grade,
    reference_type=TeacherGrade,
    # scores=["specific"], threshold=None, positive=True, negative=False,
    # repetitions=1, sealed_splits=("test",), cluster_by="example_id", group_by=("split",)
)
```

The loop:
1. While iterating, run `grasp-evals run spec.py:judge_v1_validation --split dev`.
   - `show <id> --failures` lists disagreements with the judge's explanation.
   - `compare` pairs two judge versions on the target `specific.agrees`.
2. Run `--split test` once, at the end.

| Point | Behavior |
|---|---|
| Splits | It needs a `test` split by default (`sealed_splits=("test",)`); otherwise `ValueError`, even with `--split dev`. Pass `sealed_splits=()` when the labels have no test split |
| Which run counts | Only runs with sealed (test) trials count for gates. A run *without* `--split` includes the test split, so it counts as the final test run |
| Task | The judge is the task (`ScorerTask`), so a judge failure is a *task error* here, not a scorer failure. It counts as missing |
| Recorded scores | `<s>.agrees` (bool), `<s>.judge`, `<s>.label` (text) |
| Metrics | `accuracy(s)`, `kappa(s)`, `tpr(s)`, `tnr(s)`, `confusion(s)`, plus `consistency(s)` when `repetitions > 1`, and `error_rate`, grouped per split |
| Comparison rules | Pass/fail words (`true/false`, `yes/no`, `pass/fail`, `passed/failed`) and 0/1 are read as booleans when compared with a bool. Numbers at or above `threshold` pass. A bool verdict against a text label such as `"correct"`, or a number against text, raises `LabelMismatchError`: a scorer failure, counted missing, not a disagreement |
| Labels | `{"s": value}` per score, or a bare value for the score named by `metadata.score`. When several scores are validated, a bare value with no `metadata.score` is a scorer failure |
| Intervals | Clustered on `metadata.example_id`: several labeled outputs of one example form one cluster, so intervals are wide when few examples are behind many labels |

## Probes (no labels)

```python
from grasp_agents.evals import Perturbation, judge_probes, judged_outputs

PROBES = [
    Perturbation(
        "generic",
        lambda item: item.output.model_copy(update={"feedback": "Review the material."}),
        expect="lower",  # a degradation: the verdict must fail or drop
    ),
    Perturbation(
        "padded",
        lambda item: item.output.model_copy(update={"feedback": item.output.feedback + " Keep going!"}),
        expect="same",  # an irrelevant change: the verdict must not move
    ),
]
judge_v1_probes = judge_probes(
    judge, LABELS, PROBES, sealed_splits=["test"],
    input_type=Submission, output_type=Grade, reference_type=TeacherGrade,
)
```

- **Perturbations.** `apply(item)` returns the changed output, or `None` to skip that
  item. `expect` is `"lower"`, `"changed"` (for unordered labels) or `"same"`. The
  perturbations' code is part of the run's identity.
- **Items** come from a labels file or from `judged_outputs(run, input_type=...,
  output_type=..., reference_type=...)`, which holds a run's OK, non-sealed outputs.
  Outputs that are good to begin with leave room for a degradation to show.
- **Metrics** are `sensitivity(<s>.<probe>)` and `invariance(<s>.<probe>)`.
  - A `"lower"` probe is not applicable when the original verdict already fails, and it
    needs bool or numeric verdicts.
  - If the judge fails on any variant, the whole item is a task error.
- **Sealing.** `judge_probes` seals nothing by default. Probing a labels dataset without
  `sealed_splits=["test"]` (or `--split dev`) shows its test items one by one.

## Labeling loop

```bash
grasp-evals labels sample RUN -o to_label.jsonl --score specific --against agrees \
  -n 50 [--strata topic] [--exclude labels.jsonl] [--test-share 0.4] [--seed 0] [--force]
# a person fills in each "reference": true / false, or {"specific": true}
grasp-evals labels import to_label.jsonl --into labels.jsonl --labeler alice
grasp-evals labels show labels.jsonl
grasp-evals run spec.py:judge_v1_validation --dataset labels.jsonl --split dev
```

| Point | Behavior |
|---|---|
| What is sampled | OK, non-sealed trials of a system run (not a validation, probe or pairwise run). Identical outputs of one example appear once. The judge's verdict is not included, so labeling is blind |
| Order | Disagreements between `--score` and `--against` come first. Then the sample alternates between the verdicts `--score` gave (TPR and TNR need both classes), and within a verdict between `--strata` values. Outputs the judge did not score come last |
| `--score` / `--against` | `--score` may name a score the run does not have yet (a warning; no spreading over verdicts). `--against` must exist |
| Record identity | `id` is a hash of (input, output, reference, metadata). `metadata` carries `example_id`, `repetition`, `run_id`, `score`, and `trace_id` when the run was traced |
| Splits | `test` or `dev` from a hash of the original example id, so `--test-share` is approximate and every output of one example lands in the same split. Splits found in `--exclude` files are reused |
| Import | Labels are stored per score, with `labeler` and `labeled_at` per score. Equality is typed: `true`, `"pass"` and `1` are different labels, so relabeling with another spelling reports conflicts and keeps the old value unless `--replace`. A record whose id matches but whose output differs is refused. `--into` must be `.jsonl`. Blank or null references count as unlabeled. Nothing is written when nothing changed |
| Python | `sample_for_labeling(run, n, score=..., against=...)` returns records; `grasp_agents.evals.labeling.write_records(records, path)` writes them; `import_labels(paths, into, labeler=...)` merges |

Labels from Phoenix: `grasp-evals labels pull to_label.jsonl --project P --into
labels.jsonl [--name S] [--true WORD]... [--false WORD]...`

- **Which records qualify:** only those with a `metadata.trace_id`, which means tracing
  was on during the run (see [cli.md](cli.md#phoenix)).
- **What it reads:** *human* annotations made on the trace or on any of its spans,
  keeping the newest per trace. Annotations on experiment runs are not read.
- **Annotation name:** `--name`, or each record's `metadata.score`.
- **Value mapping:**
  - `true/yes/pass(ed)` and `false/no/fail(ed)`, plus any `--true`/`--false` words,
    become booleans;
  - other labels stay text;
  - with no label, the numeric score is used.
- **Refusal:** it refuses when none of the traces are in the project.

## Validation gates

```python
from grasp_agents.evals import Evaluation, ValidationGate

grader_judged = Evaluation(
    name="grader",
    task=FunctionTask(rule_grader, version="rules-1"),
    dataset=DATA,
    reference_type=TeacherGrade,
    scorers=[agrees, judge],
    validation_gates={"specific": ValidationGate(min_kappa=0.4, labels="feedback_labels")},
)
```

| Point | Behavior |
|---|---|
| Keys | Score names (what the judge's `to_scores` emits) |
| Fields | `min_accuracy`, `min_kappa`, `min_tpr`, `min_tnr`, `max_missing=0.0`, `labels=None` (the labels dataset name: the file stem unless the file sets a name), `bound="ci_low"` (or `"value"`) |
| Which validation | The newest completed, valid validation run in the same store that covers the score and has sealed trials, from a judge whose `describe()` equals the current judge's exactly. It is summarized over the sealed trials only. A rescored validation run ranks by its parent's time |
| When checked | In `Evaluation.run` (when scoring) and `Evaluation.rescore`, before any trial runs. It raises `UnvalidatedJudgeError` (CLI exit 1). `--no-score` skips the check |
| Overrides | `--allow-unvalidated` runs anyway and records the failures in `metadata.unvalidated_judges`. Validations that were found are recorded in `metadata.judge_validations` |
| Strictness | A perfect judge on 15 test labels has accuracy `ci_low` ≈ 0.77 and κ `ci_low` ≈ 0.53. One missing verdict (an abstention or a judge failure) fails `max_missing=0.0`. Retry failed judgments with `--resume` |
| Corrected rate | For each gated score whose validation measured both TPR and TNR, `corrected_pass_rate(<s>)` is appended to the metrics (Rogan-Gladen correction, with an interval that includes the rates' uncertainty). Its value is `None` when TPR + TNR ≤ 1. It assumes the labeled outputs resemble the scored ones. `--fail-under 'corrected_pass_rate(s)=V'` is accepted |

## Pairwise A/B

```python
from grasp_agents.evals import FunctionPairwiseJudge, PairwiseContext, PairwiseVerdict


def more_specific(ctx: PairwiseContext[Submission, Grade, TeacherGrade]) -> PairwiseVerdict:
    a, b = len(ctx.first.feedback), len(ctx.second.feedback)
    if a == b:
        return PairwiseVerdict(winner="tie")
    return PairwiseVerdict(winner="first" if a > b else "second")


longer_feedback = FunctionPairwiseJudge(more_specific, name="longer", annotator="CODE")
```

Run it with `grasp-evals pairwise BASE CAND --judge spec.py:longer_feedback [--one-order]`.

For an LLM judge, use `ProcessorPairwiseJudge(agent_or_factory, to_input=...,
to_verdict=...)`.
- Its default input is `JudgedPair(input, first, second, reference, metadata)`.
- Its output must be a `PairwiseVerdict` unless `to_verdict` is given; anything else is a
  scorer failure.
- Judge `annotator` defaults to `"LLM"`.

| Point | Behavior |
|---|---|
| Pairing | By (example id, repetition), with unchanged content. Only repetitions that both runs have are paired |
| Errors | If one arm's task errored, the other arm wins without a judge call. If both errored, the pair is skipped |
| Position bias | By default each pair is judged in both orders. Agreement gives the winner; disagreement gives `"inconsistent"`, which counts as a tie (0.5). A judge that always answers "first" gets a win rate of 0.5 and a position consistency of 0 |
| Scores | `<judge>.winner` (base, candidate, tie or inconsistent), `<judge>.prefers_candidate` (1, 0.5 or 0), `<judge>.position_consistent` (bool) |
| Metrics | `win_rate(<judge>)`: examples are the unit, repetitions are averaged first, and `details` holds per-judgment counts, `examples_won`/`examples_lost` and `sign_test_p`. Also `pass_rate(<judge>.position_consistent)` |
| Types | The CLI re-validates outputs with the base run's spec (or `--spec`). Python `pairwise()` defaults to plain JSON unless `output_type=` is passed |
| The run | A `pairwise` run in the base run's store. It is shown, compared and pushed like any run, but labels cannot be sampled from it |
