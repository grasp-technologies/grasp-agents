# grasp-evals demo: a short-answer grader

A complete evaluation loop over a small grasp-agents pipeline, runnable offline.

- `grader_evals.py` — the system under test (a two-step `SequentialWorkflow`:
  analyzer → feedback writer, versions `v1` and `v2`), its scorers, a pairwise
  judge, and three `Evaluation` definitions (`grader_v1`, `grader_v2`,
  `grader_v2_strict`). `llm_grader_evaluation(llm)` swaps in an `LLMAgent`.
  For judges: a feedback-quality judge built from a processor (`v1`, `v2`;
  `llm_feedback_judge(llm)` for an `LLMAgent`), its validations and probes
  (`judge_v1_validation`, `judge_v2_validation`, `judge_v1_probes`,
  `judge_v2_probes`), and `grader_v2_judged`, which may only use a validated
  judge. For production: `grader_online`, which scores the grader's traced runs.
- `production.py` — the grader "in production": grades every student answer
  with tracing on — eight students, each a session — into the Phoenix project
  `short-answer-grader`.
- `data/short_answers.jsonl` — 24 student answers with teacher grades, `dev` and
  `test` splits (`test` is sealed), `topic`/`difficulty` strata.
- `data/feedback_labels.jsonl` — 41 graded answers whose feedback a teacher
  labeled specific or not, sampled from v1 and v2 runs on `dev` with
  `grasp-evals labels sample`; `dev` and `test` splits of their own.
- `walkthrough.sh` — the loop below, end to end.
- `../notebooks/evals_workflow.ipynb` — the same loop through the Python API.

## The loop from the shell

Run from the repository root (runs land in `$GRASP_EVALS_DIR`, default `./.evals`).
`grasp-evals` is the same as `python -m grasp_agents.evals`.

```bash
SPEC=src/grasp_agents/examples/evals/grader_evals.py

# 1. Validate the data: types, plus the evaluation's integrity checks.
grasp-evals datasets validate "${SPEC}:grader_v1"

# 2. Baseline on the dev split (3 repetitions per example → pass^3).
grasp-evals run "${SPEC}:grader_v1" --split dev

# 3. Read the failures behind the numbers.
grasp-evals show latest --failures
grasp-evals show latest --example bio-3        # input, reference, outputs, transcript

# 4. Fix, rerun, compare (paired by example; exit 1 if a score or the error rate
#    regresses significantly).
grasp-evals run "${SPEC}:grader_v2" --split dev --baseline latest:grader_v1 --fail-on-regression
grasp-evals compare latest:grader_v1 latest:grader_v2

# 5. Change the instrument, not the task: rescore stored outputs (a child run).
#    Unchanged scorers keep their scores; the changed one runs again.
grasp-evals rescore latest:grader_v2 --spec "${SPEC}:grader_v2_strict"

# 6. Pairwise A/B with an order-swapped judge (a run of its own).
grasp-evals pairwise latest:grader_v1 latest:grader_v2 --judge "${SPEC}:specific_feedback_judge"

# 7. The held-out test split: aggregates only, and always the whole split.
grasp-evals run "${SPEC}:grader_v2" --split test --fail-under 'pass_rate(agrees_with_teacher)=0.6'

# 8. History, and Phoenix (with PHOENIX_BASE_URL / PHOENIX_API_KEY set): share the
#    dataset, then mirror runs as experiments (sealed trials are withheld).
grasp-evals runs
grasp-evals datasets push src/grasp_agents/examples/evals/data/short_answers.jsonl
grasp-evals push latest:grader_v1
grasp-evals push latest                        # the test run: aggregates only
```

## Judges: validate before you trust

A judge is a scorer whose verdicts need checking too. Here the judge decides
whether the grader's feedback is specific enough to act on, and teachers' labels
say what the right verdict was.

```bash
# 9. Agreement with the labels: iterate on dev — the disagreements come with the
#    judge's explanation, and two judge versions compare like any two runs.
grasp-evals run "${SPEC}:judge_v1_validation" --split dev
grasp-evals run "${SPEC}:judge_v2_validation" --split dev
grasp-evals show latest:judge_v2_validation --failures
grasp-evals compare latest:judge_v1_validation latest:judge_v2_validation

# 10. Probes: degraded feedback must fail, padded or upper-cased feedback must not
#     change the verdict. No labels needed (the labels' test split stays sealed).
grasp-evals run "${SPEC}:judge_v1_probes"
grasp-evals run "${SPEC}:judge_v2_probes"

# 11. When the judge is good enough on dev: the sealed test split, once.
grasp-evals run "${SPEC}:judge_v2_validation" --split test

# 12. Use it. grader_v2_judged refuses to run (exit 1) unless the judge exactly
#     as it is now passed its gate on the test split of feedback_labels, and
#     also reports the pass rate corrected for the judge's measured errors.
grasp-evals run "${SPEC}:grader_v2_judged" --split dev

# 13. More labels: outputs to label (where the judge and a cheap check disagree
#     first, then both verdicts evenly; never sealed trials, nothing already
#     labeled), blind — fill in each "reference" — then merge.
LABELS=src/grasp_agents/examples/evals/data/feedback_labels.jsonl
grasp-evals labels sample latest:grader_v2_judged --score feedback_quality \
  --against names_key_issue -n 10 --exclude "$LABELS" -o to_label.jsonl --force
cp "$LABELS" my_labels.jsonl
grasp-evals labels import to_label.jsonl --into my_labels.jsonl --labeler you
grasp-evals labels show my_labels.jsonl
grasp-evals run "${SPEC}:judge_v2_validation" --dataset my_labels.jsonl --split dev
```

Labels are stored per score (`"reference": {"feedback_quality": true}`), with who
labeled each and when; a single value in a to-label file goes to the score its
`metadata.score` names. Verdicts and labels must be comparable — pass/fail words,
`true`/`false` and `1`/`0` all read as pass/fail; a label such as `correct` against
a pass/fail judge is a scorer failure, not a silent disagreement.

Labels can also come from Phoenix. Trials carry trace ids when the run is traced
into a Phoenix project — e.g. the spec module calls
`grasp_agents.telemetry.init_tracing(project_name=NAME)` and
`grasp_agents.telemetry.phoenix.init_phoenix(project_name=NAME)` with
`TELEMETRY_COLLECTOR_HTTP_ENDPOINT` set (e.g. `$PHOENIX_BASE_URL/v1/traces`). People annotate those traces (or any of
their spans) in the UI under the score's name, and `grasp-evals labels pull
to_label.jsonl --project NAME --into my_labels.jsonl` reads the newest human
annotation of each back (`--true`/`--false` map other label words to pass/fail).

Judges are improved by hand (or by the agent running this loop), one version at a
time: each prompt, model or code change is a new judge whose validation starts
over, compared on dev with the last one; the test split is the final check, not
a target.

## Production: online evaluation

The same scorers also score what the system did in production. An
`Evaluation` with `traces=TraceQuery(...)` reads the spans of a Phoenix project —
here the grader's runs, selected by processor name — and every scheduled run
covers the window since the previous one. Grasp-agents spans carry what this
needs: each processor run records its name, class, path, declared `version` and
model as attributes, and its input and output payloads as JSON in
`input.value` / `output.value` (long payloads are shortened without breaking the
JSON). Spans recorded while an evaluation runs are left out.

```bash
# 14. The grader in production (a simulation), traced into Phoenix.
export TELEMETRY_COLLECTOR_HTTP_ENDPOINT="$PHOENIX_BASE_URL/v1/traces"
python src/grasp_agents/examples/evals/production.py

# 15. Score its runs: the first online run says where the window starts; each
#     later one continues where the last ended (up to a completion buffer before
#     now, so spans still being exported are not missed). Scores are written
#     back onto the spans as annotations. The judge must be validated (step 11),
#     its pass rate is also reported corrected for its errors, and an alert
#     fires (exit 1) only when a metric's whole interval is below the line.
#     Clustered on the student (session), the intervals are wide with 8 of them.
grasp-evals online "${SPEC}:grader_online" --since 1h \
  --alert-below 'pass_rate(feedback_concise)=0.9'
grasp-evals online "${SPEC}:grader_online"      # later: the next window

# 16. Production inputs as examples for the offline evaluation — e.g. only the
#     failed runs (--status error) — leaving out inputs the dataset already has.
grasp-evals datasets from-traces "${SPEC}:grader_online" --since 1d \
  --exclude src/grasp_agents/examples/evals/data/short_answers.jsonl -o new_inputs.jsonl
```

- **What one trial is**: a span (`scope="span"`, the default), all selected spans
  of a trace (`"trace"`), or of a session (`"session"`, Phoenix `session.id`) —
  for conversations, scored once the session has been idle for
  `session_idle_s`. The default extractor reads the recorded payloads; a custom
  one (`TraceQuery(extractor=...)`) can fetch full artifacts from the
  application's database by the ids in the span attributes, and is where data
  that must not leave production is dropped.
- **Sampling** keeps a deterministic share of traces (`sample_rate`, by trace
  id, so every evaluation at the same rate sees the same traces), capped by
  `max_items` and spread over `strata`.
- **Runs**: an online run is a normal run on disk (`show`, `labels sample` and
  `rescore` work on it; `push` writes a rescore's scores to the traces) whose
  window is recorded; the next window starts at the end of the newest one that
  covered its window — a run stopped by `--max-cost` covers it too, so cap
  volume with `sample_rate` / `max_items` rather than the budget. A run whose
  extractor could not read every item is invalid; one that read none (an
  outage) failed, and its window is read again. Run one job at a time per
  evaluation. Annotations an online run failed to write are retried (best
  effort) by the next run, or with `grasp-evals push RUN`; the command still
  reports the window's gates, then exits 3. Annotations are named after the
  score and identified by the scorer version
  (`grasp-evals:feedback_quality@v2`), so writing them again replaces them;
  after an upgrade, both versions' annotations sit under the score's name.
- **Labels from production**: `grasp-evals labels sample latest:grader_online
  --score feedback_quality -o to_label.jsonl` picks production outputs to label;
  annotations people make on those traces in Phoenix come back with `labels
  pull`.

Every command takes `--json` (one JSON document on stdout, also for errors).
Progress goes to stderr; `--progress json` makes it one JSON object per finished
trial. Exit codes: 0 done, 1 a gate failed (invalid or incomplete run, missed
`--fail-under`, significant regression, unvalidated judge, invalid dataset), 2
usage error, 3 unexpected or Phoenix error (`online --alert-below` also exits 1). Runs are addressed by id, unique id
prefix, run directory, `latest`, or `latest:<name>` where the name is the run's
name or the spec attribute (`latest:grader_v1`).

## What to look for

- **v1 → v2**: agreement with teachers rises from about 0.46 to about 0.75 on
  `dev`, and naming the student's key issue from 0 to about 0.9. The writer is
  noisy on purpose, so numbers move from run to run, and with 16 dev examples the
  comparison's interval excludes zero in most runs, not all — rerun a few times
  to see the spread. v2 still misgrades confident wrong answers that use the
  right words (`phy-3`, `math-3`, `chem-3`) — the next iteration's work.
- **pass^3 ≤ pass rate**: the writer is noisy on borderline answers, so the
  same answer does not always get the same verdict.
- **Rescoring** with the stricter `feedback_quality` v2 drops a few v2 examples
  whose feedback is vague; the comparison warns that the scorer version
  changed, so the drop is the instrument, not the task.
- **Pairwise**: v2's feedback is at least as specific in every pair — it wins
  about half and ties the rest. Position consistency is trivially perfect here,
  since the stand-in judge is a deterministic function.
- **Judge v1 → v2**: v1 passes any feedback of three words or more — κ about
  −0.2 on dev, worse than chance: it fails "Good job!" on a correct answer and
  passes "Some points are missing." (TPR about 0.6, TNR about 0.2). v2 reaches κ
  about 0.55 on dev (TPR about 0.9, TNR about 0.65) and about 0.75 on test; its
  remaining false passes are feedback on correct answers that the grader marked
  wrong — the next judge version's work. Consistency below 1 is the stand-in
  judge flipping a tenth of its verdicts, as a sampled model would.
- **Intervals**: the labels' 20 test items come from a few original examples,
  and several outputs of one example are not independent, so intervals are
  clustered on the example and wide — κ's lower bound on test is about 0.4 (as
  low as 0.25 in some runs). The demo gates at 0.2; a judge you rely on needs
  more labels, from more examples.
- **Probes**: v1 is fooled by padding (a two-word "Good job!" becomes a pass)
  and passes generic feedback almost always; v2 fails generic feedback about
  nine times in ten. Its invariance is limited by its own random flips.
- **Gate and correction**: the gate reads the lower end of κ's interval on the
  test split, so a few lucky labels cannot pass it, and requires a verdict on
  every labeled item (abstentions and judge failures count against it);
  changing the judge (its version, prompt, model, settings, hooks, tools or
  code) needs a new validation. The corrected pass rate is the share of truly
  specific feedback implied by the judge's verdicts and its TPR/TNR, with an
  interval that includes their uncertainty.
- **Sealed test split**: reports, `show`, `compare` and Phoenix pushes give only
  aggregates, and a run that touches the split must include all of it (no
  `--ids`/`--limit`/`--sample` inside it). This keeps an agent working through
  the CLI from fitting to those examples; it is not access control — the dataset
  file and the run records on disk contain them.
