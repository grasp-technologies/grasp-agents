# Metrics, intervals and comparisons

Code: `evals/metrics.py`, `evals/stats.py`, `evals/compare.py`.

## Outcome taxonomy

| Outcome | How it arises | Recorded as | Pass/fail metrics (`PassRate`, `PassAtK`, `PassHatK`) | Other metrics |
|---|---|---|---|---|
| scored | The scorer returns a value | `Score` | The value | The value |
| unscored | `Score.unscored(name, reason)`, or a non-finite number | `Score(value=None, reason=...)` | `n_missing` | `n_missing` |
| not applicable | The scorer returns `None` | Nothing; the scorer is listed in `trial.scorers_run` | `n_na` | `n_na` |
| scorer failure | The scorer raises, times out (`scorer_timeout_s`), or repeats a score name another scorer already produced | `trial.scorer_failures` | `n_missing` | `n_missing` |
| task error | The task raises or times out, its output is not JSON-serializable, or `trial.record` got a non-finite value | `trial.error`; scorers are skipped unless `scores_errors=True` | A failure (0), unless `errors_as_failures=False` | `n_missing` |
| never ran | The budget ran out, or the run was cancelled | No trial | `n_missing` | `n_missing` |

- A score that is absent from an OK trial counts as not applicable only when *no*
  scorer failed on that trial. Otherwise it counts as missing, because score names are
  not tied to scorers.
- `counts.unscored` and `counts.scorer_failures` on the run add these up across trials.

## Default metrics (`metrics=None`)

| Values of score `s` across trials | Metric | Name |
|---|---|---|
| Always `bool` | `PassRate(s)` | `pass_rate(s)` |
| Numbers, or numbers mixed with bools | `Mean(s)` | `mean(s)` |
| Any `str` | `Distribution(s)` | `dist(s)` (value `None`; `counts`/`shares` in `details`) |
| Only unscored | `Mean(s)` | `mean(s)` |

Always added: `error_rate`, `p50(duration_s)` and `p95(duration_s)`, plus
`total(cost_usd)` when any trial has a cost. To keep the defaults and add more:
`metrics=lambda trials: [*default_metrics(trials), PassHatK("s", 3)]`.

`rescore()` called from Python does not inherit the parent run's metrics, so pass
`metrics=`. `Evaluation.rescore` and the CLI pass the spec's metrics.

## Catalog

| Metric | Unit (`n` counts) | Task errors | Name |
|---|---|---|---|
| `PassRate(of, threshold=None, errors_as_failures=True, reduce=mean, higher_is_better=None)` | examples | failures | `pass_rate(of)`, `pass_rate(of>=0.7)`, `pass_rate(of)[all_pass]` |
| `PassAtK(of, k)` / `PassHatK(of, k)` | examples; an example with fewer than `k` repetitions is n/a | failures | `pass@k(of)` / `pass^k(of)` |
| `Mean(of, reduce=mean, unit="example", higher_is_better=None)` | examples (or trials) | missing | `mean(of)` |
| `Pooled(of)` | examples (resampled); every weighted trial counts | missing | `pooled(of)`: `Σ value·weight / Σ weight` with `Score.weight` (1 when unset), bootstrap interval; `details.weight` is the total weight. Added by `default_metrics` for scores with weights |
| `Percentile(of, q, unit="trial")` | trials | missing | `p95(of)` |
| `ErrorRate()` | trials (clustered on the example) | what it measures | `error_rate` (`details.by_type`) |
| `Total(of)` | trials | missing | `total(of)` (no interval) |
| `Distribution(of)` | trials | missing | `dist(of)` |
| `Proportion(of, label)` | examples | missing | `share(of=label)` |
| `Consistency(of)` | examples with at least 2 scored repetitions | missing | `consistency(of)` |
| `CohenKappa`, `ClassRecall`, `ConfusionMatrix` | judge validation; see [judges.md](judges.md) | | `kappa(...)`, `recall(...)`, `confusion(...)` |
| `CorrectedPassRate(of, rates)` | examples | failures | `corrected_pass_rate(of)` |
| `WinRate(judge)` | examples (pairwise runs) | decided without the judge | `win_rate(judge)` |

- A threshold-less `PassRate` on a score with values other than 0 and 1 returns value
  `None` with `details["error"]`; the run does not fail.
- **Targets (`of`)** resolve in this order:
  1. a score name;
  2. a measurement recorded with `trial.record(name, value)` in the task (a score with
     the same name wins);
  3. a `Measure`:
     - `duration_s`: the task only, not scoring;
     - `cost_usd`: the task *plus* scorers and judges;
     - `input_tokens`, `output_tokens`, `total_tokens`: the task only;
     - `error`: 1 when the task errored.
- **Custom metric:** subclass `Metric`, set `name` and `unit`, and implement
  `compute(trials, *, clusters=None) -> MetricResult`.

## Repetitions and reducers

- Example-level metrics reduce each example's usable repetitions first, with `reduce`
  (default `mean`). So `n` counts examples, and repetitions only tighten intervals.
- Reducers: `from grasp_agents.evals.metrics import mean, median, max_, min_, all_pass,
  any_pass, at_least` (`at_least(k)`). A reducer other than `mean` appears in the name:
  `pass_rate(s)[all_pass]`.
- `PassHatK(s, 3)` is the unbiased chance that 3 fresh repetitions all pass.
  `PassRate(s, reduce=all_pass)` is the observed share of examples whose repetitions all
  passed.
- An example with only some usable repetitions still contributes; it is counted in
  `details["partial_examples"]`.

## Grouping and clustering

- **`group_by=["topic"]`** adds `result.groups["topic=<value>"]` from
  `example.metadata`. A missing key gives `topic=None`.
- **The `split` key is special.** Unless the metadata has a `split` key, it groups by the
  example's splits joined with `+` (`split=dev`, `split=dev+test`).
- **Small sealed groups.** Reports hide groups that hold 1 to 4 sealed examples. This
  check reads metadata keys only, so groups made by the `split` key are shown even when
  they are small.
- **`cluster_by="course"`** uses clustered standard errors (CR1) keyed by
  `metadata["course"]` (the example id when the key is missing). With fewer than 2
  clusters, there is no interval. Trial-level metrics always cluster repetitions on the
  example.

## Intervals

| Kind | Method | Notes |
|---|---|---|
| Shares (pass rates, error rate, recall, pass@k/pass^k, win rate) | Wilson | Exact for independent 0/1 units; on an effective sample size for fractional or clustered values. Still a real interval at 0% or 100% |
| `Mean` | Student's t | Collapses to `[v, v]` when every value is equal; no interval with `n=1` |
| `Percentile` | Cluster percentile bootstrap (seeded) | No interval unless at least 5 units lie beyond the quantile on each side: p95 needs about 100 examples, p50 about 10 |
| Cohen's κ | Wilson interval of agreement, mapped through chance agreement | `None` when every label and verdict has one value |

`confidence` defaults to 0.95 per metric.

## `compare`

`compare(base, candidate, *, targets=None, top=5, reduce=mean, confidence=0.95,
cluster_by=None)`:

1. **Pairing**: examples present in both runs with the same id *and* the same content
   hash, computed without the metadata keys either run declared descriptive. `n_changed_examples`, `n_base_only` and `n_candidate_only` count the rest.
2. **Default targets**: numeric scores (bool or number) found in *both* runs, then
   `error`, `duration_s`, and `cost_usd` when any trial has a cost. Label scores are
   never compared. From Python, pass `targets=[...]` to choose others. Numeric
   scores of the base run that the candidate lacks are listed in `missing_scores`.
   **Direction**: each score is read the way the runs recorded it in
   `EvaluationRun.directions` (higher is better unless a scorer's `lower_is_better`
   or a metric's `higher_is_better=False` says otherwise; a run without the record
   takes the other run's). A score the runs record opposite directions for is left
   out and listed in `conflicting_directions`.
3. **Per example**: repetitions are reduced with `reduce`. A score with weights
   (`Score.weight`) in either run is compared pooled instead: per run
   `Σ value·weight / Σ weight` over the paired examples, a paired bootstrap over
   examples (or clusters) for the interval and p-value (`test="pooled_bootstrap"`,
   `pooled=True`); `improved`/`regressed` compare per-example pooled values.
   - For scores that are always `bool`, a task-errored trial counts as 0.
   - Numeric scores skip errored trials, so `n_pairs` drops. The `error` target catches
     those crashes.
4. **Test**:
   - If every paired value is 0 or 1 and nothing is clustered: exact McNemar with
     Newcombe's interval.
   - Otherwise: a paired t-test. It is clustered when the *base* run had `cluster_by`
     or `cluster_by=` is given; the CLI has no flag for it.
   - If every difference is identical: the exact sign test.
   - With one pair or one cluster there is no test (`p_value=None`).
5. **Multiple testing**:
   - Holm adjustment within outcome targets (scores, `error`, recorded measures) and
     separately within resource targets (duration, cost, tokens).
   - Targets where no example changed are left out and are never significant.
   - `significant` means `p_adjusted < 1 - confidence`.
   - The Markdown table prints the unadjusted `p` and marks significance with ✱.
6. **Effect size**:
   - `diff` is candidate − base, with a CI.
   - `mde` is the smallest true difference this comparison detects with 80% power.
   - `improved`/`regressed` count examples moving in the good or bad direction (lower
     is better for `error`, the resources and lower-is-better scores).
   - `top_regressions` and `top_improvements` carry the worst repetition's explanation
     and never list sealed examples.
7. **Warnings** are raised when:
   - both sides are the same run;
   - the configuration is identical (the differences are noise);
   - nothing paired, or the runs differ in content or subsets;
   - a scorer changed version, config, fingerprint (prompt, model, declared identity)
     or code, or its code was recorded by another grasp-evals version;
   - a score of the base run is missing from the candidate, or the runs disagree on
     its direction;
   - repetitions differ;
   - task errors appear in only one run;
   - a run is invalid or incomplete;
   - all paired examples sit in one cluster.

## Subsets and groups

`compare(..., where={"source": "arxiv"})` (CLI `--where source=arxiv`, values parsed as
JSON when they parse) keeps the paired examples whose base-run metadata matches;
`group_by="source"` (`--group-by source`) adds one comparison per value under
`groups["source=<value>"]`, each with its own Holm family. A group holding one to four
sealed examples is left out. The regression gate reads only the overall comparison.

## Regression gate

`grasp_agents.evals.compare.regressions(comparison, targets=None, min_effect=0.0)` returns
failure messages (an empty list means pass). It fails when:

- nothing paired;
- a gated score is missing from the candidate (`missing_scores`) or the runs disagree
  on its direction (`conflicting_directions`);
- a gated target has no paired values, or has pairs but no p-value;
- a gated target shows a significant change in the bad direction with
  `|diff| >= min_effect`;
- a target named in `targets` is not in the comparison.

The default gated targets are the comparison's outcome targets plus the missing and
conflicting scores. With `targets=[...]` only the named ones are gated.
