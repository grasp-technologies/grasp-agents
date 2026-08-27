# Grasp Code Review

Review a change against the grasp-agents rules in `.claude/rules/`: one reviewer
per applicable rules file, then a verifier that drops anything it can't stand
up. Reviews the **working diff** by default; pass a PR number in `$ARGUMENTS`
(e.g. `/grasp-code-review 168`) for a pull request.

Invoked as `/grasp-code-review` — deliberately **not** `/code-review`, which is
a built-in that shadows project commands.

This is a review, not a fix: report findings, never edit code.

**Everything runs in one turn.** Dispatch subagents with the `Task` tool, all
in a **single message** so they run in parallel and complete before the turn
ends. Do **not** use the `Workflow` tool: it returns immediately and delivers
results via a later notification, which never arrives in a headless CI run —
the job then finishes green having posted nothing.

## 1. Scope the change

- **PR mode** (`$ARGUMENTS` has a number): the reviewable surface is exactly
  `gh pr diff <n> --name-only`. Get the full 40-char head SHA
  (`gh pr view <n> --json headRefOid`) — permalinks need it unabbreviated.
- **Working-diff mode**: `git diff --name-only $(git merge-base HEAD origin/master)...HEAD`.
  If empty, say so and stop.
- Skip entirely if the PR is closed, a draft, or an automated dependency bump.

Then establish **which round this is**: read the existing review comments.

```bash
gh api "repos/grasp-technologies/grasp-agents/pulls/<n>/comments" --jq '.[].body' | head -50
gh pr view <n> --comments
```

- **No prior Claude review → round 1.**
- **Prior review exists → later round.** Note the SHA it reviewed; the surface
  for this round is the commits added since
  (`git diff --name-only <that-sha>...<head>`).

## 2. Pick the reviewers

Choose only dimensions the diff actually contains code for. A reviewer with
nothing in scope invents findings to justify itself.

| Dimension | Rules file(s) | Applies to |
|---|---|---|
| `types_and_lint` | `.claude/rules/type_safety_and_lint.md` | any touched Python file |
| `tests` | `.claude/rules/testing_standards.md` | `tests/**`, or a behavior change with no test |
| `library_surface` | `.claude/rules/library_surface.md` | `src/grasp_agents/**/__init__.py`, `src/grasp_agents/llm_providers/**`, `pyproject.toml`, any new import of a third-party package |
| `style` | `.claude/rules/style.md`, `CONTRIBUTING.md` | any touched file |
| `bugs` | — | **always** |
| `history` | — | `git log`/`blame` on the modified lines |

Round 1: take every dimension that applies (that's the point — see §5).
Later rounds: `bugs` plus **at most one** rules dimension, and skip `history`.
A later round's surface is usually a single commit, so more reviewers mostly buy
duplicated context-reading.

## 3. Dispatch the reviewers — one message, parallel

One `Task` per dimension, all in the same message. Give each:

- the file list (**its only permitted surface**) and the diff command to re-run;
- for a rules dimension, the rules file paths, and this contract: **every
  finding must set `rule_source` to the file and `rule_quote` to the verbatim
  sentence violated.** If no sentence covers it specifically, the rule does not
  forbid it — drop the finding;
- read file contents **only from the checked-out working tree** — never from
  memory, another PR, or another repo;
- **read only what the finding needs**: the diff, plus the changed files and any
  caller you must see to judge them. Do not survey the wider codebase — each
  reviewer pays for its own context, and on a later round that duplication is
  most of the bill;
- **use `Read`, `Grep` and `Glob` for anything on disk.** CI grants a narrow
  allowlist, so shell reads (`cat`, `ls`, `head`, `tail`, `sed`, `find`) are
  denied and each attempt burns a turn before falling back. The only shell you
  may use is `git log`/`blame`/`diff`/`show`/`rev-parse`/`merge-base` and
  `gh pr diff`/`view`/`api`;
- the reporting bar in §4;
- return findings as a JSON array: `file`, `line`, `severity`, `summary`,
  `consequence`, `rule_source`, `rule_quote`, `evidence`. Empty array is a
  perfectly good answer — say nothing rather than pad.

Per-dimension focus lists are in `.claude/rules/` themselves; the highest-value
checks in this repo are: an `async def test_*` with no `pytest.mark.asyncio`
(strict asyncio mode means it never runs); a test that reaches a live provider
without `@pytest.mark.integration`; an eager top-level import of an optional
extra that reaches `import grasp_agents`; a parent-relative import
(`from ..x`); a name added to the root `__init__.py` without the lazy /
documented surface it needs; a public name renamed or removed.

## 4. The reporting bar

Severity is **`blocker` | `high` | `medium`**. There is deliberately no `nit`.
If a finding doesn't reach `medium`, it is not reported — not as a comment, not
in the summary.

A finding must name a **concrete consequence**: specific input or state, then
the wrong outcome. If you cannot, it isn't a finding.

**Drop anything that argues itself down.** If the honest phrasing needs a
clause like "only in the examples", "not on the mainline path", "unlikely in
practice", or "would be free" — the finding has failed its own test. This is
the single biggest source of review noise here.

Also never report:

- pre-existing issues, or anything on a line this diff didn't touch;
- what a linter, typechecker, formatter or the test suite would catch — CI and
  pre-commit run those; don't run or reason about build signal;
- anything under `src/grasp_agents/examples/`, `src/grasp_agents/kits/` or
  `legacy/` — excluded from lint, typecheck and the published wheel;
- abstract quality observations (coverage, generic security, docs) unless a
  rules file demands it;
- a rule violation explicitly silenced in the code with a coded `# noqa:` /
  `# type: ignore[...]` and a reason;
- changes plausibly intentional and part of the broader change;
- anything already raised or already resolved in an earlier round.

## 5. Round 1 is exhaustive; later rounds converge

**Round 1:** surface everything now. Tell the reviewers plainly there is no
later pass — a real problem held back is a problem shipped.

**Later rounds:** only `blocker` and `high`, and only in the commits added since
the last review. Do not re-scan the whole diff hunting for what earlier rounds
chose not to raise: if a `medium` wasn't worth reporting in round 1, it isn't
worth reporting in round 4. Convergence is the goal — a later round finding
nothing is a success, and should say so in one line.

## 6. Verify before posting — one Task

Dispatch a **single** verifier with the deduped findings. It adjudicates each
independently and must try to **refute** them. For each, it returns
`confidence` 0–100 plus `quote_verified` and `in_diff`.

It must:

1. open `rule_source` and confirm `rule_quote` appears **verbatim** and actually
   forbids that specific thing — a reviewer paraphrasing a rule into something
   stricter than written is the most common failure. Absent, altered, or
   not-covering ⇒ `quote_verified: false`;
2. confirm this diff introduced it (`git blame`) ⇒ else `in_diff: false`;
3. confirm something genuinely breaks — apply §4's self-refutation test again.

Drop any finding that is `quote_verified: false`, `in_diff: false`, scores
**below 80**, or comes back without a verdict. Default to a low score when
unsure: an unverifiable finding is 25, not 75.

## 7. Post

**Working-diff mode:** report via `ReportFindings`, most severe first. Empty
array when clean — say so plainly.

**PR mode, non-interactive** (you were handed a prompt and have no way to ask —
this is the CI case): post. A review generated and discarded is the failure this
replaced.
**PR mode, interactive:** show the findings and ask first.

**Never probe the environment to decide.** Do not run `env`, `echo $VAR`, or
anything with shell expansion to look for `GITHUB_ACTIONS` — the allowlist
denies expansions and `env`/`grep`, so it just burns turns. Whether you can ask
is something you already know.

Post **at most 6** inline comments, most severe first, via
`mcp__github_inline_comment__create_inline_comment`. If more survived, say how
many were held back in the summary — a capped review must never read as a clean
one.

Each comment: severity tag, then the defect and its consequence in **at most
two sentences**. Quote the rule where one applies. No preamble, no restating the
code, no hedging paragraph.

```
[Blocker] The new `async def test_streaming_cancel` has no asyncio marker, so
under `asyncio_mode = "strict"` it is skipped and the cancellation path ships
untested. `.claude/rules/testing_standards.md`: "<verbatim quote>".
```

Then **one** summary comment, grouped by severity as a checklist:

```markdown
### Code review

Round 2 · reviewers: bugs, tests · 2 findings

**Blocker**
- [ ] <one line> — `tests/llm/test_x.py:40`

**Medium**
- [ ] <one line> — `src/grasp_agents/llm/llm.py:12`

---
_Generated by [Claude Code](https://claude.ai/code)_
```

When nothing survives:

```markdown
### Code review

Round 3 · no new findings. Reviewed the 2 commits added since `<sha>` against
`.claude/rules/`.

---
_Generated by [Claude Code](https://claude.ai/code)_
```

If an inline comment can't be posted (path or line not in the diff), fold it
into the summary rather than losing it.

## Notes

- **Don't run tests, `pyright`, `ruff`, `cspell`, or any build.** CI and
  pre-commit cover them, and the reviewers are told to ignore anything they'd
  catch.
- Verify each finding's path appears verbatim in `gh pr diff --name-only`
  before posting; a path outside it means the finding is wrong, not the diff.
- Findings on untouched lines are dropped as pre-existing. Say so if a
  whole-file audit was expected.
