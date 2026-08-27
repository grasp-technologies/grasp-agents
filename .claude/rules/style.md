# Style

## Comments and docstrings

- **Reserve comments for the WHY of non-obvious behavior**: hidden constraints,
  invariants, workarounds, a false positive being suppressed. Default to none.
- **Don't write comments that justify organizational decisions** ("extracted
  from Y to avoid a cycle", "lives here so X can import it") or that narrate
  design history. Rationale belongs in the PR description.
- **No ticket ids, PR references or "introduced for X" provenance in code
  comments or docstrings** — that belongs in PR bodies and commit messages.
- Don't restate what the type annotation already says.
- Missing-docstring lint is off, so a docstring is optional — but where one
  exists it describes current behavior, not plans.
- You are free to edit existing comments and docstrings that violate these
  rules without asking.

## Spelling

`cspell.json` sets `language: "en"`, which cspell reads as US English, so
**a British spelling is a real finding: use `normalize`, `initialization`,
`parametrized`, not the `-ise` forms.** The `cspell` pre-commit hook checks
changed files and the commit message.

Add a word to `cspell.json` only for genuine vocabulary — product names, API
literals, jargon. Note `allowCompoundWords` is on, so one added base form
silently licenses every suffix of it.

## Formatting

`ruff format` owns formatting; line length is 88 and line endings are `lf`.
**Don't report formatting in review** — the formatter and the pre-commit hooks
settle it.

## Repo hygiene

- **Don't commit ad-hoc design, spec or planning documents into the repo.**
- Don't commit generated artifacts by hand: `uv.lock` and `requirements.txt`
  are refreshed and staged by the pre-commit hooks.
- No secrets, real API keys or `.env` contents in the repo — `.env.example`
  documents the variable names.
