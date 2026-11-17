---
name: public-writing
description: Rules for any text that ships in or from this open-source repo — code comments, docstrings, docs/ pages, skills, PR descriptions, commit messages and release notes. Covers audience (framework users, not implementers), no design history or plans, no private project/ticket/people names, no references to other agent frameworks, tool docstring vs agent-facing description, and a pre-publish check. Use when writing or editing comments, docs, PR descriptions or release notes.
---

# Public writing

grasp-agents is open source. Everything under version control, plus every PR
description, commit message and release note, is public.

## Audience and content

- **Write for framework users.** Comments and docstrings default to none. Add one
  when the *why* is non-obvious or there is a contract a caller must honor. Don't
  restate what the types already say.
- **Describe the current design only.** Leave out:
  - "renamed from", "previously", "used to", or how we got here;
  - plans, roadmap items, phase or milestone labels, "TODO: later", "coming soon".

  Missing features simply aren't mentioned, or appear as a current limitation.
- **No private context:**
  - no names of private projects or downstream consumers, internal hosts, URLs or
    paths (`/Users/...`);
  - no ticket ids or people;
  - no notes from planning documents or assistant memory.
- **No other agent frameworks.** Don't compare with, credit or allude to other agent
  frameworks or coding agents in code, docs, tests or PRs. The one exception is code
  copied verbatim, which gets the attribution its license requires. Vendor APIs and
  dependencies we integrate with (OpenAI, Anthropic, Gemini, LiteLLM, E2B, MCP,
  OpenTelemetry, Phoenix, Seatbelt, srt) are fine to name.
- **Tool text has two audiences.** A `BaseTool` class docstring is for developers
  (third person). The model sees only `description=`, `Field(description=...)` and
  the returned strings (second person is right there). `@function_tool` is the
  exception: its docstring *is* the agent-facing description.
- **Docs pages** follow the shape in `docs/README.md`: purpose, key types, how it
  works, **Contracts**, **Limitations**. Reference code as `module/path.py` →
  `Symbol`, never by line number. Update the page in the same change as the code.

## PR descriptions and commit messages

- Commit messages are short and imperative: "fix rollback of parked agents".
- PR descriptions use the template sections: what it does, why, how to test. Explain
  the behaviour change for users, then what changed internally, in that order. The
  same rules apply: no internal names, ticket ids, roadmap or phase labels, links to
  private documents, or "as discussed".
- Note any persisted-schema bump and any breaking API change explicitly.

## Pre-publish check

Before committing docs or posting a PR or release note, scan the text, and the
diff for comments:

```bash
git diff --cached -U0 | grep -nEi 'phase [a-h][0-9]?\b|roadmap|todo:? later|previously|renamed from|ticket|linear|/Users/|memory note' || true
```

Also scan for the names of any private downstream projects you know of, and for
other agent frameworks' names. Every hit must be either legitimate (for example
`previously` describing runtime state) or rewritten.

You may fix existing comments, docstrings or docs that break these rules without
asking.
