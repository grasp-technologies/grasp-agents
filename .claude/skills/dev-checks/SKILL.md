---
name: dev-checks
description: Run grasp-agents' lint, format, type-check and test commands correctly before handing work back — scoped ruff (the config autofixes), pyright strict, the default vs integration test suites, cspell, checking real outputs instead of trusting green tests, and making sure docs/, skills, README, CLAUDE.md, examples and model-facing text still match the change. Use after editing code in this repo, before committing, or when a check fails unexpectedly.
---

# Dev checks

Run these on every change before you report it done. Judge results by the files you
touched: the tree carries some pre-existing preview-rule lint noise that isn't yours
to chase.

## 1. Lint and format: only the files you changed

`ruff.toml` sets `fix = true` with `select = ["ALL"]` and preview rules, so a plain
`ruff check` **rewrites files**. Scope it to the files you touched:

```bash
uv run ruff check src/grasp_agents/agent/foo.py tests/agent/test_foo.py   # autofix is fine here
uv run ruff format src/grasp_agents/agent/foo.py tests/agent/test_foo.py
uv run ruff check --no-fix src tests                                      # broad, read-only
```

- Never run a fixing `ruff check` or `ruff format` on directories. It would rewrite
  untouched files and the example notebooks.
- `F401` (unused import) is unfixable by config, so remove the import by hand.
- To suppress a rule, use `# ruff: ignore[<rule-name>]` (for example
  `# ruff: ignore[import-outside-top-level]`), not a bare `# noqa`.
- Tolerated in tests (don't chase): private-member access on objects under
  test, untyped lambdas and mocks, `TYPE_CHECKING`-only import suggestions.

## 2. Types

```bash
uv run pyright src/grasp_agents          # strict; 0 errors expected
uv run pyright tests/agent/test_foo.py   # tests are included in pyrightconfig.json
```

- New code must be fully typed. Use PEP 695 syntax (`class Foo[T]`, `def f[T]`,
  `type X = ...`).
- A `type X = ...` alias is a `TypeAliasType` at runtime. `isinstance`, `except` and
  `get_args` need the real class or `X.__value__`.
- `AutoInstanceAttributesMixin` resolves generic arguments from the nearest base
  that declares `_generic_arg_to_instance_attr_map` in its own `__dict__`. A new
  intermediate generic subclass of `Processor`/`LLMAgent` must redeclare it, or the
  mapped attributes silently resolve to `object`.
- Editor diagnostics lag behind edits. Before believing an error, confirm it with
  a pyright run.

## 3. Tests

```bash
uv run pytest tests/agent/test_foo.py -q          # the file you changed
uv run pytest -q                                   # full default suite (integration deselected)
uv run pytest -m integration tests/<path> --tb=short   # live calls; needs API keys
```

- Run integration tests in the foreground and scope them by file. Don't pipe
  pytest through `tail`, which truncates tracebacks. Use `--tb=short` or `--tb=line`
  instead.
- See the `writing-tests` skill for layout, helpers and model choices.

## 4. Spelling and hooks

Pre-commit runs cspell (US spelling) on changed files and on the commit message. Add
genuine project words to `cspell.json` rather than misspelling around them.

## 5. Look at the output, not just the exit code

Green tests routinely pass on wrong output:
- **Console/UI changes.** Render real events through `EventConsole` into a truecolor
  `rich.Console(record=True)` and read the result. Strip ANSI to check text and
  layout; keep it to check colors.
- **Agent behavior.** Read the agent's actual tool calls (`ToolCallItemEvent`
  arguments) and its final answer, and confirm it did the intended thing.
- **Notebooks.** Read the executed cells' outputs.

Then encode what you saw as assertions.

## 6. Documentation matches the change

CLAUDE.md, "Keep documentation in sync", lists every text that can describe what
you changed: `docs/`, `.claude/skills/`, `README.md`, `CLAUDE.md`, examples,
model-facing strings and prompt files, and `CONTRIBUTING.md`.

1. **Search for what you renamed or removed.** For each public symbol, CLI flag, env
   var, default value or file path, search every place it could be described:

   ```bash
   grep -rn -e 'old_name' docs .claude/skills README.md CLAUDE.md CONTRIBUTING.md src/grasp_agents/examples
   grep -rn -e 'old_name' src/grasp_agents --include='*.py' --include='*.md'   # docstrings, tool descriptions, prompt files
   ```

   Fix every hit in the same change.
2. **Reread when behavior changed without a rename.** Read the matching `docs/`
   page's Contracts and Limitations, and any skill pitfall or workflow that
   covers the behavior. Update whatever now reads wrong.
3. **Check the examples.** Code examples in docs and skills must still run against
   the new signatures.

Write only the current behavior: no "previously" or "now" (see the
`public-writing` skill).

## 7. Never

- `git stash` / `git reset --hard` to test something. Work on the tree as it is.
- Blanket reformatting, or "fixing" lint in files you didn't otherwise touch.
- Wrapping a failing check in try/except or `# type: ignore` without a reason in
  the same line.
