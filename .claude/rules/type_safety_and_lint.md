# Type Safety & Lint — Boy Scout Rule

`pyright` runs in strict mode over `src/grasp_agents` and `tests`
([`pyrightconfig.json`](../../pyrightconfig.json)), and `ruff` lints with
`select = ["ALL"]` under `preview` ([`ruff.toml`](../../ruff.toml)). The rule is
simple: **leave every file you touch with zero pyright errors and zero ruff
errors**, and run both checks together. You are not required to fix pre-existing
errors in files you didn't touch, but you must not add new ones.

## Required checks after touching Python files

```bash
uv run pyright <file1> <file2> ...
uv run ruff check --fix <file1> <file2> ...
uv run ruff format <file1> <file2> ...
```

`ruff.toml` sets `fix = true`, so **a bare `ruff check` with no paths runs over
the whole repo and auto-fixes unrelated files** — always pass explicit paths.

## Suppressions

Suppression is allowed, but it must be narrow and explained:

- **Every `# noqa` carries its rule code** (`# noqa: PLC2701`) — a bare `# noqa`
  silences everything on the line and is never acceptable.
- **A suppression that applies to a whole class of files belongs in
  `per-file-ignores` in `ruff.toml`, with a comment saying why** — not repeated
  inline across dozens of call sites.
- A `# type: ignore` narrows to the specific error
  (`# type: ignore[import]`) and stands next to the reason it exists, usually a
  third-party SDK that ships no or wrong stubs.
- Widening `ignore` in `ruff.toml`, or relaxing a `reportX` setting in
  `pyrightconfig.json`, is a repo-wide policy change: it needs its own PR
  discussion, not a line in a feature PR.

## Excluded from both tools

`src/grasp_agents/examples` and `src/grasp_agents/kits` are excluded from both
`ruff` and `pyright`; `scripts` is excluded from `pyright` only; `legacy/` is a
frozen distribution with its own `pyproject.toml`. **Code under those paths is
not held to library standards, and nothing under `src/grasp_agents/` that ships
may import from them.**

`tests` runs strict pyright with a relaxed execution environment (unknown types,
private usage, and argument-type reporting are off) — test code is still
typechecked, just not held to the library's inference bar.
