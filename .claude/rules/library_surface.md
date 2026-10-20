# Library Surface — Public API, Optional Extras, Imports

`grasp_agents` is published to PyPI, so an import that works in this checkout
but not in a user's install is a shipped bug, and a renamed export breaks code
we cannot see.

## The public surface

`src/grasp_agents/__init__.py` is the headline surface — what you need to build,
run and stream an agent. Lower-tier helpers stay in their subpackages and are
imported from there (`from grasp_agents.llm_providers import OpenAIResponsesLLM`).

- **Adding a name to the root `__init__.py` makes it public API.** Do it only
  for the headline surface, and keep the grouped section comments and the
  module docstring's example list accurate.
- **Renaming or removing a name exported from `__init__.py`, or changing the
  signature of one, is a breaking change for installed users.** It must be
  deliberate and called out in the PR description — not a drive-by rename.

## Optional extras must stay lazy

Optional dependencies are declared as extras in `pyproject.toml` (`anthropic`,
`gemini`, `bedrock`, `vertex`, `mcp`, `e2b`, `tui`, `phoenix`,
`code-exec`, `notebook-edit`). A base `pip install grasp-agents` has none of
them.

**`import grasp_agents` must not import any optional dependency.** The lazy
boundary is the package `__init__.py`: `llm_providers/__init__.py` re-exports
provider classes through `_SUBMODULE_BY_NAME` plus a module `__getattr__`, with
the real imports under `if TYPE_CHECKING:` so type checkers still see them.
Modules *inside* a provider subpackage import their SDK directly — that
subpackage is only loaded when the class is accessed.

**A new provider or optional-extra feature adds a lazy entry and a
`TYPE_CHECKING` import — never a top-level import in a package that a base
install loads.**

Adding a package to `[project.dependencies]` widens every install: **a new
third-party runtime dependency belongs in an optional extra unless the core
agent loop genuinely needs it.**

## Imports

Convention, enforced by `ban-relative-imports = "parents"` in `ruff.toml`:
**same-package imports use `from .sibling`; anything reaching into another
subpackage uses the absolute `from grasp_agents.x` form.** The parent-relative
`from ..x` / `from ...x` forms are banned because they blur that boundary.

## What ships

The wheel and sdist package `src/grasp_agents` and exclude
`src/grasp_agents/examples` and `*.ipynb`. **Shipped code must not import from
`examples`, `kits`, `scripts`, `tests` or `legacy/`** — none of them are in the
distribution.

## Versioning

Releases are their own PR: bump `version` in `pyproject.toml`, merge, then tag
`vX.Y.Z` to trigger the publish workflow (see
[`CONTRIBUTING.md`](../../CONTRIBUTING.md)). **A feature or fix PR does not
touch the version.**
