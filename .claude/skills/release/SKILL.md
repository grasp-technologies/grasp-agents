---
name: release
description: How a grasp-agents version is cut and published — bumping the version in pyproject.toml, the checks to run first, tagging vX.Y.Z (PyPI trusted publishing with an allowed-pusher gate), the separate legacy-vX.Y.Z flow for the frozen grasp_agents_legacy distribution, and writing release notes. Use when asked to bump the version, prepare or publish a release, or check whether master is released.
---

# Releasing

## Versioning

- The version lives only in `pyproject.toml` (`[project] version`); there is no
  `__version__`.
- Bump the version in the PR that should ship. Patch releases are the norm; a
  breaking public-API change gets at least a minor bump, and the release notes say
  how to migrate.
- A change to the persisted checkpoint shape always ships with a
  `CURRENT_SCHEMA_VERSION` bump (see the `durability-changes` skill). Call it out in
  the notes.

## Before tagging

1. The branch is merged to `master`, and you are on an up-to-date `master`.
2. Run the checks from the `dev-checks` skill: `uv run pytest -q`,
   `uv run pyright src/grasp_agents`, and `uv run ruff check --no-fix src`. There is
   no CI for these, so a broken master is only caught here.
3. Run the relevant live suites (`-m integration`) for anything touching providers,
   the sandbox or durability.
4. Check that the version isn't already released: `git tag --list 'v*'` should not
   already contain it. A version can sit on master untagged, in which case the
   next tag should include it.
5. `uv build` succeeds and `uv run --with dist/*.whl --no-project -- python -c "import grasp_agents"` works.

## Publish

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The tag triggers `.github/workflows/workflow.yml`. It checks that the tag pusher is
on the allowed list, builds, runs an import smoke test, and publishes with PyPI
trusted publishing. An unauthorized pusher's tag is deleted automatically. Tagging is
an outward-facing action: do it only when the user explicitly asks.

## Legacy distribution

`legacy/` holds the frozen `grasp_agents_legacy` package for consumers still on the
pre-1.0 API. It versions independently (`legacy/pyproject.toml`) and publishes on
`legacy-vX.Y.Z` tags through `.github/workflows/publish-legacy.yml`. Change it only
for fixes those consumers need. Never port new features into it.

## Release notes

Write them for library users. Cover what changed, what to change in your code, and any
schema-version bump. They are public, so follow the `public-writing` skill: no
internal project names, tickets, roadmap or phase labels.
