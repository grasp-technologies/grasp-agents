## Contributing to Grasp Agents

Grasp Agents is an open-source project, and we welcome contributions from the community. Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated!

To develop and test the library locally, follow these steps:

### 1. Install UV Package Manager

Make sure [UV](https://github.com/astral-sh/uv) is installed on your system:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

Create a new virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

Install pre-commit hooks:

```bash
pre-commit install --install-hooks
```

### 3. Test Example for VS Code

- Install the [Jupyter Notebook extension](https://marketplace.visualstudio.com/items/?itemName=ms-toolsai.jupyter).

- Ensure you have a `.env` file with your OpenAI and Google AI Studio API keys set (see [.env.example](.env.example)).

```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

- Open [src/grasp_agents/examples/notebooks/basics.ipynb](https://github.com/grasp-technologies/grasp-agents/tree/master/src/grasp_agents/examples/notebooks/basics.ipynb).

You're now ready to run and experiment with the example notebook.

### 4. Recommended VS Code Extensions

- [Ruff](https://marketplace.visualstudio.com/items/?itemName=charliermarsh.ruff) -- for formatting and linting
- [Pylance](https://marketplace.visualstudio.com/items/?itemName=ms-python.vscode-pylance) -- for type checking (the project uses pyright in strict mode)

### 5. Checks before opening a pull request

```bash
uv run ruff check <changed files>      # lint; ruff.toml autofixes, so scope it to your files
uv run ruff format <changed files>     # format only what you touched
uv run pyright src/grasp_agents        # strict type check
uv run pytest                          # unit tests; live tests are marked `integration`
uv run pytest -m integration tests/<path>   # live API tests (need API keys in .env)
```

New tests go in the `tests/` subdirectory that mirrors the package they cover.
Shared helpers (a scripted `MockLLM`, response builders, sample tools) are in
`tests/_helpers.py`.

### 6. Design reference

Each subsystem has a reference page in [docs/](docs/README.md) describing its
current design and the contracts changes must preserve. Read the relevant page
before changing a subsystem. Update the page in the same pull request.

Also update any text that describes what you changed:
- the README;
- the skills in `.claude/skills/`;
- `CLAUDE.md`;
- the examples;
- tool descriptions and prompt files.

[CLAUDE.md](CLAUDE.md) ("Keep documentation in sync") lists what to check, and
summarizes the code conventions.

## Releasing a New Version (Maintainers Only)

To release a new version of the package, follow these steps:

1. Create a new branch for the release: `git checkout -b release-X.Y.Z`.

2. Update the version in `pyproject.toml`.

3. Commit the changes with a message like "Bump version to X.Y.Z". `git commit -m "Bump version to X.Y.Z"`.

4. Make a pull request to the `master` branch.

5. Once the pull request is approved and merged, checkout the `master` branch: `git checkout master`.

6. Pull the latest changes: `git pull origin master`.

7. Tag the release: `git tag vX.Y.Z`. Note that a tag name should be in the format `vX.Y.Z`.

8. Push the tag to the remote repository: `git push --tags`.

9. This will trigger the [release workflow](.github/workflows/workflow.yml), which will build and publish the package to PyPI.

The frozen `grasp_agents_legacy` distribution in `legacy/` is versioned separately and published by pushing a `legacy-vX.Y.Z` tag ([workflow](.github/workflows/publish-legacy.yml)).
