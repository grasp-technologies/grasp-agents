import pytest


@pytest.fixture(autouse=True)
def _fixed_git_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # Runs record the git state of the working directory, i.e. this checkout:
    # an edit saved mid-test would make resume checks see changed code.
    monkeypatch.setattr(
        "grasp_agents.evals._execution.git_state",
        lambda cwd=None: ("0" * 40, "test", False, None),
    )
