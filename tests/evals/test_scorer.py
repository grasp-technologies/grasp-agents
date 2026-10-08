import pytest

from grasp_agents.evals import Score, ScoreReason
from grasp_agents.evals.scorer import normalize_scores


def test_mapping_values_are_named_by_their_keys() -> None:
    scores = normalize_scores(
        "judge", {"accuracy": Score(name="draft", value=True), "tone": "warm"}
    )
    assert [s.name for s in scores] == ["accuracy", "tone"]
    assert all(s.scorer == "judge" for s in scores)


def test_non_finite_values_are_unscored() -> None:
    (from_scalar,) = normalize_scores("j", float("nan"))
    (from_score,) = normalize_scores("j", Score(name="q", value=float("inf")))
    for score in (from_scalar, from_score):
        assert score.value is None
        assert score.reason == ScoreReason.NON_FINITE_VALUE


def test_one_output_cannot_repeat_a_score_name() -> None:
    with pytest.raises(ValueError, match="several scores named"):
        normalize_scores("j", [Score(name="q", value=1.0), Score(name="q", value=0.5)])


def test_none_means_not_applicable() -> None:
    assert normalize_scores("j", None) == []
