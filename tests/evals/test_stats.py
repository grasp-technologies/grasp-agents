import math

import pytest

from grasp_agents.evals.stats import (
    betainc,
    binomial_two_sided,
    bootstrap_estimate,
    cohens_kappa,
    corrected_prevalence,
    critical_value,
    mcnemar_exact,
    mean_estimate,
    minimum_detectable_effect,
    paired_difference,
    percentile,
    proportion_estimate,
    t_ppf,
    t_sf,
)


class TestDistributions:
    def test_betainc_symmetric_point(self) -> None:
        assert betainc(2.0, 2.0, 0.5) == pytest.approx(0.5)
        assert betainc(1.0, 1.0, 0.3) == pytest.approx(0.3)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [(1, 12.7062), (2, 4.3027), (5, 2.5706), (10, 2.2281), (30, 2.0423)],
    )
    def test_t_quantiles_match_tables(self, df: int, expected: float) -> None:
        assert t_ppf(0.975, df) == pytest.approx(expected, abs=1e-3)

    def test_t_sf_and_ppf_are_inverse(self) -> None:
        t = t_ppf(0.9, 7)
        assert t_sf(t, 7) == pytest.approx(0.1, abs=1e-8)
        assert t_sf(-t, 7) == pytest.approx(0.9, abs=1e-8)

    def test_normal_critical_value(self) -> None:
        assert critical_value(0.95) == pytest.approx(1.95996, abs=1e-4)


class TestEstimates:
    def test_mean_with_t_interval(self) -> None:
        est = mean_estimate([1.0, 2.0, 3.0, 4.0])
        assert est.value == pytest.approx(2.5)
        assert est.stderr == pytest.approx(math.sqrt(5 / 3 / 4))
        half = t_ppf(0.975, 3) * est.stderr
        assert est.ci_low == pytest.approx(2.5 - half)
        assert est.ci_high == pytest.approx(2.5 + half)

    def test_single_value_has_no_interval(self) -> None:
        est = mean_estimate([0.7])
        assert est.value == pytest.approx(0.7)
        assert est.stderr is None

    def test_empty_is_nan(self) -> None:
        assert math.isnan(mean_estimate([]).value)

    def test_clustered_se_exceeds_naive_for_correlated_items(self) -> None:
        values = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        clusters = ["a", "a", "a", "b", "b", "b", "c", "c", "d", "d"]
        naive = mean_estimate(values)
        clustered = mean_estimate(values, clusters=clusters)
        assert clustered.value == naive.value
        assert clustered.stderr is not None
        assert naive.stderr is not None
        assert clustered.stderr > naive.stderr

    def test_wilson_interval(self) -> None:
        est = proportion_estimate(8, 10)
        assert est.value == pytest.approx(0.8)
        assert est.ci_low == pytest.approx(0.4902, abs=1e-3)
        assert est.ci_high == pytest.approx(0.9433, abs=1e-3)
        all_pass = proportion_estimate(10, 10)
        assert all_pass.ci_high == pytest.approx(1.0)
        assert all_pass.ci_low is not None
        assert all_pass.ci_low < 1.0

    def test_percentile_interpolates(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
        assert percentile([5.0], 95) == pytest.approx(5.0)
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 100) == pytest.approx(5.0)

    def test_bootstrap_is_seeded(self) -> None:
        values = [float(v) for v in range(50)]
        first = bootstrap_estimate(values, lambda v: percentile(v, 90), seed=3)
        second = bootstrap_estimate(values, lambda v: percentile(v, 90), seed=3)
        assert first == second
        assert first.ci_low is not None
        assert first.ci_high is not None
        assert first.ci_low <= first.value <= first.ci_high


class TestPaired:
    def test_binary_uses_mcnemar(self) -> None:
        base = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        candidate = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        est = paired_difference(base, candidate)
        assert est.test == "mcnemar"
        assert est.diff == pytest.approx(5 / 8)
        assert est.p_value == pytest.approx(mcnemar_exact(0, 5))
        assert est.mde is not None
        assert est.mde > 0

    def test_continuous_uses_paired_t(self) -> None:
        base = [0.5, 0.6, 0.7, 0.4, 0.5]
        candidate = [0.6, 0.7, 0.8, 0.5, 0.6]
        est = paired_difference(base, candidate)
        assert est.test == "paired_t"
        assert est.diff == pytest.approx(0.1)
        assert est.p_value == pytest.approx(0.0)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="paired"):
            paired_difference([1.0], [1.0, 0.0])

    def test_binomial_two_sided(self) -> None:
        assert binomial_two_sided(5, 10) == pytest.approx(1.0)
        assert binomial_two_sided(0, 5) == pytest.approx(2 / 32)
        assert binomial_two_sided(0, 0) == pytest.approx(1.0)

    def test_mde_rule_of_thumb(self) -> None:
        assert minimum_detectable_effect(1.0) == pytest.approx(2.8, abs=0.01)


class TestAgreement:
    def test_kappa(self) -> None:
        assert cohens_kappa([("a", "a"), ("b", "b")]) == pytest.approx(1.0)
        assert cohens_kappa([("a", "b"), ("b", "a")]) == pytest.approx(-1.0)
        assert cohens_kappa([]) is None

    def test_corrected_prevalence(self) -> None:
        assert corrected_prevalence(0.6, 1.0, 1.0) == pytest.approx(0.6)
        assert corrected_prevalence(0.5, 0.9, 0.8) == pytest.approx(0.3 / 0.7)
        assert corrected_prevalence(0.5, 0.5, 0.5) is None
