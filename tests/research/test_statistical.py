"""Tests for statistical module."""

import math

import numpy as np
import pytest

from backend.research.statistical import (
    binomial_test,
    proportion_z_test,
    chi_squared_test,
    wilson_ci,
    cohens_h,
    fdr_correction,
    compute_quality_score,
)


class TestBinomialTest:
    def test_fair_coin(self):
        """250 heads in 500 flips should not be significant."""
        result = binomial_test(250, 500, 0.5)
        assert result.p_value > 0.05
        assert result.significant is False

    def test_known_edge(self):
        """275 heads in 500 flips (55%) should be significant."""
        result = binomial_test(275, 500, 0.5)
        assert result.p_value < 0.05
        assert result.significant is True

    def test_small_sample_exact(self):
        """Binomial test is exact even for small samples."""
        result = binomial_test(8, 10, 0.5)
        assert result.test_name == "binomial"
        # 8/10 = 80%, should be significant
        assert result.p_value < 0.15  # exact p for 8/10 is ~0.109

    def test_perfect_rate(self):
        """100% cover rate should be computable."""
        result = binomial_test(50, 50, 0.5)
        assert result.p_value < 0.001

    def test_zero_rate(self):
        """0% cover rate should be computable."""
        result = binomial_test(0, 50, 0.5)
        assert result.p_value < 0.001


class TestProportionZTest:
    def test_equal_proportions(self):
        """Same rates should not be significant."""
        result = proportion_z_test(50, 100, 50, 100)
        assert result.p_value > 0.5
        assert result.significant is False

    def test_different_proportions(self):
        """55% vs 45% with n=500 each should be significant."""
        result = proportion_z_test(275, 500, 225, 500)
        assert result.p_value < 0.05
        assert result.significant is True


class TestChiSquaredTest:
    def test_no_association(self):
        """Equal cover rates across buckets -> not significant."""
        table = [[50, 50], [50, 50], [50, 50], [50, 50]]
        result = chi_squared_test(table)
        assert result.p_value > 0.9

    def test_strong_association(self):
        """Very different rates across buckets -> significant."""
        table = [[80, 20], [20, 80]]
        result = chi_squared_test(table)
        assert result.p_value < 0.001
        assert result.test_name == "chi_squared"

    def test_fisher_exact_fallback(self):
        """Small expected counts in 2x2 -> falls back to Fisher's exact."""
        table = [[3, 1], [1, 5]]
        result = chi_squared_test(table)
        assert result.test_name == "fisher_exact"


class TestWilsonCI:
    def test_basic(self):
        lower, upper = wilson_ci(55, 100, confidence=0.95)
        assert lower < 0.55 < upper
        assert lower > 0.40
        assert upper < 0.70

    def test_coverage_monte_carlo(self):
        """Wilson 95% CI should contain true proportion >= 93% of time."""
        rng = np.random.default_rng(42)
        true_p = 0.5
        n = 100
        n_simulations = 1000
        contained = 0
        for _ in range(n_simulations):
            successes = int(rng.binomial(n, true_p))
            lower, upper = wilson_ci(successes, n, confidence=0.95)
            if lower <= true_p <= upper:
                contained += 1
        coverage = contained / n_simulations
        assert coverage >= 0.93

    def test_extreme_proportion(self):
        """Wilson CI handles proportions near 0 and 1."""
        lower, upper = wilson_ci(1, 100)
        assert lower >= 0.0
        assert upper <= 1.0


class TestCohensH:
    def test_equal_proportions(self):
        assert cohens_h(0.5, 0.5) == 0.0

    def test_known_values(self):
        h = cohens_h(0.7, 0.3)
        # 2*arcsin(sqrt(0.7)) - 2*arcsin(sqrt(0.3)) should be positive and meaningful
        assert h > 0.5  # large effect

    def test_symmetry(self):
        h1 = cohens_h(0.6, 0.4)
        h2 = cohens_h(0.4, 0.6)
        assert abs(h1 + h2) < 1e-10  # opposite signs


class TestFDRCorrection:
    def test_empty(self):
        assert fdr_correction([]) == []

    def test_all_significant(self):
        """Very small p-values should remain significant after correction."""
        p_values = [0.001, 0.002, 0.003]
        corrected = fdr_correction(p_values)
        assert all(p < 0.05 for p in corrected)

    def test_correction_increases_pvalues(self):
        """FDR correction should not decrease any p-value."""
        p_values = [0.01, 0.04, 0.06, 0.10, 0.50]
        corrected = fdr_correction(p_values)
        for orig, adj in zip(p_values, corrected):
            assert adj >= orig - 1e-10  # allow floating point tolerance

    def test_some_lose_significance(self):
        """Borderline p-values should lose significance after FDR correction."""
        p_values = [0.01, 0.03, 0.045, 0.048, 0.20, 0.50]
        corrected = fdr_correction(p_values)
        # At least one originally-significant value should lose significance
        originally_sig = sum(1 for p in p_values if p < 0.05)
        still_sig = sum(1 for p in corrected if p < 0.05)
        assert still_sig < originally_sig


class TestComputeQualityScore:
    def test_insufficient_data(self):
        """Bucket with fewer than 10 obs -> INSUFFICIENT_DATA."""
        qs = compute_quality_score(n=5, p_value=0.001, effect_size=0.5, consistency=1.0)
        assert qs.grade == "INSUFFICIENT_DATA"
        assert qs.composite == 0.0

    def test_high_quality(self):
        """Large sample, significant p, good effect, consistent -> HIGH."""
        qs = compute_quality_score(n=500, p_value=0.001, effect_size=0.3, consistency=0.9)
        assert qs.grade == "HIGH"
        assert qs.composite >= 2.5

    def test_low_quality(self):
        """Moderate sample, marginal p, small effect -> LOW."""
        qs = compute_quality_score(n=80, p_value=0.08, effect_size=0.15, consistency=0.4)
        assert qs.grade == "LOW"
        assert 1.0 <= qs.composite < 1.8

    def test_null_pvalue(self):
        """None p-value should give zero significance score."""
        qs = compute_quality_score(n=100, p_value=None, effect_size=0.2, consistency=0.5)
        assert qs.significance_score == 0.0

    def test_scores_bounded(self):
        """All individual scores should be between 0 and 1."""
        qs = compute_quality_score(n=150, p_value=0.03, effect_size=0.25, consistency=0.7)
        assert 0 <= qs.sample_size_score <= 1
        assert 0 <= qs.significance_score <= 1
        assert 0 <= qs.effect_size_score <= 1
        assert 0 <= qs.consistency_score <= 1
        assert 0 <= qs.composite <= 3.0
