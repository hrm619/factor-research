"""Statistical test implementations for the factor research framework."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.multitest import multipletests

from backend.research.models import TestResult, QualityScore


def binomial_test(successes: int, trials: int, null_value: float = 0.5) -> TestResult:
    """Two-sided binomial test against a null proportion."""
    result = scipy_stats.binomtest(successes, trials, null_value, alternative="two-sided")
    return TestResult(
        test_name="binomial",
        statistic=successes / trials if trials > 0 else 0.0,
        p_value=float(result.pvalue),
        significant=bool(result.pvalue < 0.05),
    )


def proportion_z_test(
    successes_a: int, n_a: int, successes_b: int, n_b: int
) -> TestResult:
    """Two-proportion z-test comparing two groups."""
    count = np.array([successes_a, successes_b])
    nobs = np.array([n_a, n_b])
    stat, pval = proportions_ztest(count, nobs)
    return TestResult(
        test_name="proportion_z",
        statistic=float(stat),
        p_value=float(pval),
        significant=bool(pval < 0.05),
    )


def chi_squared_test(contingency_table: list[list[int]]) -> TestResult:
    """Chi-squared test on a contingency table.

    Falls back to Fisher's exact test for 2x2 tables when any expected
    cell frequency is below 5.
    """
    table = np.array(contingency_table)
    chi2, p, dof, expected = scipy_stats.chi2_contingency(table)

    # Fisher exact fallback for 2x2 with small expected counts
    if table.shape == (2, 2) and np.any(expected < 5):
        _, p = scipy_stats.fisher_exact(table)
        return TestResult(
            test_name="fisher_exact",
            statistic=0.0,
            p_value=float(p),
            significant=p < 0.05,
        )

    return TestResult(
        test_name="chi_squared",
        statistic=float(chi2),
        p_value=float(p),
        significant=p < 0.05,
    )


def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    alpha = 1 - confidence
    lower, upper = proportion_confint(successes, trials, alpha=alpha, method="wilson")
    return (float(lower), float(upper))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for the difference between two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def fdr_correction(p_values: list[float], method: str = "fdr_bh") -> list[float]:
    """Apply FDR correction (Benjamini-Hochberg by default) to a list of p-values."""
    if not p_values:
        return []
    reject, corrected, _, _ = multipletests(p_values, method=method)
    return [float(p) for p in corrected]


def compute_quality_score(
    n: int,
    p_value: float | None,
    effect_size: float,
    consistency: float,
    n_min_bucket: int = 10,
) -> QualityScore:
    """Compute composite quality score per spec Section 6.5.

    Weights: sample_size 30%, significance 30%, effect_size 20%, consistency 20%.
    Max composite = 3.0. Grades: HIGH >= 2.5, MEDIUM >= 1.8, LOW >= 1.0, INSUFFICIENT_DATA < 1.0.
    """
    # INSUFFICIENT_DATA if any bucket < 10 observations
    if n < n_min_bucket:
        return QualityScore(
            sample_size_score=0.0,
            significance_score=0.0,
            effect_size_score=0.0,
            consistency_score=0.0,
            composite=0.0,
            grade="INSUFFICIENT_DATA",
        )

    # Sample size score (0-1): ramps from 0 at n=10 to 1 at n=200+
    if n >= 200:
        ss_score = 1.0
    elif n >= 100:
        ss_score = 0.7 + 0.3 * (n - 100) / 100
    elif n >= 50:
        ss_score = 0.4 + 0.3 * (n - 50) / 50
    elif n >= 30:
        ss_score = 0.2 + 0.2 * (n - 30) / 20
    else:
        ss_score = 0.1 * (n - 10) / 20

    # Significance score (0-1): based on p-value
    if p_value is None:
        sig_score = 0.0
    elif p_value < 0.01:
        sig_score = 1.0
    elif p_value < 0.05:
        sig_score = 0.7 + 0.3 * (0.05 - p_value) / 0.04
    elif p_value < 0.10:
        sig_score = 0.3 + 0.4 * (0.10 - p_value) / 0.05
    else:
        sig_score = max(0.0, 0.3 * (0.20 - p_value) / 0.10) if p_value < 0.20 else 0.0

    # Effect size score (0-1): based on absolute Cohen's h
    abs_h = abs(effect_size)
    if abs_h >= 0.5:
        eff_score = 1.0
    elif abs_h >= 0.2:
        eff_score = 0.5 + 0.5 * (abs_h - 0.2) / 0.3
    elif abs_h >= 0.1:
        eff_score = 0.2 + 0.3 * (abs_h - 0.1) / 0.1
    else:
        eff_score = 0.2 * abs_h / 0.1

    # Consistency score passed directly (0-1), computed externally from time windows
    cons_score = max(0.0, min(1.0, consistency))

    # Weighted composite (max 3.0)
    composite = 3.0 * (
        0.30 * ss_score
        + 0.30 * sig_score
        + 0.20 * eff_score
        + 0.20 * cons_score
    )

    # Grade
    if composite >= 2.5:
        grade = "HIGH"
    elif composite >= 1.8:
        grade = "MEDIUM"
    elif composite >= 1.0:
        grade = "LOW"
    else:
        grade = "INSUFFICIENT_DATA"

    return QualityScore(
        sample_size_score=round(ss_score, 3),
        significance_score=round(sig_score, 3),
        effect_size_score=round(eff_score, 3),
        consistency_score=round(cons_score, 3),
        composite=round(composite, 3),
        grade=grade,
    )
