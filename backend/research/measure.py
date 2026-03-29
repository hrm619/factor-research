"""Stage 4: Measure outcomes and statistical significance."""

from __future__ import annotations

import logging
from datetime import datetime, UTC

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from backend.research.models import (
    HypothesisDefinition,
    BucketStats,
    ComparisonResult,
    QualityScore,
    MeasurementResult,
)
from backend.research.statistical import (
    binomial_test,
    proportion_z_test,
    chi_squared_test,
    wilson_ci,
    cohens_h,
    fdr_correction,
    compute_quality_score,
)

logger = logging.getLogger(__name__)


def measure(df: pd.DataFrame, definition: HypothesisDefinition) -> MeasurementResult:
    """Compute outcome statistics for each bucket and overall hypothesis."""
    outcome_col = _get_outcome_col(definition.outcome)
    margin_col = "margin_vs_spread" if definition.outcome == "ats" else None

    # Per-bucket stats
    buckets = []
    for label in sorted(df["bucket"].unique()):
        bucket_df = df[df["bucket"] == label]
        bs = compute_bucket_stats(bucket_df, label, outcome_col, margin_col)
        bs = run_statistical_test(bs, definition.statistical_test)
        buckets.append(bs)

    # FDR correction across buckets (Amendment A3)
    buckets = apply_fdr_correction(buckets)

    # Cross-bucket comparison
    comparison = None
    if len(definition.comparison_buckets) == 2:
        a_label, b_label = definition.comparison_buckets
        a_stats = next((b for b in buckets if b.bucket_label == a_label), None)
        b_stats = next((b for b in buckets if b.bucket_label == b_label), None)
        if a_stats and b_stats:
            comparison = compare_buckets(a_stats, b_stats)

    # Time-windowed analysis
    time_window_results = []
    if definition.time_windows:
        time_window_results = measure_time_windows(df, definition)

    # Quality score
    min_bucket_n = min(b.n for b in buckets) if buckets else 0
    comparison_effect = comparison.effect_size_h if comparison else 0.0
    comparison_p = comparison.p_value if comparison else (buckets[0].p_value_adjusted if buckets else None)
    consistency = _compute_consistency(time_window_results) if time_window_results else 0.5
    quality = compute_quality_score(
        n=min_bucket_n,
        p_value=comparison_p,
        effect_size=comparison_effect,
        consistency=consistency,
        n_min_bucket=10,
    )

    # Trend
    trend = _compute_trend(time_window_results) if time_window_results else None

    # Dataset summary
    dataset_summary = {
        "total_rows": len(df),
        "seasons": sorted(df["season"].unique().tolist()),
        "buckets": {b.bucket_label: b.n for b in buckets},
    }

    return MeasurementResult(
        hypothesis_name=definition.hypothesis_name,
        run_timestamp=datetime.now(UTC),
        dataset_summary=dataset_summary,
        buckets=buckets,
        comparison=comparison,
        quality_score=quality,
        time_window_results=time_window_results,
        trend_direction=trend,
    )


def compute_bucket_stats(
    df: pd.DataFrame,
    bucket_label: str,
    outcome_col: str,
    margin_col: str | None = None,
) -> BucketStats:
    """Compute per-bucket outcome statistics."""
    # Drop rows with NULL outcome
    valid = df.dropna(subset=[outcome_col])
    n = len(valid)
    covers = int(valid[outcome_col].sum()) if n > 0 else 0
    cover_rate = covers / n if n > 0 else 0.0

    win_rate = None
    if "won_game" in valid.columns:
        win_rate = float(valid["won_game"].mean()) if n > 0 else None

    avg_margin = None
    std_margin = None
    if margin_col and margin_col in valid.columns:
        margin_data = valid[margin_col].dropna()
        if len(margin_data) > 0:
            avg_margin = float(margin_data.mean())
            std_margin = float(margin_data.std()) if len(margin_data) > 1 else 0.0

    return BucketStats(
        bucket_label=bucket_label,
        n=n,
        covers=covers,
        cover_rate=cover_rate,
        win_rate=win_rate,
        avg_margin_vs_spread=avg_margin,
        std_margin_vs_spread=std_margin,
    )


def run_statistical_test(stats: BucketStats, test_name: str, null_value: float = 0.5) -> BucketStats:
    """Run the specified statistical test on bucket stats."""
    if stats.n == 0:
        return stats

    if test_name == "binomial":
        result = binomial_test(stats.covers, stats.n, null_value)
        stats.p_value = result.p_value
        stats.significant = result.significant
    # For proportion_z and chi_squared, those are cross-bucket tests handled in compare_buckets

    # Wilson CI
    if stats.n > 0:
        lower, upper = wilson_ci(stats.covers, stats.n)
        stats.ci_lower = lower
        stats.ci_upper = upper

    return stats


def apply_fdr_correction(buckets: list[BucketStats]) -> list[BucketStats]:
    """Apply FDR correction across bucket-level p-values (Amendment A3)."""
    p_values = [b.p_value for b in buckets if b.p_value is not None]
    if len(p_values) < 2:
        # No correction needed for 0 or 1 p-values
        for b in buckets:
            b.p_value_adjusted = b.p_value
        return buckets

    corrected = fdr_correction(p_values)

    idx = 0
    for b in buckets:
        if b.p_value is not None:
            b.p_value_adjusted = corrected[idx]
            b.significant = corrected[idx] < 0.05
            idx += 1
        else:
            b.p_value_adjusted = None

    return buckets


def compare_buckets(stats_a: BucketStats, stats_b: BucketStats) -> ComparisonResult:
    """Compare two buckets using proportion z-test and Cohen's h."""
    rate_diff = stats_a.cover_rate - stats_b.cover_rate

    if stats_a.n > 0 and stats_b.n > 0:
        result = proportion_z_test(stats_a.covers, stats_a.n, stats_b.covers, stats_b.n)
        p_value = result.p_value
        significant = result.significant
    else:
        p_value = 1.0
        significant = False

    effect_h = cohens_h(stats_a.cover_rate, stats_b.cover_rate) if stats_a.n > 0 and stats_b.n > 0 else 0.0

    return ComparisonResult(
        bucket_a=stats_a.bucket_label,
        bucket_b=stats_b.bucket_label,
        rate_difference=rate_diff,
        p_value=p_value,
        effect_size_h=effect_h,
        significant=significant,
    )


def measure_time_windows(
    df: pd.DataFrame, definition: HypothesisDefinition
) -> list[MeasurementResult]:
    """Re-run measurement within each time window."""
    results = []
    for window in definition.time_windows:
        window_df = df[df["season"].isin(window.seasons)]
        if window_df.empty:
            continue

        # Only compute if we have bucketed data
        if "bucket" not in window_df.columns:
            continue

        outcome_col = _get_outcome_col(definition.outcome)
        margin_col = "margin_vs_spread" if definition.outcome == "ats" else None

        buckets = []
        for label in sorted(window_df["bucket"].unique()):
            bucket_df = window_df[window_df["bucket"] == label]
            bs = compute_bucket_stats(bucket_df, label, outcome_col, margin_col)
            bs = run_statistical_test(bs, definition.statistical_test)
            buckets.append(bs)

        comparison = None
        if len(definition.comparison_buckets) == 2:
            a_label, b_label = definition.comparison_buckets
            a_stats = next((b for b in buckets if b.bucket_label == a_label), None)
            b_stats = next((b for b in buckets if b.bucket_label == b_label), None)
            if a_stats and b_stats:
                comparison = compare_buckets(a_stats, b_stats)

        result = MeasurementResult(
            hypothesis_name=f"{definition.hypothesis_name}_{window.label}",
            run_timestamp=datetime.now(UTC),
            dataset_summary={"window": window.label, "seasons": window.seasons, "total_rows": len(window_df)},
            buckets=buckets,
            comparison=comparison,
        )
        results.append(result)

    return results


def _compute_trend(window_results: list[MeasurementResult]) -> str:
    """Compute trend direction from time-windowed results.

    Uses the cover rate of the first comparison bucket across windows.
    """
    if len(window_results) < 2:
        return "insufficient_data"

    rates = []
    for wr in window_results:
        if wr.buckets:
            rates.append(wr.buckets[0].cover_rate)

    if len(rates) < 2:
        return "insufficient_data"

    # Simple linear regression on window index vs cover rate
    x = np.arange(len(rates))
    slope, _, r_value, _, _ = scipy_stats.linregress(x, rates)

    if abs(r_value) < 0.3:
        return "stable"
    elif slope > 0.005:
        return "improving"
    elif slope < -0.005:
        return "declining"
    return "stable"


def _compute_consistency(window_results: list[MeasurementResult]) -> float:
    """Compute consistency score (0-1) from time-windowed results.

    High consistency = the edge direction is consistent across windows.
    """
    if len(window_results) < 2:
        return 0.5  # Neutral if no windows

    # Check if the comparison direction is consistent across windows
    directions = []
    for wr in window_results:
        if wr.comparison:
            directions.append(1 if wr.comparison.rate_difference > 0 else -1)

    if not directions:
        return 0.5

    # Consistency = proportion of windows with same direction as majority
    majority = 1 if sum(directions) > 0 else -1
    same_direction = sum(1 for d in directions if d == majority)
    return same_direction / len(directions)


def _get_outcome_col(outcome: str) -> str:
    """Map outcome type to DataFrame column name."""
    return {
        "ats": "covered_spread",
        "su": "won_game",
        "ou": "hit_over",
    }[outcome]
