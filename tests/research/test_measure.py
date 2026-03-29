"""Tests for Stage 4: Measure."""

import numpy as np
import pandas as pd
import pytest

from backend.research.measure import (
    measure,
    compute_bucket_stats,
    run_statistical_test,
    apply_fdr_correction,
    compare_buckets,
    _compute_trend,
    _compute_consistency,
)
from backend.research.models import (
    HypothesisDefinition,
    ClassificationConfig,
    FilterConfig,
    TimeWindow,
    BucketStats,
    MeasurementResult,
    ComparisonResult,
)
from datetime import datetime, UTC


def _make_classified_df(n_per_bucket=100, cover_rates=None):
    """Create a synthetic classified DataFrame."""
    if cover_rates is None:
        cover_rates = {"Q1": 0.55, "Q2": 0.52, "Q3": 0.48, "Q4": 0.45}

    rng = np.random.default_rng(42)
    rows = []
    for bucket, rate in cover_rates.items():
        for i in range(n_per_bucket):
            covered = rng.random() < rate
            margin = rng.normal(2 if covered else -2, 5)
            rows.append({
                "season": 2024,
                "bucket": bucket,
                "covered_spread": covered,
                "won_game": rng.random() < 0.55,
                "margin_vs_spread": margin,
                "home_spread_close": -3.0,
            })
    return pd.DataFrame(rows)


def _make_definition(**overrides) -> HypothesisDefinition:
    defaults = dict(
        hypothesis_name="test",
        description="test",
        version="1.0",
        metrics=["third_down_rate_std"],
        classification=ClassificationConfig(type="quartile", metric="third_down_rate_std"),
        outcome="ats",
        filters=FilterConfig(seasons=[2024]),
        lookback="season_to_date",
        comparison_buckets=["Q1", "Q4"],
    )
    defaults.update(overrides)
    return HypothesisDefinition(**defaults)


class TestComputeBucketStats:
    def test_basic(self):
        df = pd.DataFrame({
            "covered_spread": [True, True, True, False, False],
            "won_game": [True, True, False, False, False],
            "margin_vs_spread": [3.0, 1.0, 5.0, -2.0, -4.0],
        })
        bs = compute_bucket_stats(df, "Q1", "covered_spread", "margin_vs_spread")
        assert bs.n == 5
        assert bs.covers == 3
        assert bs.cover_rate == pytest.approx(0.6)
        assert bs.avg_margin_vs_spread == pytest.approx(0.6)


class TestRunStatisticalTest:
    def test_binomial(self):
        bs = BucketStats(bucket_label="Q1", n=100, covers=55, cover_rate=0.55)
        bs = run_statistical_test(bs, "binomial")
        assert bs.p_value is not None
        assert bs.ci_lower is not None
        assert bs.ci_lower < 0.55 < bs.ci_upper


class TestApplyFDRCorrection:
    def test_adjusts_pvalues(self):
        buckets = [
            BucketStats(bucket_label="Q1", n=100, covers=56, cover_rate=0.56, p_value=0.03),
            BucketStats(bucket_label="Q2", n=100, covers=52, cover_rate=0.52, p_value=0.35),
            BucketStats(bucket_label="Q3", n=100, covers=48, cover_rate=0.48, p_value=0.35),
            BucketStats(bucket_label="Q4", n=100, covers=44, cover_rate=0.44, p_value=0.03),
        ]
        result = apply_fdr_correction(buckets)
        # Adjusted p-values should be >= raw p-values
        for b in result:
            assert b.p_value_adjusted >= b.p_value - 1e-10

    def test_single_bucket_no_correction(self):
        buckets = [BucketStats(bucket_label="Q1", n=100, covers=55, cover_rate=0.55, p_value=0.03)]
        result = apply_fdr_correction(buckets)
        assert result[0].p_value_adjusted == result[0].p_value


class TestCompareBuckets:
    def test_significant_difference(self):
        a = BucketStats(bucket_label="Q1", n=500, covers=275, cover_rate=0.55)
        b = BucketStats(bucket_label="Q4", n=500, covers=225, cover_rate=0.45)
        comp = compare_buckets(a, b)
        assert comp.rate_difference == pytest.approx(0.10)
        assert comp.p_value < 0.05
        assert comp.effect_size_h != 0
        assert comp.significant is True

    def test_no_difference(self):
        a = BucketStats(bucket_label="Q1", n=100, covers=50, cover_rate=0.50)
        b = BucketStats(bucket_label="Q4", n=100, covers=50, cover_rate=0.50)
        comp = compare_buckets(a, b)
        assert comp.rate_difference == 0.0
        assert comp.significant is False


class TestMeasure:
    def test_full_measurement(self):
        df = _make_classified_df()
        defn = _make_definition()
        result = measure(df, defn)
        assert len(result.buckets) == 4
        assert result.comparison is not None
        assert result.quality_score is not None
        assert result.hypothesis_name == "test"

    def test_bucket_labels(self):
        df = _make_classified_df()
        defn = _make_definition()
        result = measure(df, defn)
        labels = {b.bucket_label for b in result.buckets}
        assert labels == {"Q1", "Q2", "Q3", "Q4"}

    def test_with_time_windows(self):
        # Create data spanning two seasons
        rows = []
        rng = np.random.default_rng(42)
        for season in [2023, 2024]:
            for bucket in ["Q1", "Q4"]:
                for _ in range(50):
                    rows.append({
                        "season": season,
                        "bucket": bucket,
                        "covered_spread": rng.random() < 0.55,
                        "won_game": rng.random() < 0.55,
                        "margin_vs_spread": rng.normal(0, 5),
                    })
        df = pd.DataFrame(rows)

        defn = _make_definition(
            time_windows=[
                TimeWindow(label="2023", seasons=[2023]),
                TimeWindow(label="2024", seasons=[2024]),
            ],
        )
        result = measure(df, defn)
        assert len(result.time_window_results) == 2


class TestComputeTrend:
    def test_improving(self):
        results = [
            MeasurementResult("t", datetime.now(UTC), {}, [BucketStats("Q1", 100, 48, 0.48)]),
            MeasurementResult("t", datetime.now(UTC), {}, [BucketStats("Q1", 100, 52, 0.52)]),
            MeasurementResult("t", datetime.now(UTC), {}, [BucketStats("Q1", 100, 58, 0.58)]),
        ]
        assert _compute_trend(results) == "improving"

    def test_declining(self):
        results = [
            MeasurementResult("t", datetime.now(UTC), {}, [BucketStats("Q1", 100, 58, 0.58)]),
            MeasurementResult("t", datetime.now(UTC), {}, [BucketStats("Q1", 100, 52, 0.52)]),
            MeasurementResult("t", datetime.now(UTC), {}, [BucketStats("Q1", 100, 45, 0.45)]),
        ]
        assert _compute_trend(results) == "declining"

    def test_insufficient(self):
        results = [
            MeasurementResult("t", datetime.now(UTC), {}, [BucketStats("Q1", 100, 50, 0.50)]),
        ]
        assert _compute_trend(results) == "insufficient_data"
