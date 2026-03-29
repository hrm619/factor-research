"""Tests for Stage 3: Classify."""

import logging

import pandas as pd
import numpy as np
import pytest

from backend.research.classify import (
    classify,
    classify_quartile,
    classify_percentile,
    classify_binary,
    classify_custom,
)
from backend.research.models import HypothesisDefinition, ClassificationConfig, FilterConfig


def _make_df(n_per_season: int = 50, seasons: list[int] = None) -> pd.DataFrame:
    """Create a synthetic DataFrame with known metric values."""
    if seasons is None:
        seasons = [2023, 2024]
    rng = np.random.default_rng(42)
    rows = []
    for season in seasons:
        for i in range(n_per_season):
            rows.append({
                "season": season,
                "game_id": f"{season}_{i:03d}",
                "team_abbr": f"T{i % 32:02d}",
                "third_down_rate_std": rng.uniform(0.25, 0.55),
                "yards_per_game_std": rng.uniform(250, 420),
                "covered_spread": rng.choice([True, False]),
            })
    return pd.DataFrame(rows)


def _make_definition(cls_type="quartile", **cls_kwargs) -> HypothesisDefinition:
    cls_kwargs.setdefault("metric", "third_down_rate_std")
    return HypothesisDefinition(
        hypothesis_name="test",
        description="test",
        version="1.0",
        metrics=["third_down_rate_std"],
        classification=ClassificationConfig(type=cls_type, **cls_kwargs),
        outcome="ats",
        filters=FilterConfig(),
        lookback="season_to_date",
    )


class TestClassifyQuartile:
    def test_assigns_four_buckets(self):
        df = _make_df(100)
        result = classify_quartile(df, "third_down_rate_std")
        assert set(result["bucket"].unique()) == {"Q1", "Q2", "Q3", "Q4"}

    def test_within_season_ranking(self):
        """Q1 in each season should be the highest-metric teams in THAT season."""
        df = _make_df(100, seasons=[2023, 2024])
        result = classify_quartile(df, "third_down_rate_std")

        for season in [2023, 2024]:
            season_df = result[result["season"] == season]
            q1 = season_df[season_df["bucket"] == "Q1"]["third_down_rate_std"]
            q4 = season_df[season_df["bucket"] == "Q4"]["third_down_rate_std"]
            assert q1.mean() > q4.mean()

    def test_roughly_equal_bucket_sizes(self):
        df = _make_df(100)
        result = classify_quartile(df, "third_down_rate_std")
        counts = result["bucket"].value_counts()
        for bucket in ["Q1", "Q2", "Q3", "Q4"]:
            assert 40 <= counts[bucket] <= 60  # ~50 per bucket with 200 rows

    def test_ties_handled(self):
        """Identical metric values should be handled with average rank."""
        df = pd.DataFrame({
            "season": [2023] * 8,
            "third_down_rate_std": [0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.2, 0.1],
        })
        result = classify_quartile(df, "third_down_rate_std")
        assert "bucket" in result.columns
        assert len(result) == 8


class TestClassifyPercentile:
    def test_top_bottom_middle(self):
        df = _make_df(100)
        result = classify_percentile(df, "third_down_rate_std", top_pct=25.0, bottom_pct=25.0)
        assert set(result["bucket"].unique()) == {"top", "middle", "bottom"}

    def test_top_has_highest_values(self):
        df = _make_df(100)
        result = classify_percentile(df, "third_down_rate_std", top_pct=25.0, bottom_pct=25.0)
        for season in df["season"].unique():
            season_df = result[result["season"] == season]
            top_mean = season_df[season_df["bucket"] == "top"]["third_down_rate_std"].mean()
            bottom_mean = season_df[season_df["bucket"] == "bottom"]["third_down_rate_std"].mean()
            assert top_mean > bottom_mean


class TestClassifyBinary:
    def test_above_below(self):
        df = _make_df(100)
        result = classify_binary(df, "third_down_rate_std", threshold=0.40)
        assert set(result["bucket"].unique()) == {"above", "below"}

    def test_threshold_applied(self):
        df = _make_df(100)
        result = classify_binary(df, "third_down_rate_std", threshold=0.40)
        above = result[result["bucket"] == "above"]
        below = result[result["bucket"] == "below"]
        assert (above["third_down_rate_std"] >= 0.40).all()
        assert (below["third_down_rate_std"] < 0.40).all()


class TestClassifyCustom:
    def test_custom_boundaries(self):
        df = _make_df(100)
        result = classify_custom(df, "third_down_rate_std", boundaries=[0.35, 0.45])
        assert set(result["bucket"].unique()) <= {"B1", "B2", "B3"}

    def test_three_boundaries_four_buckets(self):
        df = _make_df(200)
        result = classify_custom(df, "third_down_rate_std", boundaries=[0.30, 0.40, 0.50])
        assert len(result["bucket"].unique()) <= 4


class TestClassifyDispatcher:
    def test_quartile_dispatch(self):
        df = _make_df(100)
        defn = _make_definition("quartile")
        result = classify(df, defn)
        assert "bucket" in result.columns

    def test_percentile_dispatch(self):
        df = _make_df(100)
        defn = _make_definition("percentile", top_pct=25.0, bottom_pct=25.0)
        result = classify(df, defn)
        assert "bucket" in result.columns

    def test_small_bucket_warning(self, caplog):
        """Small buckets should trigger a warning."""
        df = _make_df(5, seasons=[2023])  # very small dataset
        defn = _make_definition("quartile")
        with caplog.at_level(logging.WARNING):
            classify(df, defn)
        assert any("below min_sample_size" in msg for msg in caplog.messages)
