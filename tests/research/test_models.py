"""Tests for research dataclasses and exceptions."""

from datetime import datetime

import pytest

from backend.research.models import (
    HypothesisDefinition,
    ClassificationConfig,
    FilterConfig,
    TimeWindow,
    BucketStats,
    ComparisonResult,
    QualityScore,
    MeasurementResult,
    TestResult,
    HypothesisValidationError,
    DataExtractionError,
)


def _make_hypothesis(**overrides) -> HypothesisDefinition:
    defaults = dict(
        hypothesis_name="test_hypothesis",
        description="A test hypothesis",
        version="1.0.0",
        metrics=["third_down_rate_std"],
        classification=ClassificationConfig(type="quartile", metric="third_down_rate_std"),
        outcome="ats",
        filters=FilterConfig(seasons=[2023, 2024]),
        lookback="season_to_date",
    )
    defaults.update(overrides)
    return HypothesisDefinition(**defaults)


class TestHypothesisDefinition:
    def test_create_with_defaults(self):
        h = _make_hypothesis()
        assert h.min_sample_size == 50
        assert h.statistical_test == "binomial"
        assert h.significance_threshold == 0.05
        assert h.comparison_buckets == []
        assert h.time_windows == []

    def test_create_with_overrides(self):
        h = _make_hypothesis(min_sample_size=100, statistical_test="chi_squared")
        assert h.min_sample_size == 100
        assert h.statistical_test == "chi_squared"

    def test_frozen(self):
        h = _make_hypothesis()
        with pytest.raises(AttributeError):
            h.hypothesis_name = "changed"

    def test_time_windows(self):
        windows = [
            TimeWindow(label="2014-2018", seasons=[2014, 2015, 2016, 2017, 2018]),
            TimeWindow(label="2019-2024", seasons=[2019, 2020, 2021, 2022, 2023, 2024]),
        ]
        h = _make_hypothesis(time_windows=windows)
        assert len(h.time_windows) == 2
        assert h.time_windows[0].label == "2014-2018"


class TestClassificationConfig:
    def test_quartile(self):
        c = ClassificationConfig(type="quartile", metric="third_down_rate_std")
        assert c.type == "quartile"
        assert c.top_pct is None

    def test_percentile(self):
        c = ClassificationConfig(type="percentile", metric="yards_per_game_std", top_pct=25.0, bottom_pct=25.0)
        assert c.top_pct == 25.0


class TestBucketStats:
    def test_create(self):
        bs = BucketStats(bucket_label="Q1", n=100, covers=55, cover_rate=0.55)
        assert bs.p_value is None
        assert bs.p_value_adjusted is None
        assert bs.significant is False

    def test_mutable(self):
        bs = BucketStats(bucket_label="Q1", n=100, covers=55, cover_rate=0.55)
        bs.p_value = 0.03
        bs.significant = True
        assert bs.p_value == 0.03


class TestQualityScore:
    def test_high(self):
        qs = QualityScore(
            sample_size_score=0.9,
            significance_score=0.9,
            effect_size_score=0.6,
            consistency_score=0.6,
            composite=2.7,
            grade="HIGH",
        )
        assert qs.grade == "HIGH"
        assert qs.composite >= 2.5


class TestMeasurementResult:
    def test_create(self):
        mr = MeasurementResult(
            hypothesis_name="test",
            run_timestamp=datetime.now(),
            dataset_summary={"total_rows": 1000},
            buckets=[BucketStats(bucket_label="Q1", n=250, covers=130, cover_rate=0.52)],
        )
        assert mr.comparison is None
        assert mr.time_window_results == []


class TestExceptions:
    def test_hypothesis_validation_error(self):
        with pytest.raises(HypothesisValidationError, match="missing field"):
            raise HypothesisValidationError("missing field: hypothesis_name")

    def test_data_extraction_error(self):
        with pytest.raises(DataExtractionError, match="zero rows"):
            raise DataExtractionError("zero rows returned")
