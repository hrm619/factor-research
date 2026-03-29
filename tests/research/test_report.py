"""Tests for Stage 5: Report."""

import json
from datetime import datetime, UTC
from pathlib import Path

import pytest

from backend.research.report import (
    generate_summary,
    format_terminal_report,
    format_json_report,
    apply_confidence_filter,
    write_report,
    apply_cross_hypothesis_fdr,
)
from backend.research.models import (
    HypothesisDefinition,
    ClassificationConfig,
    FilterConfig,
    BucketStats,
    ComparisonResult,
    QualityScore,
    MeasurementResult,
)


def _make_result(grade="HIGH", composite=2.7) -> MeasurementResult:
    return MeasurementResult(
        hypothesis_name="test_hypothesis",
        run_timestamp=datetime.now(UTC),
        dataset_summary={"total_rows": 1000, "seasons": [2023, 2024]},
        buckets=[
            BucketStats("Q1", 250, 138, 0.552, p_value=0.02, p_value_adjusted=0.04, ci_lower=0.49, ci_upper=0.61),
            BucketStats("Q4", 250, 112, 0.448, p_value=0.02, p_value_adjusted=0.04, ci_lower=0.39, ci_upper=0.51),
        ],
        comparison=ComparisonResult("Q1", "Q4", 0.104, 0.01, effect_size_h=0.21, significant=True),
        quality_score=QualityScore(0.9, 0.9, 0.6, 0.6, composite, grade),
    )


def _make_definition() -> HypothesisDefinition:
    return HypothesisDefinition(
        hypothesis_name="test_hypothesis",
        description="A test",
        version="1.0",
        metrics=["third_down_rate_std"],
        classification=ClassificationConfig(type="quartile", metric="third_down_rate_std"),
        outcome="ats",
        filters=FilterConfig(),
        lookback="season_to_date",
    )


class TestGenerateSummary:
    def test_contains_hypothesis_name(self):
        result = _make_result()
        defn = _make_definition()
        summary = generate_summary(result, defn)
        assert "test_hypothesis" in summary

    def test_contains_quality_grade(self):
        result = _make_result()
        defn = _make_definition()
        summary = generate_summary(result, defn)
        assert "HIGH" in summary


class TestFormatTerminalReport:
    def test_contains_bucket_labels(self):
        result = _make_result()
        output = format_terminal_report(result)
        assert "Q1" in output
        assert "Q4" in output

    def test_contains_cover_rates(self):
        result = _make_result()
        output = format_terminal_report(result)
        assert "55.2%" in output or "0.552" in output


class TestFormatJsonReport:
    def test_structure(self):
        result = _make_result()
        defn = _make_definition()
        data = format_json_report(result, defn)
        assert data["hypothesis_name"] == "test_hypothesis"
        assert len(data["buckets"]) == 2
        assert data["comparison"]["bucket_a"] == "Q1"
        assert data["quality_score"]["grade"] == "HIGH"

    def test_serializable(self):
        result = _make_result()
        defn = _make_definition()
        data = format_json_report(result, defn)
        # Should not raise
        json.dumps(data)


class TestApplyConfidenceFilter:
    def test_high_passes(self):
        result = _make_result(grade="HIGH")
        filtered = apply_confidence_filter(result)
        assert len(filtered.buckets) == 2

    def test_low_filtered_out(self):
        result = _make_result(grade="LOW", composite=1.2)
        filtered = apply_confidence_filter(result)
        assert len(filtered.buckets) == 0
        assert "Filtered out" in filtered.summary


class TestWriteReport:
    def test_writes_json(self, tmp_path):
        result = _make_result()
        defn = _make_definition()
        path = write_report(result, defn, str(tmp_path))
        assert Path(path).exists()
        with open(path) as f:
            data = json.load(f)
        assert data["hypothesis_name"] == "test_hypothesis"

    def test_writes_csv(self, tmp_path):
        result = _make_result()
        defn = _make_definition()
        write_report(result, defn, str(tmp_path))
        csv_path = tmp_path / "summary.csv"
        assert csv_path.exists()


class TestCrossHypothesisFDR:
    def test_adjusts_comparison_pvalues(self):
        results = [
            _make_result(),
            _make_result(),
        ]
        results[0].comparison.p_value = 0.01
        results[1].comparison.p_value = 0.04
        adjusted = apply_cross_hypothesis_fdr(results)
        # Adjusted p-values should be >= raw
        assert adjusted[0].comparison.p_value_adjusted >= 0.01 - 1e-10
        assert adjusted[1].comparison.p_value_adjusted >= 0.04 - 1e-10
