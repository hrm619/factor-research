"""Tests for Stage 1: Define."""

from pathlib import Path

import pytest

from backend.research.define import load_hypothesis, validate_hypothesis, resolve_metrics
from backend.research.models import HypothesisValidationError
from backend.research.metrics_catalog import MetricsCatalog

FIXTURES = Path(__file__).parent / "fixtures"


class TestLoadHypothesis:
    def test_valid_yaml(self):
        raw = load_hypothesis(str(FIXTURES / "sample_hypothesis.yaml"))
        assert raw["hypothesis_name"] == "test_hypothesis"
        assert raw["outcome"] == "ats"

    def test_file_not_found(self):
        with pytest.raises(HypothesisValidationError, match="not found"):
            load_hypothesis("/nonexistent/path.yaml")

    def test_wrong_extension(self, tmp_path):
        bad_file = tmp_path / "test.json"
        bad_file.write_text("{}")
        with pytest.raises(HypothesisValidationError, match="Expected .yaml"):
            load_hypothesis(str(bad_file))


class TestValidateHypothesis:
    def test_valid(self):
        raw = load_hypothesis(str(FIXTURES / "sample_hypothesis.yaml"))
        defn = validate_hypothesis(raw)
        assert defn.hypothesis_name == "test_hypothesis"
        assert defn.classification.type == "quartile"
        assert defn.filters.exclude_week_1 is True
        assert defn.min_sample_size == 30

    def test_missing_name(self):
        raw = load_hypothesis(str(FIXTURES / "invalid_hypothesis_missing_name.yaml"))
        with pytest.raises(HypothesisValidationError, match="hypothesis_name"):
            validate_hypothesis(raw)

    def test_missing_classification(self):
        raw = {"hypothesis_name": "x", "description": "y", "version": "1", "metrics": ["a"],
               "outcome": "ats", "lookback": "season_to_date"}
        with pytest.raises(HypothesisValidationError, match="classification"):
            validate_hypothesis(raw)

    def test_invalid_outcome(self):
        raw = {"hypothesis_name": "x", "description": "y", "version": "1", "metrics": ["a"],
               "classification": {"type": "quartile", "metric": "a"},
               "outcome": "invalid", "lookback": "season_to_date"}
        with pytest.raises(HypothesisValidationError, match="Invalid outcome"):
            validate_hypothesis(raw)

    def test_invalid_classification_type(self):
        raw = {"hypothesis_name": "x", "description": "y", "version": "1", "metrics": ["a"],
               "classification": {"type": "invalid", "metric": "a"},
               "outcome": "ats", "lookback": "season_to_date"}
        with pytest.raises(HypothesisValidationError, match="Invalid classification type"):
            validate_hypothesis(raw)

    def test_percentile_requires_pcts(self):
        raw = {"hypothesis_name": "x", "description": "y", "version": "1", "metrics": ["a"],
               "classification": {"type": "percentile", "metric": "a"},
               "outcome": "ats", "lookback": "season_to_date"}
        with pytest.raises(HypothesisValidationError, match="top_pct"):
            validate_hypothesis(raw)

    def test_defaults_applied(self):
        raw = load_hypothesis(str(FIXTURES / "sample_hypothesis.yaml"))
        raw.pop("min_sample_size", None)
        raw.pop("statistical_test", None)
        raw.pop("significance_threshold", None)
        defn = validate_hypothesis(raw)
        assert defn.min_sample_size == 50
        assert defn.statistical_test == "binomial"
        assert defn.significance_threshold == 0.05

    def test_time_windows(self):
        raw = load_hypothesis(str(FIXTURES / "sample_hypothesis.yaml"))
        raw["time_windows"] = [
            {"label": "early", "seasons": [2023]},
            {"label": "late", "seasons": [2024]},
        ]
        defn = validate_hypothesis(raw)
        assert len(defn.time_windows) == 2
        assert defn.time_windows[0].label == "early"


class TestResolveMetrics:
    def test_valid_metrics(self):
        raw = load_hypothesis(str(FIXTURES / "sample_hypothesis.yaml"))
        defn = validate_hypothesis(raw)
        catalog = MetricsCatalog()
        resolved = resolve_metrics(defn, catalog)
        assert resolved is defn  # returns same object

    def test_unknown_metric(self):
        raw = load_hypothesis(str(FIXTURES / "sample_hypothesis.yaml"))
        raw["metrics"] = ["nonexistent_metric"]
        defn = validate_hypothesis(raw)
        catalog = MetricsCatalog()
        with pytest.raises(KeyError, match="Unknown metric"):
            resolve_metrics(defn, catalog)

    def test_unknown_classification_metric(self):
        raw = load_hypothesis(str(FIXTURES / "sample_hypothesis.yaml"))
        raw["classification"]["metric"] = "fake_metric"
        defn = validate_hypothesis(raw)
        catalog = MetricsCatalog()
        with pytest.raises(HypothesisValidationError, match="Classification metric"):
            resolve_metrics(defn, catalog)

    def test_chi_squared_compatibility(self):
        raw = load_hypothesis(str(FIXTURES / "sample_hypothesis.yaml"))
        raw["statistical_test"] = "chi_squared"
        raw["classification"]["type"] = "binary"
        raw["classification"]["threshold"] = 0.5
        defn = validate_hypothesis(raw)
        catalog = MetricsCatalog()
        with pytest.raises(HypothesisValidationError, match="chi_squared test requires"):
            resolve_metrics(defn, catalog)
