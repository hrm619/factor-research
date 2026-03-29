"""Tests for metrics catalog."""

import pytest

from backend.research.metrics_catalog import MetricsCatalog, MetricDefinition


@pytest.fixture
def catalog():
    return MetricsCatalog()


class TestMetricsCatalog:
    def test_all_expected_metrics_registered(self, catalog):
        expected = [
            "third_down_rate_std",
            "third_down_rate_l4",
            "fourth_down_rate_std",
            "fourth_down_attempts_per_game_std",
            "red_zone_td_rate_std",
            "red_zone_td_rate_l4",
            "yards_per_game_std",
            "yards_per_game_l4",
            "points_per_game_std",
            "points_per_game_l4",
            "points_allowed_per_game_std",
            "points_allowed_per_game_l4",
            "yards_allowed_per_game_std",
            "penalty_rate_std",
            "turnover_margin_std",
            "sack_rate_std",
        ]
        for name in expected:
            metric = catalog.get_metric(name)
            assert isinstance(metric, MetricDefinition)

    def test_get_metric_unknown_raises(self, catalog):
        with pytest.raises(KeyError, match="Unknown metric"):
            catalog.get_metric("nonexistent_metric")

    def test_validate_metrics_valid(self, catalog):
        defs = catalog.validate_metrics(["third_down_rate_std", "yards_per_game_std"])
        assert len(defs) == 2
        assert defs[0].name == "third_down_rate_std"

    def test_validate_metrics_invalid_raises(self, catalog):
        with pytest.raises(KeyError):
            catalog.validate_metrics(["third_down_rate_std", "fake_metric"])

    def test_list_metrics(self, catalog):
        metrics = catalog.list_metrics()
        assert len(metrics) >= 15
        assert metrics == sorted(metrics)  # should be sorted

    def test_get_lookback_variant_std(self, catalog):
        result = catalog.get_lookback_variant("third_down_rate", "season_to_date")
        assert result == "third_down_rate_std"

    def test_get_lookback_variant_l4(self, catalog):
        result = catalog.get_lookback_variant("third_down_rate", "last_4")
        assert result == "third_down_rate_l4"

    def test_get_lookback_variant_unknown_raises(self, catalog):
        with pytest.raises(KeyError):
            catalog.get_lookback_variant("nonexistent", "season_to_date")

    def test_metric_definitions_complete(self, catalog):
        """Every metric should have a non-empty description and valid formula_type."""
        valid_types = {"rate_std", "rate_l4", "per_game_std", "per_game_l4", "margin_std"}
        for name in catalog.list_metrics():
            metric = catalog.get_metric(name)
            assert metric.description, f"{name} missing description"
            assert metric.formula_type in valid_types, f"{name} has invalid formula_type"

    def test_rate_metrics_have_numerator(self, catalog):
        """Rate metrics should have a numerator column defined."""
        for name in catalog.list_metrics():
            metric = catalog.get_metric(name)
            if metric.formula_type.startswith("rate"):
                assert metric.numerator_col is not None, f"{name} missing numerator_col"
