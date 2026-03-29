"""Metrics catalog: registry of all factor metrics and their definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    formula_type: str  # rate_std, rate_l4, per_game_std, per_game_l4, margin_std
    numerator_col: str | None  # for rate metrics: numerator column in team_game_stats
    denominator_col: str | None  # for rate metrics: denominator column
    description: str
    requires_opponent_data: bool = False


# Suffix conventions:
#   _std = season-to-date (entering game)
#   _l4  = last 4 games (entering game)

_METRIC_DEFINITIONS: list[MetricDefinition] = [
    # --- Volume metrics (Section 5.1) ---
    MetricDefinition(
        name="yards_per_game_std",
        formula_type="per_game_std",
        numerator_col="total_yards",
        denominator_col=None,
        description="Average total yards per game, season-to-date entering game",
    ),
    MetricDefinition(
        name="yards_per_game_l4",
        formula_type="per_game_l4",
        numerator_col="total_yards",
        denominator_col=None,
        description="Average total yards per game, last 4 games entering game",
    ),
    MetricDefinition(
        name="points_per_game_std",
        formula_type="per_game_std",
        numerator_col="points_scored",
        denominator_col=None,
        description="Average points scored per game, season-to-date entering game",
    ),
    MetricDefinition(
        name="points_per_game_l4",
        formula_type="per_game_l4",
        numerator_col="points_scored",
        denominator_col=None,
        description="Average points scored per game, last 4 games entering game",
    ),

    # --- Conversion metrics (Section 5.2) ---
    MetricDefinition(
        name="third_down_rate_std",
        formula_type="rate_std",
        numerator_col="third_down_conversions",
        denominator_col="third_down_attempts",
        description="Third-down conversion rate (SUM/SUM), season-to-date entering game",
    ),
    MetricDefinition(
        name="third_down_rate_l4",
        formula_type="rate_l4",
        numerator_col="third_down_conversions",
        denominator_col="third_down_attempts",
        description="Third-down conversion rate (SUM/SUM), last 4 games entering game",
    ),
    MetricDefinition(
        name="fourth_down_rate_std",
        formula_type="rate_std",
        numerator_col="fourth_down_conversions",
        denominator_col="fourth_down_attempts",
        description="Fourth-down conversion rate (SUM/SUM), season-to-date entering game",
    ),
    MetricDefinition(
        name="fourth_down_attempts_per_game_std",
        formula_type="per_game_std",
        numerator_col="fourth_down_attempts",
        denominator_col=None,
        description="Fourth-down attempts per game, season-to-date entering game",
    ),
    MetricDefinition(
        name="red_zone_td_rate_std",
        formula_type="rate_std",
        numerator_col="red_zone_touchdowns",
        denominator_col="red_zone_attempts",
        description="Red zone TD rate (SUM/SUM), season-to-date entering game. NULL if red zone data unavailable.",
    ),
    MetricDefinition(
        name="red_zone_td_rate_l4",
        formula_type="rate_l4",
        numerator_col="red_zone_touchdowns",
        denominator_col="red_zone_attempts",
        description="Red zone TD rate (SUM/SUM), last 4 games entering game",
    ),

    # --- Defensive metrics (Section 5.3) ---
    MetricDefinition(
        name="points_allowed_per_game_std",
        formula_type="per_game_std",
        numerator_col="points_allowed",
        denominator_col=None,
        description="Average points allowed per game, season-to-date entering game",
    ),
    MetricDefinition(
        name="points_allowed_per_game_l4",
        formula_type="per_game_l4",
        numerator_col="points_allowed",
        denominator_col=None,
        description="Average points allowed per game, last 4 games entering game",
    ),
    MetricDefinition(
        name="yards_allowed_per_game_std",
        formula_type="per_game_std",
        numerator_col="points_allowed",  # Note: would need opponent total_yards; using points_allowed as proxy column ref
        denominator_col=None,
        description="Average yards allowed per game, season-to-date entering game",
        requires_opponent_data=True,
    ),
    MetricDefinition(
        name="sack_rate_std",
        formula_type="rate_std",
        numerator_col="sacks",
        denominator_col=None,  # denominator is opponent pass_attempts (requires opponent data)
        description="Sack rate (sacks / opponent pass attempts), season-to-date entering game",
        requires_opponent_data=True,
    ),

    # --- Discipline metrics (Section 5.4) ---
    MetricDefinition(
        name="penalty_rate_std",
        formula_type="per_game_std",
        numerator_col="penalties",
        denominator_col=None,
        description="Penalties per game, season-to-date entering game",
    ),

    # --- Turnover metrics ---
    MetricDefinition(
        name="turnover_margin_std",
        formula_type="margin_std",
        numerator_col="turnovers",
        denominator_col=None,  # margin = opponent turnovers - own turnovers (requires opponent data)
        description="Turnover margin per game, season-to-date entering game",
        requires_opponent_data=True,
    ),
]


class MetricsCatalog:
    """Registry of all available metrics."""

    def __init__(self) -> None:
        self._registry: dict[str, MetricDefinition] = {}
        for metric in _METRIC_DEFINITIONS:
            self._registry[metric.name] = metric

    def get_metric(self, name: str) -> MetricDefinition:
        """Get a metric definition by name. Raises KeyError if not found."""
        if name not in self._registry:
            raise KeyError(f"Unknown metric: '{name}'. Available: {sorted(self._registry.keys())}")
        return self._registry[name]

    def validate_metrics(self, names: list[str]) -> list[MetricDefinition]:
        """Validate that all metric names exist. Returns list of definitions."""
        return [self.get_metric(name) for name in names]

    def list_metrics(self) -> list[str]:
        """Return sorted list of all registered metric names."""
        return sorted(self._registry.keys())

    def get_lookback_variant(self, base_metric: str, lookback: str) -> str:
        """Map a base metric name and lookback type to the suffixed column name.

        Args:
            base_metric: e.g. "third_down_rate"
            lookback: "season_to_date" or "last_4"

        Returns:
            Suffixed name, e.g. "third_down_rate_std" or "third_down_rate_l4"
        """
        suffix = "_std" if lookback == "season_to_date" else "_l4"
        candidate = f"{base_metric}{suffix}"
        if candidate in self._registry:
            return candidate
        raise KeyError(
            f"No metric '{candidate}' found for base='{base_metric}', lookback='{lookback}'"
        )
