"""Stage 2: Extract data from the database for hypothesis testing."""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import Engine, text

from backend.research.models import HypothesisDefinition, DataExtractionError

logger = logging.getLogger(__name__)


def extract_data(definition: HypothesisDefinition, engine: Engine) -> pd.DataFrame:
    """Pull relevant historical data and attach outcomes.

    Returns a DataFrame where each row is a team-game observation with
    derived metrics and outcome columns.
    """
    query, params = build_extract_query(definition)
    df = pd.read_sql(text(query), engine, params=params)

    if df.empty:
        raise DataExtractionError(
            f"Zero rows returned for hypothesis '{definition.hypothesis_name}'. "
            f"Filters: seasons={definition.filters.seasons}, "
            f"game_type={definition.filters.game_type}"
        )

    initial_rows = len(df)

    # Drop rows where required metrics are NULL
    required_metrics = [definition.classification.metric] + definition.metrics
    required_metrics = list(set(required_metrics))  # dedupe
    for metric in required_metrics:
        if metric in df.columns:
            before = len(df)
            df = df.dropna(subset=[metric])
            dropped = before - len(df)
            if dropped > 0:
                logger.info("Dropped %d rows with NULL %s", dropped, metric)

    # Attach outcomes
    df = attach_outcomes(df)

    logger.info(
        "Extracted %d rows (%d after NULL drops) for '%s', seasons %s",
        initial_rows, len(df), definition.hypothesis_name,
        df["season"].unique().tolist() if len(df) > 0 else [],
    )

    if df.empty:
        raise DataExtractionError(
            f"All rows dropped due to NULL metrics for '{definition.hypothesis_name}'"
        )

    return df


def build_extract_query(definition: HypothesisDefinition) -> tuple[str, dict]:
    """Build SQL query joining games, team_game_stats, and derived_metrics."""
    params: dict = {}

    where_clauses = []

    # Season filter
    if definition.filters.seasons:
        placeholders = ", ".join(f":season_{i}" for i in range(len(definition.filters.seasons)))
        where_clauses.append(f"g.season IN ({placeholders})")
        for i, s in enumerate(definition.filters.seasons):
            params[f"season_{i}"] = s

    # Week filter
    if definition.filters.weeks:
        placeholders = ", ".join(f":week_{i}" for i in range(len(definition.filters.weeks)))
        where_clauses.append(f"g.week IN ({placeholders})")
        for i, w in enumerate(definition.filters.weeks):
            params[f"week_{i}"] = w

    # Game type filter
    if definition.filters.game_type:
        placeholders = ", ".join(f":gtype_{i}" for i in range(len(definition.filters.game_type)))
        where_clauses.append(f"g.game_type IN ({placeholders})")
        for i, gt in enumerate(definition.filters.game_type):
            params[f"gtype_{i}"] = gt

    # Exclude week 1
    if definition.filters.exclude_week_1:
        where_clauses.append("g.week > 1")

    # Require non-NULL scores for outcome computation
    where_clauses.append("g.home_score IS NOT NULL AND g.away_score IS NOT NULL")

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    # Select specific dm columns to avoid duplicate game_id/team_abbr/season/week
    dm_metrics = [
        "dm.third_down_rate_std", "dm.third_down_rate_l4",
        "dm.yards_per_game_std", "dm.yards_per_game_l4",
        "dm.points_per_game_std", "dm.points_per_game_l4",
        "dm.points_allowed_per_game_std", "dm.points_allowed_per_game_l4",
        "dm.yards_allowed_per_game_std", "dm.turnover_margin_std",
        "dm.penalty_rate_std", "dm.sack_rate_std",
        "dm.red_zone_td_rate_std", "dm.red_zone_td_rate_l4",
        "dm.fourth_down_rate_std", "dm.fourth_down_attempts_per_game_std",
    ]
    dm_cols = ",\n            ".join(dm_metrics)

    query = f"""
        SELECT
            g.game_id, g.season, g.week, g.game_date, g.game_type,
            g.home_team, g.away_team, g.home_score, g.away_score,
            g.home_spread_close, g.total_close,
            tgs.team_abbr, tgs.is_home,
            tgs.points_scored, tgs.points_allowed,
            {dm_cols}
        FROM games g
        JOIN team_game_stats tgs ON g.game_id = tgs.game_id
        LEFT JOIN derived_metrics dm ON tgs.game_id = dm.game_id AND tgs.team_abbr = dm.team_abbr
        WHERE {where_sql}
        ORDER BY g.season, g.week, g.game_id
    """

    return query, params


def compute_ad_hoc_metrics(
    df: pd.DataFrame, definition: HypothesisDefinition
) -> pd.DataFrame:
    """Compute compound metrics not in derived_metrics.

    Placeholder for Phase 1 — extend for compound factor hypotheses.
    """
    return df


def attach_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Add outcome columns: covered_spread, margin_vs_spread, won_game, hit_over.

    Spread convention: home_spread_close is negative when home is favored.
    For home team: covered = (home_score - away_score) + home_spread_close > 0
    For away team: covered = (away_score - home_score) - home_spread_close > 0
    Push (exactly 0) = non-cover.
    """
    result = df.copy()

    # Score margin from this team's perspective
    result["score_margin"] = result.apply(
        lambda r: (r["home_score"] - r["away_score"]) if r["is_home"]
        else (r["away_score"] - r["home_score"]),
        axis=1,
    )

    # Margin vs spread
    # home_spread_close < 0 means home favored (home must win by more than spread)
    # For home: margin_vs_spread = actual_margin + spread (spread is negative for favorites)
    # For away: margin_vs_spread = actual_margin - spread
    def _margin_vs_spread(row):
        if pd.isna(row["home_spread_close"]):
            return None
        if row["is_home"]:
            return row["score_margin"] + row["home_spread_close"]
        else:
            return row["score_margin"] - row["home_spread_close"]

    result["margin_vs_spread"] = result.apply(_margin_vs_spread, axis=1)

    # Covered spread: margin_vs_spread > 0 (push = non-cover)
    result["covered_spread"] = result["margin_vs_spread"].apply(
        lambda x: x > 0 if pd.notna(x) else None
    )

    # Won game straight up
    result["won_game"] = result["score_margin"] > 0

    # Over/under
    def _hit_over(row):
        if pd.isna(row.get("total_close")):
            return None
        total_points = row["home_score"] + row["away_score"]
        return total_points > row["total_close"]

    result["hit_over"] = result.apply(_hit_over, axis=1)

    return result
