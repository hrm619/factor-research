"""Compute derived metrics (season-to-date and last-4-games) from raw team stats."""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import Engine, text

from backend.models.research_models import DerivedMetrics
from backend.research.db import get_session

logger = logging.getLogger(__name__)


def compute_derived_metrics(engine: Engine, season: int) -> int:
    """Recompute all derived metrics for a season. Returns rows written."""
    session = get_session(engine)

    try:
        # Load all team game stats for the season, ordered by week
        query = text("""
            SELECT tgs.*, g.season, g.week, g.home_team, g.away_team,
                   g.home_score, g.away_score
            FROM team_game_stats tgs
            JOIN games g ON tgs.game_id = g.game_id
            WHERE g.season = :season
            ORDER BY g.week, tgs.game_id
        """)
        df = pd.read_sql(query, engine, params={"season": season})

        if df.empty:
            logger.warning("No data found for season %d", season)
            return 0

        # Delete existing derived metrics for this season
        session.query(DerivedMetrics).filter(DerivedMetrics.season == season).delete()

        rows_written = 0
        teams = df["team_abbr"].unique()

        for team in teams:
            team_df = df[df["team_abbr"] == team].sort_values("week")
            metrics = _compute_team_season_metrics(team_df, team, season, df)

            for m in metrics:
                session.add(m)
                rows_written += 1

        session.commit()
        logger.info("Computed %d derived metric rows for season %d", rows_written, season)
        return rows_written

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _compute_team_season_metrics(
    team_df: pd.DataFrame,
    team: str,
    season: int,
    all_stats_df: pd.DataFrame,
) -> list[DerivedMetrics]:
    """Compute derived metrics for one team in one season.

    Critical: all metrics represent the team's profile ENTERING a game,
    not including it. Week 1 = all NULL.
    """
    results = []
    weeks = team_df["week"].values
    game_ids = team_df["game_id"].values

    for i, (week, game_id) in enumerate(zip(weeks, game_ids)):
        if i == 0:
            # Week 1: all NULL
            dm = DerivedMetrics(
                game_id=game_id, team_abbr=team, season=season, week=int(week),
            )
            results.append(dm)
            continue

        # Season-to-date: all games before this one
        prior = team_df.iloc[:i]
        n_prior = len(prior)

        # Last 4 games: if available
        l4 = team_df.iloc[max(0, i - 4):i] if i >= 4 else None

        # Get opponent data for metrics that need it
        prior_opponents = _get_opponent_stats(prior, all_stats_df)
        l4_opponents = _get_opponent_stats(l4, all_stats_df) if l4 is not None else None

        dm = DerivedMetrics(
            game_id=game_id,
            team_abbr=team,
            season=season,
            week=int(week),
            # Conversion rates (SUM/SUM, not AVG of rates)
            third_down_rate_std=_safe_rate(prior["third_down_conversions"], prior["third_down_attempts"]),
            fourth_down_rate_std=_safe_rate(prior["fourth_down_conversions"], prior["fourth_down_attempts"]),
            red_zone_td_rate_std=_safe_rate(prior.get("red_zone_touchdowns"), prior.get("red_zone_attempts")),
            # Per-game averages (STD)
            yards_per_game_std=_safe_mean(prior["total_yards"]),
            points_per_game_std=_safe_mean(prior["points_scored"]),
            points_allowed_per_game_std=_safe_mean(prior["points_allowed"]),
            penalty_rate_std=_safe_mean(prior["penalties"]),
            fourth_down_attempts_per_game_std=_safe_mean(prior["fourth_down_attempts"]),
            # Opponent-dependent metrics
            yards_allowed_per_game_std=_safe_mean(prior_opponents["total_yards"]) if prior_opponents is not None else None,
            sack_rate_std=_safe_rate(prior["sacks"], prior_opponents["pass_attempts"]) if prior_opponents is not None else None,
            turnover_margin_std=_compute_turnover_margin_std(prior, prior_opponents, n_prior),
            # Last 4 games
            third_down_rate_l4=_safe_rate(l4["third_down_conversions"], l4["third_down_attempts"]) if l4 is not None else None,
            yards_per_game_l4=_safe_mean(l4["total_yards"]) if l4 is not None else None,
            points_per_game_l4=_safe_mean(l4["points_scored"]) if l4 is not None else None,
            points_allowed_per_game_l4=_safe_mean(l4["points_allowed"]) if l4 is not None else None,
            red_zone_td_rate_l4=_safe_rate(l4.get("red_zone_touchdowns"), l4.get("red_zone_attempts")) if l4 is not None else None,
        )
        results.append(dm)

    return results


def _get_opponent_stats(team_df: pd.DataFrame | None, all_stats_df: pd.DataFrame) -> pd.DataFrame | None:
    """Get opponent stats for each game in team_df."""
    if team_df is None or team_df.empty:
        return None

    opponent_rows = []
    for _, row in team_df.iterrows():
        game_id = row["game_id"]
        team = row["team_abbr"]
        # Find the other team's stats in the same game
        opp = all_stats_df[
            (all_stats_df["game_id"] == game_id) & (all_stats_df["team_abbr"] != team)
        ]
        if not opp.empty:
            opponent_rows.append(opp.iloc[0])

    if not opponent_rows:
        return None
    return pd.DataFrame(opponent_rows)


def _safe_rate(numerator, denominator) -> float | None:
    """Compute SUM(numerator) / SUM(denominator) safely."""
    if numerator is None or denominator is None:
        return None
    num_sum = numerator.sum()
    den_sum = denominator.sum()
    if pd.isna(num_sum) or pd.isna(den_sum) or den_sum == 0:
        return None
    return float(num_sum / den_sum)


def _safe_mean(series) -> float | None:
    """Compute mean safely, returning None if no valid data."""
    if series is None:
        return None
    result = series.mean()
    if pd.isna(result):
        return None
    return float(result)


def _compute_turnover_margin_std(
    prior: pd.DataFrame,
    prior_opponents: pd.DataFrame | None,
    n_prior: int,
) -> float | None:
    """Compute turnover margin per game (opponent turnovers - own turnovers) / games."""
    if prior_opponents is None or n_prior == 0:
        return None
    own_turnovers = prior["turnovers"].sum()
    opp_turnovers = prior_opponents["turnovers"].sum()
    if pd.isna(own_turnovers) or pd.isna(opp_turnovers):
        return None
    return float((opp_turnovers - own_turnovers) / n_prior)
