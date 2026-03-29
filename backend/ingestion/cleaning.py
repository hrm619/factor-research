"""Data cleaning and validation rules for ingested data."""

from __future__ import annotations

import logging

from backend.models.research_models import TEAM_ABBREVIATION_MAP, get_expected_game_count
from backend.ingestion.config import STAT_RANGES

logger = logging.getLogger(__name__)


def normalize_team_abbr(abbr: str) -> str:
    """Normalize a team abbreviation to canonical form."""
    return TEAM_ABBREVIATION_MAP.get(abbr.upper(), abbr.upper())


def validate_score(score: int | None) -> int | None:
    """Validate a game score. Must be non-negative integer."""
    if score is None:
        return None
    if not isinstance(score, int) or score < 0:
        logger.warning("Invalid score: %s", score)
        return None
    return score


def validate_stat_range(value: int | float | None, stat_name: str) -> int | float | None:
    """Validate a stat value is within expected range.

    Out-of-range values are set to None and logged.
    """
    if value is None:
        return None
    if stat_name not in STAT_RANGES:
        return value  # No range defined, pass through
    low, high = STAT_RANGES[stat_name]
    if value < low or value > high:
        logger.warning("Stat '%s' = %s out of range [%s, %s], setting to NULL", stat_name, value, low, high)
        return None
    return value


def validate_time_of_possession(home_top_seconds: int | None, away_top_seconds: int | None) -> bool:
    """Validate time of possession sums to approximately 60:00 (3600s).

    Allows ±120s tolerance for OT games.
    """
    if home_top_seconds is None or away_top_seconds is None:
        return True  # Can't validate if missing
    total = home_top_seconds + away_top_seconds
    if abs(total - 3600) > 120:
        logger.warning("Time of possession sum %ds deviates from 3600s by %ds", total, abs(total - 3600))
        return False
    return True


def validate_season_game_counts(season: int, game_count: int, tolerance: int = 2) -> bool:
    """Validate season game count against expected per-era counts (Amendment A1)."""
    expected = get_expected_game_count(season)
    expected_total = expected["total"]
    if abs(game_count - expected_total) > tolerance:
        logger.warning(
            "Season %d: %d games (expected %d ±%d)",
            season, game_count, expected_total, tolerance,
        )
        return False
    return True


def clean_game_data(raw: dict) -> dict:
    """Apply all cleaning rules to a parsed game data dict.

    Args:
        raw: dict with keys 'game' (metadata), 'home_stats', 'away_stats'

    Returns:
        Cleaned copy of the dict.
    """
    game = {**raw["game"]}
    home_stats = {**raw["home_stats"]}
    away_stats = {**raw["away_stats"]}

    # Normalize team abbreviations
    game["home_team"] = normalize_team_abbr(game["home_team"])
    game["away_team"] = normalize_team_abbr(game["away_team"])

    # Validate scores
    game["home_score"] = validate_score(game.get("home_score"))
    game["away_score"] = validate_score(game.get("away_score"))

    # Validate stat ranges
    for stats in (home_stats, away_stats):
        for stat_name in list(stats.keys()):
            if stat_name in STAT_RANGES:
                stats[stat_name] = validate_stat_range(stats[stat_name], stat_name)

    # Validate time of possession
    validate_time_of_possession(
        home_stats.get("time_of_possession"),
        away_stats.get("time_of_possession"),
    )

    # Compute turnovers if not present
    for stats in (home_stats, away_stats):
        if stats.get("turnovers") is None:
            ints = stats.get("interceptions_thrown") or 0
            fum = stats.get("fumbles_lost") or 0
            if ints > 0 or fum > 0:
                stats["turnovers"] = ints + fum

    return {"game": game, "home_stats": home_stats, "away_stats": away_stats}
