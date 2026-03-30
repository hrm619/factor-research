"""NFL data fetching via nflreadpy (nflverse).

Replaces PFR scraping with structured data from the nflverse project.
Provides schedule, team stats, and play-by-play aggregated stats.
"""

from __future__ import annotations

import logging
import re

import nflreadpy as nfl
import pandas as pd

from backend.ingestion.config import GAME_TYPE_MAP
from backend.models.research_models import TEAM_ABBREVIATION_MAP

logger = logging.getLogger(__name__)


def _normalize_team(abbr: str) -> str:
    """Normalize a team abbreviation to canonical form."""
    return TEAM_ABBREVIATION_MAP.get(abbr, abbr)


def fetch_schedule(seasons: list[int]) -> pd.DataFrame:
    """Fetch game schedule with scores and betting lines.

    Returns a pandas DataFrame with columns mapped to our Game model:
        game_id, season, week, game_date, game_type, home_team, away_team,
        home_score, away_score, home_spread_close, total_close, overtime
    """
    raw = nfl.load_schedules(seasons).to_pandas()

    df = pd.DataFrame({
        "game_id": raw["game_id"],
        "season": raw["season"],
        "week": raw["week"],
        "game_date": pd.to_datetime(raw["gameday"]).dt.date,
        "game_type": raw["game_type"].map(GAME_TYPE_MAP),
        "home_team": raw["home_team"].map(_normalize_team),
        "away_team": raw["away_team"].map(_normalize_team),
        "home_score": raw["home_score"].astype("Int64"),
        "away_score": raw["away_score"].astype("Int64"),
        "home_spread_close": raw["spread_line"],
        "total_close": raw["total_line"],
        "overtime": raw["overtime"].fillna(0).astype(bool),
    })

    # Drop rows with unmapped game types (e.g., Pro Bowl)
    df = df.dropna(subset=["game_type"])

    logger.info(
        "Fetched schedule: %d games across seasons %s",
        len(df), sorted(df["season"].unique()),
    )
    return df


def fetch_team_stats(seasons: list[int]) -> pd.DataFrame:
    """Fetch per-team-per-game stats from nflverse.

    Returns a pandas DataFrame with columns mapped to our TeamGameStats model.
    Join key to schedule: (season, week, team).
    """
    raw = nfl.load_team_stats(seasons).to_pandas()

    df = pd.DataFrame({
        "season": raw["season"],
        "week": raw["week"],
        "team": raw["team"].map(_normalize_team),
        "opponent_team": raw["opponent_team"].map(_normalize_team),
        "season_type": raw["season_type"],
        # Passing
        "pass_completions": raw["completions"],
        "pass_attempts": raw["attempts"],
        "pass_yards": raw["passing_yards"],
        "pass_touchdowns": raw["passing_tds"],
        "interceptions_thrown": raw["passing_interceptions"],
        # Rushing
        "rush_attempts": raw["carries"],
        "rush_yards": raw["rushing_yards"],
        "rush_touchdowns": raw["rushing_tds"],
        # Computed totals
        "total_yards": raw["passing_yards"].fillna(0) + raw["rushing_yards"].fillna(0),
        "first_downs": (
            raw["passing_first_downs"].fillna(0)
            + raw["rushing_first_downs"].fillna(0)
            + raw["receiving_first_downs"].fillna(0)
        ),
        "fumbles_lost": (
            raw["rushing_fumbles_lost"].fillna(0)
            + raw["receiving_fumbles_lost"].fillna(0)
            + raw["sack_fumbles_lost"].fillna(0)
        ),
        # Penalties
        "penalties": raw["penalties"],
        "penalty_yards": raw["penalty_yards"],
        # Defense
        "sacks": raw["def_sacks"],
        "sacks_allowed": raw["sacks_suffered"],
    })

    # Compute turnovers from components
    df["turnovers"] = df["interceptions_thrown"].fillna(0) + df["fumbles_lost"].fillna(0)

    # Cast computed int columns
    for col in ["total_yards", "first_downs", "fumbles_lost", "turnovers"]:
        df[col] = df[col].astype("Int64")

    logger.info(
        "Fetched team stats: %d team-game rows across seasons %s",
        len(df), sorted(df["season"].unique()),
    )
    return df


def _parse_drive_top(s: str | None) -> int | None:
    """Parse 'MM:SS' drive time of possession to seconds."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    match = re.match(r"(\d+):(\d+)", str(s))
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return None


def aggregate_pbp_stats(seasons: list[int]) -> pd.DataFrame:
    """Aggregate play-by-play data for stats not in team_stats.

    Processes one season at a time to manage memory.
    Returns DataFrame with columns:
        game_id, team, third_down_conversions, third_down_attempts,
        fourth_down_conversions, fourth_down_attempts,
        red_zone_attempts, red_zone_touchdowns, time_of_possession
    """
    all_results = []

    for season in seasons:
        logger.info("Aggregating PBP stats for season %d...", season)
        raw = nfl.load_pbp([season]).to_pandas()

        # Normalize team abbreviations in PBP
        raw["posteam"] = raw["posteam"].map(
            lambda x: _normalize_team(x) if pd.notna(x) else x
        )

        # Filter to actual plays (not timeouts, penalties-only, etc.)
        plays = raw[raw["play_type"].isin(["pass", "run"])].copy()

        # -- Third downs --
        third = plays[plays["down"] == 3]
        third_att = third.groupby(["game_id", "posteam"]).size().rename("third_down_attempts")
        third_conv = (
            third[third["third_down_converted"] == 1]
            .groupby(["game_id", "posteam"]).size()
            .rename("third_down_conversions")
        )

        # -- Fourth downs --
        fourth = plays[plays["down"] == 4]
        fourth_att = fourth.groupby(["game_id", "posteam"]).size().rename("fourth_down_attempts")
        fourth_conv = (
            fourth[fourth["fourth_down_converted"] == 1]
            .groupby(["game_id", "posteam"]).size()
            .rename("fourth_down_conversions")
        )

        # -- Red zone (yardline_100 <= 20) --
        rz = plays[plays["yardline_100"] <= 20]
        # Red zone attempts = unique drives that entered the red zone
        rz_att = (
            rz.groupby(["game_id", "posteam"])["drive"]
            .nunique()
            .rename("red_zone_attempts")
        )
        rz_td = (
            rz[rz["touchdown"] == 1]
            .groupby(["game_id", "posteam"]).size()
            .rename("red_zone_touchdowns")
        )

        # -- Time of possession --
        # Sum drive_time_of_possession per team per game (deduplicate by drive)
        drives = raw.dropna(subset=["drive", "posteam", "drive_time_of_possession"])
        drives = drives.drop_duplicates(subset=["game_id", "drive"])
        drives["top_seconds"] = drives["drive_time_of_possession"].apply(_parse_drive_top)

        top = (
            drives.groupby(["game_id", "posteam"])["top_seconds"]
            .sum()
            .rename("time_of_possession")
        )

        # Combine all stats
        result = pd.DataFrame(index=third_att.index).join([
            third_att, third_conv, fourth_att, fourth_conv,
            rz_att, rz_td, top,
        ], how="outer")

        result = result.fillna(0).astype(int).reset_index()
        result.columns = [
            "game_id", "team",
            "third_down_attempts", "third_down_conversions",
            "fourth_down_attempts", "fourth_down_conversions",
            "red_zone_attempts", "red_zone_touchdowns",
            "time_of_possession",
        ]

        all_results.append(result)

        # Free memory
        del raw, plays, third, fourth, rz, drives
        logger.info("Season %d: %d team-game PBP rows", season, len(result))

    combined = pd.concat(all_results, ignore_index=True)
    logger.info("Total PBP aggregated rows: %d", len(combined))
    return combined
