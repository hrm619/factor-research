"""SQLAlchemy ORM models for the factor research data foundation."""

from datetime import datetime, date, UTC

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Date,
    DateTime,
    Text,
    ForeignKey,
    ForeignKeyConstraint,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# --- Constants ---

TEAM_ABBREVIATION_MAP: dict[str, str] = {
    "STL": "LAR",
    "LA": "LAR",
    "OAK": "LVR",
    "LV": "LVR",
    "SD": "LAC",
    "SDG": "LAC",
    "GNB": "GB",
    "KAN": "KC",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "RAI": "LVR",
    "RAM": "LAR",
    "HTX": "HOU",
    "CLT": "IND",
    "RAV": "BAL",
    "OTI": "TEN",
    "CRD": "ARI",
}

CANONICAL_TEAMS: set[str] = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LVR", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}

# Per-era expected game counts (Amendment A1)
EXPECTED_GAME_COUNTS: dict[str, dict[str, int]] = {
    "2014-2019": {"regular": 256, "playoff": 10, "total": 266},
    "2020": {"regular": 256, "playoff": 12, "total": 268},
    "2021-2024": {"regular": 272, "playoff": 12, "total": 284},
}


def get_expected_game_count(season: int) -> dict[str, int]:
    """Return expected game counts for a given season."""
    if 2014 <= season <= 2019:
        return EXPECTED_GAME_COUNTS["2014-2019"]
    elif season == 2020:
        return EXPECTED_GAME_COUNTS["2020"]
    elif 2021 <= season <= 2024:
        return EXPECTED_GAME_COUNTS["2021-2024"]
    else:
        raise ValueError(f"Season {season} outside supported range 2014-2024")


# --- ORM Models ---


class Team(Base):
    __tablename__ = "teams"

    team_abbr = Column(String(4), primary_key=True)
    team_name = Column(String(50), nullable=False)
    conference = Column(String(3))  # AFC, NFC
    division = Column(String(10))  # e.g. "East", "West"


class Game(Base):
    __tablename__ = "games"

    game_id = Column(String(20), primary_key=True)  # PFR game ID, e.g. "202409050kan"
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=False)
    game_date = Column(Date, nullable=False)
    game_type = Column(String(10), nullable=False, default="regular")  # regular, wildcard, divisional, conference, superbowl
    home_team = Column(String(4), ForeignKey("teams.team_abbr"), nullable=False)
    away_team = Column(String(4), ForeignKey("teams.team_abbr"), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_spread_close = Column(Float)  # negative means home favored
    total_close = Column(Float)
    neutral_site = Column(Boolean, default=False)
    overtime = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    home_team_ref = relationship("Team", foreign_keys=[home_team])
    away_team_ref = relationship("Team", foreign_keys=[away_team])
    team_stats = relationship("TeamGameStats", back_populates="game")


class TeamGameStats(Base):
    __tablename__ = "team_game_stats"

    game_id = Column(String(20), ForeignKey("games.game_id"), primary_key=True)
    team_abbr = Column(String(4), ForeignKey("teams.team_abbr"), primary_key=True)
    is_home = Column(Boolean, nullable=False)

    # Passing
    pass_completions = Column(Integer)
    pass_attempts = Column(Integer)
    pass_yards = Column(Integer)
    pass_touchdowns = Column(Integer)
    interceptions_thrown = Column(Integer)

    # Rushing
    rush_attempts = Column(Integer)
    rush_yards = Column(Integer)
    rush_touchdowns = Column(Integer)

    # Receiving / totals
    total_yards = Column(Integer)
    first_downs = Column(Integer)

    # Turnovers
    fumbles_lost = Column(Integer)
    turnovers = Column(Integer)

    # Penalties
    penalties = Column(Integer)
    penalty_yards = Column(Integer)

    # Third/fourth downs
    third_down_conversions = Column(Integer)
    third_down_attempts = Column(Integer)
    fourth_down_conversions = Column(Integer)
    fourth_down_attempts = Column(Integer)

    # Defense
    sacks = Column(Integer)  # sacks made by this team's defense
    sacks_allowed = Column(Integer)  # sacks allowed by this team's offense

    # Red zone
    red_zone_attempts = Column(Integer)
    red_zone_touchdowns = Column(Integer)

    # Time of possession (stored as seconds)
    time_of_possession = Column(Integer)

    # Scoring
    points_scored = Column(Integer)
    points_allowed = Column(Integer)

    game = relationship("Game", back_populates="team_stats")
    team = relationship("Team")


class DerivedMetrics(Base):
    __tablename__ = "derived_metrics"

    game_id = Column(String(20), primary_key=True)
    team_abbr = Column(String(4), primary_key=True)
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=False)

    # Season-to-date (entering game)
    third_down_rate_std = Column(Float)
    yards_per_game_std = Column(Float)
    points_per_game_std = Column(Float)
    points_allowed_per_game_std = Column(Float)
    yards_allowed_per_game_std = Column(Float)
    turnover_margin_std = Column(Float)
    penalty_rate_std = Column(Float)
    sack_rate_std = Column(Float)
    red_zone_td_rate_std = Column(Float)
    fourth_down_rate_std = Column(Float)
    fourth_down_attempts_per_game_std = Column(Float)

    # Last 4 games (entering game)
    third_down_rate_l4 = Column(Float)
    yards_per_game_l4 = Column(Float)
    points_per_game_l4 = Column(Float)
    points_allowed_per_game_l4 = Column(Float)
    red_zone_td_rate_l4 = Column(Float)

    __table_args__ = (
        ForeignKeyConstraint(
            ["game_id", "team_abbr"],
            ["team_game_stats.game_id", "team_game_stats.team_abbr"],
        ),
    )


class IngestionLog(Base):
    __tablename__ = "ingestion_log"

    ingestion_id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False, default="pfr")
    season = Column(Integer)
    status = Column(String(20), nullable=False)  # running, success, partial, failed
    rows_ingested = Column(Integer, default=0)
    rows_skipped = Column(Integer, default=0)
    errors = Column(Text)  # JSON array of error strings
    started_at = Column(DateTime, default=lambda: datetime.now(UTC))
    completed_at = Column(DateTime)
