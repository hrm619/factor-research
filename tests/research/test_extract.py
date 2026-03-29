"""Tests for Stage 2: Extract."""

from datetime import date

import pytest
from sqlalchemy import create_engine

from backend.research.db import init_db, get_session
from backend.models.research_models import Game, Team, TeamGameStats, DerivedMetrics
from backend.research.extract import extract_data, attach_outcomes
from backend.research.models import (
    HypothesisDefinition,
    ClassificationConfig,
    FilterConfig,
    DataExtractionError,
)
import pandas as pd


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_db(eng)
    return eng


def _seed_teams(session):
    session.add(Team(team_abbr="KC", team_name="Chiefs", conference="AFC", division="West"))
    session.add(Team(team_abbr="BUF", team_name="Bills", conference="AFC", division="East"))
    session.flush()


def _seed_game(session, game_id, season, week, home_score, away_score, spread, game_type="regular"):
    session.add(Game(
        game_id=game_id, season=season, week=week,
        game_date=date(season, 9, week),
        game_type=game_type,
        home_team="KC", away_team="BUF",
        home_score=home_score, away_score=away_score,
        home_spread_close=spread,
    ))
    session.flush()
    for team, is_home in [("KC", True), ("BUF", False)]:
        session.add(TeamGameStats(
            game_id=game_id, team_abbr=team, is_home=is_home,
            points_scored=home_score if is_home else away_score,
            points_allowed=away_score if is_home else home_score,
        ))
        session.add(DerivedMetrics(
            game_id=game_id, team_abbr=team, season=season, week=week,
            third_down_rate_std=0.40 if team == "KC" else 0.35,
            yards_per_game_std=350.0,
        ))
    session.flush()


def _seed_full(engine, n_games=10):
    session = get_session(engine)
    _seed_teams(session)
    for i in range(n_games):
        week = i + 2  # Start at week 2 (week 1 excluded by default)
        _seed_game(session, f"2024_w{week}", 2024, week, 27, 20, -3.0)
    session.commit()
    session.close()


def _make_definition(**overrides) -> HypothesisDefinition:
    defaults = dict(
        hypothesis_name="test",
        description="test",
        version="1.0",
        metrics=["third_down_rate_std"],
        classification=ClassificationConfig(type="quartile", metric="third_down_rate_std"),
        outcome="ats",
        filters=FilterConfig(seasons=[2024], game_type=["regular"]),
        lookback="season_to_date",
    )
    defaults.update(overrides)
    return HypothesisDefinition(**defaults)


class TestExtractData:
    def test_basic_extract(self, engine):
        _seed_full(engine)
        defn = _make_definition()
        df = extract_data(defn, engine)
        assert len(df) > 0
        assert "third_down_rate_std" in df.columns
        assert "covered_spread" in df.columns

    def test_season_filter(self, engine):
        session = get_session(engine)
        _seed_teams(session)
        _seed_game(session, "2023_w2", 2023, 2, 24, 17, -3.0)
        _seed_game(session, "2024_w2", 2024, 2, 27, 20, -3.0)
        session.commit()
        session.close()

        defn = _make_definition(filters=FilterConfig(seasons=[2024], game_type=["regular"]))
        df = extract_data(defn, engine)
        assert all(df["season"] == 2024)

    def test_exclude_week_1(self, engine):
        session = get_session(engine)
        _seed_teams(session)
        _seed_game(session, "2024_w1", 2024, 1, 24, 17, -3.0)
        _seed_game(session, "2024_w2", 2024, 2, 27, 20, -3.0)
        session.commit()
        session.close()

        defn = _make_definition()
        df = extract_data(defn, engine)
        assert all(df["week"] > 1)

    def test_zero_rows_raises(self, engine):
        defn = _make_definition(filters=FilterConfig(seasons=[2099]))
        with pytest.raises(DataExtractionError, match="Zero rows"):
            extract_data(defn, engine)


class TestAttachOutcomes:
    def test_home_team_covers(self):
        """Home team wins by 7, spread is -3 -> covers (margin_vs_spread = 7 + (-3) = 4)."""
        df = pd.DataFrame([{
            "home_score": 27, "away_score": 20,
            "home_spread_close": -3.0, "total_close": 44.0,
            "is_home": True,
        }])
        result = attach_outcomes(df)
        assert result.iloc[0]["covered_spread"] == True
        assert result.iloc[0]["margin_vs_spread"] == 4.0
        assert result.iloc[0]["won_game"] == True

    def test_home_team_fails_to_cover(self):
        """Home wins by 2, spread is -3 -> doesn't cover (margin_vs_spread = 2 + (-3) = -1)."""
        df = pd.DataFrame([{
            "home_score": 22, "away_score": 20,
            "home_spread_close": -3.0, "total_close": 44.0,
            "is_home": True,
        }])
        result = attach_outcomes(df)
        assert result.iloc[0]["covered_spread"] == False
        assert result.iloc[0]["margin_vs_spread"] == -1.0

    def test_away_team_covers(self):
        """Away team loses by 2, spread is -3 (home favored by 3) -> away covers."""
        df = pd.DataFrame([{
            "home_score": 22, "away_score": 20,
            "home_spread_close": -3.0, "total_close": 44.0,
            "is_home": False,
        }])
        result = attach_outcomes(df)
        # Away score_margin = 20 - 22 = -2
        # margin_vs_spread = -2 - (-3) = 1
        assert result.iloc[0]["margin_vs_spread"] == 1.0
        assert result.iloc[0]["covered_spread"] == True

    def test_push_is_non_cover(self):
        """Exact push -> non-cover."""
        df = pd.DataFrame([{
            "home_score": 24, "away_score": 21,
            "home_spread_close": -3.0, "total_close": 44.0,
            "is_home": True,
        }])
        result = attach_outcomes(df)
        # margin_vs_spread = 3 + (-3) = 0 -> push -> non-cover
        assert result.iloc[0]["margin_vs_spread"] == 0.0
        assert result.iloc[0]["covered_spread"] == False

    def test_null_spread(self):
        """NULL spread -> NULL coverage."""
        df = pd.DataFrame([{
            "home_score": 27, "away_score": 20,
            "home_spread_close": None, "total_close": None,
            "is_home": True,
        }])
        result = attach_outcomes(df)
        assert result.iloc[0]["covered_spread"] == None
        assert result.iloc[0]["margin_vs_spread"] == None

    def test_over_under(self):
        """Total 47 with line 44 -> over."""
        df = pd.DataFrame([{
            "home_score": 27, "away_score": 20,
            "home_spread_close": -3.0, "total_close": 44.0,
            "is_home": True,
        }])
        result = attach_outcomes(df)
        assert result.iloc[0]["hit_over"] == True
