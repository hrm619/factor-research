"""Tests for SQLAlchemy ORM models and database operations."""

from datetime import date, datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.models.research_models import (
    Base,
    Team,
    Game,
    TeamGameStats,
    DerivedMetrics,
    IngestionLog,
    TEAM_ABBREVIATION_MAP,
    CANONICAL_TEAMS,
    get_expected_game_count,
)
from backend.research.db import init_db, get_session


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_db(eng)
    return eng


@pytest.fixture
def session(engine):
    sess = get_session(engine)
    yield sess
    sess.close()


def _seed_team(session: Session, abbr: str = "KC", name: str = "Kansas City Chiefs"):
    team = Team(team_abbr=abbr, team_name=name, conference="AFC", division="West")
    session.add(team)
    session.flush()
    return team


def _seed_game(session: Session, game_id: str = "202409050kan"):
    _seed_team(session, "KC", "Kansas City Chiefs")
    _seed_team(session, "BAL", "Baltimore Ravens")
    game = Game(
        game_id=game_id,
        season=2024,
        week=1,
        game_date=date(2024, 9, 5),
        game_type="regular",
        home_team="KC",
        away_team="BAL",
        home_score=27,
        away_score=20,
        home_spread_close=-3.0,
    )
    session.add(game)
    session.flush()
    return game


class TestInitDb:
    def test_creates_all_tables(self, engine):
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert tables == {"teams", "games", "team_game_stats", "derived_metrics", "ingestion_log"}


class TestTeam:
    def test_insert_and_query(self, session):
        _seed_team(session)
        team = session.query(Team).get("KC")
        assert team.team_name == "Kansas City Chiefs"
        assert team.conference == "AFC"


class TestGame:
    def test_insert_and_query(self, session):
        _seed_game(session)
        game = session.query(Game).get("202409050kan")
        assert game.season == 2024
        assert game.home_score == 27
        assert game.home_spread_close == -3.0

    def test_relationships(self, session):
        _seed_game(session)
        game = session.query(Game).get("202409050kan")
        assert game.home_team_ref.team_abbr == "KC"
        assert game.away_team_ref.team_abbr == "BAL"


class TestTeamGameStats:
    def test_insert_and_query(self, session):
        _seed_game(session)
        stats = TeamGameStats(
            game_id="202409050kan",
            team_abbr="KC",
            is_home=True,
            pass_completions=22,
            pass_attempts=34,
            pass_yards=291,
            third_down_conversions=7,
            third_down_attempts=12,
            points_scored=27,
            points_allowed=20,
        )
        session.add(stats)
        session.flush()
        result = session.query(TeamGameStats).filter_by(game_id="202409050kan", team_abbr="KC").one()
        assert result.pass_yards == 291
        assert result.third_down_conversions == 7

    def test_composite_pk_duplicate_raises(self, session):
        _seed_game(session)
        stats1 = TeamGameStats(game_id="202409050kan", team_abbr="KC", is_home=True)
        stats2 = TeamGameStats(game_id="202409050kan", team_abbr="KC", is_home=True)
        session.add(stats1)
        session.flush()
        session.add(stats2)
        with pytest.raises(IntegrityError):
            session.flush()


class TestDerivedMetrics:
    def test_insert_and_query(self, session):
        _seed_game(session)
        session.add(TeamGameStats(game_id="202409050kan", team_abbr="KC", is_home=True))
        session.flush()
        dm = DerivedMetrics(
            game_id="202409050kan",
            team_abbr="KC",
            season=2024,
            week=1,
            third_down_rate_std=None,  # Week 1 = NULL
            yards_per_game_std=None,
        )
        session.add(dm)
        session.flush()
        result = session.query(DerivedMetrics).filter_by(game_id="202409050kan", team_abbr="KC").one()
        assert result.third_down_rate_std is None
        assert result.season == 2024


class TestIngestionLog:
    def test_insert(self, session):
        log = IngestionLog(source="pfr", season=2024, status="success", rows_ingested=284)
        session.add(log)
        session.flush()
        assert log.ingestion_id is not None
        assert log.rows_ingested == 284


class TestConstants:
    def test_team_abbreviation_map(self):
        assert TEAM_ABBREVIATION_MAP["STL"] == "LAR"
        assert TEAM_ABBREVIATION_MAP["OAK"] == "LVR"
        assert TEAM_ABBREVIATION_MAP["SD"] == "LAC"

    def test_canonical_teams_count(self):
        assert len(CANONICAL_TEAMS) == 32

    def test_expected_game_counts(self):
        assert get_expected_game_count(2018)["total"] == 266
        assert get_expected_game_count(2020)["total"] == 268
        assert get_expected_game_count(2022)["total"] == 284

    def test_expected_game_count_out_of_range(self):
        with pytest.raises(ValueError):
            get_expected_game_count(2013)
