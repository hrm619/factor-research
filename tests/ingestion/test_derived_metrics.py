"""Tests for derived metrics computation."""

from datetime import date

import pytest
from sqlalchemy import create_engine

from backend.research.db import init_db, get_session
from backend.models.research_models import Game, Team, TeamGameStats, DerivedMetrics
from backend.ingestion.derived_metrics import compute_derived_metrics


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_db(eng)
    return eng


def _seed_data(engine, n_weeks=5):
    """Create a minimal dataset: 2 teams playing each other for n_weeks."""
    session = get_session(engine)

    # Teams
    session.add(Team(team_abbr="KC", team_name="Chiefs", conference="AFC", division="West"))
    session.add(Team(team_abbr="BUF", team_name="Bills", conference="AFC", division="East"))

    for week in range(1, n_weeks + 1):
        game_id = f"2024_w{week}"
        session.add(Game(
            game_id=game_id, season=2024, week=week,
            game_date=date(2024, 9, week),
            game_type="regular",
            home_team="KC", away_team="BUF",
            home_score=24 + week, away_score=20 + week,
        ))

        # KC stats: consistent ~40% third down, ~350 yards
        session.add(TeamGameStats(
            game_id=game_id, team_abbr="KC", is_home=True,
            pass_completions=20, pass_attempts=30,
            pass_yards=250, rush_attempts=25, rush_yards=100,
            total_yards=350, first_downs=20,
            third_down_conversions=4 + (week % 2),  # alternates 4 and 5
            third_down_attempts=12,
            fourth_down_conversions=1, fourth_down_attempts=2,
            penalties=5, penalty_yards=40,
            sacks=3, sacks_allowed=2,
            turnovers=1, interceptions_thrown=1, fumbles_lost=0,
            points_scored=24 + week, points_allowed=20 + week,
            time_of_possession=1800,
        ))

        # BUF stats
        session.add(TeamGameStats(
            game_id=game_id, team_abbr="BUF", is_home=False,
            pass_completions=22, pass_attempts=35,
            pass_yards=280, rush_attempts=20, rush_yards=90,
            total_yards=370, first_downs=22,
            third_down_conversions=5, third_down_attempts=11,
            fourth_down_conversions=0, fourth_down_attempts=1,
            penalties=7, penalty_yards=55,
            sacks=2, sacks_allowed=3,
            turnovers=2, interceptions_thrown=1, fumbles_lost=1,
            points_scored=20 + week, points_allowed=24 + week,
            time_of_possession=1800,
        ))

    session.commit()
    session.close()


class TestComputeDerivedMetrics:
    def test_returns_row_count(self, engine):
        _seed_data(engine, n_weeks=5)
        rows = compute_derived_metrics(engine, 2024)
        assert rows == 10  # 2 teams * 5 weeks

    def test_week_1_all_null(self, engine):
        _seed_data(engine, n_weeks=3)
        compute_derived_metrics(engine, 2024)

        session = get_session(engine)
        dm = session.query(DerivedMetrics).filter_by(
            game_id="2024_w1", team_abbr="KC"
        ).one()
        assert dm.third_down_rate_std is None
        assert dm.yards_per_game_std is None
        assert dm.points_per_game_std is None
        assert dm.third_down_rate_l4 is None
        session.close()

    def test_week_2_equals_week_1_raw(self, engine):
        """Week 2 STD metrics should equal Week 1's raw stats."""
        _seed_data(engine, n_weeks=3)
        compute_derived_metrics(engine, 2024)

        session = get_session(engine)
        dm = session.query(DerivedMetrics).filter_by(
            game_id="2024_w2", team_abbr="KC"
        ).one()

        # Week 1 KC: 4 conversions / 12 attempts (week=1, so 4 + (1%2) = 5)
        # Actually: third_down_conversions=4 + (week % 2), week=1 -> 5
        assert dm.third_down_rate_std == pytest.approx(5 / 12, rel=1e-3)

        # Week 1 KC: total_yards=350
        assert dm.yards_per_game_std == pytest.approx(350.0, rel=1e-3)

        # Week 1 KC: points_scored=25 (24+1)
        assert dm.points_per_game_std == pytest.approx(25.0, rel=1e-3)
        session.close()

    def test_l4_null_before_week_5(self, engine):
        """L4 metrics should be NULL for weeks 1-4."""
        _seed_data(engine, n_weeks=4)
        compute_derived_metrics(engine, 2024)

        session = get_session(engine)
        for week in range(1, 5):
            dm = session.query(DerivedMetrics).filter_by(
                game_id=f"2024_w{week}", team_abbr="KC"
            ).one()
            assert dm.third_down_rate_l4 is None, f"Week {week} should have NULL L4"
        session.close()

    def test_l4_populated_at_week_5(self, engine):
        """L4 metrics should be populated at week 5 (uses weeks 1-4)."""
        _seed_data(engine, n_weeks=5)
        compute_derived_metrics(engine, 2024)

        session = get_session(engine)
        dm = session.query(DerivedMetrics).filter_by(
            game_id="2024_w5", team_abbr="KC"
        ).one()
        assert dm.third_down_rate_l4 is not None
        assert dm.yards_per_game_l4 is not None
        session.close()

    def test_aggregate_rates_not_average(self, engine):
        """STD rates should use SUM/SUM, not AVG of per-game rates."""
        _seed_data(engine, n_weeks=3)
        compute_derived_metrics(engine, 2024)

        session = get_session(engine)
        # Week 3 STD should be (week1_conv + week2_conv) / (week1_att + week2_att)
        dm = session.query(DerivedMetrics).filter_by(
            game_id="2024_w3", team_abbr="KC"
        ).one()

        # KC week 1: 5/12, week 2: 4/12 (4 + 2%2 = 4)
        # SUM/SUM = (5+4)/(12+12) = 9/24 = 0.375
        expected = (5 + 4) / (12 + 12)
        assert dm.third_down_rate_std == pytest.approx(expected, rel=1e-3)
        session.close()

    def test_no_data_returns_zero(self, engine):
        rows = compute_derived_metrics(engine, 2024)
        assert rows == 0

    def test_idempotent(self, engine):
        """Running twice should produce the same results."""
        _seed_data(engine, n_weeks=3)
        rows1 = compute_derived_metrics(engine, 2024)
        rows2 = compute_derived_metrics(engine, 2024)
        assert rows1 == rows2

        session = get_session(engine)
        count = session.query(DerivedMetrics).filter_by(season=2024).count()
        assert count == 6  # 2 teams * 3 weeks, not doubled
        session.close()
