"""Tests for the pipeline harness (integration test)."""

from datetime import date

import pytest
from sqlalchemy import create_engine

from backend.research.db import init_db, get_session
from backend.models.research_models import Game, Team, TeamGameStats, DerivedMetrics
from backend.research.harness import run_hypothesis


@pytest.fixture
def db_with_data(tmp_path):
    """Create an in-memory DB with test data and return the URL."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    init_db(engine)

    session = get_session(engine)

    # Seed teams
    session.add(Team(team_abbr="KC", team_name="Chiefs", conference="AFC", division="West"))
    session.add(Team(team_abbr="BUF", team_name="Bills", conference="AFC", division="East"))
    session.add(Team(team_abbr="SF", team_name="49ers", conference="NFC", division="West"))
    session.add(Team(team_abbr="PHI", team_name="Eagles", conference="NFC", division="East"))

    import numpy as np
    rng = np.random.default_rng(42)

    teams = ["KC", "BUF", "SF", "PHI"]
    game_num = 0
    for season in [2023, 2024]:
        for week in range(2, 12):  # weeks 2-11
            for i in range(0, len(teams), 2):
                home = teams[i]
                away = teams[i + 1]
                game_num += 1
                game_id = f"{season}_w{week}_{game_num}"
                home_score = int(rng.integers(14, 38))
                away_score = int(rng.integers(14, 38))
                spread = round(float(rng.uniform(-10, 10)), 1)

                session.add(Game(
                    game_id=game_id, season=season, week=week,
                    game_date=date(season, 9, 1),
                    game_type="regular",
                    home_team=home, away_team=away,
                    home_score=home_score, away_score=away_score,
                    home_spread_close=spread,
                ))

                for team, is_home in [(home, True), (away, False)]:
                    tdr = rng.uniform(0.25, 0.55)
                    session.add(TeamGameStats(
                        game_id=game_id, team_abbr=team, is_home=is_home,
                        pass_completions=int(rng.integers(15, 30)),
                        pass_attempts=int(rng.integers(25, 45)),
                        pass_yards=int(rng.integers(150, 350)),
                        total_yards=int(rng.integers(250, 450)),
                        third_down_conversions=int(rng.integers(3, 10)),
                        third_down_attempts=int(rng.integers(10, 16)),
                        points_scored=home_score if is_home else away_score,
                        points_allowed=away_score if is_home else home_score,
                    ))
                    session.add(DerivedMetrics(
                        game_id=game_id, team_abbr=team, season=season, week=week,
                        third_down_rate_std=float(rng.uniform(0.30, 0.50)),
                        yards_per_game_std=float(rng.uniform(280, 400)),
                        points_per_game_std=float(rng.uniform(18, 30)),
                    ))

    session.commit()
    session.close()
    return db_url


@pytest.fixture
def hypothesis_path():
    return "backend/research/hypotheses/conversion_efficiency_ats_mispricing.yaml"


class TestRunHypothesis:
    def test_end_to_end(self, db_with_data, hypothesis_path, tmp_path):
        """Full pipeline should run without errors."""
        result = run_hypothesis(
            hypothesis_path,
            db_url=db_with_data,
            output_dir=str(tmp_path / "results"),
        )

        assert result.hypothesis_name == "conversion_efficiency_ats_mispricing"
        assert len(result.buckets) > 0
        assert result.quality_score is not None
        assert result.quality_score.grade in ("HIGH", "MEDIUM", "LOW", "INSUFFICIENT_DATA")

        # JSON output should exist
        import glob
        json_files = glob.glob(str(tmp_path / "results" / "*.json"))
        assert len(json_files) == 1

    def test_bucket_stats_populated(self, db_with_data, hypothesis_path, tmp_path):
        result = run_hypothesis(
            hypothesis_path,
            db_url=db_with_data,
            output_dir=str(tmp_path / "results"),
        )
        for bucket in result.buckets:
            assert bucket.n > 0
            assert 0 <= bucket.cover_rate <= 1
            assert bucket.p_value is not None
            assert bucket.ci_lower is not None
