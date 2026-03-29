"""Tests for data cleaning and validation."""

import pytest

from backend.ingestion.cleaning import (
    normalize_team_abbr,
    validate_score,
    validate_stat_range,
    validate_time_of_possession,
    validate_season_game_counts,
    clean_game_data,
)


class TestNormalizeTeamAbbr:
    def test_stl_to_lar(self):
        assert normalize_team_abbr("STL") == "LAR"

    def test_oak_to_lvr(self):
        assert normalize_team_abbr("OAK") == "LVR"

    def test_sd_to_lac(self):
        assert normalize_team_abbr("SD") == "LAC"

    def test_already_canonical(self):
        assert normalize_team_abbr("KC") == "KC"
        assert normalize_team_abbr("SF") == "SF"

    def test_case_insensitive(self):
        assert normalize_team_abbr("stl") == "LAR"

    def test_pfr_three_letter_codes(self):
        assert normalize_team_abbr("GNB") == "GB"
        assert normalize_team_abbr("KAN") == "KC"
        assert normalize_team_abbr("NWE") == "NE"
        assert normalize_team_abbr("SFO") == "SF"
        assert normalize_team_abbr("TAM") == "TB"
        assert normalize_team_abbr("HTX") == "HOU"
        assert normalize_team_abbr("NOR") == "NO"
        assert normalize_team_abbr("SDG") == "LAC"
        assert normalize_team_abbr("CRD") == "ARI"
        assert normalize_team_abbr("RAV") == "BAL"
        assert normalize_team_abbr("CLT") == "IND"
        assert normalize_team_abbr("OTI") == "TEN"


class TestValidateScore:
    def test_valid(self):
        assert validate_score(27) == 27
        assert validate_score(0) == 0

    def test_negative(self):
        assert validate_score(-1) is None

    def test_none(self):
        assert validate_score(None) is None


class TestValidateStatRange:
    def test_in_range(self):
        assert validate_stat_range(350, "total_yards") == 350

    def test_out_of_range(self):
        assert validate_stat_range(900, "total_yards") is None

    def test_none_value(self):
        assert validate_stat_range(None, "total_yards") is None

    def test_unknown_stat(self):
        assert validate_stat_range(999, "unknown_stat") == 999


class TestValidateTimeOfPossession:
    def test_valid(self):
        assert validate_time_of_possession(1800, 1800) is True

    def test_overtime_tolerance(self):
        assert validate_time_of_possession(1900, 1800) is True  # 3700 within tolerance

    def test_invalid(self):
        assert validate_time_of_possession(2400, 1800) is False  # 4200, way off

    def test_none(self):
        assert validate_time_of_possession(None, 1800) is True


class TestValidateSeasonGameCounts:
    def test_pre_2021(self):
        assert validate_season_game_counts(2018, 266) is True
        assert validate_season_game_counts(2018, 267) is True  # within tolerance
        assert validate_season_game_counts(2018, 250) is False

    def test_2020(self):
        assert validate_season_game_counts(2020, 268) is True

    def test_post_2021(self):
        assert validate_season_game_counts(2022, 284) is True
        assert validate_season_game_counts(2022, 260) is False


class TestCleanGameData:
    def test_normalizes_abbreviations(self):
        raw = {
            "game": {"game_id": "test", "home_team": "OAK", "away_team": "STL"},
            "home_stats": {},
            "away_stats": {},
        }
        cleaned = clean_game_data(raw)
        assert cleaned["game"]["home_team"] == "LVR"
        assert cleaned["game"]["away_team"] == "LAR"

    def test_validates_scores(self):
        raw = {
            "game": {"game_id": "test", "home_team": "KC", "away_team": "SF",
                      "home_score": 27, "away_score": -1},
            "home_stats": {},
            "away_stats": {},
        }
        cleaned = clean_game_data(raw)
        assert cleaned["game"]["home_score"] == 27
        assert cleaned["game"]["away_score"] is None

    def test_validates_stat_ranges(self):
        raw = {
            "game": {"game_id": "test", "home_team": "KC", "away_team": "SF"},
            "home_stats": {"total_yards": 900, "pass_yards": 300},
            "away_stats": {},
        }
        cleaned = clean_game_data(raw)
        assert cleaned["home_stats"]["total_yards"] is None
        assert cleaned["home_stats"]["pass_yards"] == 300

    def test_computes_turnovers(self):
        raw = {
            "game": {"game_id": "test", "home_team": "KC", "away_team": "SF"},
            "home_stats": {"interceptions_thrown": 2, "fumbles_lost": 1},
            "away_stats": {},
        }
        cleaned = clean_game_data(raw)
        assert cleaned["home_stats"]["turnovers"] == 3
