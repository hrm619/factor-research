"""Tests for Contract 1 import (JSON → YAML hypothesis)."""

import json
import pytest
from pathlib import Path

from backend.research.contract_import import import_contract


VALID_CONTRACT = {
    "contract_version": "1.0.0",
    "produced_at": "2026-03-30T00:00:00+00:00",
    "producer": "research-assistant",
    "hypothesis_id": "test-uuid",
    "domain_id": "test-domain",
    "domain_name": "nfl_test",
    "rich_definition": {
        "name": "defensive_test",
        "statement": "Test hypothesis",
        "factor": "points allowed",
        "classification": "quartile",
        "outcome_measure": "ATS cover rate",
        "timeframe": "2014-2024",
        "data_required": ["nflverse"],
        "data_available": True,
        "market_expression": "Bet the spread",
        "feasibility": {
            "data_gap": [],
            "knowledge_gap": [],
            "estimated_testability": "high",
        },
        "reasoning_chain": {
            "from_insight": "Expert reasoning",
            "translation_logic": "Defense underpriced",
            "assumptions_added": ["Market undervalues defense"],
            "weaknesses": ["Small sample"],
        },
    },
    "test_definition": {
        "hypothesis_name": "defensive_test",
        "description": "Test whether defensive efficiency predicts ATS outcomes",
        "version": "1.0.0",
        "metrics": ["points_allowed_per_game_std"],
        "classification": {
            "type": "quartile",
            "metric": "points_allowed_per_game_std",
        },
        "outcome": "ats",
        "lookback": "season_to_date",
        "min_sample_size": 50,
        "statistical_test": "binomial",
        "significance_threshold": 0.05,
        "comparison_buckets": ["Q1", "Q4"],
    },
}


@pytest.fixture
def contract_path(tmp_path):
    p = tmp_path / "test_contract.json"
    p.write_text(json.dumps(VALID_CONTRACT))
    return str(p)


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "hypotheses"
    d.mkdir()
    return str(d)


def test_import_valid_contract(contract_path, output_dir):
    yaml_path = import_contract(contract_path, output_dir)
    assert Path(yaml_path).exists()
    assert yaml_path.endswith("defensive_test.yaml")

    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    assert data["hypothesis_name"] == "defensive_test"
    assert data["outcome"] == "ats"
    assert data["classification"]["type"] == "quartile"
    assert data["classification"]["metric"] == "points_allowed_per_game_std"


def test_import_dry_run(contract_path, output_dir):
    yaml_path = import_contract(contract_path, output_dir, dry_run=True)
    assert not Path(yaml_path).exists()


def test_import_bad_version(tmp_path, output_dir):
    bad = {**VALID_CONTRACT, "contract_version": "2.0.0"}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Unsupported contract_version"):
        import_contract(str(p), output_dir)


def test_import_missing_test_definition(tmp_path, output_dir):
    bad = {k: v for k, v in VALID_CONTRACT.items() if k != "test_definition"}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing 'test_definition'"):
        import_contract(str(p), output_dir)


def test_import_missing_file(output_dir):
    with pytest.raises(ValueError, match="not found"):
        import_contract("/nonexistent/path.json", output_dir)


def test_import_with_filters_and_time_windows(tmp_path, output_dir):
    contract = json.loads(json.dumps(VALID_CONTRACT))
    contract["test_definition"]["filters"] = {
        "seasons": [2022, 2023, 2024],
        "game_type": ["regular"],
        "exclude_week_1": True,
    }
    contract["test_definition"]["time_windows"] = [
        {"label": "recent", "seasons": [2022, 2023, 2024]},
    ]
    p = tmp_path / "with_filters.json"
    p.write_text(json.dumps(contract))
    yaml_path = import_contract(str(p), output_dir)

    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    assert data["filters"]["seasons"] == [2022, 2023, 2024]
    assert len(data["time_windows"]) == 1
