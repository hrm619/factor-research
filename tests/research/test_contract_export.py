"""Tests for Contract 2 export (results → edge registry)."""

import json
import pytest
from pathlib import Path

import yaml

from backend.research.contract_export import export_edges, _infer_metric_direction


SAMPLE_RESULT = {
    "hypothesis_name": "defensive_test",
    "description": "Test hypothesis",
    "version": "1.0.0",
    "run_timestamp": "2026-03-30T01:30:58.908202+00:00",
    "dataset_summary": {
        "total_rows": 512,
        "seasons": [2022, 2023, 2024],
        "buckets": {"Q1": 128, "Q2": 128, "Q3": 128, "Q4": 128},
    },
    "buckets": [
        {
            "bucket": "Q1",
            "n": 128,
            "covers": 72,
            "cover_rate": 0.5625,
            "win_rate": 0.55,
            "avg_margin_vs_spread": 2.1,
            "p_value": 0.08,
            "p_value_adjusted": 0.16,
            "ci_lower": 0.47,
            "ci_upper": 0.65,
            "significant": False,
        },
        {
            "bucket": "Q4",
            "n": 128,
            "covers": 56,
            "cover_rate": 0.4375,
            "win_rate": 0.42,
            "avg_margin_vs_spread": -1.5,
            "p_value": 0.08,
            "p_value_adjusted": 0.16,
            "ci_lower": 0.35,
            "ci_upper": 0.53,
            "significant": False,
        },
    ],
    "comparison": {
        "bucket_a": "Q1",
        "bucket_b": "Q4",
        "rate_difference": 0.125,
        "p_value": 0.04,
        "p_value_adjusted": None,
        "effect_size_h": 0.25,
        "significant": True,
    },
    "quality_score": {
        "sample_size_score": 0.8,
        "significance_score": 0.9,
        "effect_size_score": 0.7,
        "consistency_score": 0.5,
        "composite": 2.3,
        "grade": "MEDIUM",
    },
    "time_windows": [],
    "trend_direction": "insufficient_data",
    "summary": "Test summary",
}

SAMPLE_HYPOTHESIS_YAML = {
    "hypothesis_name": "defensive_test",
    "description": "Test hypothesis",
    "version": "1.0.0",
    "metrics": ["points_allowed_per_game_std"],
    "classification": {
        "type": "quartile",
        "metric": "points_allowed_per_game_std",
    },
    "outcome": "ats",
    "lookback": "season_to_date",
    "filters": {
        "seasons": [2022, 2023, 2024],
        "game_type": ["regular"],
        "exclude_week_1": True,
    },
    "min_sample_size": 50,
    "statistical_test": "binomial",
    "significance_threshold": 0.05,
    "comparison_buckets": ["Q1", "Q4"],
}


@pytest.fixture
def setup_dirs(tmp_path):
    """Create results and hypotheses dirs with sample data."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hypotheses_dir = tmp_path / "hypotheses"
    hypotheses_dir.mkdir()

    # Write result JSON
    result_file = results_dir / "defensive_test_20260330_013058.json"
    result_file.write_text(json.dumps(SAMPLE_RESULT))

    # Write hypothesis YAML
    yaml_file = hypotheses_dir / "defensive_test.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(SAMPLE_HYPOTHESIS_YAML, f)

    return str(results_dir), str(hypotheses_dir)


def test_export_produces_valid_registry(setup_dirs):
    results_dir, hypotheses_dir = setup_dirs
    registry = export_edges(
        results_dir=results_dir,
        hypotheses_dir=hypotheses_dir,
    )
    assert registry["contract_version"] == "1.0.0"
    assert registry["producer"] == "factor-research"
    assert registry["domain"] == "nfl"
    assert len(registry["edges"]) == 2


def test_edge_entry_has_required_fields(setup_dirs):
    results_dir, hypotheses_dir = setup_dirs
    registry = export_edges(results_dir=results_dir, hypotheses_dir=hypotheses_dir)
    edge = registry["edges"][0]

    required = [
        "edge_id", "hypothesis_name", "hypothesis_version",
        "metric", "bucket_label", "outcome_type", "lookback",
        "measurement", "quality", "applicability",
        "decay", "last_backtested", "provenance",
    ]
    for field in required:
        assert field in edge, f"Missing field: {field}"


def test_edge_magnitude_calculation(setup_dirs):
    results_dir, hypotheses_dir = setup_dirs
    registry = export_edges(results_dir=results_dir, hypotheses_dir=hypotheses_dir)

    q1_edge = next(e for e in registry["edges"] if e["bucket_label"] == "Q1")
    assert q1_edge["measurement"]["edge_magnitude"] == round(0.5625 - 0.50, 4)

    q4_edge = next(e for e in registry["edges"] if e["bucket_label"] == "Q4")
    assert q4_edge["measurement"]["edge_magnitude"] == round(0.4375 - 0.50, 4)


def test_edge_id_format(setup_dirs):
    results_dir, hypotheses_dir = setup_dirs
    registry = export_edges(results_dir=results_dir, hypotheses_dir=hypotheses_dir)

    edge_ids = {e["edge_id"] for e in registry["edges"]}
    assert "defensive_test__Q1" in edge_ids
    assert "defensive_test__Q4" in edge_ids


def test_hypothesis_filter(setup_dirs, tmp_path):
    results_dir, hypotheses_dir = setup_dirs

    # Add a second hypothesis result
    other_result = {**SAMPLE_RESULT, "hypothesis_name": "other_hyp"}
    (Path(results_dir) / "other_hyp_20260330_013058.json").write_text(json.dumps(other_result))
    other_yaml = {**SAMPLE_HYPOTHESIS_YAML, "hypothesis_name": "other_hyp"}
    with open(Path(hypotheses_dir) / "other_hyp.yaml", "w") as f:
        yaml.dump(other_yaml, f)

    registry = export_edges(
        results_dir=results_dir,
        hypotheses_dir=hypotheses_dir,
        hypothesis_filter="defensive_test",
    )
    assert all(e["hypothesis_name"] == "defensive_test" for e in registry["edges"])


def test_latest_result_used(setup_dirs):
    results_dir, hypotheses_dir = setup_dirs

    # Add an older result with different cover rates
    older = {**SAMPLE_RESULT}
    older["buckets"][0]["cover_rate"] = 0.9999
    (Path(results_dir) / "defensive_test_20260329_010000.json").write_text(json.dumps(older))

    registry = export_edges(results_dir=results_dir, hypotheses_dir=hypotheses_dir)
    q1 = next(e for e in registry["edges"] if e["bucket_label"] == "Q1")
    # Should use the later file (013058), not the older one
    assert q1["measurement"]["cover_rate"] == 0.5625


def test_writes_to_file(setup_dirs, tmp_path):
    results_dir, hypotheses_dir = setup_dirs
    output = tmp_path / "edges" / "nfl_edges.json"

    export_edges(
        results_dir=results_dir,
        hypotheses_dir=hypotheses_dir,
        output_path=str(output),
    )
    assert output.exists()
    written = json.loads(output.read_text())
    assert written["contract_version"] == "1.0.0"


def test_quality_score_mapping(setup_dirs):
    results_dir, hypotheses_dir = setup_dirs
    registry = export_edges(results_dir=results_dir, hypotheses_dir=hypotheses_dir)
    edge = registry["edges"][0]
    assert edge["quality"]["grade"] == "MEDIUM"
    assert edge["quality"]["composite_score"] == 2.3


def test_applicability_fields(setup_dirs):
    results_dir, hypotheses_dir = setup_dirs
    registry = export_edges(results_dir=results_dir, hypotheses_dir=hypotheses_dir)
    edge = registry["edges"][0]
    app = edge["applicability"]
    assert app["applies_to"] == "either"
    assert app["classification_type"] == "quartile"
    assert app["seasons_tested"] == [2022, 2023, 2024]


def test_infer_metric_direction():
    assert _infer_metric_direction("Q1", "quartile") == "higher_is_better"
    assert _infer_metric_direction("Q4", "quartile") == "lower_is_better"
    assert _infer_metric_direction("above_threshold", "binary") == "higher_is_better"
    assert _infer_metric_direction("below_threshold", "binary") == "lower_is_better"
    assert _infer_metric_direction("top_25pct", "percentile") == "higher_is_better"
    assert _infer_metric_direction("bottom_25pct", "percentile") == "lower_is_better"
