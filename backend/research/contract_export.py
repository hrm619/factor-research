"""Export Contract 2 (Validated Edge Registry) from pipeline results."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Filename pattern: {hypothesis_name}_{YYYYMMDD}_{HHMMSS}.json
_RESULT_PATTERN = re.compile(r"^(.+)_(\d{8}_\d{6})\.json$")


def export_edges(
    results_dir: str = "backend/research/results",
    hypotheses_dir: str = "backend/research/hypotheses",
    output_path: str | None = None,
    hypothesis_filter: str | None = None,
    domain: str = "nfl",
) -> dict:
    """Build the Contract 2 Validated Edge Registry from result JSON files.

    Scans results_dir for JSON files, groups by hypothesis name, takes the
    latest result per hypothesis. Loads corresponding YAML for metadata.
    """
    results_path = Path(results_dir)
    hypotheses_path = Path(hypotheses_dir)

    # Scan and group result files by hypothesis name
    grouped: dict[str, list[Path]] = {}
    for p in results_path.glob("*.json"):
        match = _RESULT_PATTERN.match(p.name)
        if not match:
            continue
        hyp_name = match.group(1)
        if hypothesis_filter and hyp_name != hypothesis_filter:
            continue
        grouped.setdefault(hyp_name, []).append(p)

    if not grouped:
        logger.warning("No result files found in %s", results_dir)
        return _build_registry(domain, [])

    # Process each hypothesis: take latest result, load YAML, build edges
    all_edges = []
    for hyp_name, result_files in sorted(grouped.items()):
        # Sort by filename (timestamp suffix) and take latest
        latest = sorted(result_files)[-1]
        logger.info("Processing %s from %s", hyp_name, latest.name)

        with open(latest) as f:
            result = json.load(f)

        # Load corresponding hypothesis YAML
        yaml_path = hypotheses_path / f"{hyp_name}.yaml"
        if not yaml_path.exists():
            logger.warning(
                "Hypothesis YAML not found for %s at %s, skipping",
                hyp_name, yaml_path,
            )
            continue

        with open(yaml_path) as f:
            definition = yaml.safe_load(f)

        edges = _build_edges(result, definition)
        all_edges.extend(edges)

    registry = _build_registry(domain, all_edges)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(registry, f, indent=2)
        logger.info("Edge registry written to %s", out)

    return registry


def _build_registry(domain: str, edges: list[dict]) -> dict:
    return {
        "contract_version": "1.0.0",
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "producer": "factor-research",
        "domain": domain,
        "edges": edges,
    }


def _build_edges(result: dict, definition: dict) -> list[dict]:
    """Build EdgeEntry objects from a result and its hypothesis definition."""
    hyp_name = result["hypothesis_name"]
    classification = definition.get("classification", {})
    quality_score = result.get("quality_score")
    comparison = result.get("comparison")

    edges = []
    for bucket in result.get("buckets", []):
        bucket_label = bucket["bucket"]
        edge_id = f"{hyp_name}__{bucket_label}"

        cover_rate = bucket["cover_rate"]
        baseline = 0.50  # ATS baseline
        edge_magnitude = round(cover_rate - baseline, 4)

        # Get effect_size_h from comparison if this bucket is involved
        effect_size_h = None
        if comparison:
            if bucket_label in (comparison.get("bucket_a"), comparison.get("bucket_b")):
                effect_size_h = comparison.get("effect_size_h")

        measurement = {
            "n": bucket["n"],
            "covers": bucket["covers"],
            "cover_rate": cover_rate,
            "baseline_rate": baseline,
            "edge_magnitude": edge_magnitude,
            "p_value": bucket.get("p_value"),
            "p_value_adjusted": bucket.get("p_value_adjusted"),
            "ci_lower": bucket.get("ci_lower"),
            "ci_upper": bucket.get("ci_upper"),
            "effect_size_h": effect_size_h,
        }

        quality = None
        if quality_score:
            quality = {
                "grade": quality_score["grade"],
                "composite_score": quality_score["composite"],
                "sample_size_score": quality_score["sample_size_score"],
                "significance_score": quality_score["significance_score"],
                "effect_size_score": quality_score["effect_size_score"],
                "consistency_score": quality_score["consistency_score"],
            }

        # Infer metric direction from bucket label
        metric_direction = _infer_metric_direction(
            bucket_label, classification.get("type", "quartile")
        )

        # Extract seasons from dataset_summary
        seasons_tested = result.get("dataset_summary", {}).get("seasons", [])

        # Build filters_applied from definition
        filters_applied = definition.get("filters")

        applicability = {
            "metric_direction": metric_direction,
            "applies_to": "either",
            "classification_type": classification.get("type", "quartile"),
            "seasons_tested": seasons_tested,
            "filters_applied": filters_applied,
        }

        edge = {
            "edge_id": edge_id,
            "hypothesis_name": hyp_name,
            "hypothesis_version": result.get("version", definition.get("version", "1.0.0")),
            "metric": classification.get("metric", ""),
            "bucket_label": bucket_label,
            "outcome_type": definition.get("outcome", "ats"),
            "lookback": definition.get("lookback", "season_to_date"),
            "measurement": measurement,
            "quality": quality,
            "applicability": applicability,
            "decay": None,
            "last_backtested": result.get("run_timestamp"),
            "provenance": None,
        }
        edges.append(edge)

    return edges


def _infer_metric_direction(bucket_label: str, cls_type: str) -> str:
    """Infer whether higher or lower metric values correspond to this bucket.

    For quartile: Q1 = top quartile = higher_is_better,
                  Q4 = bottom quartile = lower_is_better.
    For binary: above_threshold = higher_is_better.
    """
    if cls_type == "quartile":
        if bucket_label in ("Q1", "Q2"):
            return "higher_is_better"
        return "lower_is_better"
    elif cls_type == "binary":
        if "above" in bucket_label.lower():
            return "higher_is_better"
        return "lower_is_better"
    elif cls_type == "percentile":
        if "top" in bucket_label.lower():
            return "higher_is_better"
        return "lower_is_better"
    return "unknown"
