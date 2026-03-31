"""Import Contract 1 (Hypothesis Handoff) JSON and generate hypothesis YAML."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def import_contract(
    contract_path: str,
    output_dir: str = "backend/research/hypotheses",
    dry_run: bool = False,
) -> str:
    """Read a Contract 1 JSON file and generate a hypothesis YAML.

    Returns the path to the generated YAML file.
    Raises ValueError on validation errors.
    """
    path = Path(contract_path)
    if not path.exists():
        raise ValueError(f"Contract file not found: {contract_path}")

    with open(path) as f:
        contract = json.load(f)

    # Validate contract version
    version = contract.get("contract_version", "")
    if not version.startswith("1."):
        raise ValueError(
            f"Unsupported contract_version: {version!r}. "
            "Expected major version 1."
        )

    # Extract test_definition
    test_def = contract.get("test_definition")
    if not test_def:
        raise ValueError("Contract is missing 'test_definition' section")

    # Log rich_definition fields for audit
    rich_def = contract.get("rich_definition", {})
    rich_fields = list(rich_def.keys())
    logger.info(
        "Importing hypothesis '%s' from %s. "
        "rich_definition fields present: %s",
        test_def.get("hypothesis_name", "unknown"),
        contract.get("producer", "unknown"),
        rich_fields,
    )

    # Build YAML dict from test_definition
    yaml_dict = _build_yaml_dict(test_def)

    hypothesis_name = yaml_dict["hypothesis_name"]
    yaml_path = Path(output_dir) / f"{hypothesis_name}.yaml"

    if dry_run:
        logger.info("Dry run: would write to %s", yaml_path)
        # Still validate by parsing through define.py
        _validate_generated(yaml_dict)
        return str(yaml_path)

    # Write YAML
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    # Validate the generated YAML using existing validation
    _validate_generated(yaml_dict)

    logger.info("Hypothesis YAML written to %s", yaml_path)
    return str(yaml_path)


def _build_yaml_dict(test_def: dict) -> dict:
    """Map test_definition fields to YAML hypothesis format."""
    result = {
        "hypothesis_name": test_def["hypothesis_name"],
        "description": test_def["description"],
        "version": test_def.get("version", "1.0.0"),
        "metrics": test_def["metrics"],
        "classification": test_def["classification"],
        "outcome": test_def["outcome"],
        "lookback": test_def["lookback"],
    }

    # Optional fields
    if test_def.get("filters"):
        result["filters"] = test_def["filters"]

    if test_def.get("time_windows"):
        result["time_windows"] = test_def["time_windows"]

    if test_def.get("min_sample_size") is not None:
        result["min_sample_size"] = test_def["min_sample_size"]

    if test_def.get("statistical_test"):
        result["statistical_test"] = test_def["statistical_test"]

    if test_def.get("significance_threshold") is not None:
        result["significance_threshold"] = test_def["significance_threshold"]

    if test_def.get("comparison_buckets"):
        result["comparison_buckets"] = test_def["comparison_buckets"]

    return result


def _validate_generated(yaml_dict: dict) -> None:
    """Validate the generated YAML dict using factor-research's own validation."""
    from backend.research.define import validate_hypothesis, resolve_metrics
    from backend.research.metrics_catalog import MetricsCatalog

    definition = validate_hypothesis(yaml_dict)
    catalog = MetricsCatalog()
    resolve_metrics(definition, catalog)
