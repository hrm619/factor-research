"""Stage 1: Load, validate, and resolve hypothesis definitions from YAML."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from backend.research.models import (
    HypothesisDefinition,
    ClassificationConfig,
    FilterConfig,
    TimeWindow,
    HypothesisValidationError,
)
from backend.research.metrics_catalog import MetricsCatalog

logger = logging.getLogger(__name__)

VALID_CLASSIFICATION_TYPES = {"quartile", "percentile", "binary", "custom"}
VALID_OUTCOMES = {"ats", "su", "ou"}
VALID_LOOKBACKS = {"season_to_date", "last_4"}
VALID_STATISTICAL_TESTS = {"binomial", "proportion_z", "chi_squared"}


def load_hypothesis(path: str) -> dict:
    """Load a hypothesis YAML file and return the raw dictionary."""
    filepath = Path(path)
    if not filepath.exists():
        raise HypothesisValidationError(f"Hypothesis file not found: {path}")
    if not filepath.suffix in (".yaml", ".yml"):
        raise HypothesisValidationError(f"Expected .yaml or .yml file, got: {filepath.suffix}")
    with open(filepath) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise HypothesisValidationError("YAML file must contain a mapping at the top level")
    return raw


def validate_hypothesis(raw: dict) -> HypothesisDefinition:
    """Validate a raw dictionary and return a HypothesisDefinition."""
    # Required string fields
    for field in ("hypothesis_name", "description", "version"):
        if field not in raw:
            raise HypothesisValidationError(f"Missing required field: {field}")
        if not isinstance(raw[field], str):
            raise HypothesisValidationError(f"Field '{field}' must be a string")

    # Metrics
    if "metrics" not in raw:
        raise HypothesisValidationError("Missing required field: metrics")
    if not isinstance(raw["metrics"], list) or not raw["metrics"]:
        raise HypothesisValidationError("Field 'metrics' must be a non-empty list")

    # Classification
    if "classification" not in raw:
        raise HypothesisValidationError("Missing required field: classification")
    classification = _validate_classification(raw["classification"])

    # Outcome
    if "outcome" not in raw:
        raise HypothesisValidationError("Missing required field: outcome")
    outcome = raw["outcome"]
    if outcome not in VALID_OUTCOMES:
        raise HypothesisValidationError(f"Invalid outcome '{outcome}'. Must be one of: {VALID_OUTCOMES}")

    # Lookback
    if "lookback" not in raw:
        raise HypothesisValidationError("Missing required field: lookback")
    lookback = raw["lookback"]
    if lookback not in VALID_LOOKBACKS:
        raise HypothesisValidationError(f"Invalid lookback '{lookback}'. Must be one of: {VALID_LOOKBACKS}")

    # Filters (optional)
    filters = _validate_filters(raw.get("filters"))

    # Time windows (optional)
    time_windows = []
    for tw in raw.get("time_windows", []):
        if "label" not in tw or "seasons" not in tw:
            raise HypothesisValidationError("Each time_window must have 'label' and 'seasons'")
        time_windows.append(TimeWindow(label=tw["label"], seasons=tw["seasons"]))

    # Statistical test
    stat_test = raw.get("statistical_test", "binomial")
    if stat_test not in VALID_STATISTICAL_TESTS:
        raise HypothesisValidationError(
            f"Invalid statistical_test '{stat_test}'. Must be one of: {VALID_STATISTICAL_TESTS}"
        )

    return HypothesisDefinition(
        hypothesis_name=raw["hypothesis_name"],
        description=raw["description"],
        version=raw["version"],
        metrics=raw["metrics"],
        classification=classification,
        outcome=outcome,
        filters=filters,
        lookback=lookback,
        min_sample_size=raw.get("min_sample_size", 50),
        statistical_test=stat_test,
        significance_threshold=raw.get("significance_threshold", 0.05),
        comparison_buckets=raw.get("comparison_buckets", []),
        output_breakdowns=raw.get("output_breakdowns", []),
        time_windows=time_windows,
    )


def resolve_metrics(definition: HypothesisDefinition, catalog: MetricsCatalog) -> HypothesisDefinition:
    """Verify all metrics referenced in the hypothesis exist in the catalog.

    Also checks compatibility between statistical test and classification type.
    Returns the definition unchanged if valid, raises on error.
    """
    # Validate all referenced metrics exist
    catalog.validate_metrics(definition.metrics)

    # Validate classification metric exists
    try:
        catalog.get_metric(definition.classification.metric)
    except KeyError as e:
        raise HypothesisValidationError(
            f"Classification metric '{definition.classification.metric}' not found in catalog"
        ) from e

    # Compatibility: chi_squared requires categorical classification (quartile or custom)
    if definition.statistical_test == "chi_squared":
        if definition.classification.type not in ("quartile", "custom"):
            raise HypothesisValidationError(
                "chi_squared test requires 'quartile' or 'custom' classification type, "
                f"got '{definition.classification.type}'"
            )

    logger.info("Hypothesis '%s' validated with %d metrics", definition.hypothesis_name, len(definition.metrics))
    return definition


def _validate_classification(raw: dict) -> ClassificationConfig:
    """Validate and build a ClassificationConfig from a raw dict."""
    if not isinstance(raw, dict):
        raise HypothesisValidationError("'classification' must be a mapping")
    if "type" not in raw:
        raise HypothesisValidationError("Classification missing required field: type")
    if "metric" not in raw:
        raise HypothesisValidationError("Classification missing required field: metric")

    cls_type = raw["type"]
    if cls_type not in VALID_CLASSIFICATION_TYPES:
        raise HypothesisValidationError(
            f"Invalid classification type '{cls_type}'. Must be one of: {VALID_CLASSIFICATION_TYPES}"
        )

    if cls_type == "percentile":
        if "top_pct" not in raw or "bottom_pct" not in raw:
            raise HypothesisValidationError("Percentile classification requires 'top_pct' and 'bottom_pct'")

    if cls_type == "binary":
        if "threshold" not in raw:
            raise HypothesisValidationError("Binary classification requires 'threshold'")

    if cls_type == "custom":
        if "boundaries" not in raw:
            raise HypothesisValidationError("Custom classification requires 'boundaries'")

    return ClassificationConfig(
        type=cls_type,
        metric=raw["metric"],
        top_pct=raw.get("top_pct"),
        bottom_pct=raw.get("bottom_pct"),
        threshold=raw.get("threshold"),
        boundaries=raw.get("boundaries"),
    )


def _validate_filters(raw: dict | None) -> FilterConfig:
    """Validate and build a FilterConfig from a raw dict."""
    if raw is None:
        return FilterConfig()
    if not isinstance(raw, dict):
        raise HypothesisValidationError("'filters' must be a mapping")
    return FilterConfig(
        seasons=raw.get("seasons"),
        weeks=raw.get("weeks"),
        game_type=raw.get("game_type"),
        exclude_week_1=raw.get("exclude_week_1", True),
    )
