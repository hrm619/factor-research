"""Pipeline orchestrator: runs hypothesis from definition to report."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from sqlalchemy import Engine

from backend.research.db import get_engine
from backend.research.define import load_hypothesis, validate_hypothesis, resolve_metrics
from backend.research.extract import extract_data
from backend.research.classify import classify
from backend.research.measure import measure
from backend.research.report import report, apply_cross_hypothesis_fdr
from backend.research.metrics_catalog import MetricsCatalog
from backend.research.models import MeasurementResult

logger = logging.getLogger(__name__)


def run_hypothesis(
    hypothesis_path: str,
    db_url: str | None = None,
    output_dir: str = "backend/research/results",
    high_confidence: bool = False,
) -> MeasurementResult:
    """Run the full five-stage pipeline for a single hypothesis.

    Define → Extract → Classify → Measure → Report
    """
    start = time.time()
    engine = get_engine(db_url)
    catalog = MetricsCatalog()

    # Stage 1: Define
    logger.info("Stage 1: Loading hypothesis from %s", hypothesis_path)
    raw = load_hypothesis(hypothesis_path)
    definition = validate_hypothesis(raw)
    definition = resolve_metrics(definition, catalog)

    # Stage 2: Extract
    logger.info("Stage 2: Extracting data")
    df = extract_data(definition, engine)

    # Stage 3: Classify
    logger.info("Stage 3: Classifying observations")
    df = classify(df, definition)

    # Stage 4: Measure
    logger.info("Stage 4: Computing measurements")
    result = measure(df, definition)

    # Stage 5: Report
    logger.info("Stage 5: Generating report")
    report(result, definition, output_dir=output_dir, high_confidence=high_confidence)

    elapsed = time.time() - start
    logger.info(
        "Pipeline complete for '%s' in %.1fs. Quality: %s",
        definition.hypothesis_name,
        elapsed,
        result.quality_score.grade if result.quality_score else "N/A",
    )

    return result


def run_all_hypotheses(
    hypothesis_dir: str = "backend/research/hypotheses",
    db_url: str | None = None,
    output_dir: str = "backend/research/results",
    high_confidence: bool = False,
) -> list[MeasurementResult]:
    """Run all hypothesis YAML files in a directory.

    Applies cross-hypothesis FDR correction (Amendment A3 second pass).
    """
    hyp_path = Path(hypothesis_dir)
    yaml_files = sorted(hyp_path.glob("*.yaml")) + sorted(hyp_path.glob("*.yml"))

    if not yaml_files:
        logger.warning("No hypothesis files found in %s", hypothesis_dir)
        return []

    results = []
    for yaml_file in yaml_files:
        logger.info("Running hypothesis: %s", yaml_file.name)
        try:
            result = run_hypothesis(
                str(yaml_file), db_url=db_url, output_dir=output_dir,
                high_confidence=high_confidence,
            )
            results.append(result)
        except Exception as e:
            logger.error("Failed to run %s: %s", yaml_file.name, e)

    # Cross-hypothesis FDR correction
    if len(results) > 1:
        logger.info("Applying cross-hypothesis FDR correction across %d results", len(results))
        results = apply_cross_hypothesis_fdr(results)

    return results
