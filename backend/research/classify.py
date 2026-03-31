"""Stage 3: Classify team-game observations into buckets."""

from __future__ import annotations

import logging

import pandas as pd

from backend.research.models import HypothesisDefinition

logger = logging.getLogger(__name__)


def classify(df: pd.DataFrame, definition: HypothesisDefinition) -> pd.DataFrame:
    """Dispatch to the appropriate classification method based on the hypothesis."""
    cls_type = definition.classification.type
    metric = definition.classification.metric
    min_n = definition.min_sample_size

    if cls_type == "quartile":
        result = classify_quartile(df, metric)
    elif cls_type == "percentile":
        result = classify_percentile(
            df, metric,
            top_pct=definition.classification.top_pct,
            bottom_pct=definition.classification.bottom_pct,
        )
    elif cls_type == "binary":
        result = classify_binary(df, metric, threshold=definition.classification.threshold)
    elif cls_type == "custom":
        result = classify_custom(df, metric, boundaries=definition.classification.boundaries)
    else:
        raise ValueError(f"Unknown classification type: {cls_type}")

    # Log bucket sizes and warn on small buckets
    bucket_counts = result["bucket"].value_counts()
    for bucket_label, count in bucket_counts.items():
        if count < min_n:
            logger.warning(
                "Bucket '%s' has %d observations, below min_sample_size=%d",
                bucket_label, count, min_n,
            )
    logger.info("Classification complete: %s", dict(bucket_counts))

    return result


def classify_quartile(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Assign Q1-Q4 labels based on within-season quartile rank.

    Q1 = top quartile (highest metric value), Q4 = bottom quartile.
    """
    result = df.copy()
    # Rank within season using percentile rank (0-1), average method for ties
    result["_pct_rank"] = result.groupby("season")[metric].rank(pct=True, method="average")

    result["bucket"] = pd.cut(
        result["_pct_rank"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["Q4", "Q3", "Q2", "Q1"],
        include_lowest=True,
    )
    result.drop(columns=["_pct_rank"], inplace=True)
    return result


def classify_percentile(
    df: pd.DataFrame,
    metric: str,
    top_pct: float,
    bottom_pct: float,
) -> pd.DataFrame:
    """Assign top/middle/bottom labels based on within-season percentile thresholds."""
    result = df.copy()
    result["_pct_rank"] = result.groupby("season")[metric].rank(pct=True, method="average")

    top_threshold = 1.0 - (top_pct / 100.0)
    bottom_threshold = bottom_pct / 100.0

    def _label(rank: float) -> str:
        if rank > top_threshold:
            return "top"
        elif rank <= bottom_threshold:
            return "bottom"
        return "middle"

    result["bucket"] = result["_pct_rank"].apply(_label)
    result.drop(columns=["_pct_rank"], inplace=True)
    return result


def classify_binary(
    df: pd.DataFrame,
    metric: str,
    threshold: float,
) -> pd.DataFrame:
    """Assign above/below labels based on a threshold applied within-season."""
    result = df.copy()
    result["bucket"] = result[metric].apply(lambda x: "above" if x >= threshold else "below")
    return result


def classify_custom(
    df: pd.DataFrame,
    metric: str,
    boundaries: list[float],
) -> pd.DataFrame:
    """Assign bucket labels based on custom boundary values applied within-season.

    Boundaries define cut points. N boundaries produce N+1 buckets labeled B1, B2, ..., B(N+1)
    where B1 is the lowest range.
    """
    result = df.copy()
    n_buckets = len(boundaries) + 1
    labels = [f"B{i}" for i in range(1, n_buckets + 1)]

    bins = [float("-inf")] + sorted(boundaries) + [float("inf")]
    result["bucket"] = pd.cut(result[metric], bins=bins, labels=labels, include_lowest=True)
    return result
