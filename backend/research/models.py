"""Dataclasses and exceptions for the factor research framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


class HypothesisValidationError(Exception):
    """Raised when a hypothesis definition fails validation."""


class DataExtractionError(Exception):
    """Raised when data extraction returns zero rows or encounters an unrecoverable error."""


@dataclass(frozen=True)
class ClassificationConfig:
    type: str  # quartile, percentile, binary, custom
    metric: str
    top_pct: float | None = None
    bottom_pct: float | None = None
    threshold: float | None = None
    boundaries: list[float] | None = None


@dataclass(frozen=True)
class FilterConfig:
    seasons: list[int] | None = None
    weeks: list[int] | None = None
    game_type: list[str] | None = None
    exclude_week_1: bool = True


@dataclass(frozen=True)
class TimeWindow:
    label: str
    seasons: list[int]


@dataclass(frozen=True)
class HypothesisDefinition:
    hypothesis_name: str
    description: str
    version: str
    metrics: list[str]
    classification: ClassificationConfig
    outcome: str  # ats, su, ou
    filters: FilterConfig
    lookback: str  # season_to_date, last_4
    min_sample_size: int = 50
    statistical_test: str = "binomial"
    significance_threshold: float = 0.05
    comparison_buckets: list[str] = field(default_factory=list)
    output_breakdowns: list[str] = field(default_factory=list)
    time_windows: list[TimeWindow] = field(default_factory=list)


@dataclass
class BucketStats:
    bucket_label: str
    n: int
    covers: int
    cover_rate: float
    win_rate: float | None = None
    avg_margin_vs_spread: float | None = None
    std_margin_vs_spread: float | None = None
    p_value: float | None = None
    p_value_adjusted: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    significant: bool = False


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    significant: bool


@dataclass
class ComparisonResult:
    bucket_a: str
    bucket_b: str
    rate_difference: float
    p_value: float
    p_value_adjusted: float | None = None
    effect_size_h: float = 0.0
    significant: bool = False


@dataclass
class QualityScore:
    sample_size_score: float
    significance_score: float
    effect_size_score: float
    consistency_score: float
    composite: float
    grade: str  # HIGH, MEDIUM, LOW, INSUFFICIENT_DATA


@dataclass
class MeasurementResult:
    hypothesis_name: str
    run_timestamp: datetime
    dataset_summary: dict
    buckets: list[BucketStats]
    comparison: ComparisonResult | None = None
    quality_score: QualityScore | None = None
    time_window_results: list[MeasurementResult] = field(default_factory=list)
    trend_direction: str | None = None
    summary: str = ""
