"""Stage 5: Format and output measurement results."""

from __future__ import annotations

import json
import csv
import logging
from datetime import datetime, UTC
from pathlib import Path

from backend.research.models import MeasurementResult, HypothesisDefinition, BucketStats
from backend.research.statistical import fdr_correction

logger = logging.getLogger(__name__)

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def report(
    result: MeasurementResult,
    definition: HypothesisDefinition,
    output_dir: str = "backend/research/results",
    high_confidence: bool = False,
) -> None:
    """Print terminal report and write output files."""
    if high_confidence:
        result = apply_confidence_filter(result)

    # Generate summary
    result.summary = generate_summary(result, definition)

    # Terminal output
    terminal = format_terminal_report(result)
    print(terminal)

    # Write files
    path = write_report(result, definition, output_dir)
    logger.info("Report written to %s", path)


def generate_summary(result: MeasurementResult, definition: HypothesisDefinition) -> str:
    """Generate a plain-language summary of the finding."""
    parts = [f"Hypothesis: {definition.hypothesis_name}"]

    if result.comparison:
        c = result.comparison
        parts.append(
            f"{c.bucket_a} cover rate: {_get_bucket_rate(result, c.bucket_a):.1%}, "
            f"{c.bucket_b} cover rate: {_get_bucket_rate(result, c.bucket_b):.1%}, "
            f"difference: {c.rate_difference:+.1%} (p={c.p_value:.3f})"
        )

    if result.quality_score:
        parts.append(f"Quality: {result.quality_score.grade} ({result.quality_score.composite:.2f}/3.00)")

    if result.trend_direction:
        parts.append(f"Trend: {result.trend_direction}")

    return " | ".join(parts)


def format_terminal_report(result: MeasurementResult) -> str:
    """Format a color-coded terminal report."""
    lines = []
    lines.append(f"\n{BOLD}{'=' * 70}{RESET}")
    lines.append(f"{BOLD}  {result.hypothesis_name}{RESET}")
    lines.append(f"{'=' * 70}")

    # Dataset summary
    ds = result.dataset_summary
    lines.append(f"  Rows: {ds.get('total_rows', 'N/A')}  |  Seasons: {ds.get('seasons', 'N/A')}")
    lines.append("")

    # Bucket table
    lines.append(f"  {'Bucket':<8} {'N':>6} {'Cover%':>8} {'p-raw':>8} {'p-adj':>8} {'CI 95%':>16} {'Sig':>5}")
    lines.append(f"  {'-' * 60}")
    for b in result.buckets:
        sig_marker = _significance_color(b)
        ci_str = f"[{b.ci_lower:.3f}, {b.ci_upper:.3f}]" if b.ci_lower is not None else "N/A"
        p_raw = f"{b.p_value:.4f}" if b.p_value is not None else "N/A"
        p_adj = f"{b.p_value_adjusted:.4f}" if b.p_value_adjusted is not None else "N/A"
        lines.append(
            f"  {b.bucket_label:<8} {b.n:>6} {b.cover_rate:>7.1%} {p_raw:>8} {p_adj:>8} {ci_str:>16} {sig_marker}"
        )

    # Comparison
    if result.comparison:
        c = result.comparison
        lines.append("")
        lines.append(f"  Comparison: {c.bucket_a} vs {c.bucket_b}")
        lines.append(f"  Rate difference: {c.rate_difference:+.3f}  |  p-value: {c.p_value:.4f}  |  Cohen's h: {c.effect_size_h:.3f}")

    # Quality
    if result.quality_score:
        q = result.quality_score
        grade_color = GREEN if q.grade == "HIGH" else (YELLOW if q.grade == "MEDIUM" else RED)
        lines.append(f"\n  Quality: {grade_color}{BOLD}{q.grade}{RESET} ({q.composite:.2f}/3.00)")
        lines.append(f"  [size={q.sample_size_score:.2f} sig={q.significance_score:.2f} "
                      f"eff={q.effect_size_score:.2f} cons={q.consistency_score:.2f}]")

    # Time windows
    if result.time_window_results:
        lines.append(f"\n  Time Windows:")
        for wr in result.time_window_results:
            window_label = wr.dataset_summary.get("window", "?")
            if wr.comparison:
                lines.append(f"    {window_label}: {wr.comparison.bucket_a} {_get_bucket_rate(wr, wr.comparison.bucket_a):.1%} "
                              f"vs {wr.comparison.bucket_b} {_get_bucket_rate(wr, wr.comparison.bucket_b):.1%}")
            elif wr.buckets:
                rates = ", ".join(f"{b.bucket_label}={b.cover_rate:.1%}" for b in wr.buckets)
                lines.append(f"    {window_label}: {rates}")

        if result.trend_direction:
            lines.append(f"  Trend: {result.trend_direction}")

    lines.append(f"{'=' * 70}\n")
    return "\n".join(lines)


def format_json_report(result: MeasurementResult, definition: HypothesisDefinition) -> dict:
    """Format the full structured output as a JSON-serializable dict."""
    return {
        "hypothesis_name": result.hypothesis_name,
        "description": definition.description,
        "version": definition.version,
        "run_timestamp": result.run_timestamp.isoformat(),
        "dataset_summary": result.dataset_summary,
        "buckets": [_bucket_to_dict(b) for b in result.buckets],
        "comparison": _comparison_to_dict(result.comparison) if result.comparison else None,
        "quality_score": _quality_to_dict(result.quality_score) if result.quality_score else None,
        "time_windows": [
            {
                "window": wr.dataset_summary.get("window"),
                "buckets": [_bucket_to_dict(b) for b in wr.buckets],
                "comparison": _comparison_to_dict(wr.comparison) if wr.comparison else None,
            }
            for wr in result.time_window_results
        ],
        "trend_direction": result.trend_direction,
        "summary": result.summary,
    }


def apply_confidence_filter(result: MeasurementResult, threshold: str = "MEDIUM") -> MeasurementResult:
    """Filter to only HIGH/MEDIUM quality results."""
    passing_grades = {"HIGH"} if threshold == "HIGH" else {"HIGH", "MEDIUM"}
    if result.quality_score and result.quality_score.grade not in passing_grades:
        # Return result with empty buckets to indicate it didn't pass
        return MeasurementResult(
            hypothesis_name=result.hypothesis_name,
            run_timestamp=result.run_timestamp,
            dataset_summary=result.dataset_summary,
            buckets=[],
            summary=f"Filtered out: quality grade {result.quality_score.grade} below {threshold} threshold",
            quality_score=result.quality_score,
        )
    return result


def apply_cross_hypothesis_fdr(results: list[MeasurementResult]) -> list[MeasurementResult]:
    """Apply FDR correction across hypothesis-level results (Amendment A3 second pass)."""
    p_values = []
    indices = []
    for i, r in enumerate(results):
        if r.comparison and r.comparison.p_value is not None:
            p_values.append(r.comparison.p_value)
            indices.append(i)

    if len(p_values) < 2:
        return results

    corrected = fdr_correction(p_values)
    for idx, adj_p in zip(indices, corrected):
        results[idx].comparison.p_value_adjusted = adj_p
        results[idx].comparison.significant = adj_p < 0.05

    return results


def write_report(
    result: MeasurementResult,
    definition: HypothesisDefinition,
    output_dir: str,
) -> str:
    """Write JSON report and append to summary CSV. Returns JSON path."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = result.run_timestamp.strftime("%Y%m%d_%H%M%S")
    json_path = out_path / f"{result.hypothesis_name}_{timestamp}.json"

    report_data = format_json_report(result, definition)
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Append to summary CSV
    csv_path = out_path / "summary.csv"
    _append_summary_csv(csv_path, result)

    return str(json_path)


def _append_summary_csv(csv_path: Path, result: MeasurementResult) -> None:
    """Append a row to the summary CSV."""
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "hypothesis", "timestamp", "total_rows",
                "comparison_a", "comparison_b", "rate_diff",
                "p_value", "effect_size_h", "quality_grade", "quality_score",
            ])

        comp = result.comparison
        quality = result.quality_score
        writer.writerow([
            result.hypothesis_name,
            result.run_timestamp.isoformat(),
            result.dataset_summary.get("total_rows", ""),
            comp.bucket_a if comp else "",
            comp.bucket_b if comp else "",
            f"{comp.rate_difference:.4f}" if comp else "",
            f"{comp.p_value:.4f}" if comp else "",
            f"{comp.effect_size_h:.4f}" if comp else "",
            quality.grade if quality else "",
            f"{quality.composite:.3f}" if quality else "",
        ])


def _get_bucket_rate(result: MeasurementResult, label: str) -> float:
    for b in result.buckets:
        if b.bucket_label == label:
            return b.cover_rate
    return 0.0


def _significance_color(bucket: BucketStats) -> str:
    p = bucket.p_value_adjusted if bucket.p_value_adjusted is not None else bucket.p_value
    if p is None:
        return "  -"
    if p < 0.05:
        return f"{GREEN}  *{RESET}"
    elif p < 0.10:
        return f"{YELLOW}  ~{RESET}"
    return f"{RED}   {RESET}"


def _bucket_to_dict(b: BucketStats) -> dict:
    return {
        "bucket": b.bucket_label,
        "n": b.n,
        "covers": b.covers,
        "cover_rate": round(b.cover_rate, 4),
        "win_rate": round(b.win_rate, 4) if b.win_rate is not None else None,
        "avg_margin_vs_spread": round(b.avg_margin_vs_spread, 3) if b.avg_margin_vs_spread is not None else None,
        "p_value": round(b.p_value, 6) if b.p_value is not None else None,
        "p_value_adjusted": round(b.p_value_adjusted, 6) if b.p_value_adjusted is not None else None,
        "ci_lower": round(b.ci_lower, 4) if b.ci_lower is not None else None,
        "ci_upper": round(b.ci_upper, 4) if b.ci_upper is not None else None,
        "significant": b.significant,
    }


def _comparison_to_dict(c) -> dict:
    return {
        "bucket_a": c.bucket_a,
        "bucket_b": c.bucket_b,
        "rate_difference": round(c.rate_difference, 4),
        "p_value": round(c.p_value, 6),
        "p_value_adjusted": round(c.p_value_adjusted, 6) if c.p_value_adjusted is not None else None,
        "effect_size_h": round(c.effect_size_h, 4),
        "significant": c.significant,
    }


def _quality_to_dict(q) -> dict:
    return {
        "sample_size_score": q.sample_size_score,
        "significance_score": q.significance_score,
        "effect_size_score": q.effect_size_score,
        "consistency_score": q.consistency_score,
        "composite": q.composite,
        "grade": q.grade,
    }
